"""One shared selector entry: CPU geometry/decisions and native GPU integration."""
import ast
import importlib.util
import math
import io
import json
import shutil
import subprocess
import tempfile
from argparse import ArgumentParser
from pathlib import Path
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from arguments import OptimizationParams
from utils.run_metadata import build_resolved_config, collect_run_identity, write_run_metadata
from utils.graphics_utils import normal_from_depth_image, patch_warp
from utils.loss_utils import lncc
from utils import candidate_selector as selector


def options(**overrides):
    parser = ArgumentParser()
    group = OptimizationParams(parser)
    result = group.extract(parser.parse_args([]))
    result.enable_candidate_selector = True
    result.multi_view_pixel_noise_th = 100.0
    for key, value in overrides.items():
        setattr(result, key, value)
    return result


class AnalyticCamera:
    """Known intrinsics and poses, independent of the renderer's camera loader."""
    def __init__(self, uid=0, translation=0.0, width=32, height=32,
                 ncc_scale=1, device="cpu"):
        self.uid, self.ncc_scale = uid, ncc_scale
        self.image_width, self.image_height = width, height
        self.Fx = self.Fy = 40.0
        self.Cx = self.Cy = 15.5
        self.znear = .01
        az, ay = math.pi / 6, math.pi / 9
        rz = torch.tensor([[math.cos(az), -math.sin(az), 0],
                           [math.sin(az), math.cos(az), 0], [0, 0, 1.]])
        ry = torch.tensor([[math.cos(ay), 0, math.sin(ay)], [0, 1., 0],
                           [-math.sin(ay), 0, math.cos(ay)]])
        self.world_view_transform = torch.eye(4, device=device)
        self.world_view_transform[:3, :3] = (rz @ ry).to(device)
        self.world_view_transform[3, :3] = torch.tensor(
            [.2 + translation, -.1, .3], device=device)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.nearest_id = [1, 2]
        self.depth_dict = {"depth": torch.full((height, width), 2., device=device)}
        h, w = int(height / ncc_scale), int(width / ncc_scale)
        y, x = torch.meshgrid(torch.arange(h, device=device),
                              torch.arange(w, device=device), indexing="ij")
        # Plane at camera z=2: a +0.1 source translation gives +2 full pixels.
        x = x * ncc_scale - 20 * translation
        y = y * ncc_scale
        self.gray = (.45 + .2 * torch.sin(x * 1.1 + y * .3)
                     + .15 * torch.cos(x * .4 - y * .9))[None]

    def get_k(self, scale=1):
        return self.world_view_transform.new_tensor(
            [[40 / scale, 0, 15.5 / scale], [0, 40 / scale, 15.5 / scale], [0, 0, 1]])

    def get_inv_k(self, scale=1):
        return self.get_k(scale).inverse()

    def get_image(self):
        return self.gray.expand(3, -1, -1), self.gray


def production_model_method(name):
    # Execute the actual existing method body, without importing unavailable
    # GaussianModel CUDA dependencies; native class integration is tested below.
    path = Path(__file__).resolve().parents[1] / "scene/gaussian_model.py"
    model = next(n for n in ast.parse(path.read_text(encoding="utf-8")).body
                 if isinstance(n, ast.ClassDef) and n.name == "GaussianModel")
    method = next((n for n in model.body if isinstance(n, ast.FunctionDef)
                   and n.name == name), None)
    assert method is not None, "missing production method: " + name
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[method.name]


def production_depth_sampler():
    return production_model_method("get_points_depth_in_depth_map")


class ObservedCorrespondenceTests(unittest.TestCase):
    def test_actual_donor_grid_and_confidence_producer(self):
        import os
        from utils.observed_correspondence import host_to_normalized, query_matches
        source = os.environ.get('AMBISUR_ROMAV2_SOURCE')
        if not source:
            self.skipTest('set AMBISUR_ROMAV2_SOURCE for pinned donor functions')
        path=Path(os.environ.get('AMBISUR_ROMAV2_GEOMETRY',str(Path(source)/'src/romav2/geometry.py')))
        tree=ast.parse(path.read_text(encoding='utf-8'))
        names={'get_normalized_grid','to_normalized','to_pixel','prec_mat_from_prec_params'}
        scope={'torch':torch,'np':np,'device':torch.device('cpu')}
        exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)
            and n.name in names],type_ignores=[]),str(path),'exec'),scope)
        model=Path(source)/'src/romav2/romav2.py'
        fn=next(n for n in ast.parse(model.read_text(encoding='utf-8')).body
            if isinstance(n,ast.FunctionDef) and n.name=='_map_confidence')
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(model),'exec'),scope)
        h,w=12,20
        grid=scope['get_normalized_grid'](1,h,w)
        overlap,precision=scope['_map_confidence'](confidence=torch.zeros(1,h,w,4),threshold=None)
        raw={f'{key}_{direction}':value for direction in ('AB','BA')
             for key,value in [('warp',grid),('overlap',overlap),('precision',precision)]}
        xy=np.array([[5.,4.],[10.,8.]])
        np.testing.assert_allclose(host_to_normalized(xy,(h,w)),
            scope['to_normalized'](torch.tensor(xy+.5),H=h,W=w).numpy())
        result=query_matches(raw,xy,(h,w),(h,w))
        np.testing.assert_allclose(result['source_xy'],xy,atol=1e-5)
        self.assertTrue(result['forward_valid'].all());self.assertTrue(result['cycle_valid'].all())
        np.testing.assert_allclose(result['cycle_px'],0,atol=1e-5)
        self.assertEqual(result['precision_forward'].shape,(2,2,2))
        from scripts.evaluate_depth_sources import query_dense_matches
        dense=query_dense_matches(raw,(h,w),(h,w),chunk_size=17)
        all_xy=np.c_[np.arange(h*w)%w,np.arange(h*w)//w]
        direct=query_matches(raw,all_xy,(h,w),(h,w))
        for key,value in direct.items():np.testing.assert_array_equal(dense[key],value)

    def test_reverse_padding_is_invalid_not_zero_cycle(self):
        from utils.observed_correspondence import query_matches
        y,x=torch.meshgrid(torch.arange(4),torch.arange(4),indexing='ij')
        grid=torch.stack((2*(x+.5)/4-1,2*(y+.5)/4-1),-1)[None]
        # Source host pixel 0 is in image, but outside the centers of a 4px warp
        # representing an 8px host. Reverse interpolation would require padding.
        warp=grid.clone();warp[:]=torch.tensor([-.875,0.])
        raw={f'{k}_{d}':v for d in ('AB','BA') for k,v in
            [('warp',warp if d=='AB' else grid),('overlap',torch.ones(1,4,4,1)),
             ('precision',torch.eye(2).expand(1,4,4,2,2))]}
        r=query_matches(raw,np.array([[3.,3.]]),(8,8),(8,8))
        self.assertTrue(r['forward_valid'][0]);self.assertFalse(r['cycle_valid'][0])
        self.assertTrue(np.isnan(r['cycle_px'][0]))

    def test_normalized_triangulation_and_native_pixel_sensitivity(self):
        from utils.observed_correspondence import triangulate_evidence, project_depth
        k=np.array([[100.,0,50.],[0,100.,40.],[0,0,1.]])
        tr=np.eye(4);a=tr.copy();a[3,0]=-.3;b=tr.copy();b[3,0]=.4
        xy=np.array([[50.,40.],[60.,45.]])
        z=np.array([2.,3.])
        qs=np.stack([project_depth(xy,z,k,tr,k,s,(80,100),.01,.01)[0] for s in (a,b)],1)
        r=triangulate_evidence(xy,qs,np.ones((2,2),bool),k,tr,[k,k],[a,b],
            [(80,100)]*2,.01,[.01,.01])
        np.testing.assert_allclose(r['depth'],np.broadcast_to(z[:,None],(2,3)),atol=1e-9)
        self.assertTrue(r['valid'].all());self.assertTrue(r['sensitivity_all_valid'].all())
        self.assertTrue((r['sensitivity_max_z_m']>0).all())
        pe=r['point_reprojection_px'];re=r['ray_reprojection_px']
        np.testing.assert_allclose(pe[np.isfinite(pe)],0,atol=1e-9)
        np.testing.assert_allclose(re[np.isfinite(re)],0,atol=1e-9)
        self.assertTrue(np.isnan(pe[:,0,2]).all())  # unused camera stays absent
        bad=triangulate_evidence(xy,np.stack((xy,xy),1),np.ones((2,2),bool),k,tr,[k,k],[tr,tr],
            [(80,100)]*2,.01,[.01,.01])
        self.assertFalse(bad['valid'].any())

    def test_common_support_and_no_candidate_dependent_support_selection(self):
        from utils.observed_correspondence import compare_candidates
        costs=np.array([[[1,4],[2,3],[.5,2],[0,0]],
                        [[1,8],[2,4],[.5,9],[0,0]]],float)
        valid=np.ones_like(costs,bool);valid[0,3]=False;valid[1,2,0]=False
        out=compare_candidates(costs,valid)
        np.testing.assert_array_equal(out['common3_support_count'],[2,1])
        np.testing.assert_array_equal(out['common4_support_count'],[0,1])
        np.testing.assert_array_equal(out['mvs_selected'],[True,False])
        self.assertTrue(np.isnan(out['common4_means'][0]).all())

    def test_evidence_serializer_then_cached_gt_score_and_non_overwrite(self):
        from scripts.evaluate_depth_sources import (make_depth_map_record,save_npz,write_json,
            build_correspondence_record,correspondence_score,file_sha)
        from utils.observed_correspondence import project_depth
        cam=AnalyticCamera();srcs=[AnalyticCamera(translation=.1),AnalyticCamera(translation=-.1)]
        d=torch.full((32,32),2.);n=torch.zeros((3,32,32));n[2]=-1
        data=make_depth_map_record(cam,dict(plane_depth=d[None],rendered_normal=n),
            d,n,d,torch.ones_like(d,dtype=torch.bool),srcs,[d[None],d[None]])
        ids=np.array([15*32+15,16*32+16],np.int64);xy=np.c_[ids%32,ids//32]
        matches=[]
        for i in range(2):
            q,valid=project_depth(xy,np.full(2,2.),data['camera_k'],data['camera_transform'],
                data[f'support{i}_k'],data[f'support{i}_transform'],(32,32),.01,.01)
            matches.append(dict(source_xy=q,forward_valid=valid,cycle_px=np.zeros(2),
                cycle_valid=valid,overlap_forward=np.ones((2,1))))
        probe=dict(pixel_ids=ids,accepted=np.ones(2,bool),comparable=np.ones(2,bool),mean_costs=np.zeros((2,3)))
        record=build_correspondence_record(data,np.full((32,32),2.),probe,matches)
        self.assertFalse(record['mvs_selected'].any()) # equal candidates never win
        self.assertEqual(record['common3_support_count'].tolist(),[2,2])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'matches';out.mkdir();gt=root/'gt';gt.mkdir()
            save_npz(out/'view.npz',record)
            write_json(out/'correspondence_manifest.json',dict(schema_version=1,
                config={'seed':0,'descriptive':{'cycle_max_px':1.,'overlap_min':.5}},
                records=[dict(camera='view',camera_uid=1,file='view.npz',sha256=file_sha(out/'view.npz'))]))
            save_npz(gt/'view_gt_grid.npz',dict(depth=np.full((32,32),2.),
                valid=np.ones((32,32),bool),region=np.zeros((32,32),int)))
            args=SimpleNamespace(matches=out,gt_cache=gt,output=root/'scored')
            correspondence_score(args)
            report=json.loads((args.output/'results.json').read_text())
            self.assertTrue(report['rows']);self.assertEqual(report['pixel_count'],2)
            row=next(r for r in report['rows'] if r['kind']=='triangulation' and r['subset']=='joint' and r['region']=='all')
            self.assertEqual(row['count'],2)
            self.assertAlmostEqual(row['gain_current_mean_mm'],0,places=3)
            with self.assertRaises(FileExistsError):correspondence_score(args)


class FourSourceRoutingTests(unittest.TestCase):
    def test_camera_metadata_matches_native_convention_and_text_fallback(self):
        from scripts.evaluate_depth_sources import routing_camera_metadata
        from utils.graphics_utils import getWorld2View2
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'data';poses=source/'sparse/0';poses.mkdir(parents=True)
            # Noncentral COLMAP principal point is deliberately ignored by hostCamera.
            (poses/'cameras.txt').write_text('1 PINHOLE 64 48 60 62 27 19\n')
            (poses/'images.txt').write_text('1 1 0 0 0 0.2 -0.1 0 1 view.JPG\n\n')
            baseline=root/'baseline';baseline.mkdir()
            tr=getWorld2View2(np.eye(3),np.array([.2,-.1,0])).T
            c2w=np.linalg.inv(tr.T)
            (baseline/'cameras.json').write_text(json.dumps([dict(img_name='view',width=64,height=48,
                fx=60,fy=62,rotation=c2w[:3,:3].tolist(),position=c2w[:3,3].tolist())]))
            cams,receipt=routing_camera_metadata(source,baseline,(24,32))
            self.assertEqual(receipt['pose_format'],'text')
            np.testing.assert_allclose(cams['view']['transform'],tr,atol=1e-7)
            np.testing.assert_allclose(cams['view']['k'],[[30,0,16],[0,31,12],[0,0,1]])
            (poses/'images.txt').write_text('1 1 0 0 0 1 -0.1 0 1 view.JPG\n\n')
            with self.assertRaisesRegex(ValueError,'pose'):routing_camera_metadata(source,baseline,(24,32))

    def test_new_record_roundtrip_reuses_score_schema_and_abstain(self):
        from scripts.evaluate_depth_sources import build_routing_record,save_npz,write_json,file_sha,correspondence_score
        from utils.observed_correspondence import project_depth
        cam=AnalyticCamera();ids=np.array([15*32+15,16*32+16,17*32+17],np.int64)
        xy=np.c_[ids%32,ids//32];k=cam.get_k().numpy();tr=cam.world_view_transform.numpy()
        targets=np.array([[2.,2.5,1.8,np.nan],[2.,2.,2.,2.],[2.,2.5,1.8,1.9]])
        frozen=dict(pixel_ids=ids,reference_xy=xy,candidate_depths=targets,image_size=np.array([32,32]),
            camera_k=k,camera_transform=tr,znear=np.array(.01))
        matches=[];cameras=[]
        for shift in (.1,-.1,.3,-.4):
            src=AnalyticCamera(translation=shift);st=src.world_view_transform.numpy()
            q,valid=project_depth(xy,np.array([1.8,2.,1.8]),k,tr,k,st,(32,32),.01,.01)
            matches.append(dict(pixel_ids=ids,reference_xy=xy,source_xy=q,forward_valid=valid,
                cycle_valid=valid,cycle_px=np.zeros(3),overlap_forward=np.ones((3,1))))
            cameras.append(dict(k=k,transform=st,shape=(32,32),znear=.01))
        record=build_routing_record(frozen,matches,cameras,seed=8)
        np.testing.assert_array_equal(record['candidate_depths'],targets)
        self.assertEqual(record['all_candidate_costs_px'].shape,(3,4,4))
        self.assertTrue(record['new_best2_common3_domain'][0]);self.assertFalse(record['new_best2_common4_domain'][0])
        self.assertEqual(record['new_best2_common3_winner'][0],2)
        self.assertEqual(record['new_best2_common4_winner'][1],-1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'matches';gt=root/'gt';out.mkdir();gt.mkdir()
            save_npz(out/'view.npz',record)
            write_json(out/'correspondence_manifest.json',dict(schema_version=2,audit_type='four_source_routing',
                config={},records=[dict(camera='view',file='view.npz',sha256=file_sha(out/'view.npz'))]))
            save_npz(gt/'view_gt_grid.npz',dict(depth=np.full((32,32),1.8),valid=np.ones((32,32),bool),region=np.zeros((32,32),int)))
            args=SimpleNamespace(matches=out,gt_cache=gt,output=root/'score')
            correspondence_score(args)
            result=json.loads((args.output/'results.json').read_text());self.assertEqual(result['pixel_count'],3)
            row=next(r for r in result['rows'] if r['kind']=='routing_policy' and r['source']=='selected'
                and r['subset']=='new_best2_common3_full_frozen' and r['region']=='all')
            self.assertEqual(row['count'],3);self.assertEqual(row['selected_count'],2)
            self.assertGreater(row['gain_da3_mean_mm'],0)
            self.assertTrue((args.output/'metrics.csv').is_file())
            with self.assertRaises(FileExistsError):correspondence_score(args)

    def test_cached_pair_validation_checks_ids_and_reference_coordinates(self):
        from scripts.evaluate_depth_sources import validate_routing_pair
        frozen=dict(pixel_ids=np.array([2,4],np.int64),reference_xy=np.array([[2.,0.],[4.,0.]]))
        pair=dict(**frozen,source_xy=np.array([[1.,0.],[3.,0.]]),forward_valid=np.ones(2,bool),
            cycle_valid=np.ones(2,bool),cycle_px=np.zeros(2),overlap_forward=np.ones((2,1)))
        validate_routing_pair(pair,frozen)
        pair['pixel_ids']=np.array([4,2],np.int64)
        with self.assertRaisesRegex(ValueError,'pixel'):validate_routing_pair(pair,frozen)

    def test_pair_geometry_uses_actual_rays_not_camera_heading(self):
        from utils.observed_correspondence import pair_geometry, project_depth
        k=np.array([[100.,0,50.],[0,100.,40.],[0,0,1.]])
        tr=np.eye(4);src=tr.copy();src[3,0]=-.5
        xy=np.array([[50.,40.],[60.,45.]])
        q,valid=project_depth(xy,np.array([2.,3.]),k,tr,k,src,(80,100),.01,.01)
        g=pair_geometry(xy,q,valid,k,tr,k,src,(80,100),.01,.01)
        np.testing.assert_allclose(g['depth'],[2.,3.],atol=1e-9)
        self.assertTrue(g['valid'].all());self.assertTrue((g['angle_deg']>5).all())
        np.testing.assert_allclose(g['epipolar_px'],0,atol=1e-9)
        self.assertTrue(g['sensitivity_valid'].all())
        self.assertTrue((g['sensitivity_max_z_m']>0).all())
        bad=pair_geometry(xy,xy,np.ones(2,bool),k,tr,k,tr,(80,100),.01,.01)
        self.assertFalse(bad['valid'].any())

    def test_support_selection_shared_domains_missing_spread_and_ties(self):
        from utils.observed_correspondence import routing_evidence
        # Four candidates, four supports. New supports prefer MVS, old preferDA3.
        costs=np.tile(np.array([[3,3,4,4],[1,1,3,3],[2,2,1,1],[5,5,2,2.]]),(6,1,1))
        valid=np.ones_like(costs,bool);valid[1,3]=False
        valid[2,2,2]=False # selected evidence cannot be shrunk to hide bad MVS
        costs[3]=1. # exact ties abstain
        ev={k:np.ones((6,4),bool) for k in ('forward_valid','cycle_valid','pair_valid')}
        ev.update(cycle_px=np.zeros((6,4)),overlap=np.ones((6,4)),
                  angle_deg=np.tile([1,2,8,8.],(6,1)))
        ev['cycle_px'][4,2:]=2.;ev['pair_valid'][5,1:]=False
        out=routing_evidence(costs,valid,ev,seed=3)
        np.testing.assert_array_equal(out['new_best2_support_indices'][0],[2,3])
        self.assertEqual(out['old_first2_common4_winner'][0],1)
        self.assertEqual(out['new_best2_common4_winner'][0],2)
        self.assertTrue(out['new_best2_common3_domain'][1])
        self.assertFalse(out['new_best2_common4_domain'][1])
        self.assertFalse(out['new_best2_common3_domain'][2])
        self.assertEqual(out['new_best2_common3_winner'][3],-1)
        np.testing.assert_array_equal(out['new_best2_support_indices'][4],[1,0])
        self.assertFalse(out['new_best2_common3_domain'][5])
        self.assertEqual(out['new_best2_common3_winner'][5],-1)
        for label in ('old_first2','new_best2'):
            for count in (3,4):
                p=f'{label}_common{count}';domain=out[p+'_domain']
                self.assertEqual(sorted(out[p+'_winner'][domain]),sorted(out[p+'_random_winner'][domain]))
                np.testing.assert_array_equal(out[p+'_random_winner'],routing_evidence(costs,valid,ev,seed=3)[p+'_random_winner'])
        np.testing.assert_array_equal(out['support_eligible'][2],np.ones(4,bool))


class DenseRoutingTests(unittest.TestCase):
    @staticmethod
    def evidence(n, s):
        ev={k:np.ones((n,s),bool) for k in ('forward_valid','cycle_valid','pair_valid')}
        ev.update(cycle_px=np.zeros((n,s)),overlap=np.ones((n,s)),
                  angle_deg=np.broadcast_to(np.arange(s,0,-1),(n,s)).copy())
        return ev

    def test_available_sources_shared_single_support_and_priority(self):
        from utils.observed_correspondence import dense_routing_evidence
        z=np.tile([2.,2.5,1.8,1.9],(5,1));z[0,3]=np.nan
        costs=np.tile(np.array([[3.,3.],[2.,2.],[1.,1.],[4.,4.]]),(5,1,1))
        valid=np.ones_like(costs,bool)
        valid[1,3,1]=False # four sources on first beats three on both
        valid[2,3,0]=False # four on second beats three on both
        ev=self.evidence(5,2);ev['forward_valid'][3,1]=False
        costs[4,:,0]=[.1,.2,.3,.4];costs[4,:,1]=[9.,4.,2.,0.]
        r=dense_routing_evidence(costs,valid,ev,z,seed=4)
        np.testing.assert_array_equal(r['support_count'],[2,1,1,1,2])
        np.testing.assert_array_equal(r['participating_source_bits'],[7,15,15,15,15])
        np.testing.assert_array_equal(r['support_indices'][1],[0,-1])
        np.testing.assert_array_equal(r['support_indices'][2],[1,-1])
        self.assertEqual(r['winner'][4],3) # shared mean, never each source's best view
        np.testing.assert_allclose(r['hybrid_depth'],[1.8,1.8,1.8,1.8,1.9])
        changed=dense_routing_evidence(-costs,valid,ev,z,seed=4)
        np.testing.assert_array_equal(changed['support_indices'],r['support_indices'])
        np.testing.assert_array_equal(changed['participating_source_bits'],r['participating_source_bits'])

    def test_abstain_invalid_da3_and_valid_random_source_quotas(self):
        from utils.observed_correspondence import dense_routing_evidence
        z=np.tile([2.,2.5,1.8,np.nan],(8,1));z[1,1]=np.nan
        costs=np.ones((8,4,1));costs[2:,2]=0
        valid=np.ones_like(costs,bool);valid[3:5,1]=False
        z[4,1]=np.nan;ev=self.evidence(8,1);ev['forward_valid'][0]=False
        valid[5,:3]=False
        r=dense_routing_evidence(costs,valid,ev,z,seed=1)
        self.assertEqual(r['winner'][0],-1);self.assertEqual(r['winner'][1],-1)
        self.assertTrue(np.isnan(r['hybrid_depth'][1]))
        self.assertEqual(r['fallback_reason'][0],1)
        self.assertEqual(r['fallback_reason'][1],3)
        self.assertEqual(r['fallback_reason'][5],2)
        labels=r['random_winner'];chosen=labels>=0
        self.assertTrue(r['participating_sources'][np.flatnonzero(chosen),labels[chosen]].all())
        np.testing.assert_array_equal(labels,dense_routing_evidence(costs,valid,ev,z,seed=1)['random_winner'])
        # Native DA3 validity is part of the exchangeability strata.
        for bits in np.unique(r['original_source_bits']):
            use=r['original_source_bits']==bits
            np.testing.assert_array_equal(np.sort(labels[use]),np.sort(r['winner'][use]))

    def test_geometry_lightweight_retains_exact_core(self):
        from utils.observed_correspondence import pair_geometry,project_depth
        k=np.array([[100.,0,50.],[0,100.,40.],[0,0,1.]])
        tr=np.eye(4);src=tr.copy();src[3,0]=-.5
        xy=np.array([[50.,40.],[60.,45.]])
        q,v=project_depth(xy,np.array([2.,3.]),k,tr,k,src,(80,100),.01,.01)
        full=pair_geometry(xy,q,v,k,tr,k,src,(80,100),.01,.01)
        light=pair_geometry(xy,q,v,k,tr,k,src,(80,100),.01,.01,compute_sensitivity=False)
        for key in ('depth','valid','angle_deg','raw_angle_deg','epipolar_px'):
            np.testing.assert_array_equal(full[key],light[key])
        self.assertNotIn('sensitivity_max_z_m',light)

    def test_dense_grid_chunk_queries_cover_real_donor_output(self):
        from scripts.evaluate_depth_sources import query_dense_matches
        from utils.observed_correspondence import query_matches
        h,w=5,7;y,x=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij')
        grid=torch.stack((2*(x+.5)/w-1,2*(y+.5)/h-1),-1)[None]
        raw={f'{k}_{d}':v for d in ('AB','BA') for k,v in
             [('warp',grid),('overlap',torch.ones(1,h,w,1)),('precision',torch.eye(2).expand(1,h,w,2,2))]}
        xy=np.c_[np.arange(h*w)%w,np.arange(h*w)//w]
        result=query_dense_matches(raw,(h,w),(h,w),chunk_size=8)
        expected=query_matches(raw,xy,(h,w),(h,w))
        np.testing.assert_array_equal(result['pixel_ids'],np.arange(h*w))
        for key,value in expected.items():np.testing.assert_array_equal(result[key],value)
        with self.assertRaises(ValueError):query_dense_matches(raw,(h,w),(h,w),chunk_size=0)

    def test_dense_record_real_serializer_score_domains_and_png(self):
        from scripts.evaluate_depth_sources import (build_routing_record,save_npz,write_json,file_sha,
            correspondence_score,write_dense_maps)
        from utils.observed_correspondence import project_depth
        cam=AnalyticCamera();h=w=32;ids=np.arange(h*w,dtype=np.int64);xy=np.c_[ids%w,ids//w]
        k=cam.get_k().numpy();tr=cam.world_view_transform.numpy()
        targets=np.tile([2.,2.5,1.8,np.nan],(h*w,1));targets[15*w+15,0]=np.nan
        targets[16*w+16,1]=np.nan
        frozen=dict(pixel_ids=ids,reference_xy=xy,candidate_depths=targets,image_size=np.array([w,h]),
            camera_k=k,camera_transform=tr,znear=np.array(.01))
        src=AnalyticCamera(translation=.1);st=src.world_view_transform.numpy()
        q,valid=project_depth(xy,np.full(h*w,1.8),k,tr,k,st,(h,w),.01,.01)
        match=dict(pixel_ids=ids,reference_xy=xy,source_xy=q,forward_valid=valid,
                   cycle_valid=valid,cycle_px=np.zeros(h*w),overlap_forward=np.ones((h*w,1)))
        record=build_routing_record(frozen,[match],[dict(k=k,transform=st,shape=(h,w),znear=.01)],dense=True)
        self.assertEqual(record['winner'][15*w+15],2)
        self.assertEqual(record['support_count'][15*w+15],1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);matches=root/'matches';gt=root/'gt';matches.mkdir();gt.mkdir()
            save_npz(matches/'view.npz',record);write_dense_maps(matches,'view',record)
            from PIL import Image
            with Image.open(matches/'view_sources.png') as im:self.assertEqual(im.size,(w,h))
            write_json(matches/'correspondence_manifest.json',dict(schema_version=3,audit_type='four_source_dense',
                config={},records=[dict(camera='view',file='view.npz',sha256=file_sha(matches/'view.npz'))]))
            save_npz(gt/'view_gt_grid.npz',dict(depth=np.full((h,w),1.8),valid=np.ones((h,w),bool),region=np.zeros((h,w),int)))
            args=SimpleNamespace(matches=matches,gt_cache=gt,output=root/'score')
            correspondence_score(args)
            result=json.loads((args.output/'results.json').read_text())
            self.assertEqual(result['pixel_count'],h*w)
            primary=next(r for r in result['rows'] if r['kind']=='dense_policy' and r['reference']=='da3'
                         and r['subset']=='full_image' and r['region']=='all')
            self.assertEqual(primary['count'],h*w-1) # missing current must not reduce it
            current=next(r for r in result['rows'] if r['kind']=='dense_policy' and r['reference']=='current'
                         and r['subset']=='full_image' and r['region']=='all')
            self.assertEqual(current['count'],h*w-1) # missing DA3 must not reduce it
            self.assertAlmostEqual(primary['reference_mae_mm']-primary['hybrid_mae_mm'],primary['gain_reference_mean_mm'])
            import csv
            with (args.output/'metrics.csv').open(encoding='utf-8-sig',newline='') as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),len(result['rows']))
            with self.assertRaises(FileExistsError):correspondence_score(args)

    def test_dense_preflight_reads_actual_manifests_poses_and_source_maps(self):
        from PIL import Image
        from scripts.evaluate_depth_sources import dense_routing_preflight,save_npz,write_json,file_sha
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';base=root/'base';old=root/'old';cap=root/'cap';mvs=root/'mvs'
            for d in (source/'sparse/0',source/'images',base,old,cap,mvs):d.mkdir(parents=True)
            names=['ref','a','b'];poses=[];lines=[]
            for i,name in enumerate(names):
                t=(i-1)*.1;poses.append(dict(img_name=name,width=32,height=32,fx=40,fy=40,
                    rotation=np.eye(3).tolist(),position=[-t,0,0]))
                lines.append(f'{i+1} 1 0 0 0 {t} 0 0 1 {name}.JPG\n\n')
                Image.new('RGB',(32,32),(90,100,110)).save(source/'images'/(name+'.JPG'))
            (source/'sparse/0/cameras.txt').write_text('1 PINHOLE 32 32 40 40 16 16\n')
            (source/'sparse/0/images.txt').write_text(''.join(lines))
            write_json(base/'cameras.json',poses)
            (base/'multi_view.json').write_text(json.dumps(dict(ref_name='ref',nearest_name=['a','b']))+'\n')
            metadata={x:file_sha(base/x) for x in ('cameras.json','multi_view.json')}
            k=np.array([[40,0,16],[0,40,16],[0,0,1]],np.float32);tr=np.eye(4,dtype=np.float32);tr[3,0]=-.1
            data=dict(current_depth=np.full((32,32),2.),da3_depth=np.full((32,32),2.5),
                spread_depth=np.full((32,32),1.9),spread_valid=np.ones((32,32),bool),
                current_normal=np.zeros((3,32,32)),prior_normal=np.zeros((3,32,32)),
                camera_k=k,camera_transform=tr,image_size=np.array([32,32]),znear=np.array(.01))
            for i in range(2):data.update({f'support{i}_depth':np.ones((32,32)),f'support{i}_k':k})
            save_npz(cap/'depth.npz',data);save_npz(mvs/'mvs.npz',dict(depth=np.full((32,32),1.8),camera_k=k))
            write_json(cap/'capture_manifest.json',dict(snapshot_inputs={'native7k':{'metadata_sha256':metadata}},
                records=[dict(camera='ref',snapshot_iteration=7000,camera_uid=1,supports=['a','b'],depth_file='depth.npz')]))
            write_json(mvs/'inference_manifest.json',dict(metadata_sha256=metadata,records=[dict(camera='ref',file='mvs.npz')]))
            save_npz(old/'ref.npz',dict(camera_k=k,camera_transform=tr))
            model={'commit':'example'}
            write_json(old/'input_receipt.json',dict(images={n:dict(sha256=file_sha(source/'images'/(n+'.JPG'))) for n in names}))
            write_json(old/'correspondence_manifest.json',dict(model_commit='example',config={'model':model},
                input_receipt_sha256=file_sha(old/'input_receipt.json'),records=[dict(camera='ref',file='ref.npz',
                sha256=file_sha(old/'ref.npz'),supports=['a','b'])]))
            cfg=dict(model=model,views=[dict(reference='ref',supports=['a','b'])],native_shape=[32,32],rgb_size=[32,32],
                pair_count=2,query_chunk_size=17,baseline_metadata_sha256=metadata,
                existing_manifest_sha256=file_sha(old/'correspondence_manifest.json'),
                capture_manifest_sha256=file_sha(cap/'capture_manifest.json'),mvs_manifest_sha256=file_sha(mvs/'inference_manifest.json'),
                source_arrays={'ref':dict(depth_file='depth.npz',depth_sha256=file_sha(cap/'depth.npz'),
                    mvs_file='mvs.npz',mvs_sha256=file_sha(mvs/'mvs.npz'))})
            prepared,cameras,receipt=dense_routing_preflight(cfg,source,old,base,cap,mvs)
            self.assertEqual(len(receipt['images']),3);self.assertEqual(receipt['pair_count'],2)
            self.assertEqual(prepared[0]['frozen']['candidate_depths'].shape,(1024,4))
            self.assertEqual(prepared[0]['frozen']['pixel_ids'][-1],1023)
            # Exercise the actual dispatch -> dense query -> pair cache -> manifest.
            # Only GPU forward/bootstrap is replaced; capture meta has no sha256.
            from contextlib import ExitStack,nullcontext
            from scripts import evaluate_depth_sources as audit
            vendor=root/'vendor';vendor.mkdir()
            cfg.update(audit_type='four_source_dense',seed=0,support_quality=dict(cycle_max_px=1.,overlap_min=.5))
            write_json(root/'config.json',cfg)
            args=SimpleNamespace(config=root/'config.json',source=source,existing=old,baseline=base,
                capture=cap,mvs=mvs,vendor=vendor,output=root/'dense')
            y,x=torch.meshgrid(torch.arange(32),torch.arange(32),indexing='ij')
            grid=torch.stack((2*(x+.5)/32-1,2*(y+.5)/32-1),-1)[None]
            raw={f'{k}_{d}':v for d in ('AB','BA') for k,v in
                [('warp',grid),('overlap',torch.ones(1,32,32,1)),('precision',torch.eye(2).expand(1,32,32,2,2))]}
            calls=[]
            def forward(a,b):calls.append((a,b));return raw
            original_check=subprocess.check_output
            def check(command,*a,**kw):
                if kw.get('cwd')==vendor.resolve():return 'example\n' if command[-1]=='HEAD' else ''
                return original_check(command,*a,**kw)
            with ExitStack() as stack:
                stack.enter_context(patch.object(audit.subprocess,'check_output',side_effect=check))
                stack.enter_context(patch.object(audit,'offline_correspondence_hub',return_value=nullcontext()))
                stack.enter_context(patch.object(audit,'load_correspondence_model',return_value=SimpleNamespace(match=forward)))
                for method in ('reset_peak_memory_stats','synchronize','max_memory_allocated','max_memory_reserved'):
                    stack.enter_context(patch.object(torch.cuda,method,return_value=0))
                audit.routing_match(args)
            manifest=json.loads((args.output/'correspondence_manifest.json').read_text())
            self.assertEqual(len(calls),2);self.assertEqual(manifest['audit_type'],'four_source_dense')
            self.assertEqual(manifest['records'][0]['original_record_sha256'],file_sha(old/'ref.npz'))
            (cap/'depth.npz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'depth'):dense_routing_preflight(cfg,source,old,base,cap,mvs)

    def test_dense_model_bootstrap_is_cache_only_and_restores_hub(self):
        from scripts.evaluate_depth_sources import offline_correspondence_hub
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            hub=Path(tmp);checkpoint=hub/'checkpoints/romav2.0.1.pt'
            checkpoint.parent.mkdir();checkpoint.write_bytes(b'abc')
            dino=hub/'facebookresearch_dinov3_fixed';dino.mkdir()
            (dino/'hubconf.py').write_text('def marker(**kwargs): return kwargs\n')
            cfg=dict(dinov3_commit='fixed',model=dict(weights_bytes=3,weights_sha256=hashlib.sha256(b'abc').hexdigest()))
            before_load=torch.hub.load;before_download=torch.hub.download_url_to_file
            with patch('torch.hub.get_dir',return_value=str(hub)):
                with offline_correspondence_hub(cfg):
                    result=torch.hub.load('facebookresearch/dinov3:fixed','marker',skip_validation=True,pretrained=False)
                    self.assertEqual(result,dict(pretrained=False))
                    with self.assertRaisesRegex(RuntimeError,'offline'):torch.hub.download_url_to_file('https://unused','unused')
                self.assertIs(torch.hub.load,before_load);self.assertIs(torch.hub.download_url_to_file,before_download)
                checkpoint.unlink()
                with self.assertRaises(FileNotFoundError):
                    with offline_correspondence_hub(cfg):pass

    def test_random_fallback_preserves_positive_subnear_da3_coverage(self):
        from utils.observed_correspondence import dense_routing_evidence
        z=np.tile([2.,np.nan,1.8,np.nan],(12,1));z[:6,1]=.005
        costs=np.ones((12,4,1));costs[6:,2]=0
        r=dense_routing_evidence(costs,np.ones_like(costs,bool),self.evidence(12,1),z,seed=1)
        self.assertTrue(np.isfinite(r['hybrid_depth']).all())
        self.assertTrue(np.isfinite(r['random_hybrid_depth']).all())


class DepthSourceAuditTests(unittest.TestCase):
    def test_mvs_support_cap_preserves_real_saved_order_without_padding(self):
        from scripts.evaluate_depth_sources import select_mvs_supports
        names = ['s'+str(i) for i in range(8)]
        cameras = dict.fromkeys(['many','five','four',*names])
        saved = {'many':names.copy(), 'five':names[:5], 'four':names[:4]}
        result = select_mvs_supports(['many','five','four'], saved, cameras, 7)
        self.assertEqual(result, {'many':names[:7], 'five':names[:5], 'four':names[:4]})
        self.assertEqual(saved['many'], names)
        self.assertEqual([len(set(result[n])) for n in result], [7,5,4])

    def test_mvs_support_preflight_rejects_empty_or_invalid_saved_list(self):
        from scripts.evaluate_depth_sources import select_mvs_supports
        for saved in ([], ['ref'], ['s','s'], ['missing']):
            with self.subTest(saved=saved), self.assertRaisesRegex(ValueError, 'ref'):
                select_mvs_supports(['ref'], {'ref':saved}, {'ref':{},'s':{}}, 7)

    def test_actual_donor_config_does_not_parse_outer_audit_arguments(self):
        import contextlib
        import importlib
        import os
        import sys
        vendor = os.environ.get('AMBISUR_MVSANYWHERE_SOURCE')
        if not vendor:
            self.skipTest('set AMBISUR_MVSANYWHERE_SOURCE for the pinned donor interface test')
        vendor = Path(vendor)
        sha = subprocess.check_output(['git', '-C', str(vendor), 'rev-parse', 'HEAD'], text=True).strip()
        from scripts.evaluate_depth_sources import VENDOR_COMMIT
        self.assertEqual(sha, VENDOR_COMMIT)
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'scripts/evaluate_depth_sources.py').read_text(encoding='utf-8'))
        infer = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'infer')
        adapter = next(n for n in infer.body if isinstance(n, ast.ClassDef) and n.name == 'CachedInference')
        init = next(n for n in adapter.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        call = next(n for n in init.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                    and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == 'parse_and_merge_options')
        argv = ['evaluate_depth_sources.py', 'infer', '--source', '/readonly/dataset', '--device', 'cuda']
        with patch.object(sys, 'path', [str(vendor / 'src'), *sys.path]):
            donor = importlib.import_module('mvsanywhere.options')
            handler = donor.OptionsHandler()
            with patch.object(sys, 'argv', argv.copy()), contextlib.redirect_stderr(io.StringIO()):
                exec(compile(ast.Module(body=[call], type_ignores=[]), '<actual adapter options call>', 'exec'),
                     {'handler': handler, 'vendor': vendor})
                self.assertEqual(sys.argv, argv)
            self.assertEqual(handler.options.image_width, 640)
            self.assertEqual(handler.options.image_height, 480)
            self.assertEqual(handler.options.image_encoder_name, 'dinov2_vitb14')

    def test_common_mean_admission_not_opportunistic_best_view(self):
        from scripts.evaluate_depth_sources import common_mean_admission
        z = np.array([[.4, .4], [.4, .4], [.4, .4]])
        p = np.array([[.3, .3], [.3, .3], [.3, .3]])
        q = np.array([[.1, .9], [.2, .2], [.1, .1]])
        valid = np.ones_like(z, bool)
        qvalid = valid.copy(); qvalid[2] = False
        accept, comparable, means = common_mean_admission(z,p,q,valid,valid,qvalid)
        np.testing.assert_array_equal(accept,[False,True,False])
        np.testing.assert_array_equal(comparable,[True,True,False])
        self.assertAlmostEqual(means[0,2],.5)

    def test_target_replacement_gain_and_missing_values(self):
        from scripts.evaluate_depth_sources import paired_target_stats
        result = paired_target_stats(np.array([3.,1.,1.]),np.array([1.,2.,1.]),
            np.array([2.,1.,np.nan]),np.ones(3),np.ones(3,bool),np.ones(3,bool))
        self.assertEqual(result['count'],2)
        self.assertAlmostEqual(result['gain_current_mean_mm'],500.)
        self.assertAlmostEqual(result['gain_da3_mean_mm'],0.)
        self.assertAlmostEqual(result['hybrid_da3_mean_gain_mm'],0.)
        self.assertEqual(result['improved_da3_fraction'],.5)

    def test_depth_map_producer_real_tensor_shapes(self):
        from scripts.evaluate_depth_sources import make_depth_map_record, validate_depth_map
        cam = AnalyticCamera()
        depth = torch.full((32,32),2.)
        normal = torch.zeros((3,32,32));normal[2] = -1
        preview = dict(plane_depth=depth[None],rendered_normal=normal,
                       rendered_distance=depth[None])
        data = make_depth_map_record(cam,preview,depth,normal,depth,
            torch.ones_like(depth,dtype=torch.bool),[cam,cam],[depth[None],depth[None]])
        validate_depth_map(data)
        self.assertEqual(data['current_depth'].shape,(32,32))
        self.assertEqual(data['support0_depth'].dtype,np.float32)
        data['spread_valid'] = np.ones((2,2),bool)
        with self.assertRaises(ValueError):validate_depth_map(data)

    def test_inference_camera_is_c2w_and_intrinsics_copy(self):
        from scripts.evaluate_depth_sources import inference_camera
        cam = dict(rotation=np.eye(3).tolist(),position=[1.,2.,3.],
            width=100,height=80,fx=60.,fy=65.)
        pose,k = inference_camera(cam)
        np.testing.assert_array_equal(pose[:3,3],[1.,2.,3.])
        k[0] *= .5
        self.assertEqual(inference_camera(cam)[1][0,0],60.)

    def test_mvs_regrid_preserves_slanted_plane_rays_not_image_corners(self):
        from scripts.evaluate_depth_sources import regrid_depth,regrid_mvs
        y,x=np.mgrid[:32,:32]
        ks=np.array([[40.,0,16.],[0,40.,16.],[0,0,1]],np.float32)
        kt=ks.copy();kt[:2]*=2
        z=(2/(1-.2*(x-16)/40)).astype(np.float32)
        mapped=regrid_depth(z,ks,kt,(64,64),np.ones_like(z,bool))
        yt,xt=np.mgrid[:64,:64];expected=2/(1-.2*(xt-32)/80)
        np.testing.assert_allclose(mapped[::2,::2],expected[::2,::2],atol=1e-6)
        self.assertTrue(np.isnan(mapped[:,-1]).all())
        masked=z.copy();masked[4,5]=np.nan
        prediction=dict(depth=z,depth_masked=masked,camera_k=ks)
        shared=regrid_mvs(prediction,ks,(32,32))
        self.assertTrue(np.isnan(shared[4,5]))
        self.assertAlmostEqual(float(shared[8,9]),float(z[8,9]))

    def test_ncc_probe_uses_real_sampler_and_equal_target_cannot_win(self):
        from scripts.evaluate_depth_sources import make_depth_map_record,ncc_probe
        cam=AnalyticCamera();d=torch.full((32,32),2.)
        n=torch.zeros((3,32,32));n[2]=-1
        data=make_depth_map_record(cam,dict(plane_depth=d[None],rendered_normal=n,
            rendered_distance=d[None]),d,n,d,torch.ones_like(d,dtype=torch.bool),
            [cam,cam],[d[None],d[None]])
        result=ncc_probe(data,d.numpy(),dict(patch_radius=3,budget=128,seed=0,max_cost=.5))
        self.assertEqual(len(result['pixel_ids']),128)
        self.assertTrue(result['comparable'].any())
        self.assertFalse(result['accepted'].any())
        empty=ncc_probe(data,np.zeros((32,32),np.float32),dict(patch_radius=3,budget=128,seed=0,max_cost=.5))
        self.assertEqual(len(empty['pixel_ids']),0)

    def test_visible_depth_grid_is_camera_z_after_origin_shift(self):
        from scripts.evaluate_candidate_actions import GTQueries
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'plane.ply'
            path.write_text('ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\n'
                'property float y\nproperty float z\nelement face 2\n'
                'property list uchar int vertex_indices\nend_header\n'
                '-10 -10 2\n10 -10 2\n10 10 2\n-10 10 2\n3 0 1 2\n3 0 2 3\n')
            query=GTQueries(path)
            frame=query.depth_grid(np.array([[10.,0,8],[0,10.,8],[0,0,1]],np.float32),
                np.eye(4,dtype=np.float32),np.array([16,16]))
            self.assertTrue(frame['valid'].all())
            np.testing.assert_allclose(frame['depth'],2.,atol=1e-5)

    def test_real_depth_source_score_serializer_and_no_overwrite(self):
        from scripts.evaluate_depth_sources import make_depth_map_record,save_npz,score
        from scripts.evaluate_candidate_actions import GTQueries
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cap=root/'capture';mvs=root/'mvs'
            cap.mkdir();mvs.mkdir()
            cam=AnalyticCamera();d=torch.full((32,32),2.)
            n=torch.zeros((3,32,32));n[2]=-1
            data=make_depth_map_record(cam,dict(plane_depth=d[None],rendered_normal=n,
                rendered_distance=d[None]),d,n,d,torch.ones_like(d,dtype=torch.bool),
                [cam,cam],[d[None],d[None]])
            save_npz(cap/'depth.npz',data);save_npz(mvs/'prediction.npz',
                {'depth':d.numpy(),'camera_k':data['camera_k']})
            config={'ncc_probe':dict(patch_radius=3,budget=64,seed=0,max_cost=.5)}
            (root/'config.json').write_text(json.dumps(config))
            hashes={'cameras.json':'same-cameras','multi_view.json':'same-supports'}
            (cap/'capture_manifest.json').write_text(json.dumps(dict(records=[dict(camera='view',
                camera_uid=0,snapshot_iteration=7000,depth_file='depth.npz')],
                snapshot_inputs={'native7k':dict(metadata_sha256=hashes)})))
            (mvs/'inference_manifest.json').write_text(json.dumps(dict(config=config,
                records=[dict(camera='view',file='prediction.npz')],metadata_sha256=hashes)))
            gt_dir=root/'ground_truth';gt_dir.mkdir()
            gt=gt_dir/'plane.ply'
            gt.write_text('ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\n'
                'property float y\nproperty float z\nelement face 2\n'
                'property list uchar int vertex_indices\nend_header\n'
                '-10 -10 2\n10 -10 2\n10 10 2\n-10 10 2\n3 0 1 2\n3 0 2 3\n')
            args=SimpleNamespace(config=root/'config.json',capture=cap,mvs=mvs,gt=gt,
                output=root/'scored',segments=None,annotations=None)
            score(args)
            rows=json.loads((args.output/'results.json').read_text())['rows']
            self.assertEqual(len(rows),13)
            self.assertTrue(all(r['count']==1024 for r in rows if r['kind']=='source'))
            self.assertEqual(next(r for r in rows if r['subset']=='accepted')['count'],0)
            self.assertTrue((args.output/'metrics.csv').is_file())
            with self.assertRaises(FileExistsError):score(args)


class AnalyticGaussians:
    get_points_depth_in_depth_map = production_depth_sampler()

    def __init__(self, reference, count=4, z=3.0):
        transform = reference.world_view_transform
        camera_points = transform.new_tensor([[0., 0., z]]).repeat(count, 1)
        self._xyz = (camera_points - transform[3, :3]) @ transform[:3, :3].T
        self.scores = transform.new_full((count,), 2.)
        self.normals = transform.new_tensor([[0., 0., -1.]]) @ transform[:3, :3].T

    @property
    def get_xyz(self):
        return self._xyz

    def compute_weighted_sh_norm(self, weighting):
        assert weighting == "equal"
        return self.scores

    def get_normal(self, reference):
        return self.normals.expand(self._xyz.shape[0], -1).clone()


def analytic_fixture(**source_kwargs):
    ref = AnalyticCamera()
    sources = [AnalyticCamera(1, .1, **source_kwargs),
               AnalyticCamera(2, -.1, **source_kwargs)]
    model = AnalyticGaussians(ref)
    depth = ref.depth_dict["depth"]
    normal = normal_from_depth_image(depth, ref.get_k(),
                                     ref.world_view_transform.T).permute(2, 0, 1)
    return ref, sources, model, depth, normal


class CandidateMathTests(unittest.TestCase):
    def test_cli_defaults_and_real_metadata_without_old_core(self):
        parser = ArgumentParser()
        group = OptimizationParams(parser)
        default = group.extract(parser.parse_args([]))
        self.assertFalse(default.enable_candidate_selector)
        self.assertEqual(default.candidate_selector_budget, 4096)
        self.assertEqual(default.seed, 0)
        enabled = group.extract(parser.parse_args(["--enable_candidate_selector", "--seed", "5"]))
        self.assertEqual(enabled.seed, 5)
        record = build_resolved_config(SimpleNamespace(), enabled, SimpleNamespace())
        self.assertTrue(record["optimization"]["enable_candidate_selector"])
        self.assertNotIn("core", record)
        self.assertEqual(record["training_path"], "candidate_selector")
        self.assertFalse(hasattr(default, "core_shadow_mode"))

    def test_activation_and_unsupported_inputs(self):
        for opts, iteration, expected in [(options(), 7000, False), (options(), 7001, True),
                (options(unc_from_iter=0), 1000, False), (options(unc_from_iter=0), 1001, True),
                (options(enable_candidate_selector=False), 7001, False)]:
            with self.subTest(iteration=iteration, expected=expected):
                self.assertEqual(selector.candidate_selector_active(iteration, opts), expected)
        for changes in ({"candidate_selector_budget": 0}, {"candidate_selector_budget": 1.5},
                        {"use_mono": True}, {"enable_observation_calibration": True},
                        {"core_shadow_mode": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                selector.candidate_selector_active(7001, options(**changes))

    def test_original_sh_strict_quantiles_and_low_cap(self):
        ref = AnalyticCamera()
        model = AnalyticGaussians(ref, 5)
        model.scores = torch.tensor([0., 1., 2., 3., 4.])
        labels = selector.compute_sh_ambiguity_labels(model, options(
            sh_ambi_upper_ratio=.75, sh_ambi_lower_ratio=.5, sh_unc_lower_max=1.))
        self.assertEqual(labels.tolist(), [True, False, False, False, True])

    def test_best_source_raw_cost_common_support_and_strict_thresholds(self):
        cases = [([.95, .10], [.20, .80], [1, 1], False),
                 ([.95, .90], [.20, .80], [1, 1], True),
                 ([.95, .10], [.20, .80], [1, 0], True),
                 ([.95, .90], [.20, .80], [0, 0], False),
                 ([.95, .90], [.50, .80], [1, 1], False),
                 ([.20, .90], [.20, .80], [1, 1], False)]
        for g, p, valid, expected in cases:
            with self.subTest(g=g, p=p, valid=valid):
                decision = selector._best_source_decision(torch.tensor([g]), torch.tensor([p]),
                                                          torch.tensor([valid], dtype=torch.bool))
                self.assertEqual(decision.item(), expected)

    def test_sampling_is_bounded_repeatable_and_does_not_consume_global_rng(self):
        pool = torch.arange(5000)
        before = torch.random.get_rng_state().clone()
        py_before, np_before = random.getstate(), np.random.get_state()
        ids = selector._sample_candidate_ids(pool, 4096, seed=0, iteration=7001, camera_uid=17)
        again = selector._sample_candidate_ids(pool, 4096, seed=0, iteration=7001, camera_uid=17)
        self.assertEqual(ids.numel(), 4096)
        self.assertEqual(ids.unique().numel(), 4096)
        self.assertTrue(torch.equal(ids, again))
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual(py_before, random.getstate())
        np.testing.assert_equal(np_before, np.random.get_state())
        self.assertEqual(selector._sample_candidate_ids(pool[:3], 4096,
            seed=0, iteration=7001, camera_uid=17).tolist(), [0, 1, 2])

    def test_known_plane_homography_nonidentity_pose_and_sign_pair(self):
        ref, sources, _, _, _ = analytic_fixture()
        n = torch.tensor([[0., 0., -1.]])
        d = torch.tensor([2.])
        uv = torch.tensor([[[15.5, 15.5], [10., 20.]]])
        h = selector._plane_homography(ref, sources[0], n, d)
        projected = patch_warp(h, uv)
        torch.testing.assert_close(projected, torch.tensor([[[17.5, 15.5], [12., 20.]]]))
        torch.testing.assert_close(h, selector._plane_homography(ref, sources[0], -n, -d))

    def test_real_ncc_returns_bad_g_raw_cost_and_constant_patch_rejection(self):
        pattern = torch.arange(49).float()[None] / 49
        good, valid_good = lncc(pattern, pattern)
        bad, valid_bad = lncc(pattern, torch.ones_like(pattern))
        self.assertEqual(good.shape, (1, 1))
        self.assertLess(good.item(), .001)
        self.assertTrue(valid_good.item())
        self.assertGreater(bad.item(), .9)
        self.assertFalse(valid_bad.item())
        self.assertTrue(selector._best_source_decision(bad, good, torch.ones_like(bad, dtype=torch.bool)).item())

    def run_selector(self, ref, sources, model, depth, normal, **changes):
        calls = []
        def source_depth(camera):
            self.assertFalse(torch.is_grad_enabled())
            calls.append(camera.uid)
            return depth.new_full((1, camera.image_height, camera.image_width), 2.)
        result = selector.select_candidate_labels(model, ref, ref.gray, depth, normal,
            sources, options(**changes), seed=0, iteration=7001, support_depth_fn=source_depth)
        return result, calls

    def test_full_selector_real_warp_ncc_and_depth_sampler_adds_labels(self):
        ref, sources, model, depth, normal = analytic_fixture()
        (labels, extra, stats), calls = self.run_selector(ref, sources, model, depth, normal)
        self.assertEqual(calls, [1, 2])
        self.assertTrue(extra.all())
        self.assertTrue(torch.equal(labels, extra))
        self.assertEqual(int(stats["sampled"]), 4)
        self.assertEqual(int(stats["comparable"]), 4)
        self.assertEqual(int(stats["added"]), 4)
        self.assertEqual(labels.dtype, torch.bool)
        self.assertFalse(labels.requires_grad)

    def test_each_source_uses_its_own_dimensions_and_ncc_scale(self):
        ref, sources, model, depth, normal = analytic_fixture(width=40, height=36, ncc_scale=2)
        (labels, extra, stats), calls = self.run_selector(ref, sources, model, depth, normal)
        self.assertEqual(calls, [1, 2])
        self.assertEqual(int(stats["comparable"]), 4)
        self.assertTrue(extra.all())

    def test_no_sources_empty_pool_and_invalid_prior_fall_back(self):
        for case in ("one_source", "empty_pool", "bad_prior", "bad_normal", "border"):
            ref, sources, model, depth, normal = analytic_fixture()
            if case == "one_source": sources = sources[:1]
            if case == "empty_pool": model._xyz[:, 2] = -10
            if case == "bad_prior": depth[:] = float("nan")
            if case == "bad_normal": normal[:] = 0
            if case == "border": model._xyz[:] = (torch.tensor([[-.775, 0., 2.]]) - ref.world_view_transform[3, :3]) @ ref.world_view_transform[:3, :3].T
            with self.subTest(case=case):
                (labels, extra, _), calls = self.run_selector(ref, sources, model, depth, normal)
                self.assertFalse(extra.any())
                self.assertTrue(torch.equal(labels, selector.compute_sh_ambiguity_labels(model, options())))
                self.assertEqual(calls, [])

    def test_bad_sampled_candidates_are_not_refilled_and_a_hits_survive(self):
        ref, sources, model, depth, normal = analytic_fixture()
        model.scores = torch.tensor([0., 2., 2., 20.])
        depth[:] = -1
        (labels, extra, stats), calls = self.run_selector(ref, sources, model, depth, normal,
                                                       candidate_selector_budget=1)
        self.assertEqual(int(stats["sampled"]), 1)
        self.assertEqual(labels.tolist(), [True, False, False, True])
        self.assertFalse(extra.any())
        self.assertEqual(calls, [])

    def test_invalid_depth_in_normal_stencil_is_rejected_before_source_render(self):
        ref, sources, model, depth, normal = analytic_fixture()
        # Bilinear normal at (15.5,15.5) depends on this left-hand depth even
        # when the sampled center depth and resulting normal are finite.
        depth[15, 14] = -1
        normal = normal_from_depth_image(depth, ref.get_k(),
            ref.world_view_transform.T).permute(2, 0, 1)
        (labels, extra, stats), calls = self.run_selector(ref, sources, model, depth, normal)
        self.assertEqual(calls, [])
        self.assertFalse(extra.any())

    def test_common_geometry_rejects_bad_source_patches_and_respects_occ_flag(self):
        for case, expected_count in (("one_source", 4), ("outside", 0),
                                     ("behind", 0), ("roundtrip", 0), ("wo_occ", 4)):
            ref, sources, model, depth, normal = analytic_fixture()
            changes = {}
            if case == "one_source": sources[1] = AnalyticCamera(2, 10.)
            if case == "outside": sources = [AnalyticCamera(1, 10.), AnalyticCamera(2, 20.)]
            if case == "behind":
                for source in sources: source.world_view_transform[3, 2] = -5
            if case in ("roundtrip", "wo_occ"):
                changes["multi_view_pixel_noise_th"] = .1
                changes["wo_use_geo_occ_aware"] = case == "wo_occ"
            with self.subTest(case=case):
                (labels, extra, stats), calls = self.run_selector(ref, sources, model, depth, normal, **changes)
                self.assertEqual(int(stats["comparable"]), expected_count)
                self.assertEqual(int(extra.sum()), expected_count)

    def test_model_interface_and_callback_errors_are_exposed(self):
        ref, sources, model, depth, normal = analytic_fixture()
        with self.assertRaises(FloatingPointError):
            selector.select_candidate_labels(model, ref, ref.gray, depth, normal, sources,
                options(), seed=0, iteration=7001,
                support_depth_fn=lambda c: depth.new_full((1, c.image_height, c.image_width), float("nan")))
        with self.assertRaisesRegex(RuntimeError, "source failure"):
            selector.select_candidate_labels(model, ref, ref.gray, depth, normal, sources,
                options(), seed=0, iteration=7001,
                support_depth_fn=lambda _: (_ for _ in ()).throw(RuntimeError("source failure")))
        with self.assertRaises(ValueError):
            self.run_selector(ref, sources, model, depth[:-1], normal)
        model._xyz[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            self.run_selector(ref, sources, model, depth, normal)


class CandidateRelocationTests(unittest.TestCase):
    stat_names = ("xyz_gradient_accum", "xyz_gradient_accum_abs", "denom",
                  "denom_abs", "max_radii2D", "max_weight")

    def fixture(self, count=12):
        ref, sources, model, depth, normal = analytic_fixture()
        model = AnalyticGaussians(ref, count)
        transform = ref.world_view_transform
        camera_xyz = torch.stack((torch.linspace(-.25, .25, count),
            torch.linspace(.15, -.15, count), torch.full((count,), 3.)), -1)
        model._xyz = torch.nn.Parameter((camera_xyz - transform[3, :3]) @ transform[:3, :3].T)
        model._rotation = torch.nn.Parameter(torch.ones(count, 4))
        model.relocate_selected = production_model_method("relocate_selected").__get__(model)
        model.optimizer = torch.optim.Adam([
            dict(params=[model._xyz], name="xyz", lr=0.),
            dict(params=[model._rotation], name="rotation", lr=.001)], eps=1e-15)
        (model._xyz.square().sum() + model._rotation.square().sum()).backward()
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)
        model.optimizer.param_groups[0]["lr"] = .00016
        for name in self.stat_names:
            values = torch.arange(1, count + 1, dtype=model._xyz.dtype)
            setattr(model, name, values if name in ("max_radii2D", "max_weight") else values[:, None])
        return ref, sources, model, depth, normal

    def test_relocation_defaults_schedule_and_invalid_configuration(self):
        parser = ArgumentParser()
        group = OptimizationParams(parser)
        opts = group.extract(parser.parse_args([]))
        self.assertFalse(getattr(opts, "enable_candidate_relocation", True))
        self.assertEqual(opts.candidate_relocation_interval, 100)
        opts = group.extract(parser.parse_args(["--enable_candidate_selector",
            "--enable_candidate_relocation", "--candidate_relocation_interval", "100"]))
        record = build_resolved_config(SimpleNamespace(), opts, SimpleNamespace())
        self.assertTrue(record["optimization"]["enable_candidate_relocation"])
        self.assertEqual(record["optimization"]["candidate_relocation_interval"], 100)
        for iteration in (0, 7000, 7001, 7101, 30000):
            self.assertFalse(selector.candidate_relocation_active(iteration, opts))
        for iteration in (7100, 15000, 29900):
            self.assertTrue(selector.candidate_relocation_active(iteration, opts))
        for changes in ({"candidate_relocation_interval": 0},
                        {"candidate_relocation_interval": True},
                        {"candidate_relocation_interval": 1.5},
                        {"enable_candidate_selector": False}, {"use_mono": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                selector.candidate_relocation_active(0, options(enable_candidate_relocation=True, **changes))

    def test_relocation_cutoff_preserves_full_v2_and_early_schedule(self):
        parser = ArgumentParser()
        group = OptimizationParams(parser)
        default = group.extract(parser.parse_args([]))
        self.assertEqual(getattr(default, "candidate_relocation_until_iter", None), 0)
        early = group.extract(parser.parse_args(["--enable_candidate_selector",
            "--enable_candidate_relocation", "--candidate_relocation_until_iter", "15000"]))
        record = build_resolved_config(SimpleNamespace(), early, SimpleNamespace())
        self.assertEqual(record["optimization"]["candidate_relocation_until_iter"], 15000)
        actual = [i for i in range(30001) if selector.candidate_relocation_active(i, early)]
        self.assertEqual(actual, list(range(7100, 15000, 100)))
        self.assertEqual(len(actual), 79)
        full = options(enable_candidate_relocation=True)
        legacy = options(enable_candidate_relocation=True)
        del legacy.candidate_relocation_until_iter
        for opts in (full, legacy):
            actual = [i for i in range(30001) if selector.candidate_relocation_active(i, opts)]
            self.assertEqual(actual, list(range(7100, 30000, 100)))
            self.assertEqual(len(actual), 229)
        for iteration in (15000, 20000, 29900, 30000):
            self.assertFalse(selector.candidate_relocation_active(iteration, early))
            self.assertTrue(selector.candidate_selector_active(iteration, early))
        for invalid in (-1, True, 1.5, "15000"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                selector.candidate_relocation_active(0, options(enable_candidate_relocation=True,
                    candidate_relocation_until_iter=invalid))
        self.assertFalse(selector.candidate_relocation_active(7100, options(
            enable_candidate_relocation=False, candidate_relocation_until_iter=-1)))

    def test_real_selector_targets_keep_id_order_and_v1_result(self):
        ref, sources, model, depth, normal = self.fixture()
        model.scores[0], model.scores[3] = 0., 4.
        opts = options(candidate_selector_budget=7, enable_candidate_relocation=True)
        calls = []
        def source_depth(camera):
            calls.append(camera.uid)
            return depth.new_full((1, camera.image_height, camera.image_width), 2.)
        kwargs = dict(seed=0, iteration=7100, support_depth_fn=source_depth)
        rng = torch.get_rng_state().clone()
        old = selector.select_candidate_labels(model, ref, ref.gray, depth, normal, sources, opts, **kwargs)
        new = selector.select_candidate_labels_with_targets(model, ref, ref.gray, depth, normal, sources, opts, **kwargs)
        self.assertEqual(len(old), 3)
        self.assertEqual(len(new), 4)
        for i in (0, 1):
            self.assertTrue(torch.equal(old[i], new[i]))
        self.assertEqual(old[2], new[2])
        self.assertEqual(calls, [1, 2, 1, 2])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        ids, targets = new[3]
        self.assertGreater(ids.numel(), 1)
        self.assertFalse(torch.equal(ids, ids.sort().values), "exercise nonsorted sampled IDs")
        self.assertEqual(ids.dtype, torch.long)
        self.assertEqual(set(ids.tolist()), set(new[1].nonzero().flatten().tolist()))
        original_labels = selector.compute_sh_ambiguity_labels(model, opts)
        self.assertTrue(original_labels.any())
        self.assertFalse(original_labels[ids].any())
        self.assertTrue(new[0][original_labels].all())
        transform = ref.world_view_transform
        original = model._xyz.detach()[ids] @ transform[:3, :3] + transform[3, :3]
        expected_camera = original * (2. / original[:, 2:3])
        expected = (expected_camera - transform[3, :3]) @ transform[:3, :3].T
        torch.testing.assert_close(targets, expected)
        self.assertFalse(targets.requires_grad)
        parameter = model._xyz
        state = model.optimizer.state[parameter]
        before_xyz = parameter.detach().clone()
        before_moments = {k: v.clone() for k, v in state.items()}
        before_stats = {k: getattr(model, k).clone() for k in self.stat_names}
        other_state = {k: v.clone() for k, v in model.optimizer.state[model._rotation].items()}
        rotation = model._rotation.detach().clone()
        self.assertEqual(model.relocate_selected(ids, targets), len(ids))
        self.assertIs(model._xyz, parameter)
        self.assertIs(model.optimizer.param_groups[0]["params"][0], parameter)
        keep = torch.ones(len(parameter), dtype=torch.bool); keep[ids] = False
        torch.testing.assert_close(parameter[ids], expected)
        self.assertTrue(torch.equal(parameter[keep], before_xyz[keep]))
        self.assertTrue(torch.equal(state["step"], before_moments["step"]))
        self.assertEqual(model.optimizer.param_groups[0]["lr"], .00016)
        for key in ("exp_avg", "exp_avg_sq"):
            self.assertEqual(int(torch.count_nonzero(state[key][ids])), 0)
            self.assertTrue(torch.equal(state[key][keep], before_moments[key][keep]))
        for key in self.stat_names:
            self.assertEqual(int(torch.count_nonzero(getattr(model, key)[ids])), 0)
            self.assertTrue(torch.equal(getattr(model, key)[keep], before_stats[key][keep]))
        self.assertTrue(torch.equal(model._rotation, rotation))
        for key, value in other_state.items():
            self.assertTrue(torch.equal(model.optimizer.state[model._rotation][key], value))
        parameter.square().sum().backward()
        model.optimizer.step()
        self.assertTrue(torch.isfinite(parameter).all())
        empty = selector.select_candidate_labels_with_targets(model, ref, ref.gray,
            depth, normal, [], opts, **kwargs)
        self.assertIsNone(empty[3])
        self.assertFalse(empty[1].any())

    def test_relocation_invalid_inputs_do_not_partially_write(self):
        for case in ("dtype", "shape", "range", "duplicate", "nan", "binding", "state", "stats"):
            with self.subTest(case=case):
                _, _, model, _, _ = self.fixture(4)
                ids = torch.tensor([2, 0]); targets = model._xyz.detach()[ids] + .2
                if case == "dtype": ids = ids.float()
                if case == "shape": targets = targets[:, :2]
                if case == "range": ids[0] = 9
                if case == "duplicate": ids[0] = 0
                if case == "nan": targets[0, 0] = float("nan")
                if case == "binding": model.optimizer.param_groups[0]["params"] = [model._rotation]
                if case == "state": model.optimizer.state[model._xyz]["exp_avg"] = torch.zeros(1, 3)
                if case == "stats": model.denom = torch.ones(2, 1)
                xyz = model._xyz.detach().clone()
                moments = {k: v.clone() for k, v in model.optimizer.state[model._xyz].items()}
                with self.assertRaises((ValueError, TypeError)):
                    model.relocate_selected(ids, targets)
                self.assertTrue(torch.equal(model._xyz, xyz))
                for key, value in moments.items():
                    self.assertTrue(torch.equal(model.optimizer.state[model._xyz][key], value))

    def test_relocation_empty_and_uninitialized_adam(self):
        _, _, model, _, _ = self.fixture(4)
        before = model._xyz.detach().clone()
        state = model.optimizer.state[model._xyz]["exp_avg"].clone()
        self.assertEqual(model.relocate_selected(torch.empty(0, dtype=torch.long), before[:0]), 0)
        self.assertTrue(torch.equal(model._xyz, before))
        self.assertTrue(torch.equal(model.optimizer.state[model._xyz]["exp_avg"], state))
        del model.optimizer.state[model._xyz]
        model.relocate_selected(torch.tensor([1]), before[1:2] + .1)
        self.assertNotIn(model._xyz, model.optimizer.state)

    def test_real_training_helper_moves_before_main_render_only_on_event(self):
        root = Path(__file__).resolve().parents[1]
        renderer_tree = ast.parse((root / "gaussian_renderer/__init__.py").read_text(encoding="utf-8"))
        normal_node = next(n for n in renderer_tree.body if isinstance(n, ast.FunctionDef) and n.name == "render_normal")
        normal_namespace = dict(normal_from_depth_image=normal_from_depth_image)
        exec(compile(ast.Module(body=[normal_node], type_ignores=[]), "<real render_normal>", "exec"), normal_namespace)
        camera_tree = ast.parse((root / "scene/cameras.py").read_text(encoding="utf-8"))
        camera_class = next(n for n in camera_tree.body if isinstance(n, ast.ClassDef) and n.name == "Camera")
        calibration = next(n for n in camera_class.body if isinstance(n, ast.FunctionDef) and n.name == "get_calib_matrix_nerf")
        calibration_namespace = dict(torch=torch)
        exec(compile(ast.Module(body=[calibration], type_ignores=[]), "<real camera calibration>", "exec"), calibration_namespace)
        tree = ast.parse((root / "train.py").read_text(encoding="utf-8"))
        helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_render_training_reference")
        for iteration, enabled, invalid, until, should_move in (
                (7100, True, False, 0, True), (7101, True, False, 0, False),
                (7100, False, False, 0, False), (7100, True, True, 0, False),
                (14900, True, False, 15000, True), (15000, True, False, 15000, False),
                (20000, True, False, 15000, False), (30000, True, False, 15000, False)):
            with self.subTest(iteration=iteration, enabled=enabled, invalid=invalid, until=until):
                ref, sources, model, _, _ = self.fixture(4)
                ref.get_calib_matrix_nerf = calibration_namespace[calibration.name].__get__(ref)
                ref._candidate_support_cameras = sources
                if invalid: ref.depth_dict["depth"].fill_(0)
                calls = []
                def render_boundary(camera, current, *args, **kwargs):
                    calls.append(camera.uid)
                    if camera.uid != ref.uid:
                        self.assertFalse(torch.is_grad_enabled())
                        return {"plane_depth": current._xyz.new_full((1, 32, 32), 2.)}
                    self.assertTrue(torch.is_grad_enabled())
                    camera_xyz = current._xyz @ ref.world_view_transform[:3, :3] + ref.world_view_transform[3, :3]
                    torch.testing.assert_close(camera_xyz[:, 2], camera_xyz.new_full((4,), 2. if should_move else 3.))
                    return {"plane_depth": camera_xyz[:, 2].mean().expand(1, 32, 32)}
                namespace = dict(torch=torch, F=torch.nn.functional, render=render_boundary,
                    render_normal=normal_namespace["render_normal"],
                    candidate_selector_active=selector.candidate_selector_active,
                    candidate_relocation_active=getattr(selector, "candidate_relocation_active", None),
                    select_candidate_labels=selector.select_candidate_labels,
                    select_candidate_labels_with_targets=getattr(selector, "select_candidate_labels_with_targets", None))
                exec(compile(ast.Module(body=[helper], type_ignores=[]), "<real training helper>", "exec"), namespace)
                opts = options(enable_candidate_relocation=enabled, ray_color_lambda=-1,
                               candidate_relocation_until_iter=until)
                rng = torch.get_rng_state().clone()
                pkg, data = namespace[helper.name](ref, model, None, None, None, opts, iteration=iteration, seed=0)
                self.assertEqual(calls, [0] if invalid else [1, 2, 0])
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertEqual(data["stats"].get("relocated", 0), 4 if should_move else 0)
                if not invalid:
                    self.assertEqual(int(data["extra"].sum()), 4)
                pkg["plane_depth"].sum().backward()
                model.optimizer.step()
                self.assertTrue(torch.isfinite(model._xyz).all())


def native_unavailable():
    if not torch.cuda.is_available():
        return "CUDA unavailable"
    for module in ("diff_plane_rasterization_ambisur", "simple_knn", "pytorch3d",
                   "cv2", "scipy", "open3d", "plyfile"):
        if importlib.util.find_spec(module) is None:
            return "native integration dependency missing: " + module
    return ""


@unittest.skipIf(native_unavailable(), native_unavailable())
class CandidateIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from gaussian_renderer import render, render_normal
        from scene.cameras import Camera
        from scene.gaussian_model import GaussianModel
        from train import _render_training_reference, _compute_alr_loss
        cls.render = staticmethod(render)
        cls.normal = staticmethod(render_normal)
        cls.reference_render = staticmethod(_render_training_reference)
        cls.alr = staticmethod(_compute_alr_loss)
        cls.Camera, cls.GaussianModel = Camera, GaussianModel

    def test_installed_rasterizer_matches_clean_baseline(self):
        import diff_plane_rasterization_ambisur as rasterizer
        expected = Path(__file__).resolve().parents[1] / "submodules/diff-plane-rasterization-ambisur/diff_plane_rasterization_ambisur/__init__.py"
        actual = Path(rasterizer.__file__)
        self.assertEqual(ast.dump(ast.parse(actual.read_text(encoding="utf-8"))),
                         ast.dump(ast.parse(expected.read_text(encoding="utf-8"))),
                         "installed rasterizer Python wrapper is from a different checkout")
        self.assertFalse(hasattr(rasterizer._C, "rasterize_gaussians_with_evidence"),
                         "old first-run evidence extension is still installed")

    def fixture(self, invalid_prior=False):
        from pytorch3d.transforms import matrix_to_quaternion
        device = "cuda"
        cameras = []
        for uid, delta in enumerate((0., .1, -.1)):
            a = AnalyticCamera(uid, delta, device=device)
            camera = self.Camera(uid, a.world_view_transform[:3, :3].cpu().numpy(),
                a.world_view_transform[3, :3].cpu().numpy(), 2 * math.atan(32 / 80),
                2 * math.atan(32 / 80), 32, 32, "unused", str(uid), uid,
                preload_img=False, depth_dict={"depth": np.full((32, 32), 2., np.float32)})
            camera.preload_img = True
            camera.original_image = a.gray.expand(3, -1, -1).contiguous()
            camera.original_image_gray = a.gray
            camera.nearest_id = [1, 2]
            cameras.append(camera)
        ref = cameras[0]
        model = self.GaussianModel(3)
        cam_xyz = torch.tensor([[-.25, 0., 3.], [0., 0., 3.], [.25, 0., 3.]], device=device)
        model._xyz = torch.nn.Parameter((cam_xyz - ref.world_view_transform[3, :3]) @ ref.world_view_transform[:3, :3].T)
        model._features_dc = torch.nn.Parameter(torch.full((3, 1, 3), .2, device=device))
        model._features_rest = torch.nn.Parameter(torch.ones((3, 15, 3), device=device)
            * torch.tensor([0., .2, 2.], device=device)[:, None, None])
        model._scaling = torch.nn.Parameter(torch.tensor([[.35, .35, .02]], device=device).log().repeat(3, 1))
        tilt = ref.world_view_transform.new_tensor([
            [math.cos(.15), 0, math.sin(.15)], [0, 1., 0], [-math.sin(.15), 0, math.cos(.15)]])
        model._rotation = torch.nn.Parameter(matrix_to_quaternion(
            ref.world_view_transform[:3, :3] @ tilt).repeat(3, 1))
        model._opacity = torch.nn.Parameter(torch.full((3, 1), 2., device=device))
        model.disable_trunc, model.trunc_sigma = False, 2.
        if invalid_prior:
            with torch.no_grad():
                ref.depth_dict["depth"].copy_(-1 + .01 * torch.arange(32, device=device)[None].expand(32, 32))
        pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
        # The real nearest order is passed through the train helper, without a scene loader.
        ref._candidate_support_cameras = cameras[1:]
        return ref, cameras[1:], model, pipe, torch.zeros(3, device=device)

    def test_real_normal_producer_is_already_in_reference_coordinates(self):
        ref, sources, model, pipe, bg = self.fixture()
        depth = ref.depth_dict["depth"].detach()
        actual = self.normal(ref, depth)
        torch.testing.assert_close(actual[:, 10:22, 10:22],
            actual.new_tensor([0., 0., -1.])[:, None, None].expand(3, 12, 12), atol=1e-5, rtol=1e-5)
        self.assertFalse(torch.allclose(actual[:, 16, 16] @ ref.world_view_transform[:3, :3], actual[:, 16, 16]))

    def test_static_audit_uses_real_native_depth_without_moving_model(self):
        from scripts import evaluate_candidate_actions as audit
        ref, sources, model, pipe, bg = self.fixture()
        original = model.get_xyz.detach().clone()
        depth = ref.depth_dict['depth'].detach()
        with torch.no_grad():
            record, stats = audit.observe_selector(model,ref,ref.get_image()[1],depth,
                self.normal(ref,depth),sources,options(),seed=0,iteration=7100,
                support_depth_fn=lambda c:self.render(c,model,pipe,bg,
                    return_plane=True,return_depth_normal=False,ray_reg=False)['plane_depth'])
        audit.validate_record(record)
        self.assertGreater(stats['added'],0)
        self.assertTrue(torch.equal(model.get_xyz,original))

    def test_real_selector_to_renderer_to_alr_backward_and_step(self):
        ref, sources, model, pipe, bg = self.fixture()
        opt = options()
        depth = ref.depth_dict["depth"].detach()
        normals = self.normal(ref, depth)
        with patch("utils.candidate_selector.lncc", wraps=lncc):
            labels, extra, stats = selector.select_candidate_labels(model, ref,
                ref.original_image_gray, depth, normals, sources, opt, seed=0,
                iteration=7001, support_depth_fn=lambda c: self.render(c, model,
                    pipe, bg, return_depth_normal=False, ray_reg=False)["plane_depth"])
        a = selector.compute_sh_ambiguity_labels(model, opt)
        outputs = [self.render(ref, model, pipe, bg, opt=opt, ambiguity_labels=b)
                   for b in (labels, a, extra)]
        self.assertTrue(extra.any(), "fixture must exercise additional labels")
        torch.testing.assert_close(outputs[0]["rendered_unc"],
            outputs[1]["rendered_unc"] + outputs[2]["rendered_unc"])
        loss = self.alr(ref, outputs[0], depth, 1, -1, 7001, opt, normals)
        optimizer = torch.optim.Adam([model._xyz, model._rotation], lr=1e-3)
        loss.backward()
        self.assertGreater(float(loss.detach()), 0)
        for grad in (model._xyz.grad, model._rotation.grad):
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(float(grad.abs().sum()), 0)
        optimizer.step()
        self.assertTrue(torch.isfinite(model.get_xyz).all())

    def test_v2_real_render_relocation_alr_backward_and_step(self):
        ref, sources, model, pipe, bg = self.fixture()
        opt = options(enable_candidate_relocation=True, ray_color_lambda=-1)
        model.spatial_lr_scale = 1.
        model.training_setup(opt)
        model.max_radii2D = model._xyz.new_zeros(len(model._xyz))
        model.max_weight = model.max_radii2D.clone()
        parameter = model._xyz
        before = parameter.detach().clone()
        pkg, data = self.reference_render(ref, model, pipe, bg, None, opt, iteration=7100, seed=0)
        self.assertGreater(data["stats"]["relocated"], 0)
        self.assertIs(model._xyz, parameter)
        self.assertTrue(torch.equal(parameter[~data["extra"]], before[~data["extra"]]))
        camera_xyz = parameter[data["extra"]] @ ref.world_view_transform[:3, :3] + ref.world_view_transform[3, :3]
        torch.testing.assert_close(camera_xyz[:, 2], torch.full_like(camera_xyz[:, 2], 2.))
        loss = ((pkg["render"] - .3) ** 2).mean() + self.alr(ref, pkg,
            data["prior_depth"], 1, -1, 7100, opt, data["prior_normal"])
        loss.backward()
        self.assertTrue(torch.isfinite(parameter.grad).all())
        model.optimizer.step()
        self.assertTrue(torch.isfinite(parameter).all())

    def test_spread_real_render_to_alr_backward_and_step(self):
        ref,sources,model,pipe,bg = self.fixture()
        opt = options(candidate_depth_source='spread',enable_candidate_relocation=True,
                      candidate_relocation_until_iter=15000,ray_color_lambda=-1,
                      candidate_selector_valid_pool=True,candidate_relocation_position_check=True)
        model.spatial_lr_scale=1.;model.training_setup(opt)
        model.max_radii2D=model._xyz.new_zeros(len(model._xyz));model.max_weight=model.max_radii2D.clone()
        raw=ref.depth_dict['depth'].clone()
        pkg,data=self.reference_render(ref,model,pipe,bg,None,opt,iteration=7100,seed=0)
        torch.testing.assert_close(data['prior_depth'],raw)
        self.assertIn('anchors',data['stats']);self.assertIn('spread_pixels',data['stats'])
        self.assertIn('position_checked',data['stats'])
        self.assertEqual(int(data['stats']['position_passed']), data['stats']['relocated'])
        self.assertTrue(torch.isfinite(pkg['plane_depth']).all())
        loss=pkg['render'].mean()+self.alr(ref,pkg,data['prior_depth'],1,-1,7100,opt,data['prior_normal'])
        loss.backward()
        self.assertTrue(torch.isfinite(model._xyz.grad).all())
        model.optimizer.step()
        self.assertTrue(torch.isfinite(model.get_xyz).all())

    def test_real_training_helper_off_ezero_and_active_path(self):
        ref, sources, model, pipe, bg = self.fixture(invalid_prior=True)
        from train import prepare_output_and_logger
        with tempfile.TemporaryDirectory() as output:
            dataset = SimpleNamespace(model_path=output, source_path="synthetic", sh_degree=3)
            writer = prepare_output_and_logger(dataset, options(), pipe)
            if writer is not None:
                writer.close()
            config = json.loads((Path(output) / "resolved_config.json").read_text())
            identity = json.loads((Path(output) / "run_identity.json").read_text())
            self.assertEqual(config["training_path"], "candidate_selector")
            self.assertNotIn("core", config)
            self.assertEqual(identity["seed"], 0)
            self.assertEqual(len(identity["git_commit"]), 40)
        opts = options(ray_color_lambda=-1)
        before = torch.cuda.get_rng_state().clone()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        pkg, data = self.reference_render(ref, model, pipe, bg, None, opts, iteration=7001, seed=0)
        end.record()
        torch.cuda.synchronize()
        self.assertFalse(data["extra"].any())
        self.assertTrue(torch.equal(before, torch.cuda.get_rng_state()))
        self.assertTrue(torch.isfinite(pkg["plane_depth"]).all())
        self.assertGreaterEqual(start.elapsed_time(end), 0)
        print("selector-active-ms", start.elapsed_time(end), "peak-bytes", torch.cuda.max_memory_allocated())
        off = options(enable_candidate_selector=False, ray_color_lambda=-1)
        direct = self.render(ref, model, pipe, bg, return_plane=True,
            return_depth_normal=True, ray_reg=off.ray_color_lambda, opt=off)
        self.assertTrue(selector.compute_sh_ambiguity_labels(model, off).any())
        for opts in (off, options(ray_color_lambda=-1)):
            actual, metadata = self.reference_render(ref, model, pipe, bg, None,
                opts, iteration=7001, seed=0)
            torch.testing.assert_close(actual["rendered_unc"], direct["rendered_unc"])
            for result in (actual, direct):
                model._xyz.grad = None
                model._rotation.grad = None
                loss = self.alr(ref, result, ref.depth_dict["depth"].detach(), 1, -1, 7001, opts)
                self.assertGreater(float(loss.detach()), 0)
                loss.backward(retain_graph=True)
                grad = (model._xyz.grad.clone(), model._rotation.grad.clone())
                for parameter_grad in grad:
                    self.assertTrue(torch.isfinite(parameter_grad).all())
                    self.assertGreater(float(parameter_grad.abs().sum()), 0)
                if result is actual: actual_grad = grad
                else:
                    for first, second in zip(actual_grad, grad):
                        torch.testing.assert_close(first, second)
            if opts is off: self.assertIsNone(metadata)

    def test_renderer_rejects_bad_labels_and_training_callback_preserves_grad_path(self):
        ref, sources, model, pipe, bg = self.fixture()
        for labels in (torch.zeros(3, device="cuda"), torch.zeros(2, device="cuda", dtype=torch.bool),
                       torch.zeros(3, dtype=torch.bool)):
            with self.subTest(shape=labels.shape, dtype=labels.dtype), self.assertRaises(ValueError):
                self.render(ref, model, pipe, bg, opt=options(), ambiguity_labels=labels)
        # The helper performs no-grad forwards only; a separate ordinary source
        # render still has its original differentiable depth path.
        from gaussian_renderer import render as actual_render
        calls = []
        def traced(camera, *args, **kwargs):
            calls.append((camera.uid, torch.is_grad_enabled(), kwargs))
            return actual_render(camera, *args, **kwargs)
        with patch("train.render", side_effect=traced):
            pkg, data = self.reference_render(ref, model, pipe, bg, None,
                options(), iteration=7001, seed=0)
        self.assertEqual([uid for uid, grad, kw in calls], [1, 2, 0])
        for uid, grad, kw in calls[:2]:
            self.assertFalse(grad)
            self.assertFalse(kw["ray_reg"])
            self.assertFalse(kw["return_depth_normal"])
        self.assertTrue(calls[-1][1])
        differentiable_source = self.render(sources[0], model, pipe, bg, return_depth_normal=False)
        self.assertTrue(differentiable_source["plane_depth"].requires_grad)
        ref.depth_dict = None
        for opt, iteration in ((options(enable_candidate_selector=False), 7001), (options(), 7000)):
            calls.clear()
            with patch("train.render", side_effect=traced):
                _, metadata = self.reference_render(ref, model, pipe, bg, None,
                    opt, iteration=iteration, seed=0)
            self.assertIsNone(metadata)
            self.assertEqual([uid for uid, grad, kw in calls], [0])


class NativeCheckpointTests(unittest.TestCase):
    def test_native_capture_to_training_restore_keeps_adam_and_exact_app_iteration(self):
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root / "scene/gaussian_model.py").read_text(encoding="utf-8"))
        model = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GaussianModel")
        methods = [n for n in model.body if isinstance(n, ast.FunctionDef)
                   and n.name in ("capture", "restore")]
        namespace = {"torch": torch}
        exec(compile(ast.Module(body=methods, type_ignores=[]), "<native Gaussian methods>", "exec"), namespace)
        parameter_names = ("_xyz", "_knn_f", "_features_dc", "_features_rest",
                           "_scaling", "_rotation", "_opacity")
        # Match Scene -> create_from_pcd -> capture: radius is a NumPy scalar,
        # which torch.load's restricted default does not accept.
        reader_tree = ast.parse((root / "scene/dataset_readers.py").read_text(encoding="utf-8"))
        normalize = next(n for n in reader_tree.body
                         if isinstance(n, ast.FunctionDef) and n.name == "getNerfppNorm")
        from utils.graphics_utils import getWorld2View2
        normalization_namespace = dict(np=np, getWorld2View2=getWorld2View2)
        exec(compile(ast.Module(body=[normalize], type_ignores=[]),
                     "<native normalization>", "exec"), normalization_namespace)
        cameras = [SimpleNamespace(R=np.eye(3), T=np.array(t))
                   for t in ([0., 0., 0.], [1., 0., 0.])]
        radius = normalization_namespace["getNerfppNorm"](cameras)["radius"]
        self.assertIsInstance(radius, np.generic)

        def setup(instance, opts):
            instance.optimizer = torch.optim.Adam(
                [getattr(instance, name) for name in parameter_names], lr=opts.position_lr_init)

        cpu_model = type("CpuNativeGaussians", (), {
            "capture": namespace["capture"], "restore": namespace["restore"], "training_setup": setup})

        def make_model(value):
            instance = cpu_model()
            for name in parameter_names:
                setattr(instance, name, torch.nn.Parameter(torch.full((2, 3), value)))
            for name in ("max_radii2D", "max_weight", "xyz_gradient_accum", "xyz_gradient_accum_abs",
                         "denom", "denom_abs"):
                setattr(instance, name, torch.full((2, 1), value))
            instance.active_sh_degree, instance.spatial_lr_scale = 3, radius
            instance.training_setup(SimpleNamespace(position_lr_init=.004))
            return instance

        def step(instance):
            sum(getattr(instance, name).square().sum() for name in parameter_names).backward()
            instance.optimizer.step()
            instance.optimizer.zero_grad(set_to_none=True)

        # Execute the real AppModel producers/consumer, bypassing only its CUDA constructor.
        app_namespace = {}
        exec(compile((root / "scene/app_model.py").read_text(encoding="utf-8"),
                     "<native AppModel>", "exec"), app_namespace)
        app_class = app_namespace["AppModel"]

        def make_app(value):
            app = app_class.__new__(app_class)
            torch.nn.Module.__init__(app)
            app.appear_ab = torch.nn.Parameter(torch.full((2, 2), value))
            return app

        original, restored = make_model(1.), make_model(8.)
        step(original)
        train_tree = ast.parse((root / "train.py").read_text(encoding="utf-8"))
        training = next(n for n in train_tree.body if isinstance(n, ast.FunctionDef) and n.name == "training")
        restore_block = next(n for n in training.body if isinstance(n, ast.If) and ast.unparse(n.test) == "checkpoint")
        with tempfile.TemporaryDirectory() as temporary:
            prefix, destination = Path(temporary) / "prefix", Path(temporary) / "new_candidate"
            prefix.mkdir()
            checkpoint = prefix / "chkpnt7000.pth"
            torch.save((original.capture(), 7000), checkpoint)
            make_app(.2).save_weights(str(prefix), 7000)
            make_app(9.).save_weights(str(prefix), 30000)
            make_app(8.).save_weights(str(destination), 30000)
            app = make_app(0.)
            context = dict(torch=torch, os=__import__("os"), checkpoint=str(checkpoint),
                           gaussians=restored, app_model=app, scene=SimpleNamespace(model_path=str(destination)),
                           opt=SimpleNamespace(position_lr_init=.123, iterations=30000), first_iter=0)
            exec(compile(ast.Module(body=[restore_block], type_ignores=[]), "<real training restore>", "exec"), context)
            self.assertEqual(context["first_iter"], 7000)
            self.assertEqual(context["opt"].iterations, 30000)
            self.assertIs(type(restored.spatial_lr_scale), type(radius))
            self.assertEqual(restored.spatial_lr_scale, radius)
            torch.testing.assert_close(app.appear_ab, torch.full((2, 2), .2))
            self.assertEqual(restored.optimizer.param_groups[0]["lr"], .004)
            torch.testing.assert_close(restored.xyz_gradient_accum, original.xyz_gradient_accum)
            torch.testing.assert_close(restored.denom_abs, original.denom_abs)
            for name in parameter_names:
                first, second = getattr(original, name), getattr(restored, name)
                torch.testing.assert_close(first, second)
                for key in ("step", "exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(original.optimizer.state[first][key], restored.optimizer.state[second][key])
            step(original)
            step(restored)
            for name in parameter_names:
                torch.testing.assert_close(getattr(original, name), getattr(restored, name), rtol=0, atol=0)


class RunMetadataTests(unittest.TestCase):
    def test_resolved_config_keeps_baseline_arguments_without_core(self):
        dataset = SimpleNamespace(source_path=Path("data/scene"), sh_degree=3)
        optimization = SimpleNamespace(iterations=500, seed=0)
        pipeline = SimpleNamespace(debug=False)

        resolved = build_resolved_config(dataset, optimization, pipeline)

        self.assertEqual(resolved["schema_version"], 1)
        self.assertEqual(
            resolved["model"],
            {"sh_degree": 3, "source_path": str(Path("data/scene"))},
        )
        self.assertEqual(
            resolved["optimization"], {"iterations": 500, "seed": 0}
        )
        self.assertEqual(resolved["pipeline"], {"debug": False})
        self.assertNotIn("core", resolved)
        self.assertEqual(resolved["training_path"], "baseline")

    def test_metadata_writer_creates_only_named_json_files(self):
        resolved = {"schema_version": 1, "training_path": "baseline"}
        identity = {"git_commit": "abc", "git_dirty": False}

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            write_run_metadata(output, resolved, identity)

            self.assertEqual(
                sorted(path.name for path in output.iterdir()),
                ["resolved_config.json", "run_identity.json"],
            )
            self.assertEqual(
                json.loads((output / "resolved_config.json").read_text()), resolved
            )
            self.assertEqual(
                json.loads((output / "run_identity.json").read_text()), identity
            )

    @unittest.skipUnless(shutil.which("git"), "git executable is required")
    def test_run_identity_reports_commit_and_dirty_state(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = Path(temporary_directory)
            subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
            tracked = repository / "tracked.txt"
            tracked.write_text("baseline\n", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.txt"], cwd=repository, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=E0 Test",
                    "-c",
                    "user.email=e0@example.invalid",
                    "commit",
                    "-qm",
                    "baseline",
                ],
                cwd=repository,
                check=True,
            )

            clean_identity = collect_run_identity(repository)
            tracked.write_text("changed\n", encoding="utf-8")
            dirty_identity = collect_run_identity(repository)

        self.assertEqual(len(clean_identity["git_commit"]), 40)
        self.assertFalse(clean_identity["git_dirty"])
        self.assertTrue(dirty_identity["git_dirty"])


class SeedTests(unittest.TestCase):
    def test_safe_state_default_zero_and_explicit_seed_replay(self):
        from utils.general_utils import safe_state
        python_before, numpy_before = random.getstate(), np.random.get_state()
        torch_before = torch.random.get_rng_state()
        try:
            for seed in (0, 7):
                with self.subTest(seed=seed), patch("torch.cuda.set_device"), patch("sys.stdout", io.StringIO()):
                    if seed == 0:
                        safe_state(True)
                    else:
                        safe_state(True, seed=seed)
                    self.assertEqual(random.random(), random.Random(seed).random())
                    self.assertEqual(float(np.random.rand()), float(np.random.RandomState(seed).rand()))
                    torch.testing.assert_close(torch.rand(4), torch.rand(4, generator=torch.Generator().manual_seed(seed)))
        finally:
            random.setstate(python_before)
            np.random.set_state(numpy_before)
            torch.random.set_rng_state(torch_before)



class StaticActionAuditTests(unittest.TestCase):
    def test_pose_reader_matches_baseline_text_fallback_order_and_binary_priority(self):
        from scripts import evaluate_candidate_actions as audit
        import struct
        spec = importlib.util.spec_from_file_location('actual_colmap_reader',
            Path(__file__).resolve().parents[1]/'scene/colmap_loader.py')
        loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loader)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, location in enumerate(('sparse/0','colmap/sparse/0','sparse')):
                source = root/str(index)
                pose = source/location; pose.mkdir(parents=True)
                (pose/'images.txt').write_text('9 1 0 0 0 .1 .2 .3 4 B.JPG\n\n2 1 0 0 0 .4 .5 .6 4 A.JPG\n\n')
                (pose/'cameras.txt').write_text('4 PINHOLE 32 24 40 41 16 12\n')
                selected, extrinsics, intrinsics, kind = audit.read_pose_files(source,loader)
                self.assertEqual(selected,pose)
                self.assertEqual(kind,'text')
                self.assertEqual(list(extrinsics),[9,2])
                self.assertEqual(extrinsics[9].name,'B.JPG')
                np.testing.assert_allclose(extrinsics[2].tvec,[.4,.5,.6])
                np.testing.assert_array_equal(intrinsics[4].params,[40,41,16,12])
                (pose/'images.bin').write_bytes(b'bad')
                self.assertEqual(audit.read_pose_files(source,loader)[3],'text')
                with (pose/'images.bin').open('wb') as f:
                    f.write(struct.pack('<Q',2))
                    for image_id,name in ((7,'D.JPG'),(3,'C.JPG')):
                        f.write(struct.pack('<idddddddi',image_id,1.,0.,0.,0.,.7,.8,.9,4))
                        f.write(name.encode()+b'\0'+struct.pack('<Q',0))
                (pose/'cameras.bin').write_bytes(struct.pack('<QiiQQdddd',1,4,1,32,24,45.,46.,16.,12.))
                _, extrinsics, intrinsics, kind = audit.read_pose_files(source,loader)
                self.assertEqual(kind,'binary')
                self.assertEqual(list(extrinsics),[7,3])
                np.testing.assert_array_equal(intrinsics[4].params,[45,46,16,12])

    def test_observer_mixed_acceptance_exports_exact_production_targets(self):
        from scripts import evaluate_candidate_actions as audit
        ref, sources, _, depth, normal = analytic_fixture()
        model = AnalyticGaussians(ref, count=16)
        rng = torch.Generator().manual_seed(0)
        rotation, _ = torch.linalg.qr(torch.randn(3,3,generator=rng))
        for camera in [ref,*sources]:
            camera.world_view_transform[:3,:3] = rotation
            camera.camera_center = camera.world_view_transform.inverse()[3,:3]
        tr = ref.world_view_transform
        model.normals = tr.new_tensor([[0.,0.,-1.]]) @ rotation.T
        points = (torch.rand(16,3,generator=rng)-.5)*.2
        points[:,2] = torch.tensor([3.,2.]*8)
        model._xyz = (points-tr[3,:3]) @ rotation.T
        fn = lambda c: depth.new_full((1,c.image_height,c.image_width),2.)
        direct = selector.select_candidate_labels_with_targets(model,ref,ref.gray,depth,
            normal,sources,options(),seed=0,iteration=7100,support_depth_fn=fn)
        self.assertGreater(int(direct[1].sum()),0)
        self.assertLess(int(direct[1].sum()),16)
        record, _ = audit.observe_selector(model,ref,ref.gray,depth,normal,sources,
            options(),seed=0,iteration=7100,support_depth_fn=fn)
        np.testing.assert_array_equal(record['ids'][record['accepted']],direct[3][0].numpy())
        np.testing.assert_array_equal(record['prior_world'][record['accepted']],direct[3][1].numpy())

    def test_observer_calls_real_selector_and_preserves_results_state_and_rng(self):
        from scripts import evaluate_candidate_actions as audit
        ref, sources, model, depth, normal = analytic_fixture()
        opts = options()
        fn = lambda c: depth.new_full((1, c.image_height, c.image_width), 2.)
        before, rng = model.get_xyz.clone(), torch.random.get_rng_state().clone()
        direct = selector.select_candidate_labels_with_targets(model, ref, ref.gray, depth,
            normal, sources, opts, seed=0, iteration=7100, support_depth_fn=fn)
        record, stats = audit.observe_selector(model, ref, ref.gray, depth, normal,
            sources, opts, seed=0, iteration=7100, support_depth_fn=fn)
        np.testing.assert_array_equal(record['ids'][record['accepted']], direct[3][0].numpy())
        np.testing.assert_allclose(record['prior_world'][record['accepted']], direct[3][1].numpy())
        self.assertEqual(stats['added'], int(direct[1].sum()))
        self.assertTrue(torch.equal(before, model.get_xyz))
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertIsNone(__import__('sys').getprofile())

    def test_observer_empty_path_and_restores_profiler_on_failure(self):
        from scripts import evaluate_candidate_actions as audit
        ref, sources, model, depth, normal = analytic_fixture()
        bad = lambda c: (_ for _ in ()).throw(RuntimeError('source failure'))
        with self.assertRaisesRegex(RuntimeError, 'source failure'):
            audit.observe_selector(model, ref, ref.gray, depth, normal, sources,
                options(), seed=0, iteration=7100, support_depth_fn=bad)
        self.assertIsNone(__import__('sys').getprofile())
        depth.fill_(float('nan'))
        record, stats = audit.observe_selector(model, ref, ref.gray, depth, normal,
            sources, options(), seed=0, iteration=7100, support_depth_fn=bad)
        self.assertEqual(len(record['ids']), 0)
        self.assertEqual(stats['sampled'], 4)

    def test_view_extension_and_literal_options(self):
        from scripts import evaluate_candidate_actions as audit
        names = ['v%02d' % i for i in range(20)]
        plan = audit.choose_views(names, ['v02', 'v17'], 5)
        self.assertEqual(plan['coverage'], ['v00','v05','v10','v14','v19'])
        self.assertEqual(plan['illustrative'], ['v02','v17'])
        self.assertEqual(len(set(plan['all'])), len(plan['all']))
        self.assertEqual(audit.literal_namespace('Namespace(seed=0, use_mono=False)'),
                         dict(seed=0, use_mono=False))
        with self.assertRaises(ValueError): audit.literal_namespace("Namespace(seed=__import__('os'))")

    @unittest.skipIf(importlib.util.find_spec('open3d') is None, 'Open3D unavailable')
    def test_real_selector_npz_to_gt_score_and_publication(self):
        from scripts import evaluate_candidate_actions as audit
        import open3d as o3d
        ref, sources, model, depth, normal = analytic_fixture()
        record, stats = audit.observe_selector(model, ref, ref.gray, depth, normal,
            sources, options(), seed=0, iteration=7100,
            support_depth_fn=lambda c: depth.new_full((1, c.image_height, c.image_width), 2.))
        camera_points = np.array([[-100.,-100.,2.], [100.,-100.,2.],
                                  [100.,100.,2.], [-100.,100.,2.]])
        tr = ref.world_view_transform.numpy()
        world = (camera_points-tr[3,:3]) @ tr[:3,:3].T
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            gt = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(world),
                o3d.utility.Vector3iVector([[0,1,2],[0,2,3]]))
            (p/'gt').mkdir()
            gt_path = p/'gt/mesh.ply'
            o3d.io.write_triangle_mesh(str(gt_path), gt)
            capture = p/'capture'; capture.mkdir()
            audit.write_capture_record(capture/'one.npz', record)
            manifest = dict(schema_version=1, records=[dict(file='one.npz',snapshot='clean7k',
                snapshot_iteration=7000,probe_iteration=7100,camera='analytic',
                groups=['illustrative','coverage'],stats=stats)])
            (capture/'capture_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            result = audit.score_capture(capture, gt_path, p/'score')
            row = next(x for x in result['rows'] if x['subset']=='accepted' and x['region']=='ALL')
            self.assertEqual(row['count'], 4)
            self.assertGreater(row['nearest_mean_gain_mm'], 999.)
            self.assertGreater(row['visible_z_mean_gain_mm'], 999.)
            self.assertEqual(row['nearest_improved_fraction'], 1.)
            self.assertLess(row['prior_normal_median_deg'], .1)
            self.assertTrue((p/'score/results.json').is_file())
            self.assertTrue((p/'score/metrics.csv').is_file())
            with self.assertRaises(FileExistsError): audit.score_capture(capture,gt_path,p/'score')


class SpreadIntegrationTests(unittest.TestCase):
    def test_real_training_helper_preview_cache_relocation_and_raw_prior(self):
        from utils.depth_spread import build_spread_proposal
        ref,sources,model,depth,normal = CandidateRelocationTests().fixture(4)
        ref.depth_dict['depth'] = depth * 2
        ref._candidate_support_cameras = sources
        ref.get_calib_matrix_nerf = lambda scale=1: (ref.get_k(scale),ref.world_view_transform.T)
        calls=[]
        def boundary(camera,current,*args,**kwargs):
            grad = torch.is_grad_enabled()
            calls.append((camera.uid,grad))
            if not grad:
                return dict(plane_depth=depth[None].clone(),rendered_normal=normal.clone(),
                            rendered_distance=depth[None].clone())
            z = (current.get_xyz@ref.world_view_transform[:3,:3]+ref.world_view_transform[3,:3])[:,2]
            return {'plane_depth':z.mean().expand(1,32,32)}
        root=Path(__file__).resolve().parents[1]
        n=next(x for x in ast.parse((root/'gaussian_renderer/__init__.py').read_text(encoding='utf-8')).body
               if isinstance(x,ast.FunctionDef) and x.name=='render_normal')
        ns=dict(torch=torch,F=torch.nn.functional,render=boundary,
                normal_from_depth_image=normal_from_depth_image,
                candidate_selector_active=selector.candidate_selector_active,
                candidate_relocation_active=selector.candidate_relocation_active,
                select_candidate_labels=selector.select_candidate_labels,
                select_candidate_labels_with_targets=selector.select_candidate_labels_with_targets,
                build_spread_proposal=build_spread_proposal)
        exec(compile(ast.Module(body=[n],type_ignores=[]),'<production normal>','exec'),ns)
        f=next(x for x in ast.parse((root/'train.py').read_text(encoding='utf-8')).body
               if isinstance(x,ast.FunctionDef) and x.name=='_render_training_reference')
        exec(compile(ast.Module(body=[f],type_ignores=[]),'<production training helper>','exec'),ns)
        opts=options(candidate_depth_source='spread',enable_candidate_relocation=True,
                     candidate_relocation_until_iter=15000,multi_view_sample_num=12,
                     candidate_selector_valid_pool=True,candidate_relocation_position_check=True)
        pkg,data=ns[f.name](ref,model,None,None,None,opts,iteration=7100,seed=0)
        self.assertEqual(calls,[(0,False),(1,False),(2,False),(0,True)])
        self.assertEqual(data['stats']['relocated'],4)
        self.assertEqual(int(data['stats']['position_passed']),4)
        torch.testing.assert_close(data['prior_depth'],depth*2)
        torch.testing.assert_close(data['prior_normal'],normal)
        torch.testing.assert_close(pkg['plane_depth'],depth.new_full((1,32,32),2.),atol=3e-5,rtol=2e-5)
        pkg['plane_depth'].sum().backward();model.optimizer.step()
        self.assertTrue(torch.isfinite(model.get_xyz).all())
        before = model.get_xyz.detach().clone()
        _, late = ns[f.name](ref,model,None,None,None,opts,iteration=15000,seed=0)
        self.assertNotIn('position_checked',late['stats'])
        self.assertNotIn('relocated',late['stats'])
        self.assertTrue(torch.equal(model.get_xyz,before))

    def test_shared_producer_to_real_observer_and_capture_config_override(self):
        from utils.depth_spread import build_spread_proposal
        from scripts import evaluate_candidate_actions as audit
        ref,sources,model,depth,normal=analytic_fixture()
        opts=options(candidate_depth_source='spread',multi_view_sample_num=12)
        geometry=dict(plane_depth=depth[None],rendered_normal=normal,rendered_distance=depth[None])
        source=lambda c: depth[None]
        proposal,valid,stats=build_spread_proposal(model,ref,sources,depth*2,normal,
            geometry,opts,seed=0,iteration=7100,support_depth_fn=source)
        kw=dict(seed=0,iteration=7100,support_depth_fn=source,
                proposal_depth=proposal,proposal_valid=valid)
        direct=selector.select_candidate_labels_with_targets(model,ref,ref.gray,depth*2,normal,
            sources,opts,**kw)
        record,counts=audit.observe_selector(model,ref,ref.gray,depth*2,normal,sources,opts,**kw)
        audit.validate_record(record)
        self.assertEqual(counts['added'],4)
        np.testing.assert_array_equal(record['prior_world'][record['accepted']],direct[3][1].numpy())
        # Exercise actual CLI parsing into capture, without invoking native CUDA.
        config=Path(__file__).resolve().parents[1]/'experiments/candidate_actions/toolroom_static_v1.json'
        with tempfile.TemporaryDirectory() as temp:
            source_path=Path(temp)
            argv=['audit','capture','--config',str(config),'--source',temp,
                  '--output',str(source_path/'new'),'--candidate-depth-source','spread']
            for name in ('native7k','c0_30k','v2_30k','early_30k'):
                argv.extend(['--snapshot',name+'='+temp])
            with patch('sys.argv',argv),patch.object(audit,'capture') as capture:
                audit.main()
            self.assertEqual(capture.call_args.args[0]['candidate_depth_source'],'spread')


class DepthSpreadTests(unittest.TestCase):
    def test_tilted_plane_from_single_anchor_and_normal_boundary(self):
        from utils.depth_spread import propagate_depth
        y,x = torch.meshgrid(torch.arange(13),torch.arange(13),indexing='ij')
        rays = torch.stack(((x-6)/40, (y-6)/40, torch.ones_like(x)), -1).float()
        n = torch.tensor([.1,.05,-1.]); n = n/n.norm()
        normals = n[:,None,None].expand(3,13,13).clone()
        truth = -2/(rays*n).sum(-1)
        start = torch.full_like(truth, 90.); start[6,6] = truth[6,6]
        anchors = torch.zeros_like(truth,dtype=torch.bool); anchors[6,6] = True
        usable = torch.ones_like(anchors)
        output, mask = propagate_depth(start,normals,rays,anchors,usable)
        self.assertTrue(mask.all())
        torch.testing.assert_close(output,truth,rtol=2e-5,atol=2e-5)
        empty, empty_mask = propagate_depth(start,normals,rays,anchors & False,usable)
        self.assertFalse(empty_mask.any()); self.assertFalse(empty.any())
        anchors.zero_(); anchors[6,2] = True; start[6,2] = truth[6,2]
        normals[:,:,6:] = torch.tensor([0.,.8,-.6])[:,None,None]
        bounded, reached = propagate_depth(start,normals,rays,anchors,usable)
        self.assertTrue(reached[:,:4].any()); self.assertFalse(reached[:,6:].any())
        normals[:,6,2] = float('nan')
        invalid, valid = propagate_depth(start,normals,rays,anchors,usable)
        self.assertTrue(torch.isfinite(invalid).all()); self.assertFalse(valid.any())
        grazing=torch.zeros_like(normals);grazing[0]=1.
        anchors.zero_();anchors[6,6]=True
        invalid,valid=propagate_depth(start,grazing,rays,anchors,usable)
        self.assertTrue(torch.isfinite(invalid).all());self.assertFalse(valid.any())

    def test_anchor_rgb_full_images_respect_ncc_scale_and_source_dimensions(self):
        from utils.depth_spread import build_spread_proposal
        ref=AnalyticCamera(ncc_scale=2)
        sources=[AnalyticCamera(1,.1,width=40,height=28,ncc_scale=2),
                 AnalyticCamera(2,-.1,ncc_scale=2)]
        for camera in [ref,*sources]:
            shift=camera.world_view_transform[3,0].item()-.2
            full=AnalyticCamera(camera.uid,shift,width=camera.image_width,height=camera.image_height)
            rgb=full.gray.expand(3,-1,-1).clone();gray=camera.gray
            camera.get_image=lambda rgb=rgb,gray=gray:(rgb,gray)
        model=AnalyticGaussians(ref)
        depth=ref.depth_dict['depth']
        normal=normal_from_depth_image(depth,ref.get_k(),ref.world_view_transform.T).permute(2,0,1)
        geometry=dict(plane_depth=depth[None],rendered_normal=normal,rendered_distance=depth[None])
        proposal,valid,stats=build_spread_proposal(model,ref,sources,depth,normal,
            geometry,options(),seed=0,iteration=7100,
            support_depth_fn=lambda c:depth.new_full((1,c.image_height,c.image_width),2.))
        self.assertGreater(int(stats['anchors']),0)
        self.assertTrue(valid[15:17,15:17].all())
        torch.testing.assert_close(proposal[valid],torch.full_like(proposal[valid],2.),atol=3e-5,rtol=2e-5)

    def test_real_rgb_anchor_producer_to_selector_targets_and_rng(self):
        from utils.depth_spread import build_spread_proposal
        ref,sources,model,depth,normal = analytic_fixture()
        opts = options(candidate_depth_source='spread',multi_view_sample_num=12)
        geometry = dict(plane_depth=depth[None],rendered_normal=normal,
                        rendered_distance=depth[None])
        rng = torch.get_rng_state().clone()
        source_calls=[]
        def source_depth(camera):
            source_calls.append(camera.uid)
            return depth[None]
        with patch.object(model,'get_normal',side_effect=AssertionError('anchor is a rendered plane')):
            proposal,valid,stats = build_spread_proposal(model,ref,sources,depth*2,normal,
                geometry,opts,seed=0,iteration=7100,support_depth_fn=source_depth)
        self.assertEqual(source_calls,[1])
        self.assertGreater(int(stats['anchors']),0)
        self.assertLessEqual(int(stats['anchors']),12)
        self.assertTrue(valid[15:17,15:17].all())
        torch.testing.assert_close(proposal[valid],torch.full_like(proposal[valid],2.),atol=3e-5,rtol=2e-5)
        self.assertTrue(torch.equal(torch.get_rng_state(),rng))
        result = selector.select_candidate_labels_with_targets(model,ref,ref.gray,
            depth*2,normal,sources,opts,seed=0,iteration=7100,support_depth_fn=source_depth,
            proposal_depth=proposal,proposal_valid=valid)
        self.assertEqual(int(result[1].sum()),4)
        camera_target = result[3][1]@ref.world_view_transform[:3,:3]+ref.world_view_transform[3,:3]
        torch.testing.assert_close(camera_target[:,2],torch.full((4,),2.),atol=3e-5,rtol=2e-5)
        missing = dict(geometry,plane_depth=torch.zeros_like(depth[None]))
        no_depth,no_mask,no_stats = build_spread_proposal(model,ref,sources,depth*2,normal,
            missing,opts,seed=0,iteration=7100,support_depth_fn=source_depth)
        self.assertFalse(no_mask.any()); self.assertFalse(no_depth.any())
        self.assertEqual(int(no_stats['anchors']),0)


class PositionCheckTests(unittest.TestCase):
    def fixture(self, count=4, budget=16384):
        ref, sources, model, depth, normal = CandidateRelocationTests().fixture(count)
        transform = ref.world_view_transform
        with torch.no_grad():
            points = depth.new_tensor([[0., 0., 2.2]]).repeat(count, 1)
            model._xyz.copy_((points-transform[3, :3]) @ transform[:3, :3].T)
        # Real reference/source images depict z=2. A tilted old plane at z=2.2
        # matches better than the SAME tilted plane moved to z=2; front-facing
        # rows benefit from moving. This uses the actual warp/NCC and sampler.
        normals = depth.new_tensor([[0., 0., -1.]]).repeat(count, 1)
        normals[1::2, 0] = -8.
        model.normals = normals @ transform[:3, :3].T
        opt = options(candidate_depth_source='spread', candidate_selector_budget=budget,
            enable_candidate_relocation=True, candidate_relocation_until_iter=15000,
            candidate_selector_valid_pool=True, candidate_relocation_position_check=True)
        kw = dict(seed=0, iteration=7100, support_depth_fn=lambda c: depth[None],
            proposal_depth=depth, proposal_valid=torch.ones_like(depth, dtype=torch.bool))
        return ref, sources, model, depth, normal, opt, kw

    def test_real_cli_and_metadata_preserve_opt_in_and_early_schedule(self):
        parser = ArgumentParser()
        group = OptimizationParams(parser)
        default = group.extract(parser.parse_args([]))
        for name in ('candidate_selector_valid_pool', 'candidate_relocation_position_check'):
            self.assertIs(getattr(default, name, None), False)
        opt = group.extract(parser.parse_args(['--enable_candidate_selector',
            '--candidate_depth_source', 'spread', '--candidate_selector_valid_pool',
            '--candidate_relocation_position_check', '--enable_candidate_relocation',
            '--candidate_relocation_until_iter', '15000', '--candidate_selector_budget', '16384']))
        metadata = build_resolved_config(SimpleNamespace(), opt, SimpleNamespace())['optimization']
        self.assertTrue(metadata['candidate_selector_valid_pool'])
        self.assertTrue(metadata['candidate_relocation_position_check'])
        self.assertEqual([i for i in range(30001) if selector.candidate_relocation_active(i, opt)],
                         list(range(7100, 15000, 100)))
        for changes in (dict(enable_candidate_selector=False), dict(candidate_depth_source='da3')):
            ref, sources, model, depth, normal, opt, kw = self.fixture()
            for key, value in changes.items(): setattr(opt, key, value)
            with self.assertRaises(ValueError):
                selector.candidate_selector_active(0, opt)

    def test_valid_proposal_pool_is_filtered_before_budget_without_rng_consumption(self):
        ref, sources, model, depth, normal = analytic_fixture()
        model = AnalyticGaussians(ref, count=80)
        centers = depth.new_full((80, 2), 20.)
        centers[71:75] = 12.
        points = torch.cat((centers, torch.ones(80, 1)), -1) @ ref.get_inv_k().T * 3
        transform = ref.world_view_transform
        model._xyz = (points-transform[3, :3]) @ transform[:3, :3].T
        model.scores[0] = 100.
        mask = torch.zeros_like(depth, dtype=torch.bool)
        mask[10:15, 10:15] = True
        opt = options(candidate_depth_source='spread', candidate_selector_budget=4,
                      candidate_selector_valid_pool=True)
        rng = torch.get_rng_state().clone()
        result = selector.select_candidate_labels_with_targets(model, ref, ref.gray,
            depth, normal, sources, opt, seed=0, iteration=7100,
            support_depth_fn=lambda c: depth[None], proposal_depth=depth, proposal_valid=mask)
        self.assertEqual(result[2]['sampled'], 4)
        self.assertIsNotNone(result[3], 'valid candidates must fill this budget before matching')
        self.assertEqual(result[3][0].tolist(), [71, 72, 73, 74])
        self.assertEqual(result[1].nonzero().flatten().tolist(), [71, 72, 73, 74])
        self.assertTrue(result[0][0])  # Original A survives the new pool filter.
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        mask.zero_()
        with patch.object(selector, '_source_cost', wraps=selector._source_cost) as cost:
            empty = selector.select_candidate_labels_with_targets(model, ref, ref.gray,
                depth, normal, sources, opt, seed=0, iteration=7100,
                support_depth_fn=lambda c: depth[None], proposal_depth=depth, proposal_valid=mask)
        self.assertEqual(cost.call_count, 0)
        self.assertEqual(empty[2]['sampled'], 0)
        self.assertEqual(empty[0].nonzero().flatten().tolist(), [0])
        self.assertIsNone(empty[3])

    def test_joint_win_position_loss_keeps_alr_but_returns_no_move(self):
        ref, sources, model, depth, normal, opt, kw = self.fixture()
        model.normals[:] = depth.new_tensor([[-8., 0., -1.]]) @ ref.world_view_transform[:3, :3].T
        before = model.get_xyz.detach().clone()
        result = selector.select_candidate_labels_with_targets(model, ref, ref.gray,
            depth, normal, sources, opt, **kw)
        self.assertEqual(int(result[1].sum()), 4)
        self.assertTrue(result[0].all())
        self.assertIsNone(result[3])
        self.assertEqual(int(result[2]['position_checked']), 4)
        self.assertEqual(int(result[2]['position_passed']), 0)
        self.assertTrue(torch.equal(model.get_xyz, before))

    def test_position_decision_is_strict_common_support_without_extra_cost_threshold(self):
        current = torch.tensor([[.8,.9],[.2,.7],[.2,.1],[.5,.8],[.3,.4]])
        proposed = torch.tensor([[.6,.7],[.9,.1],[.2,.1],[0.,.9],[float('nan'),0.]])
        valid = torch.tensor([[True,True],[True,False],[True,True],[False,True],[True,False]])
        self.assertEqual(selector._best_source_decision(current,proposed,valid,max_cost=None).tolist(),
                         [True,False,False,False,False])

    def test_prior_normal_invalidity_does_not_hide_a_valid_current_move_support(self):
        # The geometric core is real; only the image-cost boundary is controlled
        # to exercise a support whose proposal-normal warp is invalid.
        ref, sources, model, depth, normal, opt, kw = self.fixture()
        calls = iter(((.8, True), (.3, True), (.1, True), (float('inf'), False),
                      (.6, True), (.9, True)))
        def source_cost(*args):
            value, valid = next(calls)
            distances = args[8]
            return torch.full_like(distances, value), torch.full_like(distances, valid, dtype=torch.bool)
        with patch.object(selector, '_source_cost', side_effect=source_cost):
            result = selector.select_candidate_labels_with_targets(model, ref, ref.gray,
                depth, normal, sources, opt, **kw)
        self.assertEqual(int(result[1].sum()), 4)  # Joint .3 < .8, unchanged ALR.
        self.assertIsNone(result[3])  # Position min(.6,.9) > min(.8,.1).
        self.assertEqual(int(result[2]['position_passed']), 0)

    def test_partial_movement_preserves_unsorted_target_pairs_and_other_rows(self):
        ref, sources, model, depth, normal, opt, kw = self.fixture(count=12, budget=7)
        before = model.get_xyz.detach().clone()
        old_normal = model.normals.clone()
        result = selector.select_candidate_labels_with_targets(model, ref, ref.gray,
            depth, normal, sources, opt, **kw)
        joint_ids = set(result[1].nonzero().flatten().tolist())
        move_ids, targets = result[3]
        self.assertEqual(set(move_ids.tolist()), {i for i in joint_ids if i % 2 == 0})
        self.assertGreater(len(joint_ids), move_ids.numel())
        self.assertGreater(move_ids.numel(), 0)
        self.assertEqual(int(result[2]['position_passed']), move_ids.numel())
        with torch.no_grad(): model.relocate_selected(move_ids, targets)
        keep = torch.ones(12, dtype=torch.bool); keep[move_ids] = False
        self.assertTrue(torch.equal(model.get_xyz[keep], before[keep]))
        self.assertTrue(torch.equal(model.normals, old_normal))
        camera = model.get_xyz[move_ids] @ ref.world_view_transform[:3, :3] + ref.world_view_transform[3, :3]
        torch.testing.assert_close(camera[:, 2], torch.full((move_ids.numel(),), 2.))

    def test_labels_only_skips_position_cost_and_target_entry_reuses_source_renders(self):
        ref, sources, model, depth, normal, opt, kw = self.fixture()
        calls = []
        def source(camera):
            calls.append(camera.uid)
            return depth[None]
        kw['support_depth_fn'] = source
        with patch.object(selector, '_source_cost', wraps=selector._source_cost) as cost:
            labels = selector.select_candidate_labels(model, ref, ref.gray, depth, normal, sources, opt, **kw)
        self.assertEqual(cost.call_count, 4)
        self.assertEqual(calls, [1, 2])
        calls.clear()
        with patch.object(selector, '_source_cost', wraps=selector._source_cost) as cost:
            targets = selector.select_candidate_labels_with_targets(model, ref, ref.gray,
                depth, normal, sources, opt, **kw)
        self.assertEqual(cost.call_count, 6)
        self.assertEqual(calls, [1, 2])
        self.assertTrue(torch.equal(labels[0], targets[0]))
        self.assertTrue(torch.equal(labels[1], targets[1]))

    def test_disabled_options_match_frozen_v3_real_selector(self):
        frozen = subprocess.check_output(['git', 'show', '8b1def9:utils/candidate_selector.py'], text=True)
        namespace = {'__name__': 'frozen_v3'}
        exec(compile(frozen, '<frozen V3>', 'exec'), namespace)
        ref, sources, model, depth, normal, opt, kw = self.fixture(count=12, budget=7)
        opt.candidate_selector_valid_pool = False
        opt.candidate_relocation_position_check = False
        old = namespace['select_candidate_labels_with_targets'](model, ref, ref.gray,
            depth, normal, sources, opt, **kw)
        new = selector.select_candidate_labels_with_targets(model, ref, ref.gray,
            depth, normal, sources, opt, **kw)
        for a, b in zip(old[:2], new[:2]): self.assertTrue(torch.equal(a, b))
        self.assertEqual(old[2], new[2])
        for a, b in zip(old[3], new[3]): self.assertTrue(torch.equal(a, b))

    def test_real_observer_acceptance_is_alr_and_targets_can_be_a_subset(self):
        from scripts import evaluate_candidate_actions as audit
        ref, sources, model, depth, normal, opt, kw = self.fixture(count=12, budget=7)
        result = selector.select_candidate_labels_with_targets(model, ref, ref.gray,
            depth, normal, sources, opt, **kw)
        record, stats = audit.observe_selector(model, ref, ref.gray, depth, normal, sources, opt, **kw)
        audit.validate_record(record)
        self.assertEqual(int(record['accepted'].sum()), stats['added'])
        self.assertIn('position_passed', stats)
        self.assertGreater(stats['added'], stats['position_passed'])
        rows = {int(i): j for j, i in enumerate(record['ids'])}
        np.testing.assert_array_equal(record['prior_world'][[rows[int(i)] for i in result[3][0]]],
                                      result[3][1].numpy())


class ProcessTraceTests(unittest.TestCase):
    def test_capture_schedule_bounds_storage_and_keeps_late_observation(self):
        from scripts.trace_candidate_training import trace_iteration, probe_iteration
        chosen = [i for i in range(7001, 30001) if trace_iteration(i)]
        self.assertEqual(chosen, [7001] + list(range(7100,15000,100)) + list(range(15000,30001,1000)))
        self.assertEqual(len(chosen), 96)
        self.assertEqual([i for i in chosen if probe_iteration(i)], [7001]+list(range(8000,30001,1000)))

    def test_actual_selector_trace_copies_pre_move_and_records_narrowed_targets(self):
        from scripts.trace_candidate_training import CandidateTrainingTrace
        from scripts import evaluate_candidate_actions as audit
        ref,sources,model,depth,normal,opt,kw=PositionCheckTests().fixture(count=12,budget=7)
        ref.image_name='sample'
        before=model.get_xyz.clone();rng=torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as temp:
            trace=CandidateTrainingTrace(Path(temp)/'trace')
            with trace:
                result=selector.select_candidate_labels_with_targets(model,ref,ref.gray,depth,
                    normal,sources,opt,**kw)
            manifest=json.loads((trace.output/'trace_manifest.json').read_text())
            event=manifest['events'][0]
            with np.load(trace.output/event['candidate_file']) as z:
                record={k:z[k] for k in z.files}
            audit.validate_record(record)
            with np.load(trace.output/event['selection_file']) as z:
                moved=z['move_accepted'];targets=z['target_xyz']
                self.assertEqual(int(moved.sum()),len(result[3][0]))
                self.assertLess(int(moved.sum()),int(record['accepted'].sum()))
                np.testing.assert_array_equal(z['target_ids'],result[3][0].numpy())
                np.testing.assert_array_equal(targets,result[3][1].numpy())
            np.testing.assert_array_equal(record['current_world'],before[record['ids']].detach().numpy())
            with torch.no_grad(): model._xyz.add_(1)
            np.testing.assert_array_equal(record['current_world'],before[record['ids']].detach().numpy())
            self.assertFalse(event['reference_complete'])
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertIsNone(__import__('sys').getprofile())

    def test_late_joint_labels_are_not_misreported_as_moves_and_empty_path_works(self):
        from scripts.trace_candidate_training import CandidateTrainingTrace
        ref,sources,model,depth,normal,opt,kw=PositionCheckTests().fixture()
        ref.image_name='sample';kw['iteration']=15000
        with tempfile.TemporaryDirectory() as temp:
            with CandidateTrainingTrace(Path(temp)/'trace') as trace:
                result=selector.select_candidate_labels(model,ref,ref.gray,depth,normal,sources,opt,**kw)
            event=json.loads((trace.output/'trace_manifest.json').read_text())['events'][0]
            with np.load(trace.output/event['selection_file']) as z:self.assertFalse(z['move_accepted'].any())
            self.assertGreater(int(result[1].sum()),0)
        kw['iteration']=7100;kw['proposal_valid']=torch.zeros_like(depth,dtype=torch.bool)
        with tempfile.TemporaryDirectory() as temp:
            with CandidateTrainingTrace(Path(temp)/'trace') as trace:
                selector.select_candidate_labels_with_targets(model,ref,ref.gray,depth,normal,sources,opt,**kw)
            event=json.loads((trace.output/'trace_manifest.json').read_text())['events'][0]
            with np.load(trace.output/event['candidate_file']) as z:self.assertEqual(len(z['ids']),0)
            with np.load(trace.output/event['selection_file']) as z:self.assertEqual(len(z['target_ids']),0)

    def test_real_spread_inputs_replay_exactly_and_profile_restores_on_exception(self):
        from scripts.trace_candidate_training import CandidateTrainingTrace
        from utils.depth_spread import build_spread_proposal,propagate_depth
        ref,sources,model,depth,normal=analytic_fixture();ref.image_name='sample'
        opts=options(candidate_depth_source='spread',multi_view_sample_num=12)
        geometry=dict(plane_depth=depth[None],rendered_normal=normal,rendered_distance=depth[None])
        with tempfile.TemporaryDirectory() as temp:
            with CandidateTrainingTrace(Path(temp)/'trace') as trace:
                out,valid,stats=build_spread_proposal(model,ref,sources,depth*2,normal,
                    geometry,opts,seed=0,iteration=7100,support_depth_fn=lambda c:depth[None])
            event=json.loads((trace.output/'trace_manifest.json').read_text())['events'][0]
            with np.load(trace.output/event['spread_file']) as z:
                replay,rv=propagate_depth(torch.from_numpy(z['model_depth_before']),torch.from_numpy(z['prior_normal']),
                    torch.from_numpy(z['rays']),torch.from_numpy(z['anchor_mask']),torch.from_numpy(z['normal_valid']))
                self.assertTrue(torch.equal(replay,out));self.assertTrue(torch.equal(rv,valid))
                self.assertEqual(int(z['anchor_mask'].sum()),int(stats['anchors']))
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RuntimeError,'original failure'):
                with CandidateTrainingTrace(Path(temp)/'trace'):
                    raise RuntimeError('original failure')
            self.assertIsNone(__import__('sys').getprofile())
            with self.assertRaises(FileExistsError):CandidateTrainingTrace(Path(temp)/'trace')

    def test_actual_training_helper_captures_post_move_and_readonly_fixed_probe(self):
        from scripts.trace_candidate_training import CandidateTrainingTrace
        from utils.depth_spread import build_spread_proposal
        ref,sources,model,depth,normal,opt,kw=PositionCheckTests().fixture()
        ref.image_name='sample';ref._candidate_support_cameras=sources
        ref.get_calib_matrix_nerf=lambda scale=1:(ref.get_k(scale),ref.world_view_transform.T)
        calls=[]
        def boundary(camera,current,*args,**kwargs):
            calls.append((camera.uid,kwargs.get('return_depth_normal'),kwargs.get('ambiguity_labels')))
            z=(current.get_xyz@ref.world_view_transform[:3,:3]+ref.world_view_transform[3,:3])[:,2].mean()-.2
            d=(z if camera.uid==ref.uid else z*0+2).expand(1,32,32)
            return dict(plane_depth=d,rendered_normal=normal,rendered_distance=d,
                depth_normal=normal,rendered_unc=d*0,render=ref.get_image()[0])
        root=Path(__file__).resolve().parents[1]
        ns=dict(torch=torch,F=torch.nn.functional,render=boundary,
            normal_from_depth_image=normal_from_depth_image,
            candidate_selector_active=selector.candidate_selector_active,
            candidate_relocation_active=selector.candidate_relocation_active,
            select_candidate_labels=selector.select_candidate_labels,
            select_candidate_labels_with_targets=selector.select_candidate_labels_with_targets,
            build_spread_proposal=build_spread_proposal)
        n=next(x for x in ast.parse((root/'gaussian_renderer/__init__.py').read_text(encoding='utf-8')).body
            if isinstance(x,ast.FunctionDef) and x.name=='render_normal')
        exec(compile(ast.Module(body=[n],type_ignores=[]),str(root/'gaussian_renderer/__init__.py'),'exec'),ns)
        f=next(x for x in ast.parse((root/'train.py').read_text(encoding='utf-8')).body
            if isinstance(x,ast.FunctionDef) and x.name=='_render_training_reference')
        exec(compile(ast.Module(body=[f],type_ignores=[]),str(root/'train.py'),'exec'),ns)
        def training():
            scene=SimpleNamespace(getTrainCameras=lambda:[ref,*sources])
            return ns[f.name](ref,model,None,None,None,opt,iteration=8000,seed=0)
        rng=torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as temp:
            with CandidateTrainingTrace(Path(temp)/'trace',probe_view='sample') as trace:
                result,data=training()
            m=json.loads((trace.output/'trace_manifest.json').read_text());e=m['events'][0]
            self.assertTrue(e['reference_complete']);self.assertEqual(len(m['probes']),1)
            self.assertEqual(data['stats']['relocated'],2)
            self.assertEqual([x[:2] for x in calls],[(0,False),(1,False),(2,False),(0,True),(0,True)])
            np.testing.assert_array_equal(calls[-1][2].numpy(),selector.compute_sh_ambiguity_labels(model,opt).numpy())
            with np.load(trace.output/e['reference_file']) as z:
                np.testing.assert_array_equal(z['plane_depth'],result['plane_depth'].detach().numpy())
                self.assertEqual(len(z['actual_target_ids']),2)
            with np.load(trace.output/e['spread_file']) as z:
                self.assertGreater(float(z['model_depth_before'].mean()),float(result['plane_depth'].detach().mean()))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_launcher_preserves_original_arguments_and_restores_argv_on_failure(self):
        from scripts import trace_candidate_training as trace
        import sys
        before=sys.argv
        with tempfile.TemporaryDirectory() as temp:
            original=['--source_path','data','--model_path','new','--iterations','30000']
            def run(path,run_name):
                self.assertEqual(Path(path),trace.REPOSITORY/'train.py')
                self.assertEqual(run_name,'__main__')
                self.assertEqual(sys.argv[1:],original)
                raise RuntimeError('training failure')
            with patch.object(trace.runpy,'run_path',side_effect=run):
                with self.assertRaisesRegex(RuntimeError,'training failure'):
                    trace.main(['--trace-output',str(Path(temp)/'trace'),'--',*original])
            self.assertIs(sys.argv,before);self.assertIsNone(sys.getprofile())
            m=json.loads((Path(temp)/'trace/trace_manifest.json').read_text())
            self.assertFalse(m['successful']);self.assertEqual(m['error'],'training failure')


class SpreadContractTests(unittest.TestCase):
    def test_gray_source_matches_frozen_pre_spread_implementation(self):
        frozen=subprocess.check_output(['git','show','43e2ea0:utils/candidate_selector.py'],text=True)
        old=next(x for x in ast.parse(frozen).body if isinstance(x,ast.FunctionDef) and x.name=='_source_cost')
        ns=dict(vars(selector))
        exec(compile(ast.Module(body=[old],type_ignores=[]),'<frozen source matcher>','exec'),ns)
        ref,sources,model,depth,_=analytic_fixture()
        centers=torch.tensor([[2.,2.],[12.,15.],[16.,16.],[30.,30.]])
        patches=centers[:,None]+selector.patch_offsets(3,'cpu')
        points=(torch.cat((centers,torch.ones(4,1)),-1)@ref.get_inv_k().T)*3
        normal=torch.tensor([[0.,0.,-1.]]).expand(4,-1)
        args=(model,ref,sources[0],patches,selector._sample_map(ref.gray,patches).squeeze(-1),
              centers,points,normal,torch.full((4,),3.),depth[None],sources[0].gray,options())
        old_cost,old_valid=ns['_source_cost'](*args)
        new_cost,new_valid=selector._source_cost(*args)
        self.assertTrue(torch.equal(old_cost,new_cost));self.assertTrue(torch.equal(old_valid,new_valid))

    def test_source_option_and_old_sampling_stream(self):
        parser = ArgumentParser()
        group = OptimizationParams(parser)
        default = group.extract(parser.parse_args([]))
        self.assertEqual(getattr(default, "candidate_depth_source", None), "da3")
        spread = group.extract(parser.parse_args([
            "--enable_candidate_selector", "--candidate_depth_source", "spread"]))
        self.assertEqual(build_resolved_config(SimpleNamespace(), spread,
            SimpleNamespace())["optimization"]["candidate_depth_source"], "spread")
        with self.assertRaises(ValueError):
            selector.candidate_selector_active(7100, options(candidate_depth_source="typo"))
        pool = torch.arange(100)
        rng = torch.get_rng_state().clone()
        old = selector._sample_candidate_ids(pool, 12, seed=0, iteration=7100, camera_uid=2)
        same = selector._sample_candidate_ids(pool, 12, seed=0, iteration=7100,
            camera_uid=2, stream="ambisur-selector-v1")
        other = selector._sample_candidate_ids(pool, 12, seed=0, iteration=7100,
            camera_uid=2, stream="selector-v3-spread-anchors")
        self.assertTrue(torch.equal(old, same))
        self.assertFalse(torch.equal(old, other))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_proposal_targets_and_uncovered_rows_keep_a_without_matching(self):
        ref, sources, model, depth, normal = analytic_fixture()
        model.scores[0] = 100.
        raw_prior = depth * 2
        source_depth = lambda c: depth[None]
        kw = dict(seed=0, iteration=7100, support_depth_fn=source_depth,
                  proposal_depth=depth, proposal_valid=torch.ones_like(depth, dtype=torch.bool))
        result = selector.select_candidate_labels_with_targets(model, ref, ref.gray,
            raw_prior, normal, sources, options(candidate_depth_source="spread"), **kw)
        self.assertEqual(int(result[1].sum()), 3)
        target_camera = result[3][1] @ ref.world_view_transform[:3,:3] + ref.world_view_transform[3,:3]
        torch.testing.assert_close(target_camera[:,2], depth.new_full((3,), 2.))
        for mask in (torch.zeros_like(depth, dtype=torch.bool), kw['proposal_valid'].clone()):
            # Gaussian projects to(15.5,15.5); one missing corner forbids bilinear mixing.
            mask[16,16] = False
            with patch.object(selector, '_source_cost', wraps=selector._source_cost) as match:
                missing = selector.select_candidate_labels_with_targets(model, ref, ref.gray,
                    raw_prior, normal, sources, options(candidate_depth_source="spread"),
                    **dict(kw, proposal_valid=mask))
            self.assertEqual(match.call_count, 0)
            self.assertTrue(torch.equal(missing[0], selector.compute_sh_ambiguity_labels(model, options())))
            self.assertFalse(missing[1].any())
            self.assertIsNone(missing[3])
        with self.assertRaises(ValueError):
            selector.select_candidate_labels_with_targets(model, ref, ref.gray, depth,
                normal, sources, options(candidate_depth_source="spread"),
                seed=0, iteration=7100, support_depth_fn=source_depth)

    def test_rgb_cost_is_mean_of_real_channel_ncc_and_geometry(self):
        ref, sources, model, depth, _ = analytic_fixture()
        src = sources[0]
        centers = torch.tensor([[12.,15.], [16.,16.]])
        patches = centers[:,None] + selector.patch_offsets(3, 'cpu')
        rays = torch.cat((centers, torch.ones(2,1)), -1) @ ref.get_inv_k().T
        points = 2 * rays
        normal = torch.tensor([[0.,0.,-1.]]).expand(2,-1)
        distance = torch.full((2,), 2.)
        rgb = torch.cat((ref.gray, ref.gray.square(), 1-ref.gray), 0)
        src_rgb = torch.cat((src.gray, src.gray.square(), 1-src.gray), 0)
        args = (model, ref, src, patches)
        tail = (centers, points, normal, distance, depth[None])
        colors, valid = selector._source_cost(*args, selector._sample_map(rgb, patches),
            *tail, src_rgb, options())
        channels = []
        for c in range(3):
            costs, mask = selector._source_cost(*args,
                selector._sample_map(rgb[c:c+1], patches).squeeze(-1), *tail,
                src_rgb[c:c+1], options())
            self.assertTrue(torch.equal(valid, mask))
            channels.append(costs)
        torch.testing.assert_close(colors, torch.stack(channels, -1).mean(-1))
        geometry_valid, error = selector._roundtrip_geometry(model, ref, src,
            centers, points, depth[None])
        self.assertTrue(geometry_valid.all())
        self.assertLess(float(error.max()), 1e-4)


if __name__ == "__main__":
    unittest.main()
