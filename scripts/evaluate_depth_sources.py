"""Offline depth-source inference and paired target audit; never a training gate.

Capture uses evaluate_candidate_actions --save-depth-maps. GT is imported only
in score(). The fixed-normal NCC probe reuses production warping/sampling and
is explicitly not independent correspondence evidence or a training mask.
"""
import argparse
import ast
import csv
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from utils.run_metadata import collect_run_identity, write_run_metadata

VENDOR_COMMIT = '5bd49bba0992aa56060e1aa3bb40b2150b04726f'
WEIGHTS_URL = 'https://storage.googleapis.com/niantic-lon-static/research/mvsanywhere/mvsanywhere_hero.ckpt'


def array(value):
    return value.detach().cpu().numpy().copy()


def save_npz(path, data):
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream,**data)


def read_npz(path):
    with np.load(path,allow_pickle=False) as data:
        return dict(data)


def write_json(path, data):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(data,stream,indent=2,allow_nan=False)


def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def make_depth_map_record(ref, preview, prior, prior_normal, spread, spread_valid,
                          supports, source_depths):
    data = dict(current_depth=array(preview['plane_depth'][0]),da3_depth=array(prior),
        spread_depth=array(spread),spread_valid=array(spread_valid),
        current_normal=array(preview['rendered_normal']),prior_normal=array(prior_normal),
        camera_k=array(ref.get_k()),camera_transform=array(ref.world_view_transform),
        reference_gray=array(ref.get_image()[1]),image_size=np.array([ref.image_width,ref.image_height]),
        ncc_scale=np.array(ref.ncc_scale),znear=np.array(ref.znear))
    if len(supports)!=2 or len(source_depths)!=2:
        raise ValueError('depth source audit needs two recorded supports')
    for i,(cam,depth) in enumerate(zip(supports,source_depths)):
        data.update({f'support{i}_depth':array(depth[0]),
            f'support{i}_gray':array(cam.get_image()[1]),
            f'support{i}_k':array(cam.get_k()),
            f'support{i}_transform':array(cam.world_view_transform),
            f'support{i}_ncc_scale':np.array(cam.ncc_scale),
            f'support{i}_znear':np.array(cam.znear)})
    validate_depth_map(data)
    return data


def validate_depth_map(data):
    w,h=map(int,data['image_size'])
    for key in ('current_depth','da3_depth','spread_depth','spread_valid'):
        if data[key].shape!=(h,w):raise ValueError('invalid depth map shape: '+key)
    if data['spread_valid'].dtype!=np.bool_:raise ValueError('spread mask must be boolean')
    if data['current_normal'].shape!=(3,h,w) or data['prior_normal'].shape!=(3,h,w):
        raise ValueError('invalid normal map shape')
    for key in ('camera_k','camera_transform'):
        if not np.isfinite(data[key]).all():raise ValueError('non-finite '+key)
    if data['camera_k'].shape!=(3,3) or data['camera_transform'].shape!=(4,4):
        raise ValueError('invalid camera shape')
    for i in range(2):
        if data[f'support{i}_depth'].ndim!=2 or data[f'support{i}_k'].shape!=(3,3):
            raise ValueError('invalid source depth/camera')


def inference_camera(camera):
    pose=np.eye(4,dtype=np.float32)
    pose[:3,:3]=camera['rotation'];pose[:3,3]=camera['position']
    k=np.eye(4,dtype=np.float32)
    k[0,0],k[1,1]=camera['fx'],camera['fy']
    k[0,2],k[1,2]=camera['width']/2,camera['height']/2
    return pose,k


def outside_inputs(output, inputs):
    target=Path(output).resolve()
    for item in inputs:
        source=Path(item).resolve()
        if target==source or source in target.parents:
            raise ValueError('audit output must be outside read-only inputs')
    if target.exists():raise FileExistsError(target)


def regrid_depth(depth,source_k,target_k,target_shape,valid):
    """Sample the same camera rays using calibrated K-to-K coordinates.

    Pixel-center grids are not aligned by their image corners. Reject rays
    without a full valid interpolation footprint rather than extrapolating.
    """
    import torch
    h,w=map(int,target_shape);sh,sw=depth.shape
    if min(h,w,sh,sw)<2 or valid.shape!=depth.shape:raise ValueError('invalid regrid shapes')
    if (h,w)==(sh,sw) and np.array_equal(source_k,target_k):
        return np.where(valid&np.isfinite(depth)&(depth>0),depth,np.nan).copy()
    y,x=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij')
    pixels=torch.stack((x,y,torch.ones_like(x)),-1).float()
    projected=pixels@torch.tensor(target_k,dtype=torch.float32).inverse().T@torch.tensor(source_k,dtype=torch.float32).T
    xy=projected[...,:2]/projected[...,2:]
    grid=xy.clone();grid[...,0]=2*grid[...,0]/(sw-1)-1;grid[...,1]=2*grid[...,1]/(sh-1)-1
    good=valid&np.isfinite(depth)&(depth>0)
    sampled=torch.nn.functional.grid_sample(torch.tensor(np.where(good,depth,0),dtype=torch.float32)[None,None],
        grid[None],align_corners=True,padding_mode='zeros')[0,0]
    footprint=torch.nn.functional.grid_sample(torch.tensor(good,dtype=torch.float32)[None,None],
        grid[None],align_corners=True,padding_mode='zeros')[0,0]
    usable=(footprint>=1-1e-6)&torch.isfinite(sampled)&(sampled>0)&torch.isfinite(grid).all(-1)
    return np.where(usable.numpy(),sampled.numpy(),np.nan)


def regrid_mvs(prediction, target_k, target_shape):
    """Original MVS score validity, shared by the correspondence adapter."""
    depth=prediction['depth'];masked=prediction.get('depth_masked',depth)
    if depth.ndim!=2 or masked.shape!=depth.shape:raise ValueError('invalid MVS producer shape')
    valid=np.isfinite(depth)&(depth>0)&np.isfinite(masked)&(masked>0)
    return regrid_depth(depth,prediction['camera_k'],target_k,target_shape,valid)


def common_mean_admission(z,p,q,vz,vp,vq,max_cost=.5):
    costs=np.stack((z,p,q),-1)
    common=vz&vp&vq&np.isfinite(costs).all(-1)
    n=common.sum(-1)
    means=np.full((len(z),3),np.inf)
    good=n>0
    means[good]=np.where(common[...,None],costs,0).sum(1)[good]/n[good,None]
    accepted=good&(means[:,2]<means[:,0])&(means[:,2]<means[:,1])&(means[:,2]<max_cost)
    return accepted,good,means


def paired_target_stats(z,p,q,gt,domain,selected):
    domain=domain&np.isfinite(gt)&(gt>0)&np.isfinite(z)&(z>0)&np.isfinite(p)&(p>0)
    eligible=domain&np.isfinite(q)&(q>0)
    use=eligible&selected
    g0=(np.abs(z[use]-gt[use])-np.abs(q[use]-gt[use]))*1000
    gp=(np.abs(p[use]-gt[use])-np.abs(q[use]-gt[use]))*1000
    def mean(a):return float(a.mean()) if len(a) else None
    return dict(count=int(use.sum()),domain_count=int(domain.sum()),eligible_count=int(eligible.sum()),
        coverage=float(use.sum()/domain.sum()) if domain.any() else None,
        gain_current_mean_mm=mean(g0),gain_current_median_mm=float(np.median(g0)) if len(g0) else None,
        gain_da3_mean_mm=mean(gp),gain_da3_median_mm=float(np.median(gp)) if len(gp) else None,
        improved_current_fraction=mean(g0>0),improved_da3_fraction=mean(gp>0),
        harmed_da3_over5mm_fraction=mean(gp < -5),
        gain_da3_p05_mm=float(np.quantile(gp,.05)) if len(gp) else None,
        hybrid_da3_mean_gain_mm=float(gp.sum()/domain.sum()) if domain.any() else None)


def _camera(k,tr,depth,gray,ncc_scale,znear):
    import torch
    h,w=depth.shape
    cam=SimpleNamespace(image_width=w,image_height=h,world_view_transform=torch.tensor(tr),
        ncc_scale=float(ncc_scale),znear=float(znear),Fx=float(k[0,0]),Fy=float(k[1,1]),
        Cx=float(k[0,2]),Cy=float(k[1,2]))
    def get_k(scale=1):
        value=torch.tensor(k).clone();value[:2]/=scale
        return value
    cam.get_k=get_k;cam.get_inv_k=lambda scale=1:get_k(scale).inverse()
    return cam


def _offline_depth_sampler():
    """Execute the exact production method body, avoiding native CUDA imports.

    Same import adaptation already used by the shared selector CPU tests. The
    mathematical sampler is not copied or reimplemented; source hash is saved.
    """
    import torch
    path=ROOT/'scene/gaussian_model.py'
    cls=next(n for n in ast.parse(path.read_text(encoding='utf-8')).body
        if isinstance(n,ast.ClassDef) and n.name=='GaussianModel')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef)
        and n.name=='get_points_depth_in_depth_map')
    scope={'torch':torch}
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),scope)
    return type('ProductionDepthSampler',(),{'get_points_depth_in_depth_map':scope[method.name]})()


def ncc_probe(data,proposal,config,camera_uid=0):
    import torch
    from utils import candidate_selector as core
    from utils.graphics_utils import patch_offsets
    from scripts.evaluate_candidate_actions import _default_options
    validate_depth_map(data)
    z,p=data['current_depth'],data['da3_depth']
    if proposal.shape!=z.shape:raise ValueError('proposal grid mismatch')
    ref=_camera(data['camera_k'],data['camera_transform'],z,data['reference_gray'],
        data['ncc_scale'],data['znear'])
    _,opt,_=_default_options()
    # Do not reject a new target only because current rendered source depth
    # disagrees. Existing finite/positive projection checks remain active.
    opt.wo_use_geo_occ_aware=True
    radius=int(config['patch_radius']);opt.multi_view_patch_size=radius
    h,w=z.shape
    yy,xx=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij')
    all_centers=torch.stack((xx,yy),-1).float().reshape(-1,2)
    normals=torch.tensor(data['current_normal']).permute(1,2,0).reshape(-1,3)
    pool=torch.tensor(np.isfinite(z)&(z>ref.znear)&np.isfinite(p)&(p>ref.znear)
        &np.isfinite(proposal)&(proposal>ref.znear)).flatten()
    pool &= torch.isfinite(normals).all(-1)&(normals.norm(dim=-1)>1e-6)
    gh,gw=data['reference_gray'].shape[-2:]
    xy=all_centers/ref.ncc_scale
    pool &= (xy[:,0]>=radius)&(xy[:,0]<=gw-1-radius)&(xy[:,1]>=radius)&(xy[:,1]<=gh-1-radius)
    ids=core._sample_candidate_ids(pool.nonzero(as_tuple=True)[0],int(config['budget']),
        seed=int(config['seed']),iteration=7000,camera_uid=camera_uid,stream='depth-source-position-probe')
    centers=all_centers[ids];n=normals[ids];n=n/n.norm(dim=-1,keepdim=True)
    if not len(ids):
        return dict(pixel_ids=ids.numpy(),accepted=np.empty(0,bool),comparable=np.empty(0,bool),
                    mean_costs=np.empty((0,3)))
    rays=torch.cat((centers,torch.ones((len(ids),1))),-1)@ref.get_inv_k().T
    patch=centers[:,None]/ref.ncc_scale+patch_offsets(radius,torch.device('cpu')).float()
    ref_values=core._sample_map(torch.tensor(data['reference_gray']),patch).squeeze(-1)
    sampler=_offline_depth_sampler()
    all_cost,all_valid=[],[]
    with torch.no_grad():
        for target in (z,p,proposal):
            points=rays*torch.tensor(target).flatten()[ids,None]
            oriented,distance,plane_valid=core._oriented_plane(points,n)
            costs=[];valids=[]
            for i in range(2):
                sd=data[f'support{i}_depth'];gray=torch.tensor(data[f'support{i}_gray'])
                src=_camera(data[f'support{i}_k'],data[f'support{i}_transform'],sd,gray,
                    data[f'support{i}_ncc_scale'],data[f'support{i}_znear'])
                c,v=core._source_cost(sampler,ref,src,patch,ref_values,centers,points,
                    oriented,distance,torch.tensor(sd)[None],gray,opt)
                costs.append(c.numpy());valids.append((v&plane_valid).numpy())
            all_cost.append(np.stack(costs,-1));all_valid.append(np.stack(valids,-1))
    accepted,comparable,means=common_mean_admission(*all_cost,*all_valid,max_cost=config['max_cost'])
    return dict(pixel_ids=ids.numpy(),accepted=accepted,comparable=comparable,mean_costs=means)


def select_mvs_supports(views,neighbors,cameras,max_supports):
    """Use at most the saved ordered prefix, without padding or re-selection."""
    maximum=int(max_supports)
    if maximum<1:raise ValueError('MVS support maximum must be positive')
    planned={}
    for name in views:
        names=list(neighbors.get(name,[]))[:maximum]
        if (name not in cameras or not names or name in names or
                len(set(names))!=len(names) or any(n not in cameras for n in names)):
            raise ValueError('invalid or empty saved MVS neighbors for '+name)
        planned[name]=names
    return planned


def infer(args):
    import torch
    from PIL import Image
    import torchvision.transforms as transforms
    vendor=args.vendor.resolve(strict=True)
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=vendor,text=True).strip()
    if sha!=VENDOR_COMMIT:raise ValueError('fixed official MVSAnywhere commit required')
    if subprocess.check_output(['git','status','--porcelain'],cwd=vendor,text=True).strip():
        raise ValueError('vendor working tree must be clean')
    config=json.loads(args.config.read_text(encoding='utf-8'))
    outside_inputs(args.output,[args.source,args.baseline,vendor])
    cameras=json.loads((args.baseline/'cameras.json').read_text(encoding='utf-8'))
    by_name={c['img_name']:c for c in cameras}
    from scripts.evaluate_candidate_actions import choose_views
    views=choose_views(list(by_name),config['illustrative_views'],config['coverage_count'])['all']
    if args.views:views=args.views
    neighbors={r['ref_name']:r['nearest_name'] for r in map(json.loads,
        (args.baseline/'multi_view.json').read_text(encoding='utf-8').splitlines())}
    planned_supports=select_mvs_supports(views,neighbors,by_name,config['mvs_supports'])
    print('MVS planned support counts:',json.dumps({n:len(s) for n,s in planned_supports.items()}),flush=True)
    args.output.mkdir(parents=True,exist_ok=False)
    write_run_metadata(args.output,dict(command=sys.argv,config=config),collect_run_identity(ROOT))
    sys.path.insert(0,str(vendor/'src'))
    spec=importlib.util.spec_from_file_location('fixed_mvs_hub',vendor/'hubconf.py')
    hub=importlib.util.module_from_spec(spec);spec.loader.exec_module(hub)
    if not args.weights.is_file():
        args.weights.parent.mkdir(parents=True,exist_ok=True)
        torch.hub.download_url_to_file(WEIGHTS_URL,str(args.weights))
    # Inherit the official forward and preprocess verbatim; only replace the
    # constructor's unconditional checkpoint download with a configured file.
    class CachedInference(hub.MVSAnywhereInference):
        def __init__(self):
            torch.nn.Module.__init__(self)
            handler=hub.options.OptionsHandler()
            handler.parse_and_merge_options(config_filepaths=str(vendor/'configs/models/mvsanywhere_model.yaml'),
                                            ignore_cl_args=True)
            opts=handler.options;opts.load_weights_from_checkpoint=str(args.weights)
            self.model=hub.load_model_inference(opts,hub.get_model_class(opts)).eval()
            self.normalize=transforms.Normalize(mean=[.485,.456,.406],std=[.229,.224,.225])
            self.input_transform=transforms.Compose([transforms.ToTensor(),self.normalize])
    model=CachedInference().to(args.device).eval()
    records=[]
    with torch.inference_mode():
        for name in views:
            names=planned_supports[name]
            pose,k=inference_camera(by_name[name])
            inputs=[inference_camera(by_name[n]) for n in names]
            image=Image.open(args.source/'images'/(name+'.JPG')).convert('RGB')
            if image.size!=(by_name[name]['width'],by_name[name]['height']):
                raise ValueError('saved camera and RGB dimensions differ')
            images=[Image.open(args.source/'images'/(n+'.JPG')).convert('RGB') for n in names]
            out=model.preprocess_and_run(image,images,pose,np.stack([x[0] for x in inputs]),
                k.copy(),np.stack([x[1] for x in inputs]).copy(),depth_range=None,
                num_refinement_steps=int(config['mvs_refinement_steps']),
                target_image_size=config['mvs_image_size'],device=args.device)
            depth=array(out['depth']).astype(np.float32)
            if depth.ndim!=2 or not (np.isfinite(depth)&(depth>0)).any():
                raise ValueError('invalid actual MVS depth output')
            filename=name+'.npz'
            payload={key:array(value) for key,value in out.items()}
            _,prediction_k=inference_camera(by_name[name])
            prediction_k[0]*=depth.shape[1]/by_name[name]['width']
            prediction_k[1]*=depth.shape[0]/by_name[name]['height']
            payload['camera_k']=prediction_k[:3,:3]
            save_npz(args.output/filename,payload)
            records.append(dict(camera=name,file=filename,supports=names,support_count=len(names),
                available_support_count=len(neighbors[name]),shape=list(depth.shape)))
            print('MVSAnywhere',name,depth.shape,flush=True)
    write_json(args.output/'inference_manifest.json',dict(schema_version=1,vendor_commit=sha,
        records=records,config=config,baseline=str(args.baseline),source=str(args.source),
        support_policy=dict(maximum=int(config['mvs_supports']),selection='saved_order_prefix_without_padding'),
        metadata_sha256={n:file_sha(args.baseline/n) for n in ('cameras.json','multi_view.json')},
        units='camera-z in the coordinate scale of the provided saved cameras; no GT alignment',
        weights=str(args.weights),weights_bytes=args.weights.stat().st_size,
        weights_sha256=file_sha(args.weights)))


def score(args):
    import torch
    from scripts.evaluate_candidate_actions import GTQueries
    capture=json.loads((args.capture/'capture_manifest.json').read_text(encoding='utf-8'))
    inference=json.loads((args.mvs/'inference_manifest.json').read_text(encoding='utf-8'))
    config=json.loads(args.config.read_text(encoding='utf-8'))
    outside_inputs(args.output,[args.capture,args.mvs,args.gt.parent])
    predictions={r['camera']:r for r in inference['records']}
    if len(predictions)!=len(inference['records']) or set(predictions)!={r['camera'] for r in capture['records']}:
        raise ValueError('capture and MVS view sets must agree exactly')
    for snapshot in capture['snapshot_inputs'].values():
        if any(snapshot['metadata_sha256'][n]!=inference['metadata_sha256'][n]
               for n in ('cameras.json','multi_view.json')):
            raise ValueError('MVS and native capture camera/support identities differ')
    if inference['config']!=config:raise ValueError('MVS config differs from frozen score config')
    # Validate every real producer output before constructing any GT object.
    prepared=[]
    for meta in capture['records']:
        if meta['snapshot_iteration']!=7000 or not meta.get('depth_file'):
            raise ValueError('native7K complete depth-map capture required')
        data=read_npz(args.capture/meta['depth_file']);validate_depth_map(data)
        predicted=read_npz(args.mvs/predictions[meta['camera']]['file'])
        q=regrid_mvs(predicted,data['camera_k'],data['current_depth'].shape)
        probe=ncc_probe(data,q,config['ncc_probe'],meta['camera_uid'])
        prepared.append((meta,data,q,probe))
    args.output.mkdir(parents=True,exist_ok=False)
    # Save the non-GT mask/evidence first. This is an evaluation probe only.
    for meta,_,_,probe in prepared:save_npz(args.output/(meta['camera']+'_probe.npz'),probe)
    write_json(args.output/'frozen_protocol.json',dict(config=config,capture=capture,inference=inference,
        scope='Static source and fixed-normal NCC proxy audit. Not an observed-correspondence gate, training mask or surface benefit.'))
    query=GTQueries(args.gt,args.segments,args.annotations)
    rows=[]
    for meta,data,q,probe in prepared:
        frame=query.depth_grid(data['camera_k'],data['camera_transform'],data['image_size'])
        gt,visible=frame['depth'],frame['valid']
        sources={'current':data['current_depth'],'da3':data['da3_depth'],
            'spread':data['spread_depth'],'mvs':q}
        domains={name:visible&np.isfinite(d)&(d>0) for name,d in sources.items()}
        domains['spread'] &= data['spread_valid']
        common=np.logical_and.reduce(list(domains.values()))
        regions=[('ALL',np.ones(gt.shape,bool))]
        if query.face_ids is not None:
            regions += [(str(r)+':'+query.labels.get(int(r),'unlabeled'),frame['region']==r)
                for r in np.unique(frame['region'][visible])]
        for name,depth in sources.items():
            for label,mask in [('native',domains[name]),('common4',common)]:
                for region,rmask in regions:
                    use=mask&rmask;error=np.abs(depth[use]-gt[use])*1000
                    rows.append(dict(camera=meta['camera'],groups='|'.join(meta.get('groups',[])),kind='source',source=name,subset=label,
                        region=region,count=int(use.sum()),visible_count=int((visible&rmask).sum()),
                        abs_mean_mm=float(error.mean()) if len(error) else None,
                        abs_median_mm=float(np.median(error)) if len(error) else None,
                        abs_p90_mm=float(np.quantile(error,.9)) if len(error) else None))
        ids=probe['pixel_ids'];z=data['current_depth'].ravel()[ids]
        p=data['da3_depth'].ravel()[ids];qp=q.ravel()[ids];truth=gt.ravel()[ids]
        eligible=probe['comparable']
        rng=np.random.default_rng(int(config['ncc_probe']['seed'])+int(meta['camera_uid']))
        random_mask=np.zeros(len(ids),bool)
        candidates=np.flatnonzero(eligible)
        count=int(probe['accepted'].sum())
        if count:random_mask[rng.choice(candidates,count,replace=False)]=True
        for label,selected in [('all_comparable',eligible),('accepted',probe['accepted']),
            ('rejected_comparable',eligible&~probe['accepted']),('random_same_count',random_mask)]:
            for region,rmask in regions:
                row=dict(camera=meta['camera'],groups='|'.join(meta.get('groups',[])),kind='admission',source='mvs',subset=label,region=region)
                row.update(paired_target_stats(z,p,qp,truth,eligible&rmask.ravel()[ids],selected));rows.append(row)
        sparse_mask=np.zeros(gt.shape,bool);sparse_mask.ravel()[ids[probe['accepted']]]=True
        for region,rmask in regions:
            row=dict(camera=meta['camera'],groups='|'.join(meta.get('groups',[])),kind='sampled_updates_full_image',
                source='mvs',subset='accepted_sampled_pixels_only',region=region)
            row.update(paired_target_stats(data['current_depth'],data['da3_depth'],q,gt,visible&rmask,sparse_mask))
            rows.append(row)
        print('Scored',meta['camera'],'common4',int(common.sum()),'accepted',count,flush=True)
    write_run_metadata(args.output,dict(command=sys.argv,config=config),collect_run_identity(ROOT))
    summaries=[]
    keys=sorted({(r['kind'],r['source'],r['subset'],r['region'],g) for r in rows
        for g in ['all',*filter(None,r['groups'].split('|'))]})
    for kind,source,subset,region,group in keys:
        part=[r for r in rows if (r['kind'],r['source'],r['subset'],r['region'])==(kind,source,subset,region)
              and (group=='all' or group in r['groups'].split('|'))]
        total=sum(r['count'] for r in part)
        item=dict(kind=kind,source=source,subset=subset,region=region,group=group,views=len(part),count=total)
        for metric in ('abs_mean_mm','gain_current_mean_mm','gain_da3_mean_mm',
                       'improved_current_fraction','improved_da3_fraction','harmed_da3_over5mm_fraction'):
            good=[r for r in part if r.get(metric) is not None and r['count']]
            if good:item[metric]=sum(r[metric]*r['count'] for r in good)/sum(r['count'] for r in good)
        if kind=='source':
            item['visible_count']=sum(r['visible_count'] for r in part)
            item['coverage']=total/item['visible_count'] if item['visible_count'] else None
        else:
            denom=sum(r['domain_count'] for r in part)
            item['domain_count']=denom;item['coverage']=total/denom if denom else None
            item['hybrid_da3_mean_gain_mm']=sum((r['hybrid_da3_mean_gain_mm'] or 0)*r['domain_count'] for r in part)/denom if denom else None
        summaries.append(item)
    write_json(args.output/'results.json',dict(schema_version=1,rows=rows,summaries=summaries,
        interpretation='Paired visible first-hit z errors; overlapping pixels/views are not independent replicates. Admission hybrid gain/coverage is within sampled comparable pixels. sampled_updates_full_image instead uses the full baseline-valid visible image denominator, changing only the actually probed/accepted pixels; it is not an inferred dense mask. No target statistic proves final mesh gain. Source coverage uses all visible pixels; means pooled by counts, medians retained per view only.'))
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (args.output/'metrics.csv').open('x',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def server(args):
    """Three finite steps for the user's AutoDL process; no training or GT."""
    import torch
    import torchvision
    outside_inputs(args.output,[args.source,args.baseline,args.vendor,args.runtime])
    identity=collect_run_identity(ROOT)
    if identity['git_dirty']:raise RuntimeError('server audit requires committed clean code')
    args.output.mkdir(parents=True,exist_ok=False)
    write_run_metadata(args.output,dict(command=sys.argv),identity)
    env=os.environ.copy()
    cache=args.runtime.parent/'mvsanywhere-cache'
    cache.mkdir(parents=True,exist_ok=True)
    env.update(TORCH_HOME=str(cache/'torch'),HF_HOME=str(cache/'huggingface'),
        PIP_CACHE_DIR=str(cache/'pip'),OMP_NUM_THREADS='1',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1')
    if args.rasterizer:
        env['PYTHONPATH']=str(args.rasterizer)+os.pathsep+env.get('PYTHONPATH','')
    def run(name,command):
        print('START',name,flush=True)
        with (args.output/(name+'.log')).open('x',encoding='utf-8') as stream:
            result=subprocess.run([str(x) for x in command],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        (args.output/(name+'.exitcode')).write_text(str(result.returncode)+'\n')
        if result.returncode:raise subprocess.CalledProcessError(result.returncode,command)
        print('DONE',name,flush=True)
    rc=1
    try:
        run('capture',[sys.executable,'-u','-B',ROOT/'scripts/evaluate_candidate_actions.py','capture',
            '--config',args.config,'--source',args.source,'--snapshot',
            'native7k='+str(args.baseline/'chkpnt7000.pth'),'--save-depth-maps','--output',args.output/'capture'])
        if not args.vendor.exists():
            run('vendor_clone',['git','clone','https://github.com/nianticlabs/mvsanywhere.git',args.vendor])
            run('vendor_checkout',['git','-C',args.vendor,'checkout','--detach',VENDOR_COMMIT])
        else:
            sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.vendor,text=True).strip()
            if sha!=VENDOR_COMMIT:raise ValueError('existing vendor must be at pinned commit')
        if not (args.runtime/'pyvenv.cfg').is_file():
            run('runtime',[sys.executable,'-m','venv','--system-site-packages',args.runtime])
        python=args.runtime/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
        constraints=args.output/'preserve_torch.txt'
        constraints.write_text('torch=='+torch.__version__+'\ntorchvision=='+torchvision.__version__+'\n')
        run('dependencies',[python,'-m','pip','install','-c',constraints,'-r',
            ROOT/'experiments/depth_sources/inference-requirements.txt','--disable-pip-version-check'])
        run('freeze',[python,'-m','pip','freeze'])
        run('inference',[python,'-u','-B',Path(__file__),'infer','--config',args.config,
            '--source',args.source,'--baseline',args.baseline,'--vendor',args.vendor,
            '--weights',args.weights,'--output',args.output/'mvs','--device','cuda'])
        rc=0
    except subprocess.CalledProcessError as exc:
        rc=exc.returncode
        raise
    finally:
        (args.output/'audit.exitcode').write_text(str(rc)+'\n')


def build_correspondence_record(data, mvs_depth, probe, matches, seed=0, camera_uid=0):
    """Freeze non-GT evidence at the original pixel IDs, without new sampling."""
    from utils.observed_correspondence import project_depth,compare_candidates,triangulate_evidence
    validate_depth_map(data)
    ids=probe['pixel_ids'];h,w=data['current_depth'].shape
    if ids.dtype!=np.int64 or len(np.unique(ids))!=len(ids) or np.any((ids<0)|(ids>=h*w)):
        raise ValueError('invalid frozen pixel IDs')
    if len(matches)!=2 or mvs_depth.shape!=(h,w):raise ValueError('two supports and native MVS grid required')
    xy=np.c_[ids%w,ids//w].astype(float)
    targets=np.stack([data['current_depth'].ravel()[ids],data['da3_depth'].ravel()[ids],
        mvs_depth.ravel()[ids],np.where(data['spread_valid'],data['spread_depth'],np.nan).ravel()[ids]],1)
    costs=[];valid=[];ks=[];trs=[];shapes=[];nears=[]
    for i,match in enumerate(matches):
        k=data[f'support{i}_k'];tr=data[f'support{i}_transform'];shape=data[f'support{i}_depth'].shape
        near=float(data[f'support{i}_znear']);ks.append(k);trs.append(tr);shapes.append(shape);nears.append(near)
        c=[];v=[]
        for depth in targets.T:
            q,good=project_depth(xy,depth,data['camera_k'],data['camera_transform'],k,tr,shape,float(data['znear']),near)
            c.append(np.linalg.norm(q-match['source_xy'],axis=-1));v.append(good&match['forward_valid'])
        costs.append(np.stack(c,1));valid.append(np.stack(v,1))
    costs=np.stack(costs,2);valid=np.stack(valid,2)
    record=dict(pixel_ids=ids,reference_xy=xy,candidate_depths=targets,
        candidate_costs_px=costs,candidate_projection_valid=valid,image_size=data['image_size'],
        camera_k=data['camera_k'],camera_transform=data['camera_transform'],
        old_ncc_accepted=probe['accepted'],old_ncc_comparable=probe['comparable'],old_ncc_mean_costs=probe['mean_costs'])
    record.update(compare_candidates(costs,valid))
    for common in (3,4):
        count=record[f'common{common}_support_count']
        accepted=record['mvs_selected' if common==3 else 'mvs_selected_common4']
        for support_count in (1,2):
            eligible=count==support_count;pool=np.flatnonzero(eligible)
            random_mask=np.zeros(len(ids),bool)
            rng=np.random.default_rng(int(seed)+int(camera_uid))
            budget=int((accepted&eligible).sum())
            if budget:random_mask[rng.choice(pool,budget,replace=False)]=True
            record[f'common{common}_{support_count}support_random']=random_mask
    for i,match in enumerate(matches):record.update({f'support{i}_{key}':value for key,value in match.items()})
    tri=triangulate_evidence(xy,np.stack([m['source_xy'] for m in matches],1),
        np.stack([m['forward_valid'] for m in matches],1),data['camera_k'],data['camera_transform'],
        ks,trs,shapes,float(data['znear']),nears)
    record.update({'tri_'+key:value for key,value in tri.items()})
    return record


@contextmanager
def offline_correspondence_hub(config):
    """Dense batch consumes the existing official caches, never downloads."""
    import torch
    hub=Path(torch.hub.get_dir());weights=hub/'checkpoints/romav2.0.1.pt'
    if not weights.is_file():raise FileNotFoundError('missing cached RoMa checkpoint: '+str(weights))
    model=config['model']
    if weights.stat().st_size!=model['weights_bytes'] or file_sha(weights)!=model['weights_sha256']:
        raise ValueError('cached official RoMa checkpoint differs')
    commit=config['dinov3_commit'];repo='facebookresearch/dinov3:'+commit
    directory=hub/('facebookresearch_dinov3_'+commit)
    if not (directory/'hubconf.py').is_file():raise FileNotFoundError('missing pinned DINO source cache: '+str(directory))
    original_load=torch.hub.load;original_download=torch.hub.download_url_to_file
    def cached_load(repo_or_dir, model, *args, **kwargs):
        if repo_or_dir!=repo:raise RuntimeError('offline batch encountered an uncached hub repository')
        for key in ('skip_validation','trust_repo','force_reload','source'):kwargs.pop(key,None)
        return original_load(str(directory),model,*args,source='local',**kwargs)
    def no_download(*args,**kwargs):raise RuntimeError('offline batch requires an existing cache; downloads disabled')
    torch.hub.load=cached_load;torch.hub.download_url_to_file=no_download
    try:
        yield
    finally:
        torch.hub.load=original_load;torch.hub.download_url_to_file=original_download


def load_correspondence_model(config, vendor):
    """Shared unchanged official precise initialization for old and new batches."""
    import torch
    if not torch.cuda.is_available():raise RuntimeError('approved precise batch requires CUDA; no silent CPU fallback')
    torch.set_float32_matmul_precision('highest')
    cache=Path(torch.hub.get_dir())/'checkpoints'/'romav2.0.1.pt';cache.parent.mkdir(parents=True,exist_ok=True)
    model_config=config['model']
    if not cache.exists():torch.hub.download_url_to_file(model_config['weights_url'],str(cache),hash_prefix=model_config['weights_sha256'])
    if cache.stat().st_size!=model_config['weights_bytes'] or file_sha(cache)!=model_config['weights_sha256']:
        raise ValueError('official RoMaV2 checkpoint identity differs')
    sys.path.insert(0,str(vendor/'src'))
    from romav2 import RoMaV2
    import romav2.romav2 as donor_model
    import romav2.local_correlation as corr
    if not Path(donor_model.__file__).resolve().is_relative_to(vendor):raise ValueError('wrong installed RoMaV2')
    if corr.local_corr is None:raise RuntimeError('approved Linux fused-local-corr backend missing; no silent fallback')
    print('Initializing official RoMaV2 precise; DINOv3 code may be cached/downloaded.',flush=True)
    model=RoMaV2(RoMaV2.Cfg(setting='precise',compile=False))
    if (model.H_lr,model.W_lr,model.H_hr,model.W_hr,model.bidirectional)!=(800,800,1280,1280,True):
        raise ValueError('official precise settings differ')
    return model


def correspondence_match(args):
    """Eight actual RGB pairs; no GT argument, model sampling, or training."""
    import time
    import torch
    from PIL import Image
    from utils.observed_correspondence import query_matches
    config=json.loads(args.config.read_text(encoding='utf-8'))
    vendor=args.vendor.resolve(strict=True)
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=vendor,text=True).strip()
    if sha!=config['model']['commit']:raise ValueError('pinned RoMaV2 source required')
    if subprocess.check_output(['git','status','--porcelain'],cwd=vendor,text=True).strip():
        raise ValueError('RoMaV2 source must be clean')
    outside_inputs(args.output,[args.source,args.capture,args.mvs,args.probes,vendor])
    cap_path=args.capture/'capture_manifest.json';mvs_path=args.mvs/'inference_manifest.json'
    capture=json.loads(cap_path.read_text(encoding='utf-8'));mvs=json.loads(mvs_path.read_text(encoding='utf-8'))
    cap={r['camera']:r for r in capture['records']};pred={r['camera']:r for r in mvs['records']}
    for snapshot in capture['snapshot_inputs'].values():
        if any(snapshot['metadata_sha256'][n]!=mvs['metadata_sha256'][n] for n in ('cameras.json','multi_view.json')):
            raise ValueError('MVS/capture camera identities differ')
    prepared=[];image_receipts={}
    for view in config['views']:
        name=view['reference'];meta=cap[name];pm=pred[name]
        if meta['snapshot_iteration']!=7000 or meta['supports']!=view['supports']:
            raise ValueError('original native7K and saved supports required')
        dp=args.capture/meta['depth_file'];mp=args.mvs/pm['file'];pp=args.probes/(name+'_probe.npz')
        if file_sha(pp)!=view['probe_sha256']:raise ValueError('frozen original probe differs: '+name)
        data=read_npz(dp);validate_depth_map(data);probe=read_npz(pp)
        if len(probe['pixel_ids'])!=config['pixels_per_reference']:raise ValueError('frozen pixel count differs')
        if list(data['current_depth'].shape)!=config['native_shape']:raise ValueError('native shape differs')
        for im_name in [name,*view['supports']]:
            image=args.source/'images'/(im_name+'.JPG')
            if im_name not in image_receipts:
                with Image.open(image) as rgb:
                    if list(rgb.size)!=config['rgb_size']:raise ValueError('RGB header differs')
                image_receipts[im_name]=dict(bytes=image.stat().st_size,sha256=file_sha(image))
        raw=read_npz(mp)
        q=regrid_mvs(raw,data['camera_k'],data['current_depth'].shape)
        prepared.append((view,meta,data,probe,q,dict(depth_sha256=file_sha(dp),mvs_sha256=file_sha(mp),probe_sha256=file_sha(pp))))
    args.output.mkdir(parents=True,exist_ok=False)
    write_run_metadata(args.output,dict(command=sys.argv,config=config),collect_run_identity(ROOT))
    write_json(args.output/'input_receipt.json',dict(capture_sha256=file_sha(cap_path),mvs_manifest_sha256=file_sha(mvs_path),
        images=image_receipts,records=[dict(camera=x[0]['reference'],**x[-1]) for x in prepared]))
    model_config=config['model']
    print('Inputs verified: 4 references / 8 pairs / 65536 frozen pixels. Checking RoMa checkpoint.',flush=True)
    model=load_correspondence_model(config,vendor)
    print('Model initialized. Beginning actual 8-pair batch.',flush=True)
    records=[]
    for view,meta,data,probe,q,receipt in prepared:
        name=view['reference'];h,w=data['current_depth'].shape;ids=probe['pixel_ids'];xy=np.c_[ids%w,ids//w]
        matches=[];timing=[]
        for i,support in enumerate(view['supports']):
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
            print('START RoMaV2',name,'->',support,flush=True)
            with torch.inference_mode():
                prediction=model.match(args.source/'images'/(name+'.JPG'),args.source/'images'/(support+'.JPG'))
                match=query_matches(prediction,xy,(h,w),data[f'support{i}_depth'].shape)
            torch.cuda.synchronize();elapsed=time.monotonic()-start
            pair_file=name+'_'+support+'.npz';save_npz(args.output/pair_file,dict(pixel_ids=ids,**match))
            timing.append(dict(support=support,seconds=elapsed,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),file=pair_file,sha256=file_sha(args.output/pair_file)))
            matches.append(match);del prediction
            print('DONE',name,'->',support,'forward_valid',int(match['forward_valid'].sum()),'seconds',round(elapsed,2),flush=True)
        record=build_correspondence_record(data,q,probe,matches,config['seed'],meta['camera_uid']);filename=name+'.npz'
        save_npz(args.output/filename,record)
        records.append(dict(camera=name,camera_uid=int(meta['camera_uid']),supports=view['supports'],
            file=filename,sha256=file_sha(args.output/filename),pairs=timing,**receipt))
        print('FROZEN observed geometry',name,flush=True)
    write_json(args.output/'correspondence_manifest.json',dict(schema_version=1,config=config,records=records,
        model_commit=sha,backend='official fused-local-corr',torch=torch.__version__,
        weights_sha256=model_config['weights_sha256'],input_receipt_sha256=file_sha(args.output/'input_receipt.json'),
        interpretation='No GT read. RGB-only observed matches; original 7K candidate depths. Not independent evidence or a training mask.'))


def validate_routing_pair(pair, frozen):
    """A reused match must refer to precisely the frozen reference pixels."""
    ids=frozen['pixel_ids'];n=len(ids)
    if not np.array_equal(pair['pixel_ids'],ids):raise ValueError('cached pair pixel IDs differ')
    if not np.array_equal(pair['reference_xy'],frozen['reference_xy']):raise ValueError('cached pair reference pixels differ')
    for key,shape in (('source_xy',(n,2)),('forward_valid',(n,)),('cycle_valid',(n,)),('cycle_px',(n,))):
        if pair[key].shape!=shape:raise ValueError('invalid cached pair field: '+key)
    if pair['overlap_forward'].shape not in ((n,),(n,1)):raise ValueError('invalid overlap shape')


def routing_camera_metadata(source, baseline, native_shape):
    """CPU-only native host camera convention; binary/text reader is shared."""
    from scripts.evaluate_candidate_actions import read_pose_files
    from utils.graphics_utils import getWorld2View2
    spec=importlib.util.spec_from_file_location('routing_colmap_loader',ROOT/'scene/colmap_loader.py')
    loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
    pose_path,extr,intr,kind=read_pose_files(source,loader)
    archive={c['img_name']:c for c in json.loads((baseline/'cameras.json').read_text(encoding='utf-8'))}
    h,w=map(int,native_shape);cameras={}
    for e in extr.values():
        name=Path(e.name).stem
        if name not in archive:continue
        c=intr[e.camera_id];saved=archive[name]
        if c.model not in ('PINHOLE','SIMPLE_PINHOLE'):raise ValueError('undistorted PINHOLE source required')
        fx=float(c.params[0]);fy=float(c.params[1] if c.model=='PINHOLE' else c.params[0])
        if (c.width,c.height)!=(saved['width'],saved['height']) or not np.allclose([fx,fy],[saved['fx'],saved['fy']],atol=1e-3,rtol=0):
            raise ValueError('archive camera intrinsics differ: '+name)
        rotation=loader.qvec2rotmat(e.qvec)
        tr=getWorld2View2(rotation.T,np.asarray(e.tvec)).T
        expected_pose,_=inference_camera(saved)
        if not np.allclose(np.linalg.inv(tr.T),expected_pose,atol=5e-6,rtol=0):raise ValueError('archive pose differs: '+name)
        k=np.array([[fx*w/c.width,0,w/2],[0,fy*h/c.height,h/2],[0,0,1]],np.float32)
        cameras[name]=dict(k=k,transform=tr,shape=(h,w),znear=.01,image_name=Path(e.name).name)
    if set(cameras)!=set(archive):raise ValueError('COLMAP and baseline camera names differ')
    suffix='.bin' if kind=='binary' else '.txt'
    return cameras,dict(pose_format=kind,pose_files={name+suffix:file_sha(pose_path/(name+suffix)) for name in ('images','cameras')})


def build_routing_record(frozen, matches, cameras, seed=0, cycle_max_px=1., overlap_min=.5, dense=False):
    """Freeze variable-support evidence around the unchanged depth hypotheses."""
    from utils.observed_correspondence import project_depth,pair_geometry,routing_evidence
    n=len(frozen['pixel_ids']);s=len(matches)
    if s!=len(cameras) or s<(1 if dense else 2):raise ValueError('insufficient matching support cameras')
    ids=frozen['pixel_ids'];w,h=map(int,frozen['image_size']);xy=frozen['reference_xy']
    if ids.dtype!=np.int64 or len(np.unique(ids))!=n or np.any((ids<0)|(ids>=h*w)):
        raise ValueError('invalid frozen pixel IDs')
    if not np.array_equal(xy,np.c_[ids%w,ids//w]):raise ValueError('frozen reference coordinates differ')
    targets=frozen['candidate_depths']
    if targets.shape!=(n,4):raise ValueError('four frozen depth sources required')
    ref_k=frozen['camera_k'];ref_tr=frozen['camera_transform'];near=float(frozen.get('znear',.01))
    costs=[];valid=[];geometry=[]
    for match,cam in zip(matches,cameras):
        validate_routing_pair(match,frozen)
        geom=pair_geometry(xy,match['source_xy'],match['forward_valid'],ref_k,ref_tr,
            cam['k'],cam['transform'],cam['shape'],near,cam['znear'],compute_sensitivity=not dense);geometry.append(geom)
        c=[];v=[]
        for z in targets.T:
            projected,good=project_depth(xy,z,ref_k,ref_tr,cam['k'],cam['transform'],cam['shape'],near,cam['znear'])
            c.append(np.linalg.norm(projected-match['source_xy'],axis=1));v.append(good&match['forward_valid'])
        costs.append(np.stack(c,1));valid.append(np.stack(v,1))
    costs=np.stack(costs,2);valid=np.stack(valid,2)
    evidence={key:np.stack(values,1) for key,values in dict(
        forward_valid=[m['forward_valid'] for m in matches],cycle_valid=[m['cycle_valid'] for m in matches],
        cycle_px=[m['cycle_px'] for m in matches],overlap=[m['overlap_forward'].reshape(n) for m in matches],
        pair_valid=[g['valid'] for g in geometry],angle_deg=[g['angle_deg'] for g in geometry]).items()}
    record={key:frozen[key].copy() for key in ('pixel_ids','reference_xy','candidate_depths','image_size','camera_k','camera_transform')}
    record.update(all_candidate_costs_px=costs,all_candidate_projection_valid=valid,znear=np.array(near))
    record.update({'observed_'+key:value for key,value in evidence.items()})
    record.update({'pair_'+key:np.stack([g[key] for g in geometry],1) for key in geometry[0]})
    record.update(support_ks=np.stack([c['k'] for c in cameras]),
        support_transforms=np.stack([c['transform'] for c in cameras]))
    if dense:
        from utils.observed_correspondence import dense_routing_evidence
        record.update(dense_routing_evidence(costs,valid,evidence,targets,seed,cycle_max_px,overlap_min,near))
    else:
        record.update(routing_evidence(costs,valid,evidence,seed,cycle_max_px,overlap_min))
    return record


def query_dense_matches(predictions, reference_shape, source_shape, chunk_size=65536):
    """Query every native pixel once; chunks never trigger extra model forwards."""
    from utils.observed_correspondence import query_matches
    if not isinstance(chunk_size,int) or chunk_size<1:raise ValueError('positive integer query chunk size required')
    h,w=map(int,reference_shape);ids=np.arange(h*w,dtype=np.int64);parts={}
    for start in range(0,len(ids),chunk_size):
        use=ids[start:start+chunk_size];xy=np.c_[use%w,use//w]
        for key,value in query_matches(predictions,xy,reference_shape,source_shape).items():
            parts.setdefault(key,[]).append(value)
    return dict(pixel_ids=ids,**{k:np.concatenate(v,axis=0) for k,v in parts.items()})


def write_dense_maps(output, name, data):
    """Lossless float/label maps plus a fixed legend visualization, no GT."""
    from PIL import Image
    w,h=map(int,data['image_size']);shape=(h,w)
    save_npz(output/(name+'_maps.npz'),{
        k:data[k].reshape(shape) for k in ('hybrid_depth','winner','support_count',
            'original_source_bits','participating_source_bits','fallback_reason')})
    palette=np.array([[0,0,0],[50,120,255],[55,190,80],[245,90,55],[185,70,220]],np.uint8)
    Image.fromarray(palette[data['winner'].reshape(shape)+1]).save(output/(name+'_sources.png'))
    Image.fromarray((data['support_count'].reshape(shape)*100).astype(np.uint8)).save(output/(name+'_supports.png'))


def target_error_stats(target, truth, use):
    error=np.abs(target[use]-truth[use])*1000
    return dict(abs_mean_mm=float(error.mean()) if len(error) else None,
        abs_median_mm=float(np.median(error)) if len(error) else None,
        abs_p90_mm=float(np.quantile(error,.9)) if len(error) else None)


def dense_score_rows(data, truth, visible, regions):
    """Shared paired statistics with each actual reference's own valid domain."""
    n=len(truth);all_pixels=np.ones(n,bool);names=('current','da3','mvs','spread')
    labels=data['winner'];count=data['support_count'];bits=data['participating_source_bits']
    groups=[('full_image',all_pixels)]
    groups.extend((f'{i}_support',count==i) for i in (0,1,2))
    number=data['participating_sources'].sum(1)
    groups.extend((f'{i}_sources',number==i) for i in (2,3,4))
    groups.extend((f'selected_{name}',labels==i) for i,name in enumerate(names))
    groups.append(('fallback_da3',labels<0))
    groups.extend((f'fallback_reason_{i}',data['fallback_reason']==i) for i in (1,2,3))
    groups.append(('da3_not_compared',(count>0)&((bits&2)==0)))
    region_masks=[('all',all_pixels),*[(f'region_{int(i)}',regions==i) for i in np.unique(regions)]]
    for subset,mask in groups:
        comparisons=[('dense_policy',i,data['hybrid_depth']) for i in (range(4) if subset=='full_image' else (1,))]
        if subset=='full_image':comparisons.append(('dense_random',1,data['random_hybrid_depth']))
        for kind,index,target in comparisons:
            reference=data['candidate_depths'][:,index]
            for region,rmask in region_masks:
                domain=visible&mask&rmask
                # Both reference inputs deliberately mean X, not actual current/DA3.
                stats=paired_target_stats(reference,reference,target,truth,domain,all_pixels)
                use=domain&np.isfinite(reference)&(reference>0)&np.isfinite(target)&(target>0)
                row=dict(kind=kind,reference=names[index],subset=subset,region=region,
                    count=stats['count'],domain_count=stats['domain_count'],eligible_count=stats['eligible_count'],
                    coverage=stats['coverage'],visible_count=int(domain.sum()),
                    gain_reference_mean_mm=stats['gain_da3_mean_mm'],
                    gain_reference_median_mm=stats['gain_da3_median_mm'],
                    improved_reference_fraction=stats['improved_da3_fraction'],
                    harmed_reference_over5mm_fraction=stats['harmed_da3_over5mm_fraction'],
                    gain_reference_p05_mm=stats['gain_da3_p05_mm'])
                for prefix,values in (('reference',reference),('hybrid',target)):
                    errors=target_error_stats(values,truth,use)
                    row.update({prefix+'_'+k.removeprefix('abs_').replace('mean','mae'):v for k,v in errors.items()})
                yield row


def routing_preflight(config, source, existing, baseline):
    """Validate the approved cache/neighbor/photo identities before GPU work."""
    from PIL import Image
    manifest_path=existing/'correspondence_manifest.json'
    if file_sha(manifest_path)!=config['existing_manifest_sha256']:raise ValueError('original8 manifest differs')
    old=json.loads(manifest_path.read_text(encoding='utf-8'))
    if old['model_commit']!=config['model']['commit'] or old['config']['model']!=config['model']:
        raise ValueError('original RoMa model configuration differs')
    receipt_path=existing/'input_receipt.json'
    if file_sha(receipt_path)!=old['input_receipt_sha256']:raise ValueError('original image receipt differs')
    original_receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
    for name,expected in config['baseline_metadata_sha256'].items():
        if file_sha(baseline/name)!=expected:raise ValueError('baseline metadata differs: '+name)
    neighbors={r['ref_name']:r['nearest_name'] for r in map(json.loads,(baseline/'multi_view.json').read_text().splitlines())}
    cameras,pose_receipt=routing_camera_metadata(source,baseline,config['native_shape'])
    records={r['camera']:r for r in old['records']};prepared=[];images={};pair_count=0;reused=0
    for view in config['views']:
        name=view['reference'];meta=records[name];supports=view['supports']
        if supports!=neighbors[name] or len(set(supports))!=len(supports) or supports[:2]!=meta['supports']:
            raise ValueError('original full ordered neighbor list required: '+name)
        path=existing/meta['file']
        if file_sha(path)!=meta['sha256']:raise ValueError('frozen candidate record differs: '+name)
        frozen=read_npz(path);n=len(frozen['pixel_ids']);w,h=map(int,frozen['image_size'])
        if n!=config['pixels_per_reference'] or [h,w]!=config['native_shape']:
            raise ValueError('original reference pixel grid differs')
        reference=cameras[name]
        if not np.allclose(frozen['camera_k'],reference['k'],rtol=0,atol=1e-3) or not np.allclose(
                frozen['camera_transform'],reference['transform'],rtol=0,atol=5e-6):
            raise ValueError('native reference camera differs: '+name)
        cached={}
        for pair in meta['pairs']:
            pp=existing/pair['file']
            if file_sha(pp)!=pair['sha256']:raise ValueError('cached pair hash differs: '+pair['file'])
            data=read_npz(pp);validate_routing_pair(data,frozen)
            cached[pair['support']]=(data,pair,pp)
        if set(cached)!=set(meta['supports']) or len(cached)!=2:raise ValueError('exactly original two cached supports required')
        for image_name in [name,*supports]:
            if image_name not in images:
                path=source/'images'/cameras[image_name]['image_name']
                with Image.open(path) as image:
                    if list(image.size)!=config['rgb_size']:raise ValueError('RGB header differs: '+image_name)
                sha=file_sha(path)
                if image_name in original_receipt['images'] and sha!=original_receipt['images'][image_name]['sha256']:
                    raise ValueError('original photograph differs: '+image_name)
                images[image_name]=dict(file=path.name,bytes=path.stat().st_size,sha256=sha)
        pair_count+=len(supports);reused+=len(cached)
        prepared.append(dict(view=view,meta=meta,frozen=frozen,cached=cached))
    if (pair_count,reused,pair_count-reused)!=(config['pair_count'],config['reused_pairs'],config['new_pairs']):
        raise ValueError('approved pair budget differs')
    receipt=dict(existing_manifest_sha256=file_sha(manifest_path),existing_receipt_sha256=file_sha(receipt_path),
        baseline_metadata_sha256=config['baseline_metadata_sha256'],images=images,
        pair_count=pair_count,reused_pairs=reused,new_pairs=pair_count-reused,**pose_receipt)
    return prepared,cameras,receipt


def dense_routing_preflight(config, source, existing, baseline, capture, mvs):
    """Read the original complete depth producers; freeze inputs without GT."""
    from PIL import Image
    if int(config['query_chunk_size'])<1:raise ValueError('positive query chunk required')
    manifest_path=existing/'correspondence_manifest.json';receipt_path=existing/'input_receipt.json'
    if file_sha(manifest_path)!=config['existing_manifest_sha256']:raise ValueError('original routing manifest differs')
    old=json.loads(manifest_path.read_text(encoding='utf-8'))
    if old['model_commit']!=config['model']['commit'] or old['config']['model']!=config['model']:
        raise ValueError('original model differs')
    if file_sha(receipt_path)!=old['input_receipt_sha256']:raise ValueError('original image receipt differs')
    original=json.loads(receipt_path.read_text(encoding='utf-8'))
    for filename,expected in config['baseline_metadata_sha256'].items():
        if file_sha(baseline/filename)!=expected:raise ValueError('baseline metadata differs: '+filename)
    for directory,filename,key in ((capture,'capture_manifest.json','capture_manifest_sha256'),
                                    (mvs,'inference_manifest.json','mvs_manifest_sha256')):
        if file_sha(directory/filename)!=config[key]:raise ValueError('source manifest differs: '+filename)
    cap=json.loads((capture/'capture_manifest.json').read_text(encoding='utf-8'))
    pred=json.loads((mvs/'inference_manifest.json').read_text(encoding='utf-8'))
    if pred['metadata_sha256']!=config['baseline_metadata_sha256']:raise ValueError('MVS cameras differ')
    for snapshot in cap['snapshot_inputs'].values():
        if any(snapshot['metadata_sha256'][k]!=v for k,v in config['baseline_metadata_sha256'].items()):
            raise ValueError('capture cameras differ')
    cap_records={r['camera']:r for r in cap['records']};mvs_records={r['camera']:r for r in pred['records']}
    old_records={r['camera']:r for r in old['records']}
    neighbors={r['ref_name']:r['nearest_name'] for r in map(json.loads,(baseline/'multi_view.json').read_text().splitlines())}
    cameras,pose_receipt=routing_camera_metadata(source,baseline,config['native_shape'])
    prepared=[];images={};sources={};pair_count=0
    for view in config['views']:
        name=view['reference'];meta=cap_records[name];saved=old_records[name];supports=view['supports']
        if supports!=neighbors[name] or supports!=saved['supports'] or len(set(supports))!=len(supports):
            raise ValueError('original ordered neighbor list differs: '+name)
        if meta['snapshot_iteration']!=7000 or meta['supports']!=supports[:2]:raise ValueError('native7K capture required')
        identity=config['source_arrays'][name];pm=mvs_records[name]
        if meta['depth_file']!=identity['depth_file'] or pm['file']!=identity['mvs_file']:
            raise ValueError('source filenames differ: '+name)
        dp=capture/meta['depth_file'];mp=mvs/pm['file'];old_path=existing/saved['file']
        for path,key in ((dp,'depth_sha256'),(mp,'mvs_sha256')):
            if file_sha(path)!=identity[key]:raise ValueError('source '+key+' differs: '+name)
        if file_sha(old_path)!=saved['sha256']:raise ValueError('original record differs: '+name)
        data=read_npz(dp);validate_depth_map(data);native=cameras[name];prior=read_npz(old_path)
        h,w=data['current_depth'].shape
        if [h,w]!=config['native_shape']:raise ValueError('full native grid differs')
        for frozen in (data,prior):
            if not np.allclose(frozen['camera_k'],native['k'],atol=1e-3,rtol=0) or not np.allclose(
                    frozen['camera_transform'],native['transform'],atol=5e-6,rtol=0):
                raise ValueError('source camera differs: '+name)
        q=regrid_mvs(read_npz(mp),data['camera_k'],(h,w));ids=np.arange(h*w,dtype=np.int64)
        frozen=dict(pixel_ids=ids,reference_xy=np.c_[ids%w,ids//w],
            candidate_depths=np.stack([data['current_depth'].ravel(),data['da3_depth'].ravel(),q.ravel(),
                np.where(data['spread_valid'],data['spread_depth'],np.nan).ravel()],1),
            image_size=data['image_size'],camera_k=data['camera_k'],camera_transform=data['camera_transform'],znear=data['znear'])
        for image_name in [name,*supports]:
            if image_name in images:continue
            path=source/'images'/cameras[image_name]['image_name']
            with Image.open(path) as image:
                if list(image.size)!=config['rgb_size']:raise ValueError('RGB header differs: '+image_name)
            sha=file_sha(path)
            if sha!=original['images'][image_name]['sha256']:raise ValueError('original photograph differs: '+image_name)
            images[image_name]=dict(file=path.name,bytes=path.stat().st_size,sha256=sha)
        prepared.append(dict(view=view,meta=meta,frozen=frozen,cached={}))
        sources[name]=identity;pair_count+=len(supports)
    if pair_count!=config['pair_count']:raise ValueError('approved dense pair budget differs')
    return prepared,cameras,dict(existing_manifest_sha256=config['existing_manifest_sha256'],
        capture_manifest_sha256=config['capture_manifest_sha256'],mvs_manifest_sha256=config['mvs_manifest_sha256'],
        images=images,source_arrays=sources,pair_count=pair_count,reused_pairs=0,new_pairs=pair_count,**pose_receipt)


def routing_match(args):
    """Reuse 8 frozen RGB pairs, acquire only approved21, never load GT/model depths."""
    import shutil
    import time
    import torch
    from utils.observed_correspondence import query_matches
    config=json.loads(args.config.read_text(encoding='utf-8'));vendor=args.vendor.resolve(strict=True)
    dense=config.get('audit_type')=='four_source_dense'
    inputs=[args.source,args.existing,args.baseline,vendor]
    if dense:
        if args.capture is None or args.mvs is None:raise ValueError('dense routing needs original --capture and --mvs')
        inputs.extend([args.capture,args.mvs])
    outside_inputs(args.output,inputs)
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=vendor,text=True).strip()!=config['model']['commit']:
        raise ValueError('pinned RoMaV2 source required')
    if subprocess.check_output(['git','status','--porcelain'],cwd=vendor,text=True).strip():raise ValueError('RoMaV2 source must be clean')
    if dense:
        prepared,cameras,receipt=dense_routing_preflight(config,args.source,args.existing,args.baseline,args.capture,args.mvs)
    else:
        prepared,cameras,receipt=routing_preflight(config,args.source,args.existing,args.baseline)
    args.output.mkdir(parents=True,exist_ok=False)
    write_run_metadata(args.output,dict(command=sys.argv,config=config),collect_run_identity(ROOT))
    write_json(args.output/'input_receipt.json',receipt)
    print(f"Inputs verified: {len(prepared)} references / {receipt['pair_count']} pairs / "
          f"{receipt['reused_pairs']} reused / {receipt['new_pairs']} new. No training or GT.",flush=True)
    if dense:
        with offline_correspondence_hub(config):model=load_correspondence_model(config,vendor)
    else:
        model=load_correspondence_model(config,vendor)
    records=[]
    for item in prepared:
        view=item['view'];meta=item['meta'];frozen=item['frozen'];name=view['reference']
        xy=frozen['reference_xy'];ids=frozen['pixel_ids'];w,h=map(int,frozen['image_size'])
        matches=[];timings=[];supports=[]
        for support in view['supports']:
            filename=name+'_'+support+'.npz';path=args.output/filename;cam=cameras[support]
            if support in item['cached']:
                match,old_pair,old_path=item['cached'][support]
                with old_path.open('rb') as src,path.open('xb') as dst:shutil.copyfileobj(src,dst)
                timing=dict(support=support,file=filename,sha256=file_sha(path),reused=True,
                    original_sha256=old_pair['sha256'])
                print('REUSE',name,'->',support,flush=True)
            else:
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
                print('START RoMaV2',name,'->',support,flush=True)
                with torch.inference_mode():
                    prediction=model.match(args.source/'images'/cameras[name]['image_name'],args.source/'images'/cam['image_name'])
                    match=(query_dense_matches(prediction,(h,w),cam['shape'],int(config['query_chunk_size'])) if dense
                           else query_matches(prediction,xy,(h,w),cam['shape']))
                torch.cuda.synchronize();seconds=time.monotonic()-start
                match=dict(pixel_ids=ids,**match);validate_routing_pair(match,frozen);save_npz(path,match)
                timing=dict(support=support,file=filename,sha256=file_sha(path),reused=False,seconds=seconds,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
                del prediction
                print('DONE',name,'->',support,'seconds',round(seconds,2),flush=True)
            matches.append(match);timings.append(timing);supports.append(cam)
        record=build_routing_record(frozen,matches,supports,int(config['seed'])+int(meta['camera_uid']),
                                    dense=dense,**config['support_quality'])
        filename=name+'.npz';save_npz(args.output/filename,record)
        if dense:write_dense_maps(args.output,name,record)
        records.append(dict(camera=name,camera_uid=meta['camera_uid'],supports=view['supports'],pairs=timings,
            file=filename,sha256=file_sha(args.output/filename),original_record_sha256=meta['sha256']))
        print('FROZEN four-source routing',name,flush=True)
    write_json(args.output/'correspondence_manifest.json',dict(schema_version=3 if dense else 2,
        audit_type='four_source_dense' if dense else 'four_source_routing',
        config=config,records=records,model_commit=config['model']['commit'],backend='official fused-local-corr',
        torch=torch.__version__,input_receipt_sha256=file_sha(args.output/'input_receipt.json'),
        interpretation='Frozen offline source ranks and shared support evidence. No GT, no training teacher mask, no surface updates.'))


def routing_score_specs(data):
    """Describe policy/per-source paired statistics; original scorer serializes."""
    targets=data['candidate_depths'];n=len(targets);all_pixels=np.ones(n,bool)
    names=('current','da3','mvs','spread')
    for count in (3,4):
        for protocol,other in (('old_first2','new_best2'),('new_best2','old_first2')):
            p=f'{protocol}_common{count}';domain=data[p+'_domain'];labels=data[p+'_winner']
            random=data[p+'_random_winner'];paired=domain&data[f'{other}_common{count}_domain']
            def target_for(which):
                target=targets[:,1].copy();ids=np.flatnonzero(which>=0)
                target[ids]=targets[ids,which[ids]]
                return target
            for suffix,scope in (('full_frozen',all_pixels),('native',domain),('paired',paired)):
                subset=p+'_'+suffix
                for name,target in (('selected',target_for(labels)),('random_source_quota',target_for(random)),
                                    ('da3',targets[:,1]),('current',targets[:,0])):
                    extra={'policy_labels':random if name=='random_source_quota' else labels}
                    yield 'routing_policy',name,subset,scope,scope,target,extra
                if suffix=='full_frozen':continue
                for j in range(count):
                    yield 'routing_source',names[j],subset,scope,scope,targets[:,j],{}
                    yield 'routing_selection',names[j],subset,scope,scope&(labels==j),targets[:,j],{}
                yield 'routing_selection','abstain',subset,scope,scope&(labels<0),targets[:,1],{}


def correspondence_score(args):
    """Read only completed evidence and cached GT; reuse paired target statistics."""
    outside_inputs(args.output,[args.matches,args.gt_cache])
    manifest_path=args.matches/'correspondence_manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'));config=manifest['config']
    prepared=[]
    # Validate all non-GT records first. Do not query mesh or produce GT caches.
    for meta in manifest['records']:
        path=args.matches/meta['file']
        if file_sha(path)!=meta['sha256']:raise ValueError('frozen correspondence evidence differs')
        prepared.append((meta,read_npz(path)))
    args.output.mkdir(parents=True,exist_ok=False)
    write_run_metadata(args.output,dict(command=sys.argv,manifest_sha256=file_sha(manifest_path)),collect_run_identity(ROOT))
    rows=[];pixel_count=0;gt_receipts={}
    for meta,data in prepared:
        name=meta['camera'];gt_path=args.gt_cache/(name+'_gt_grid.npz')
        grid=read_npz(gt_path);gt_receipts[name]=file_sha(gt_path)
        if tuple(grid['depth'].shape)!=tuple(data['image_size'][::-1]):raise ValueError('cached GT native grid differs')
        ids=data['pixel_ids'];truth=grid['depth'].ravel()[ids]
        visible=grid['valid'].ravel()[ids]&np.isfinite(truth)&(truth>0)
        region_ids=grid['region'].ravel()[ids];pixel_count+=len(ids)
        if manifest.get('audit_type')=='four_source_dense':
            if not np.array_equal(ids,np.arange(grid['depth'].size,dtype=np.int64)):
                raise ValueError('dense score requires every native pixel in order')
            rows.extend(dict(camera=name,**row) for row in dense_score_rows(data,truth,visible,region_ids))
            print('Scored full native grid',name,'pixels',len(ids),flush=True)
            continue
        z,p,q,spread=data['candidate_depths'].T
        regs=[('all',np.ones(len(ids),bool)),*[(f'region_{int(r)}',region_ids==r) for r in np.unique(region_ids)]]
        def append(kind,source,subset,domain,selected,target,extra=None):
            for region,mask in regs:
                row=dict(camera=name,kind=kind,source=source,subset=subset,region=region)
                row.update(paired_target_stats(z,p,target,truth,domain&visible&mask,selected))
                use=domain&visible&mask&selected&np.isfinite(target)&(target>0)
                row.update(target_error_stats(target,truth,use))
                if extra:
                    labels=extra['policy_labels'];use_domain=domain&visible&mask
                    use_domain&=np.isfinite(z)&(z>0)&np.isfinite(p)&(p>0)&np.isfinite(target)&(target>0)
                    row.update(selected_count=int((use_domain&(labels>=0)).sum()),
                        abstain_count=int((use_domain&(labels<0)).sum()))
                rows.append(row)
        if manifest.get('audit_type')=='four_source_routing':
            for kind,source,subset,domain,selected,target,extra in routing_score_specs(data):
                append(kind,source,subset,domain,selected,target,extra)
            print('Scored cached routing GT',name,'pixels',len(ids),flush=True)
            continue
        # Random controls fixed without GT. Same non-GT domain and exact budget.
        for common in (3,4):
            counts=data[f'common{common}_support_count'];means=data[f'common{common}_means']
            for support_count in (1,2):
                eligible=counts==support_count
                selected=(means[:,2]<means[:,0])&(means[:,2]<means[:,1])&eligible
                random_mask=data[f'common{common}_{support_count}support_random']
                common_mask=data[f'common{common}_support_mask']
                cycle=np.stack([data[f'support{i}_cycle_valid']&
                    (data[f'support{i}_cycle_px']<=config['descriptive']['cycle_max_px']) for i in range(2)],1)
                overlap=np.stack([data[f'support{i}_overlap_forward'].ravel()>=config['descriptive']['overlap_min'] for i in range(2)],1)
                masks=[('all',eligible),('accepted',selected),('rejected',eligible&~selected),('random_same_count',random_mask),
                    ('old_ncc_accepted',eligible&data['old_ncc_accepted']),
                    ('cycle_le1_descriptive',eligible&((cycle&common_mask).sum(1)==support_count)),
                    ('overlap_ge_half_descriptive',eligible&((overlap&common_mask).sum(1)==support_count))]
                for label,mask in masks:
                    append('admission','mvs',f'common{common}_{support_count}support_{label}',eligible,mask,q)
                for s,target in enumerate((z,p,q,spread)[:common]):
                    append('source',('current','da3','mvs','spread')[s],f'common{common}_{support_count}support',eligible,eligible,target)
        for s,target in enumerate((z,p,q,spread)):
            eligible=np.isfinite(target)&(target>0)
            append('source',('current','da3','mvs','spread')[s],'native',np.ones(len(ids),bool),eligible,target)
        for j,label in enumerate(('pair0','pair1','joint')):
            append('triangulation',label,label,np.ones(len(ids),bool),data['tri_valid'][:,j],data['tri_depth'][:,j])
        print('Scored cached GT',name,'pixels',len(ids),flush=True)
    interpretation=('Four-source routing on original frozen pixels. Full/native/paired domains distinct; '
        'source-quota random labels frozen before GT. Current is a static comparison, not a self-supervision target. '
        'No learned-match independence, training mask or surface/mesh benefit established.' if manifest.get('audit_type')=='four_source_routing' else
        'Original sampled pixels, not dense coverage. Positive gains mean lower same-ray camera-z error. One/two shared supports separate. Random selected before GT. Descriptive groups are not validated gates. Learned matches and repeated photos are not independent. No surface or mesh benefit established.')
    if manifest.get('audit_type')=='four_source_dense':
        interpretation=('All native pixels, variable available depth sources on the same one/two supports. '
            'Each reference_X pair uses its own valid domain; primary DA3 table includes DA3 fallbacks. '
            'Source-quota random labels stratified before GT by native validity, participating sources, '
            'support count and fallback validity. Current winners are static targets, not self-supervision. '
            'No training gate, independent evidence or rendered-surface gain established.')
    write_json(args.output/'results.json',dict(schema_version=1,pixel_count=pixel_count,rows=rows,gt_cache_sha256=gt_receipts,interpretation=interpretation))
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (args.output/'metrics.csv').open('x',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='mode',required=True)
    i=sub.add_parser('infer')
    for flag in ('config','source','baseline','vendor','weights','output'):i.add_argument('--'+flag,type=Path,required=True)
    i.add_argument('--device',default='cuda');i.add_argument('--views',nargs='+')
    s=sub.add_parser('score')
    for flag in ('config','capture','mvs','gt','output'):s.add_argument('--'+flag,type=Path,required=True)
    s.add_argument('--segments',type=Path);s.add_argument('--annotations',type=Path)
    r=sub.add_parser('server')
    for flag in ('config','source','baseline','vendor','weights','runtime','output'):
        r.add_argument('--'+flag,type=Path,required=True)
    r.add_argument('--rasterizer',type=Path)
    m=sub.add_parser('match',help='fixed RGB correspondence and normalized triangulation; no GT')
    for flag in ('config','source','capture','mvs','probes','vendor','output'):
        m.add_argument('--'+flag,type=Path,required=True)
    c=sub.add_parser('correspondence-score',help='score frozen correspondence evidence using existing GT grids')
    for flag in ('matches','gt-cache','output'):c.add_argument('--'+flag,type=Path,required=True)
    m=sub.add_parser('routing-match',help='offline routing; dense config queries every native pixel, no GT')
    for flag in ('config','source','existing','baseline','vendor','output'):m.add_argument('--'+flag,type=Path,required=True)
    for flag in ('capture','mvs'):m.add_argument('--'+flag,type=Path)
    c=sub.add_parser('routing-score',help='same cached-GT scorer for frozen four-source routing')
    for flag in ('matches','gt-cache','output'):c.add_argument('--'+flag,type=Path,required=True)
    args=p.parse_args()
    if args.mode=='infer':infer(args)
    elif args.mode=='score':score(args)
    elif args.mode=='server':server(args)
    elif args.mode=='match':correspondence_match(args)
    elif args.mode=='routing-match':routing_match(args)
    else:correspondence_score(args)


if __name__=='__main__':main()
