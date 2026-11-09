"""Offline depth-source inference and paired target audit; never a training gate.

Capture uses evaluate_candidate_actions --save-depth-maps. GT is imported only
in score(). The fixed-normal NCC probe reuses production warping/sampling and
is explicitly not independent correspondence evidence or a training mask.
"""
import argparse
import ast
import csv
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
        pred=predicted['depth']
        masked=predicted.get('depth_masked',pred)
        source_valid=np.isfinite(pred)&(pred>0)&np.isfinite(masked)&(masked>0)
        if pred.ndim!=2 or masked.shape!=pred.shape:raise ValueError('invalid MVS producer shape')
        q=regrid_depth(pred,predicted['camera_k'],data['camera_k'],data['current_depth'].shape,source_valid)
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
    args=p.parse_args()
    if args.mode=='infer':infer(args)
    elif args.mode=='score':score(args)
    else:server(args)


if __name__=='__main__':main()
