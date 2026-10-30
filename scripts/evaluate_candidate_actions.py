"""Static real-selector capture (native GPU) and read-only GT scoring (CPU).

No optimizer updates, relocation, training loop, or GT input exists in capture. A scoped
Python return observer reads the existing selector's locals without rewriting
sampling, validity, NCC, or acceptance. Final snapshots are counterfactual
probes, not reconstructions of historical training decisions.
"""
import argparse
import ast
import csv
import gc
import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
from utils.run_metadata import collect_run_identity, write_run_metadata


def literal_namespace(text):
    try:
        node = ast.parse(text.strip(), mode='eval').body
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != 'Namespace' or node.args:
            raise ValueError('expected literal Namespace keyword arguments')
        if any(k.arg is None for k in node.keywords):
            raise ValueError('namespace expansion is unsupported')
        return {k.arg: ast.literal_eval(k.value) for k in node.keywords}
    except (SyntaxError, TypeError, ValueError) as exc:
        raise ValueError('expected saved literal Namespace') from exc


def choose_views(names, illustrative, coverage_count):
    names = sorted(names)
    if len(set(names)) != len(names) or not names or coverage_count <= 0:
        raise ValueError('unique camera names and positive coverage_count required')
    if len(set(illustrative)) != len(illustrative) or any(n not in names for n in illustrative):
        raise ValueError('illustrative views must exist and be unique')
    ix = np.rint(np.linspace(0, len(names)-1, min(coverage_count, len(names)))).astype(int)
    coverage = [names[i] for i in ix]
    return dict(illustrative=list(illustrative), coverage=coverage,
                all=list(dict.fromkeys([*illustrative, *coverage])))


def _array(tensor):
    return tensor.detach().cpu().numpy().copy()


def observe_selector(model, reference, gray, depth, normal, supports, opt, *,
                     seed, iteration, support_depth_fn, proposal_depth=None, proposal_valid=None):
    """Observe the actual core once; restore profiling and preserve tensor state."""
    import torch
    from utils import candidate_selector as selector
    code = inspect.unwrap(selector._select_candidate_geometry).__code__
    if sys.getprofile() is not None:
        raise RuntimeError('another Python profiler is active')
    captured = {}
    def observer(frame, event, arg):
        if event == 'return' and frame.f_code is code:
            captured.update(frame.f_locals)
    before = (id(model.get_xyz), model.get_xyz._version)
    sys.setprofile(observer)
    try:
        result = selector.select_candidate_labels_with_targets(model, reference, gray,
            depth, normal, supports, opt, seed=seed, iteration=iteration,
            support_depth_fn=support_depth_fn, proposal_depth=proposal_depth,
            proposal_valid=proposal_valid)
    finally:
        sys.setprofile(None)
    if before != (id(model.get_xyz), model.get_xyz._version):
        raise RuntimeError('static selector changed model xyz')
    labels, extra, stats, targets = result
    stats = {k: int(v) for k, v in stats.items()}
    stats.update(gaussian_count=len(model.get_xyz), sh_count=int((labels & ~extra).sum()))
    tr = reference.world_view_transform
    record = dict(camera_k=_array(reference.get_k()), camera_transform=_array(tr),
                  image_size=np.array([reference.image_width, reference.image_height], np.int64))
    if 'accepted' not in captured:
        record.update(ids=np.empty(0, np.int64), current_world=np.empty((0,3)),
            prior_world=np.empty((0,3)), current_normal=np.empty((0,3)), prior_normal=np.empty((0,3)),
            centers=np.empty((0,2)), accepted=np.empty(0,bool), comparable=np.empty(0,bool),
            cost_current=np.empty((0,2)), cost_prior=np.empty((0,2)), confidence=np.empty(0))
        return record, stats
    ids, accepted = captured['ids'], captured['accepted']
    world_p = (captured['points_p']-tr[3,:3]) @ tr[:3,:3].T
    if targets is not None:
        # FP32 matmul may round differently for full vs accepted-only batches.
        if not torch.equal(ids[accepted], targets[0]) or not torch.allclose(world_p[accepted], targets[1],rtol=1e-5,atol=1e-6):
            raise RuntimeError('observer target order differs from actual returned targets')
        world_p[accepted] = targets[1]  # export the literal production action
    record.update(ids=_array(ids), current_world=_array(model.get_xyz[ids]), prior_world=_array(world_p),
        current_normal=_array(captured['normal_g']), prior_normal=_array(captured['normal_p']),
        centers=_array(captured['centers']), accepted=_array(accepted),
        comparable=_array(captured['common'].any(-1)),
        cost_current=_array(torch.stack(captured['costs_g'], -1).masked_fill(~captured['common'],float('inf'))),
        cost_prior=_array(torch.stack(captured['costs_p'], -1).masked_fill(~captured['common'],float('inf'))))
    conf = getattr(reference, 'depth_dict', {}).get('conf')
    if conf is None:
        record['confidence'] = np.full(len(ids), np.nan, np.float32)
    else:
        conf = torch.nn.functional.interpolate(conf.to(depth.device)[None,None],
            size=depth.shape, mode='bilinear', align_corners=True).squeeze()
        record['confidence'] = _array(selector._sample_map(conf[None],captured['centers'][:,None])[:,0,0])
    if int(accepted.sum()) != stats['added'] or stats['comparable'] != int(record['comparable'].sum()):
        raise RuntimeError('actual selector counts differ from capture')
    return record, stats


def validate_record(record):
    expected = {'camera_k','camera_transform','image_size','ids','current_world','prior_world',
                'current_normal','prior_normal','centers','accepted','comparable',
                'cost_current','cost_prior','confidence'}
    if set(record) != expected:
        raise ValueError('unexpected candidate record fields')
    n = len(record['ids'])
    shapes = dict(camera_k=(3,3),camera_transform=(4,4),image_size=(2,),ids=(n,),
        current_world=(n,3),prior_world=(n,3),current_normal=(n,3),prior_normal=(n,3),
        centers=(n,2),accepted=(n,),comparable=(n,),cost_current=(n,2),cost_prior=(n,2),confidence=(n,))
    if any(np.asarray(record[k]).shape != shape for k,shape in shapes.items()):
        raise ValueError('candidate array shapes do not match')
    if record['accepted'].dtype != np.bool_ or record['comparable'].dtype != np.bool_:
        raise ValueError('candidate decision arrays must be boolean')
    if np.any(record['accepted'] & ~record['comparable']) or len(np.unique(record['ids'])) != n:
        raise ValueError('invalid candidate decision/ID relation')
    for k in ('current_world','prior_world','current_normal','prior_normal','centers','camera_k','camera_transform'):
        if not np.isfinite(record[k]).all():
            raise ValueError('non-finite candidate geometry: '+k)


def write_capture_record(path, record):
    validate_record(record)
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **record)


def _default_options():
    from arguments import ModelParams, OptimizationParams, PipelineParams
    p = argparse.ArgumentParser()
    m, o, pipeline = ModelParams(p), OptimizationParams(p), PipelineParams(p)
    args = p.parse_args([])
    return m.extract(args), o.extract(args), pipeline.extract(args)


def read_pose_files(source, loader):
    """Match readColmapSceneInfo's lookup/format fallback, without Scene writes."""
    poses = next((p for p in (source/'sparse/0',source/'colmap/sparse/0',source/'sparse')
                  if p.exists()),None)
    if poses is None:
        raise FileNotFoundError('Can not find COLMAP outcome: '+str(source))
    try:
        extrinsics = loader.read_extrinsics_binary(str(poses/'images.bin'))
        intrinsics = loader.read_intrinsics_binary(str(poses/'cameras.bin'))
        kind = 'binary'
    except Exception:
        extrinsics = loader.read_extrinsics_text(str(poses/'images.txt'))
        intrinsics = loader.read_intrinsics_text(str(poses/'cameras.txt'))
        kind = 'text'
    return poses, extrinsics, intrinsics, kind


def capture(config, source, snapshots, output):
    import torch
    from scene import colmap_loader
    from scene.dataset_readers import readColmapCameras
    from scene.gaussian_model import GaussianModel
    from utils.camera_utils import loadCam
    from gaussian_renderer import render, render_normal
    from utils.depth_spread import build_spread_proposal
    if not torch.cuda.is_available():
        raise RuntimeError('native CUDA capture required')
    if output.resolve() == source.resolve() or source.resolve() in output.resolve().parents:
        raise ValueError('capture output must be outside source data')
    for path in snapshots.values():
        run = path.parent if path.is_file() else path
        if output.resolve() == run.resolve() or run.resolve() in output.resolve().parents:
            raise ValueError('capture output must be outside model input directories')
    output.mkdir(parents=True, exist_ok=False)
    # Read production cameras directly: Scene initialization would write source PLY.
    poses, extrinsics, intrinsics, pose_format = read_pose_files(source,colmap_loader)
    priors = source/'sparse_da3_aligned/0'
    infos = readColmapCameras(extrinsics,intrinsics,str(source/'images'),str(source),str(priors),True)
    infos = {c.image_name: c for c in infos}
    views = choose_views(list(infos), config['illustrative_views'],config['coverage_count'])
    identity = collect_run_identity(REPOSITORY)
    if identity['git_dirty']:
        raise RuntimeError('capture requires committed clean code')
    write_run_metadata(output, dict(config=config, source=str(source),snapshots={k:str(v) for k,v in snapshots.items()},views=views), identity)
    records = []
    snapshot_inputs = {}
    for snapshot in config['snapshots']:
        name = snapshot['name']
        path = snapshots[name]
        run = path.parent if path.is_file() else path
        ds, opt, pipe = _default_options()
        ds.__dict__.update(literal_namespace((run/'cfg_args').read_text(encoding='utf-8')))
        opt.__dict__.update(literal_namespace((run/'cfg_opts').read_text(encoding='utf-8')))
        opt.enable_candidate_selector, opt.candidate_selector_budget = True, config['budget']
        if 'candidate_depth_source' in config:
            opt.candidate_depth_source = config['candidate_depth_source']
        opt.enable_candidate_relocation = False
        ds.source_path, ds.preload_img = str(source), False
        native = GaussianModel(ds.sh_degree)
        native.disable_trunc, native.trunc_sigma = ds.disable_trunc, ds.trunc_sigma
        if path.is_file():
            state, iteration = torch.load(path,map_location='cuda',weights_only=False)
            if iteration != snapshot['iteration']:
                raise ValueError('checkpoint iteration mismatch')
            native.restore(state,opt)
            del state
        else:
            ply = run/'point_cloud'/('iteration_'+str(snapshot['iteration']))/'point_cloud.ply'
            native.load_ply(str(ply))
        model_file = path if path.is_file() else ply
        snapshot_inputs[name] = dict(path=str(model_file),bytes=model_file.stat().st_size,
            modified_ns=model_file.stat().st_mtime_ns,
            metadata_sha256={f:hashlib.sha256((run/f).read_bytes()).hexdigest()
                for f in ('cfg_args','cfg_opts','cameras.json','multi_view.json')})
        recorded = {c['img_name']:c for c in json.loads((run/'cameras.json').read_text(encoding='utf-8'))}
        neighbors = {r['ref_name']:r['nearest_name'][:2]
            for r in map(json.loads,(run/'multi_view.json').read_text(encoding='utf-8').splitlines())}
        bg = torch.tensor([1.,1.,1.] if ds.white_background else [0.,0.,0.],device='cuda')
        for view in views['all']:
            supports = neighbors[view]
            if len(supports) != 2:
                raise ValueError('need two recorded supports for '+view)
            # Camera wraps entries in-place; keep reader arrays reusable/read-only.
            cams = [loadCam(ds,infos[n].global_id,
                infos[n]._replace(depth_dict=dict(infos[n].depth_dict)),1.) for n in [view,*supports]]
            ref = cams[0]
            c2w = ref.world_view_transform.inverse().T.detach().cpu().numpy()
            expected = recorded[view]
            if not (np.allclose(c2w[:3,:3],expected['rotation'],atol=1e-5) and
                    np.allclose(c2w[:3,3],expected['position'],atol=1e-5) and
                    np.isclose(ref.Fx,expected['fx']*ref.image_width/expected['width']) and
                    np.isclose(ref.Fy,expected['fy']*ref.image_height/expected['height'])):
                raise ValueError('source pose differs from saved model cameras')
            with torch.no_grad():
                depth = torch.nn.functional.interpolate(ref.depth_dict['depth'].detach()[None,None],
                    size=(ref.image_height,ref.image_width),mode='bilinear',align_corners=True).squeeze()
                normal = render_normal(ref,depth)
                spread = getattr(opt,'candidate_depth_source','da3') == 'spread'
                source_cache = {} if spread else None
                def source_depth(camera):
                    if source_cache is not None and camera.uid in source_cache:
                        return source_cache[camera.uid]
                    value=render(camera,native,pipe,bg,return_plane=True,return_depth_normal=False,ray_reg=False)['plane_depth']
                    if source_cache is not None:
                        source_cache[camera.uid]=value
                    return value
                proposal_args, spread_stats = {}, {}
                if spread:
                    preview=render(ref,native,pipe,bg,return_plane=True,return_depth_normal=False,ray_reg=False)
                    proposal,usable,spread_stats=build_spread_proposal(native,ref,cams[1:],depth,normal,
                        preview,opt,seed=config['seed'],iteration=snapshot['probe_iteration'],support_depth_fn=source_depth)
                    proposal_args=dict(proposal_depth=proposal,proposal_valid=usable)
                    del preview
                record, stats = observe_selector(native,ref,ref.get_image()[1],depth,normal,cams[1:],opt,
                    seed=config['seed'],iteration=snapshot['probe_iteration'],support_depth_fn=source_depth,**proposal_args)
                stats.update({k:int(v) for k,v in spread_stats.items()})
                if source_cache is not None:
                    source_cache.clear()
            filename = name+'_'+view+'.npz'
            write_capture_record(output/filename,record)
            records.append(dict(file=filename,snapshot=name,snapshot_iteration=snapshot['iteration'],
                probe_iteration=snapshot['probe_iteration'],camera=view,camera_uid=ref.uid,supports=supports,
                groups=[g for g in ('illustrative','coverage') if view in views[g]],stats=stats,
                model_commit=snapshot['model_commit'],options=vars(opt)))
            print(name,view,stats,flush=True)
            del record, cams, ref, depth, normal
        del native
        gc.collect();torch.cuda.empty_cache()
    manifest = dict(schema_version=1,records=records,views=views,config=config,snapshot_inputs=snapshot_inputs,
        capture_identity=identity,source=str(source),pose_source=str(poses),pose_format=pose_format,
        scope='Static counterfactual selector observations; no GT, optimization or relocation. Late probes are not historical decisions.')
    with (output/'capture_manifest.json').open('x',encoding='utf-8') as f:
        json.dump(manifest,f,indent=2,allow_nan=False)
    return manifest


def _face_regions(mesh, segments, annotations):
    seg = np.asarray(json.loads(segments.read_text(encoding='utf-8'))['segIndices'])
    groups = json.loads(annotations.read_text(encoding='utf-8'))['segGroups']
    if len(seg) != len(mesh['vertices']):
        raise ValueError('GT segmentation vertex count differs')
    mapping = {}
    for g in groups:
        for s in g['segments']:
            if s in mapping and mapping[s] != g['id']:
                raise ValueError('conflicting GT segments')
            mapping[s] = g['id']
    vertex = np.array([mapping.get(int(x),-1) for x in seg])
    faces = vertex[mesh['triangles']]
    ids = np.where((faces[:,0]==faces[:,1]) & (faces[:,0]==faces[:,2]),faces[:,0],-1)
    return ids, {g['id']:g['label'] for g in groups}


class GTQueries:
    def __init__(self, path, segments=None, annotations=None):
        import open3d as o3d
        from scripts.evaluate_mesh_geometry import _read_mesh, TriangleDistance
        mesh, self.origin = _read_mesh(path,1.)
        self.query = TriangleDistance(mesh)
        self.o3d, self.views = o3d, {}
        self.face_ids, self.labels = (None,{}) if segments is None else _face_regions(mesh,segments,annotations)

    def evaluate(self, record):
        import torch
        from utils.candidate_selector import _sample_map
        from utils.graphics_utils import normal_from_depth_image
        n = len(record['ids'])
        if not n: return None
        k, tr = record['camera_k'], record['camera_transform']
        r = np.c_[record['centers'],np.ones(n)] @ np.linalg.inv(k).T
        center = np.linalg.inv(tr)[3,:3]-self.origin
        direction = r @ tr[:3,:3].T
        rays = np.c_[np.broadcast_to(center,direction.shape),direction].astype(np.float32)
        hit = self.query.scene.cast_rays(self.o3d.core.Tensor(rays))
        z = hit['t_hit'].numpy()
        visible = np.isfinite(z)&(z>0)
        e_g = self.query.distances(record['current_world']-self.origin)
        e_p = self.query.distances(record['prior_world']-self.origin)
        current_z = (record['current_world'] @ tr[:3,:3]+tr[3,:3])[:,2]
        prior_z = (record['prior_world'] @ tr[:3,:3]+tr[3,:3])[:,2]
        key = (tuple(record['image_size']),k.tobytes(),tr.tobytes())
        if key not in self.views:
            w,h = record['image_size']
            u,v = np.meshgrid(np.arange(w),np.arange(h))
            ray = np.stack([u,v,np.ones_like(u)],-1) @ np.linalg.inv(k).T
            d = ray @ tr[:3,:3].T
            full = np.concatenate([np.broadcast_to(center,d.shape),d],-1).astype(np.float32)
            hits = self.query.scene.cast_rays(self.o3d.core.Tensor(full))
            depth = hits['t_hit'].numpy()
            good = np.isfinite(depth)&(depth>0)
            stencil = np.zeros_like(good)
            stencil[1:-1,1:-1] = good[1:-1,1:-1]&good[:-2,1:-1]&good[2:,1:-1]&good[1:-1,:-2]&good[1:-1,2:]
            if self.face_ids is not None:
                obj = np.full(depth.shape,-1,np.int64)
                obj[good] = self.face_ids[hits['primitive_ids'].numpy()[good]]
                for dy,dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                    stencil[1:-1,1:-1] &= obj[1:-1,1:-1]==obj[1+dy:h-1+dy,1+dx:w-1+dx]
            normal = normal_from_depth_image(torch.tensor(np.where(good,depth,0)),
                torch.tensor(k),torch.eye(4)).permute(2,0,1)
            self.views[key] = (normal,torch.tensor(stencil,dtype=torch.float32)[None])
        normal, stencil = self.views[key]
        pixels = torch.tensor(record['centers'],dtype=torch.float32)[:,None]
        gt_normal = _sample_map(normal,pixels)[:,0].numpy()
        mag = np.linalg.norm(gt_normal,axis=-1)
        normal_valid = visible & (mag>.5) & (_sample_map(stencil,pixels)[:,0,0].numpy()>=1-1e-6)
        gt_normal /= np.maximum(mag[:,None],1e-12)
        ag = np.degrees(np.arccos(np.clip((record['current_normal']*gt_normal).sum(-1),-1,1)))
        ap = np.degrees(np.arccos(np.clip((record['prior_normal']*gt_normal).sum(-1),-1,1)))
        region = np.full(n,-1,np.int64)
        if self.face_ids is not None:
            region[visible] = self.face_ids[hit['primitive_ids'].numpy()[visible]]
        return dict(nearest_g=e_g*1000,nearest_p=e_p*1000,
            visible=visible,z_g=np.abs(current_z-z)*1000,z_p=np.abs(prior_z-z)*1000,
            normal_valid=normal_valid,normal_g=ag,normal_p=ap,region=region,
            displacement=np.linalg.norm(record['prior_world']-record['current_world'],axis=-1)*1000)


def _summary(record, scored, mask):
    result = dict(count=int(mask.sum()),visible_count=int((mask&scored['visible']).sum()),
                  normal_count=int((mask&scored['normal_valid']).sum()))
    gain = scored['nearest_g'][mask]-scored['nearest_p'][mask]
    result.update(nearest_mean_gain_mm=float(gain.mean()),nearest_median_gain_mm=float(np.median(gain)),
        nearest_improved_fraction=float(np.mean(gain>0)),nearest_worsened_over5mm_fraction=float(np.mean(gain < -5)),
        displacement_median_mm=float(np.median(scored['displacement'][mask])))
    vz = mask&scored['visible']
    gain = scored['z_g'][vz]-scored['z_p'][vz]
    result['visible_z_mean_gain_mm'] = float(gain.mean()) if len(gain) else None
    result['visible_z_improved_fraction'] = float(np.mean(gain>0)) if len(gain) else None
    nv = mask&scored['normal_valid']
    for source in ('current','prior'):
        result[source+'_normal_median_deg'] = float(np.median(scored['normal_g' if source=='current' else 'normal_p'][nv])) if nv.any() else None
    result['normal_improved_fraction'] = float(np.mean(scored['normal_p'][nv] < scored['normal_g'][nv])) if nv.any() else None
    result['visible_position_improved_normal_not_fraction'] = float(np.mean(
        (scored['z_p'][nv]<scored['z_g'][nv]) & (scored['normal_p'][nv]>=scored['normal_g'][nv]))) if nv.any() else None
    result['visible_normal_improved_position_not_fraction'] = float(np.mean(
        (scored['normal_p'][nv]<scored['normal_g'][nv]) & (scored['z_p'][nv]>=scored['z_g'][nv]))) if nv.any() else None
    evidence = mask&record['comparable']
    margin = record['cost_current'][evidence].min(-1)-record['cost_prior'][evidence].min(-1)
    result['ncc_margin_median'] = float(np.median(margin)) if len(margin) else None
    conf = record['confidence'][mask];conf = conf[np.isfinite(conf)]
    result['confidence_median'] = float(np.median(conf)) if len(conf) else None
    return result


def score_capture(capture_dir, gt_path, output, segments=None, annotations=None):
    manifest = json.loads((capture_dir/'capture_manifest.json').read_text(encoding='utf-8'))
    if manifest['schema_version'] != 1 or not manifest['records']:
        raise ValueError('complete capture_manifest schema1 required')
    if (segments is None) != (annotations is None):
        raise ValueError('segments and annotations must be provided together')
    if output.resolve() == gt_path.resolve().parent or gt_path.resolve().parent in output.resolve().parents:
        raise ValueError('score output must be outside GT input directory')
    output.mkdir(parents=True,exist_ok=False)
    query = GTQueries(gt_path,segments,annotations)
    rows = []
    for meta in manifest['records']:
        filename = Path(meta['file'])
        if filename.name != str(filename) or filename.suffix != '.npz':
            raise ValueError('capture record must be an immediate NPZ filename')
        with np.load(capture_dir/filename,allow_pickle=False) as f:
            record = dict(f)
        validate_record(record)
        if int(record['accepted'].sum()) != meta['stats']['added']:
            raise ValueError('capture stats/actual accepted rows differ')
        scored = query.evaluate(record)
        if scored is None: continue
        regions = [('ALL',np.ones(len(record['ids']),bool))]
        if query.face_ids is not None:
            regions += [(str(r)+':'+query.labels.get(r,'mixed_or_unlabeled'),scored['region']==r)
                        for r in np.unique(scored['region'])]
        for subset, selected in [('accepted',record['accepted']),
                                 ('rejected_comparable',record['comparable']&~record['accepted'])]:
            for region, region_mask in regions:
                mask = selected&region_mask
                if not mask.any(): continue
                rows.append(dict(snapshot=meta['snapshot'],snapshot_iteration=meta['snapshot_iteration'],
                    probe_iteration=meta['probe_iteration'],camera=meta['camera'],groups='|'.join(meta['groups']),
                    subset=subset,region=region,**_summary(record,scored,mask)))
        print(meta['snapshot'],meta['camera'],'scored',len(record['ids']),flush=True)
    if not rows:
        raise ValueError('no scoreable candidate observations')
    summaries = []
    for snapshot in sorted({r['snapshot'] for r in rows}):
        for group in ('illustrative','coverage'):
            for subset in ('accepted','rejected_comparable'):
                part = [r for r in rows if r['snapshot']==snapshot and group in r['groups'].split('|')
                        and r['subset']==subset and r['region']=='ALL']
                if not part: continue
                count = sum(r['count'] for r in part)
                summaries.append(dict(snapshot=snapshot,group=group,subset=subset,views=len(part),count=count,
                    nearest_pooled_mean_gain_mm=sum(r['nearest_mean_gain_mm']*r['count'] for r in part)/count,
                    nearest_macro_view_mean_gain_mm=float(np.mean([r['nearest_mean_gain_mm'] for r in part])),
                    nearest_pooled_improved_fraction=sum(r['nearest_improved_fraction']*r['count'] for r in part)/count))
    result = dict(schema_version=1,rows=rows,summaries=summaries,capture=str(capture_dir),gt=str(gt_path),
        capture_manifest_sha256=hashlib.sha256((capture_dir/'capture_manifest.json').read_bytes()).hexdigest(),
        interpretation='Positive gain means candidate is closer to GT. Candidate-view observations may repeat Gaussians; not independent samples, formal CD, historical causal effects, or training labels.',
        normal_domain='GT first-hit depth-derived normal, finite five-point stencil; unanimous semantic neighborhood when annotations supplied. Boundary/hidden GT domains reported by valid counts.',
        visibility_domain='Nearest-GT distance and same-reference first-hit-z errors are separate; neither alone certifies Gaussian-to-surface ownership.',
        evidence='NCC costs use the actual common-valid support intersection, as in original acceptance. Illustrative and systematic coverage groups are separate. No significance inference from repeated observations.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_run_metadata(output,dict(command=sys.argv,capture=str(capture_dir),gt=str(gt_path)),collect_run_identity(REPOSITORY))
    with (output/'metrics.csv').open('x',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (output/'results.json').open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2,allow_nan=False)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode',required=True)
    c = sub.add_parser('capture');c.add_argument('--config',type=Path,required=True)
    c.add_argument('--source',type=Path,required=True);c.add_argument('--snapshot',action='append',required=True)
    c.add_argument('--output',type=Path,required=True)
    c.add_argument('--candidate-depth-source',choices=('da3','spread'))
    s = sub.add_parser('score');s.add_argument('--capture',type=Path,required=True)
    s.add_argument('--gt',type=Path,required=True);s.add_argument('--output',type=Path,required=True)
    s.add_argument('--segments',type=Path);s.add_argument('--annotations',type=Path)
    args = p.parse_args()
    if args.mode == 'score':
        score_capture(args.capture,args.gt,args.output,args.segments,args.annotations)
    else:
        snapshots = {}
        for value in args.snapshot:
            name,sep,path = value.partition('=')
            if not sep or not name or name in snapshots: raise ValueError('unique NAME=PATH snapshots required')
            snapshots[name] = Path(path).resolve(strict=True)
        config=json.loads(args.config.read_text(encoding='utf-8'))
        if args.candidate_depth_source is not None:
            config['candidate_depth_source']=args.candidate_depth_source
        if set(snapshots) != {s['name'] for s in config['snapshots']}:
            raise ValueError('snapshot names must match config')
        capture(config,args.source.resolve(strict=True),snapshots,args.output)


if __name__ == '__main__':
    main()
