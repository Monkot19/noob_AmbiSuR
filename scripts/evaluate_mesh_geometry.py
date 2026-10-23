"""Reusable full-surface mesh evaluation; see the versioned protocol/design.

PRF equations follow scripts/tnt_eval/evaluation.py::get_f1_score_histo2.
Closest-point query follows the archived offline_g1 kernel without its routing
dependencies. This is NOT the complete DTU/TnT/ScanNet++ official protocol.
Only NumPy and Open3D are required for geometry; no renderer or training import.
"""
import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

# Support both `python scripts/...py` (from any cwd) and module imports.
REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
from utils.run_metadata import collect_run_identity, write_run_metadata


def validate_protocol(config):
    expected = {'protocol_name', 'domain', 'coordinate_frame', 'unit_scale_to_m',
                'sample_count', 'seed', 'chunk_size', 'thresholds_m', 'primary_threshold_m'}
    if set(config) != expected:
        raise ValueError(f'protocol keys must be {sorted(expected)}')
    for key in ('protocol_name', 'coordinate_frame'):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f'{key} must be explicit')
    if config['domain'] != 'full_surface':
        raise ValueError('only full_surface is implemented; no implicit crop')
    for key in ('sample_count', 'chunk_size', 'seed'):
        if type(config[key]) is not int or config[key] < (0 if key == 'seed' else 1):
            raise ValueError(f'invalid {key}')
    for value in (config['unit_scale_to_m'], config['primary_threshold_m']):
        if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError('scale and primary threshold must be finite and positive')
    thresholds = np.asarray(config['thresholds_m'], dtype=np.float64)
    if (thresholds.ndim != 1 or not thresholds.size or not np.isfinite(thresholds).all()
            or np.any(thresholds <= 0) or np.any(np.diff(thresholds) <= 0)):
        raise ValueError('thresholds_m must be strictly increasing finite positive values')
    if config['primary_threshold_m'] not in thresholds:
        raise ValueError('primary threshold must occur in thresholds_m')
    return config


def summarize_distances(recon_to_gt, gt_to_recon, thresholds_m):
    arrays = [np.asarray(d, dtype=np.float64) for d in (recon_to_gt, gt_to_recon)]
    for d in arrays:
        if d.ndim != 1 or not d.size or not np.isfinite(d).all() or np.any(d < 0):
            raise ValueError('distances must be nonempty finite nonnegative vectors')
    a, b = arrays
    acc, comp = float(a.mean() * 1000), float(b.mean() * 1000)
    prf = []
    for threshold in thresholds_m:
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError('threshold must be finite and positive')
        p, r = float(np.mean(a < threshold)), float(np.mean(b < threshold))
        prf.append(dict(threshold_m=float(threshold), precision=p, recall=r,
                        f1=2*p*r/(p+r) if p+r else 0.))
    if not prf:
        raise ValueError('at least one threshold required')
    return dict(accuracy_mm=acc, completeness_mm=comp,
                surface_chamfer_mm=(acc+comp)/2, prf=prf)


def prepare_mesh(vertices, triangles, chunk_size=65536):
    """Validate once and calculate triangle area CDF with bounded temporaries."""
    v, t = np.asarray(vertices, dtype=np.float64), np.asarray(triangles)
    if v.ndim != 2 or v.shape[1] != 3 or not len(v) or not np.isfinite(v).all():
        raise ValueError('mesh vertices must be nonempty finite [V,3]')
    if (t.ndim != 2 or t.shape[1] != 3 or not len(t)
            or not np.issubdtype(t.dtype, np.integer) or np.any(t < 0) or np.any(t >= len(v))):
        raise ValueError('mesh requires valid integer triangle indices')
    if chunk_size <= 0:
        raise ValueError('chunk_size must be positive')
    areas = np.empty(len(t), dtype=np.float64)
    for start in range(0, len(t), chunk_size):
        p = v[t[start:start+chunk_size]]
        areas[start:start+len(p)] = np.linalg.norm(np.cross(p[:,1]-p[:,0], p[:,2]-p[:,0]), axis=1)/2
    if not np.isfinite(areas).all():
        raise ValueError('nonfinite triangle area')
    valid = areas > 0
    removed = int(np.count_nonzero(~valid))
    if removed:
        t, areas = t[valid], areas[valid]
    if not len(t):
        raise ValueError('mesh has no positive-area triangles')
    cdf = np.cumsum(areas, dtype=np.float64)
    total = float(cdf[-1])
    if not np.isfinite(total) or total <= 0:
        raise ValueError('invalid total surface area')
    cdf /= total
    return dict(vertices=v, triangles=t, area_cdf=cdf, area_m2=total,
                removed_zero_area_faces=removed)


def surface_samples(mesh, count, seed, chunk_size=65536):
    if count <= 0 or chunk_size <= 0:
        raise ValueError('count/chunk_size must be positive')
    rng = np.random.Generator(np.random.PCG64(seed))
    points = np.empty((count, 3), dtype=np.float64)
    for start in range(0, count, chunk_size):
        n = min(chunk_size, count-start)
        # One three-column random array preserves RNG stream across chunk sizes.
        u = rng.random((n, 3))
        face = np.searchsorted(mesh['area_cdf'], u[:,0], side='right')
        corners = mesh['vertices'][mesh['triangles'][face]]
        root = np.sqrt(u[:,1:2])
        points[start:start+n] = ((1-root)*corners[:,0] + root*(1-u[:,2:3])*corners[:,1]
                                + root*u[:,2:3]*corners[:,2])
    return points


def _open3d():
    import open3d as o3d
    return o3d


class TriangleDistance:
    """CPU triangle BVH, Float32 geometry/query and Float64 distance accumulation."""
    def __init__(self, mesh):
        o3d = _open3d()
        self.scene = o3d.t.geometry.RaycastingScene()
        # Direct tensors avoid another complete legacy-mesh copy for large inputs.
        vertices = o3d.core.Tensor(np.asarray(mesh['vertices'], dtype=np.float32))
        triangles = o3d.core.Tensor(np.asarray(mesh['triangles'], dtype=np.uint32))
        self.scene.add_triangles(vertices, triangles)

    def distances(self, points, chunk_size=65536):
        q = np.asarray(points, dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != 3 or not len(q) or not np.isfinite(q).all():
            raise ValueError('queries must be nonempty finite [N,3]')
        if chunk_size <= 0:
            raise ValueError('chunk_size must be positive')
        o3d = _open3d()
        result = np.empty(len(q), dtype=np.float64)
        for start in range(0, len(q), chunk_size):
            batch = q[start:start+chunk_size]
            near = self.scene.compute_closest_points(o3d.core.Tensor(batch.astype(np.float32)))
            result[start:start+len(batch)] = np.linalg.norm(
                batch - near['points'].numpy().astype(np.float64), axis=1)
        if not np.isfinite(result).all():
            raise ValueError('nonfinite point-to-triangle distance')
        return result


def closest_triangle_distances(points, mesh, chunk_size=65536):
    return TriangleDistance(mesh).distances(points, chunk_size)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _read_mesh(path, scale, origin=None, chunk_size=65536):
    mesh = _open3d().io.read_triangle_mesh(str(path), enable_post_processing=False)
    vertices = np.asarray(mesh.vertices).copy()
    triangles = np.asarray(mesh.triangles).copy()
    del mesh
    vertices *= scale
    if not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError(f'empty/nonfinite mesh: {path}')
    if origin is None:
        origin = (vertices.min(axis=0) + vertices.max(axis=0))/2
    vertices -= origin
    return prepare_mesh(vertices, triangles, chunk_size), origin


def _mesh_record(path, mesh, origin):
    return dict(path=str(path), size_bytes=path.stat().st_size, sha256=_sha256(path),
                vertex_count=len(mesh['vertices']), triangle_count=len(mesh['triangles']),
                removed_zero_area_faces=mesh['removed_zero_area_faces'], area_m2=mesh['area_m2'],
                bounds_m=[(mesh['vertices'].min(axis=0)+origin).tolist(),
                          (mesh['vertices'].max(axis=0)+origin).tolist()])


def _deltas(metrics, baseline):
    result = {key: metrics[key]-baseline[key] for key in
              ('accuracy_mm', 'completeness_mm', 'surface_chamfer_mm')}
    result['prf'] = [dict(threshold_m=p['threshold_m'],
                          **{key: p[key]-b[key] for key in ('precision','recall','f1')})
                     for p,b in zip(metrics['prf'], baseline['prf'])]
    return result


def evaluate_meshes(gt_path, models, protocol, output, scene, model_commits):
    """Real shared producer -> existing metadata writer -> JSON/CSV publication.

    models/model_commits are insertion-ordered name -> path/commit mappings.
    First model is the paired baseline; no model/GT files are modified.
    """
    started = time.perf_counter()
    validate_protocol(protocol)
    if not scene.strip() or not models or set(models) != set(model_commits):
        raise ValueError('explicit scene and matching named meshes/model commits required')
    if any(not str(c).strip() for c in model_commits.values()):
        raise ValueError('model commit must be explicit')
    gt_path = Path(gt_path).resolve(strict=True)
    models = {name: Path(path).resolve(strict=True) for name,path in models.items()}
    output = Path(output).resolve()
    for path in [gt_path, *models.values()]:
        if not path.is_file():
            raise ValueError(f'not a file: {path}')
        if output == path.parent or path.parent in output.parents:
            raise ValueError('output must be outside input mesh directories')
    # Existing directory is never a destination, including a failed/partial run.
    output.mkdir(parents=True, exist_ok=False)
    resolved = dict(protocol=protocol, scene=scene, gt=str(gt_path),
                    models={k:str(v) for k,v in models.items()}, model_commits=model_commits,
                    command=sys.argv, started_utc=datetime.now(timezone.utc).isoformat())
    identity = collect_run_identity(REPOSITORY)
    identity.update(numpy_version=np.__version__, open3d_version=_open3d().__version__)
    write_run_metadata(output, resolved, identity)
    n, seed, chunk = (protocol[k] for k in ('sample_count','seed','chunk_size'))
    scale = protocol['unit_scale_to_m']
    print(f'GT: loading and sampling {n} points', flush=True)
    gt_mesh, origin = _read_mesh(gt_path, scale, chunk_size=chunk)
    gt_record = _mesh_record(gt_path, gt_mesh, origin)
    gt_points = surface_samples(gt_mesh, n, seed, chunk)
    gt_hash = hashlib.sha256(gt_points.astype('<f8',copy=False).tobytes()).hexdigest()
    gt_query = TriangleDistance(gt_mesh)
    del gt_mesh
    report = dict(schema_version=1, protocol=protocol, scene=scene,
                  evaluator_identity=identity, gt=gt_record, common_origin_m=origin.tolist(),
                  gt_samples_sha256=gt_hash, baseline_model=next(iter(models)), models=[])
    for name,path in models.items():
        print(f'{name}: loading, sampling and querying both directions', flush=True)
        mesh, _ = _read_mesh(path, scale, origin, chunk)
        record = _mesh_record(path, mesh, origin)
        # Same local random stream for every reconstruction, independent of order.
        points = surface_samples(mesh, n, seed, chunk)
        accuracy_distances = gt_query.distances(points, chunk)
        completeness_distances = closest_triangle_distances(gt_points, mesh, chunk)
        metrics = summarize_distances(accuracy_distances, completeness_distances, protocol['thresholds_m'])
        baseline = report['models'][0]['metrics'] if report['models'] else metrics
        report['models'].append(dict(model=name, model_commit=model_commits[name], input=record,
                                     metrics=metrics, delta_from_baseline=_deltas(metrics, baseline)))
        primary = next(p for p in metrics['prf'] if p['threshold_m'] == protocol['primary_threshold_m'])
        print(f"{name}: CD={metrics['surface_chamfer_mm']:.6f}mm, F1@{primary['threshold_m']:g}m={primary['f1']:.6f}", flush=True)
        del mesh, points, accuracy_distances, completeness_distances
        gc.collect()
    report['elapsed_seconds'] = time.perf_counter()-started
    # JSON last: its existence marks successful metric/CSV publication.
    with (output/'metrics.csv').open('x',encoding='utf-8',newline='') as stream:
        fields = ['model','model_commit','accuracy_mm','completeness_mm','surface_chamfer_mm',
                  'threshold_m','is_primary','precision','recall','f1',
                  'delta_accuracy_mm','delta_completeness_mm','delta_surface_chamfer_mm',
                  'delta_precision','delta_recall','delta_f1']
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for model in report['models']:
            means = {k:v for k,v in model['metrics'].items() if k != 'prf'}
            delta = {f'delta_{k}':v for k,v in model['delta_from_baseline'].items() if k != 'prf'}
            for prf,dp in zip(model['metrics']['prf'],model['delta_from_baseline']['prf']):
                writer.writerow(dict(model=model['model'],model_commit=model['model_commit'],
                                     **means,**delta,**prf,
                                     is_primary=prf['threshold_m']==protocol['primary_threshold_m'],
                                     **{f'delta_{k}':dp[k] for k in ('precision','recall','f1')}))
    with (output/'results.json').open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n')
    return report


def _named(values):
    result = {}
    for value in values:
        name, sep, path = value.partition('=')
        if not sep or not name.strip() or not path.strip() or name in result:
            raise ValueError('expected unique NAME=VALUE entries')
        result[name] = path
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--scene',required=True)
    parser.add_argument('--gt',type=Path,required=True)
    parser.add_argument('--mesh',action='append',required=True,help='NAME=PATH; first is baseline')
    parser.add_argument('--model-commit',action='append',required=True,help='NAME=TRAINING_COMMIT')
    parser.add_argument('--output',type=Path,required=True,help='new independent directory')
    args = parser.parse_args(argv)
    with args.config.open(encoding='utf-8') as stream:
        protocol = json.load(stream)
    evaluate_meshes(args.gt, _named(args.mesh), protocol, args.output,args.scene,_named(args.model_commit))
    print(f'Completed: {args.output}/results.json',flush=True)


if __name__ == '__main__':
    main()
