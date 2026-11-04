"""Explicit, read-only process capture around the unchanged training entry.

Only this launcher installs a return observer. No GT, extra backward, training
hooks or candidate decisions live here. Static schema1 remains jointly accepted
ALR candidates; auxiliary arrays identify actual executed relocation separately.
"""
import argparse
from contextlib import AbstractContextManager
import hashlib
import inspect
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import torch

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
from scripts.evaluate_candidate_actions import _array, selection_record, write_capture_record
from utils import candidate_selector
from utils.depth_spread import build_spread_proposal
from utils.run_metadata import collect_run_identity, write_run_metadata


def trace_iteration(iteration):
    return (iteration == 7001 or 7000 < iteration < 15000 and iteration % 100 == 0
            or 15000 <= iteration <= 30000 and iteration % 1000 == 0)


def probe_iteration(iteration):
    return iteration == 7001 or 7000 < iteration <= 30000 and iteration % 1000 == 0


def _npz(path, arrays):
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)


def _render_arrays(result):
    keys = ('plane_depth', 'depth_normal', 'rendered_unc', 'render')
    return {key: _array(result[key]) for key in keys if key in result}


class CandidateTrainingTrace(AbstractContextManager):
    def __init__(self, output, *, probe_view=None):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.events, self.probes, self._targets = {}, [], {}
        self.probe_view = probe_view
        self._selector_code = inspect.unwrap(candidate_selector._select_candidate_geometry).__code__
        self._spread_code = inspect.unwrap(build_spread_proposal).__code__

    def __enter__(self):
        if sys.getprofile() is not None:
            raise RuntimeError('another Python profiler is active')
        sys.setprofile(self._observe)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.setprofile(None)
        manifest = dict(schema_version=1, successful=exc_type is None,
            error=None if exc_value is None else str(exc_value),
            semantics='Actual reference returns before optimizer step; schema1 accepted is joint ALR; auxiliary move mask is execution only. Fixed probe is A-only.',
            events=list(self.events.values()), probes=self.probes)
        with (self.output/'trace_manifest.json').open('x', encoding='utf-8') as stream:
            json.dump(manifest, stream, indent=2, allow_nan=False)
        return False

    def _event(self, iteration, camera):
        if iteration not in self.events:
            self.events[iteration] = dict(iteration=iteration,
                camera=getattr(camera, 'image_name', str(camera.uid)),
                camera_uid=int(camera.uid), reference_complete=False)
        return self.events[iteration]

    def _save(self, iteration, kind, arrays):
        filename = f'{iteration:06d}_{kind}.npz'
        _npz(self.output/filename, arrays)
        return filename

    def _observe(self, frame, event, returned):
        if event != 'return':
            return
        code, local = frame.f_code, frame.f_locals
        if code is self._spread_code:
            iteration = local['iteration']
            if trace_iteration(iteration) and returned is not None:
                self._spread(local, returned)
        elif code is self._selector_code:
            iteration = local['iteration']
            if trace_iteration(iteration) and returned is not None:
                self._selection(local, returned)
        elif (code.co_name == '_render_training_reference'
              and Path(code.co_filename).resolve() == REPOSITORY/'train.py'):
            iteration = local['iteration']
            if trace_iteration(iteration) and returned is not None:
                self._reference(local, returned)
            if self.probe_view and probe_iteration(iteration) and returned is not None:
                self._probe(frame)

    def _spread(self, local, result):
        iteration, camera = local['iteration'], local['reference']
        event = self._event(iteration, camera)
        arrays = dict(proposal_depth=_array(result[0]), proposal_valid=_array(result[1]),
            prior_depth=_array(local['prior_depth']), prior_normal=_array(local['prior_normal']))
        for name in ('depth', 'normal_valid', 'rays', 'source_depth', 'ids', 'cost', 'valid', 'confidence'):
            value = local.get(name)
            if isinstance(value, torch.Tensor):
                arrays['model_depth_before' if name == 'depth' else 'anchor_sample_ids' if name == 'ids' else name] = _array(value)
        mask = np.zeros(result[1].shape, dtype=bool)
        if 'anchors' in local:
            mask.reshape(-1)[_array(local['anchors'])] = True
        arrays['anchor_mask'] = mask
        if 'ids' in local and 'error_map' in local:
            arrays['anchor_roundtrip_error'] = _array(local['error_map'][local['ids']])
        for name in ('rendered_normal', 'rendered_distance'):
            arrays[name+'_before'] = _array(local['reference_geometry'][name])
        event['spread_file'] = self._save(iteration, 'spread', arrays)
        event['spread_stats'] = {k:int(v) for k,v in result[2].items()}

    def _selection(self, local, result):
        iteration, camera = local['iteration'], local['reference_camera']
        event = self._event(iteration, camera)
        record, stats = selection_record(local['gaussians'], camera,
            local['prior_depth'], result, local)
        filename = f'{iteration:06d}_candidates.npz'
        write_capture_record(self.output/filename, record)
        event.update(candidate_file=filename, selector_stats=stats,
            include_targets=bool(local['include_targets']),
            supports=[getattr(c,'image_name',str(c.uid)) for c in local['support_cameras'][:2]])
        targets = result[3]
        move = np.zeros(len(record['ids']), dtype=bool)
        if targets is not None:
            move = _array(local.get('move_accepted', local['accepted']))
            target_ids, target_xyz = _array(targets[0]), _array(targets[1])
        else:
            target_ids = np.empty(0, np.int64)
            target_xyz = np.empty((0,3), record['current_world'].dtype)
        arrays = dict(move_accepted=move, target_ids=target_ids, target_xyz=target_xyz)
        if local.get('costs_move'):
            arrays.update(position_rows=_array(local['rows']),
                cost_move=_array(torch.stack(local['costs_move'], -1)),
                common_move=_array(torch.stack(local['common_move'], -1)))
        event['selection_file'] = self._save(iteration, 'selection', arrays)
        self._targets[iteration] = (target_ids, target_xyz)

    def _reference(self, local, result):
        iteration, camera = local['iteration'], local['viewpoint_cam']
        event = self._event(iteration, camera)
        arrays = _render_arrays(result[0])
        ids, expected = self._targets.pop(iteration, (np.empty(0,np.int64),np.empty((0,3))))
        model = local['gaussians']
        with torch.no_grad():
            actual = _array(model.get_xyz[torch.as_tensor(ids, device=model.get_xyz.device)])
        if len(ids) and not np.array_equal(actual, expected):
            raise RuntimeError('recorded targets differ from actual post-relocation xyz')
        arrays.update(actual_target_ids=ids, actual_target_xyz=actual)
        event['reference_file'] = self._save(iteration, 'reference_after_move', arrays)
        event['reference_complete'] = True

    def _probe(self, frame):
        local = frame.f_locals
        scene = frame.f_back.f_locals.get('scene') if frame.f_back else None
        if scene is None:
            raise RuntimeError('fixed probe requires the original training scene')
        camera = next((c for c in scene.getTrainCameras() if c.image_name == self.probe_view), None)
        if camera is None:
            raise ValueError('fixed probe view missing: '+self.probe_view)
        model, opt = local['gaussians'], local['opt']
        version = (id(model.get_xyz),model.get_xyz._version)
        with torch.no_grad():
            labels = candidate_selector.compute_sh_ambiguity_labels(model,opt)
            result = frame.f_globals['render'](camera,model,local['pipe'],local['bg'],
                app_model=local['app_model'],return_plane=True,return_depth_normal=True,
                ray_reg=False,opt=opt,ambiguity_labels=labels)
        if version != (id(model.get_xyz),model.get_xyz._version):
            raise RuntimeError('read-only fixed probe changed xyz')
        iteration = local['iteration']
        arrays = _render_arrays(result)
        arrays.update(camera_k=_array(camera.get_k()),camera_transform=_array(camera.world_view_transform))
        filename = self._save(iteration, 'fixed_probe', arrays)
        self.probes.append(dict(iteration=iteration,camera=self.probe_view,file=filename,
            label_scope='A-only; not joint ALR labels of this probe view'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace-output', required=True, type=Path)
    parser.add_argument('--probe-view', default='DSC06642')
    parser.add_argument('training_args', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    training_args = args.training_args
    if training_args[:1] == ['--']:
        training_args = training_args[1:]
    if not training_args:
        parser.error('supply original train.py arguments after --')
    with CandidateTrainingTrace(args.trace_output, probe_view=args.probe_view) as trace:
        write_run_metadata(trace.output,dict(entry='trace_candidate_training',
            original_training_args=training_args,probe_view=args.probe_view,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
            collect_run_identity(REPOSITORY))
        old_argv = sys.argv
        try:
            sys.argv = [str(REPOSITORY/'train.py'), *training_args]
            runpy.run_path(str(REPOSITORY/'train.py'), run_name='__main__')
        finally:
            sys.argv = old_argv


if __name__ == '__main__':
    main()
