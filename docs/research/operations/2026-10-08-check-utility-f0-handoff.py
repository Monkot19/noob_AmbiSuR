"""Local fake-subprocess checks only. Never run AutoDL, Torch or real data."""
import contextlib
import copy
from datetime import datetime, timedelta, timezone
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).with_name('2026-10-08-launch-utility-f0.py')
spec = importlib.util.spec_from_file_location('f0_handoff', SCRIPT)
handoff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handoff)


class HandoffChecks(unittest.TestCase):
    def test_digest_sidecar_is_guarded_before_any_payload_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'digest'
            path.write_text('a' * 64 + '\n')
            firewall = types.ModuleType('reliability.utility_gt_firewall')
            from unittest.mock import Mock
            firewall._read_bytes_guarded = Mock(side_effect=ValueError('protected alias'))
            with patch.dict('sys.modules', {'reliability.utility_gt_firewall': firewall}):
                with self.assertRaisesRegex(ValueError, 'protected alias'):
                    getattr(handoff, 'read_digest', lambda p: p.read_text().strip())(path)
            firewall._read_bytes_guarded.assert_called_once_with(path, (handoff.GT,))

    def test_dangling_new_probe_link_is_rejected_without_normalization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            probe = root / 'probe'
            try:
                probe.symlink_to(root / 'absent')
            except OSError:
                self.skipTest('symlink privilege unavailable; must run on AutoDL')
            record = {'runs': [], 'probe_targets': {'output_dir': str(probe)}}
            with self.assertRaisesRegex(RuntimeError, 'target exists'):
                getattr(handoff, 'require_absent_targets', lambda r: None)(record)

    def test_existing_qualification_digest_blocks_preregistration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = {key: str(root / key) for key in ('run_dir', 'view_dir', 'state_file',
                'launcher_dir', 'qualification_path')}
            Path(row['qualification_path'] + '.sha256').write_text('already reserved')
            with self.assertRaisesRegex(RuntimeError, 'target exists'):
                getattr(handoff, 'require_absent_targets', lambda r: None)(
                    {'runs': [row], 'probe_targets': {}})

    def test_import_failure_creates_no_recovery_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(handoff, 'ROOT', root), \
                 patch.object(handoff, 'clean_checkout'), \
                 patch.object(handoff, 'import_smoke', side_effect=RuntimeError('import failed')):
                with self.assertRaisesRegex(RuntimeError, 'import failed'):
                    handoff.recover()
            self.assertEqual(list(root.iterdir()), [])

    def test_startup_failure_requires_no_checkpoint_or_refresh_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, launch = root / 'run', root / 'launch'
            run.mkdir(); launch.mkdir()
            (run / 'train.log').write_text(
                'MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1\n')
            (launch / 'exit_code.txt').write_text('2\n')
            (launch / 'launcher.log').write_text('train_return_code=1\n')
            row = dict(run_dir=str(run), launcher_dir=str(launch))
            read = lambda path: Path(path).read_bytes()
            before = {path: path.read_bytes() for path in root.rglob('*') if path.is_file()}
            handoff.validate_startup_failure(row, read)
            self.assertEqual(before, {path: path.read_bytes() for path in before})
            (run / 'chkpnt3000.pth').write_bytes(b'evidence')
            with self.assertRaisesRegex(RuntimeError, 'training evidence'):
                handoff.validate_startup_failure(row, read)

    def test_recovery_binding_rejects_formula_change_before_inventory(self):
        old = dict(confirmation_id='old')
        new = dict(confirmation_id='new', created_utc='stamp', protocol={'changed': True})
        firewall = types.ModuleType('reliability.utility_gt_firewall')
        firewall._read_verified = lambda identity, **kw: json.dumps(
            old if identity['sha256'] == handoff.OLD_DIGEST else new).encode()
        confirmation = types.ModuleType('reliability.g1_prior_transfer_confirmation')
        confirmation._validate_record = lambda *a, **kw: None
        with patch.dict('sys.modules', {'reliability.utility_gt_firewall': firewall,
            'reliability.g1_prior_transfer_confirmation': confirmation}), \
             patch.object(handoff, 'recovery_record', return_value={**new, 'protocol': {'frozen': True}}), \
             patch.object(handoff, 'failed_inventory') as inventory:
            with self.assertRaisesRegex(RuntimeError, 'frozen contract'):
                handoff.recovery_binding('test-digest')
        inventory.assert_not_called()

    def test_threading_environment_reaches_real_child_without_parent_mutation(self):
        parent = dict(os.environ, MKL_THREADING_LAYER='INTEL',
                      MKL_SERVICE_FORCE_INTEL='1', PYTHONPATH='unsafe', PYTHONHOME='unsafe',
                      OMP_NUM_THREADS='16')
        before = dict(parent)
        # Without the fix this exercises the existing inherited environment.
        environment = getattr(handoff, 'runtime_environment', dict)(parent)
        result = __import__('subprocess').run(
            [__import__('sys').executable, '-E', '-B', '-c',
             'import os,json; print(json.dumps(dict(os.environ)))'],
            env=environment, text=True, capture_output=True, check=True)
        child = json.loads(result.stdout)
        self.assertEqual(child['MKL_THREADING_LAYER'], 'GNU')
        self.assertEqual(child['OMP_NUM_THREADS'], '1')
        for key in ('MKL_SERVICE_FORCE_INTEL', 'PYTHONPATH', 'PYTHONHOME'):
            self.assertNotIn(key, child)
        self.assertEqual(parent, before)

    def test_recovery_record_changes_only_attempt_identity_and_paths(self):
        old = {'confirmation_id': 'utility_prior_transfer_softv4_20261008_v2',
               'created_utc': 'old', 'protocol': {'solver': 'frozen'},
               'repository': {'commit': 'frozen'}, 'snapshot_record': {'sha256': 'frozen'},
               'runs': [], 'probe_targets': {'output_dir': '/tmp/v2',
                   'staging_dir': '/tmp/stage-v2', 'access_log_path': '/tmp/log-v2'}}
        for seed in (0, 1, 2):
            old['runs'].append(dict(seed=seed, run_dir=f'/tmp/run-v2-{seed}',
                view_dir=f'/tmp/view-v2-{seed}/colmap_undistorted',
                state_file=f'/tmp/state-v2-{seed}', launcher_dir=f'/tmp/launch-v2-{seed}',
                qualification_path=f'/tmp/qual-v2-{seed}',
                training_argv=['python', 'train.py', '--source_path',
                    f'/tmp/view-v2-{seed}/colmap_undistorted', '--model_path', f'/tmp/run-v2-{seed}',
                    '--seed', str(seed)]))
        original = copy.deepcopy(old)
        new = getattr(handoff, 'recovery_record', lambda value, stamp: value)(old, 'new')
        self.assertEqual(new['confirmation_id'], 'utility_prior_transfer_softv4_20261008_v3')
        self.assertEqual(new['created_utc'], 'new')
        self.assertEqual(old, original)
        for key in ('protocol', 'repository', 'snapshot_record'):
            self.assertEqual(new[key], old[key])
        for row in new['runs']:
            self.assertIn('_v3_seed' + str(row['seed']), row['run_dir'])
            before = list(old['runs'][row['seed']]['training_argv'])
            after = list(row['training_argv'])
            for flag in ('--source_path', '--model_path'):
                after[after.index(flag) + 1] = before[before.index(flag) + 1]
            self.assertEqual(after, before)

    def test_completed_attempt_cannot_be_recovered_as_startup_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, launch = root / 'run', root / 'launch'
            run.mkdir(); launch.mkdir()
            (run / 'train.log').write_text('Training complete.\n')
            (launch / 'exit_code.txt').write_text('0\n')
            read = lambda path: Path(path).read_bytes()
            with self.assertRaises(RuntimeError):
                getattr(handoff, 'validate_startup_failure', lambda *a: None)(
                    dict(run_dir=str(run), launcher_dir=str(launch)), read)

    def test_worker_receipts_match_existing_completion_contract(self):
        self.run_worker()

    def test_monitor_failure_stops_child_and_records_failure(self):
        self.run_worker(monitor_failure=True)

    def test_wrong_binding_never_starts_training(self):
        self.run_worker(wrong_binding=True)

    def run_worker(self, *, monitor_failure=False, wrong_binding=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            launch = root / 'utility_prior_transfer_softv4_20261008_v3_seed0.launch'
            launch.mkdir()
            run = root / 'run'
            run.mkdir()
            argv = ['fake-python', 'train.py', '--seed', '0']
            row = dict(seed=0, run_dir=str(run), view_dir=str(root / 'view'),
                       state_file=str(root / 'state'), launcher_dir=str(launch),
                       training_argv=argv)
            record = dict(row, schema_version=1, attempt=1, resumed=False,
                          replaces_completed_run=False, confirmation_sha256=handoff.DIGEST)
            if wrong_binding:
                record['training_argv'] = [*argv, '--unexpected']
            (launch / 'launch_record.json').write_text(json.dumps(record), encoding='utf-8')
            confirmation = {'runs': [row]}
            firewall = types.ModuleType('reliability.utility_gt_firewall')
            firewall._read_bytes_guarded = lambda path, protected: path.read_bytes()
            firewall._read_verified = lambda *a, **kw: json.dumps(confirmation).encode()
            times = [datetime(2026, 10, 8, tzinfo=timezone.utc),
                     datetime(2026, 10, 8, tzinfo=timezone.utc) + timedelta(seconds=8)]
            clock = types.SimpleNamespace(now=lambda tz: times.pop(0))
            process = types.SimpleNamespace(pid=12345678, returncode=None, terminated=False)
            polls = [None, None, 0]
            def poll():
                value = polls.pop(0) if polls else 0
                if value is not None:
                    process.returncode = value
                return value
            process.poll = poll
            process.terminate = lambda: setattr(process, 'terminated', True)
            process.wait = lambda **kw: 0
            process.kill = lambda: None
            with patch.dict('sys.modules', {'reliability.utility_gt_firewall': firewall}), \
                 patch.object(handoff, 'ROOT', root), \
                 patch.object(handoff, 'clean_checkout'), \
                 patch.object(handoff, 'frozen_trees') as trees, \
                 patch.object(handoff, 'datetime', clock), \
                 patch.object(handoff.subprocess, 'Popen', return_value=process) as start, \
                 patch.object(handoff.time, 'sleep'), \
                 patch.object(handoff, 'gpu_memory', side_effect=
                              RuntimeError('monitor unavailable') if monitor_failure else [200, 400]), \
                 contextlib.redirect_stdout(io.StringIO()):
                if wrong_binding or monitor_failure:
                    with self.assertRaises(RuntimeError):
                        handoff.worker(launch)
                else:
                    handoff.worker(launch)
            if wrong_binding:
                start.assert_not_called()
                self.assertFalse((launch / 'start_utc.txt').exists())
                return
            start.assert_called_once()
            self.assertEqual(start.call_args.args[0], argv)
            self.assertEqual(start.call_args.kwargs.get('env', {}).get('MKL_THREADING_LAYER'), 'GNU')
            self.assertFalse((run / '.training_active').exists())
            self.assertFalse((launch / '.training_active').exists())
            self.assertEqual((launch / 'exit_code.txt').read_text().strip(),
                             '2' if monitor_failure else '0')
            self.assertEqual((launch / 'wall_seconds.txt').read_text().strip(), '8')
            self.assertEqual(process.terminated, monitor_failure)
            self.assertEqual(trees.call_count, 1 if monitor_failure else 2)
            if not monitor_failure:
                self.assertEqual((launch / 'gpu_peak_mib.txt').read_text().strip(), '400')
                self.assertNotEqual((launch / 'launcher.pid').read_text(),
                                    (launch / 'training.pid').read_text())

    def test_snapshot_handles_contain_only_path_and_sha(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = dict(source_root=temporary, files=[dict(path='images/a.jpg')])
            snapshot = dict(snapshot_root=temporary, derived_manifest=[], snapshot_sha256=
                            '307b176e41111af403a565db94cfe8ada0a7d739361e1e380dcfaca08f49fc22')
            firewall = types.ModuleType('reliability.utility_gt_firewall')
            def read(identity, **kw):
                self.assertEqual(set(identity), {'path', 'sha256'})
                return json.dumps(source if identity['path'] == 'source' else snapshot).encode()
            firewall._read_verified = read
            confirmation = dict(source_record=dict(path='source', sha256='a'),
                                snapshot_record=dict(path='snapshot', sha256='b', snapshot_sha256='c'))
            with patch.dict('sys.modules', {'reliability.utility_gt_firewall': firewall}), \
                 patch.object(handoff, 'verified_tree') as trees:
                handoff.frozen_trees(confirmation, dict(view_dir=temporary))
            self.assertEqual(trees.call_count, 3)

    def test_unlisted_file_rejected_before_any_payload_read_or_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'extra').write_bytes(b'not an admitted input')
            firewall = types.ModuleType('reliability.utility_gt_firewall')
            from unittest.mock import Mock
            firewall._read_verified = Mock()
            firewall._reject_aliases = Mock()
            assets = types.ModuleType('reliability.prior_transfer_assets')
            assets._file_inventory = lambda path: {'extra'}
            target = root / 'not-created'
            with patch.dict('sys.modules', {'reliability.utility_gt_firewall': firewall,
                                           'reliability.prior_transfer_assets': assets}):
                with self.assertRaisesRegex(RuntimeError, 'inventory mismatch'):
                    handoff.verified_tree(root, [], destination=target)
            firewall._read_verified.assert_not_called()
            self.assertFalse(target.exists())

    def test_directory_symlink_rejected_before_payload_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            actual = root / 'unrelated'
            actual.mkdir()
            tree = root / 'tree'
            tree.mkdir()
            try:
                (tree / 'subtree').symlink_to(actual, target_is_directory=True)
            except OSError:
                self.skipTest('local account cannot create symlinks; static guard retained')
            firewall = types.ModuleType('reliability.utility_gt_firewall')
            from unittest.mock import Mock
            firewall._read_verified = Mock()
            firewall._reject_aliases = Mock()
            assets = types.ModuleType('reliability.prior_transfer_assets')
            assets._file_inventory = lambda path: set()
            with patch.dict('sys.modules', {'reliability.utility_gt_firewall': firewall,
                                           'reliability.prior_transfer_assets': assets}):
                with self.assertRaisesRegex(RuntimeError, 'symlink'):
                    handoff.verified_tree(tree, [])
            firewall._read_verified.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
