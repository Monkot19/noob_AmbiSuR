"""Local fake-subprocess checks only. Never run AutoDL, Torch or real data."""
import contextlib
from datetime import datetime, timedelta, timezone
import importlib.util
import io
import json
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
    def test_worker_receipts_match_existing_completion_contract(self):
        self.run_worker()

    def test_monitor_failure_stops_child_and_records_failure(self):
        self.run_worker(monitor_failure=True)

    def test_wrong_binding_never_starts_training(self):
        self.run_worker(wrong_binding=True)

    def run_worker(self, *, monitor_failure=False, wrong_binding=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            launch = root / 'utility_prior_transfer_softv4_20261008_v2_seed0.launch'
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
