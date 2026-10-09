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
    def test_seed1_admission_preserves_completed_seed0_and_blocks_later_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [dict(seed=seed, **{key: str(root / f'{seed}-{key}') for key in
                    ('run_dir', 'view_dir', 'state_file', 'launcher_dir', 'qualification_path')})
                    for seed in (0, 1, 2)]
            Path(rows[0]['run_dir']).mkdir()
            record = {'runs': rows, 'probe_targets': {'output_dir': str(root / 'probe')}}
            check = getattr(handoff, 'require_seed_targets', handoff.require_absent_targets)
            check(record, seed=1)
            for value in (rows[1]['qualification_path'] + '.sha256',
                          rows[2]['state_file'], record['probe_targets']['output_dir']):
                Path(value).write_text('reserved')
                with self.assertRaises(RuntimeError):
                    check(record, seed=1)
                Path(value).unlink()
            with self.assertRaises(RuntimeError):
                check(record, seed=3)

    def test_seed2_admission_preserves_predecessors_and_blocks_probe_and_sidecar(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [dict(seed=seed, **{key: str(root / f'{seed}-{key}') for key in
                    ('run_dir', 'view_dir', 'state_file', 'launcher_dir', 'qualification_path')})
                    for seed in (0, 1, 2)]
            for row in rows[:2]:
                Path(row['run_dir']).mkdir()
            record = dict(runs=rows, probe_targets={'output_dir': str(root / 'probe')})
            handoff.require_seed_targets(record, seed=2)
            for value in (rows[2]['qualification_path'] + '.sha256', record['probe_targets']['output_dir']):
                Path(value).write_text('reserved')
                with self.assertRaises(RuntimeError):
                    handoff.require_seed_targets(record, seed=2)
                Path(value).unlink()

    def test_seed2_worker_reuses_exact_row_and_receipts(self):
        self.run_worker(seed=2)

    def test_seed2_wrong_binding_never_starts_training(self):
        self.run_worker(seed=2, wrong_binding=True)

    def test_seed2_monitor_failure_stops_child(self):
        self.run_worker(seed=2, monitor_failure=True)

    def test_seed2_prepare_preserves_both_qualified_runs(self):
        self.prepare_seed1(seed=2)

    def test_seed2_changed_seed1_stops_before_targets(self):
        self.prepare_seed1(seed=2, bad_predecessor=True)

    def test_seed2_low_space_stops_before_targets(self):
        self.prepare_seed1(seed=2, low_space=True)

    def test_seed1_worker_reuses_exact_argv_environment_and_receipts(self):
        self.run_worker(seed=1)

    def test_seed1_wrong_binding_never_starts_training(self):
        self.run_worker(seed=1, wrong_binding=True)

    def test_seed1_monitor_failure_stops_child(self):
        self.run_worker(seed=1, monitor_failure=True)

    def test_seed1_prepare_creates_only_seed1_using_unchanged_confirmation(self):
        self.prepare_seed1()

    def test_seed1_bad_predecessor_stops_before_any_target(self):
        self.prepare_seed1(bad_predecessor=True)

    def prepare_seed1(self, *, bad_predecessor=False, seed=1, low_space=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for row_seed in (0, 1, 2):
                row = dict(seed=row_seed, run_dir=str(root / f'run{row_seed}'),
                    view_dir=str(root / f'view{row_seed}' / 'colmap_undistorted'),
                    launcher_dir=str(root / f'launch{row_seed}'), state_file=str(root / f'state{row_seed}'),
                    qualification_path=str(root / f'qualification{row_seed}'))
                row['training_argv'] = [handoff.PYTHON, 'train.py', '--seed', str(row_seed),
                    '--source_path', row['view_dir'], '--model_path', row['run_dir']]
                rows.append(row)
            Path(rows[0]['run_dir']).mkdir()
            payload = Path(rows[0]['run_dir']) / 'original'
            payload.write_bytes(b'qualified seed0')
            if seed == 2:
                Path(rows[1]['run_dir']).mkdir()
                (Path(rows[1]['run_dir']) / 'original').write_bytes(b'qualified seed1')
            confirmation = dict(runs=rows, probe_targets={'output_dir': str(root / 'probe')},
                repository=dict(root=str(handoff.REPO), commit=handoff.COMMIT, clean=True))
            frozen = json.dumps(confirmation, sort_keys=True).encode()
            firewall = types.ModuleType('reliability.utility_gt_firewall')
            firewall._read_verified = lambda *a, **kw: frozen
            schema = types.ModuleType('reliability.g1_prior_transfer_confirmation')
            schema._canonical_bytes = lambda record: frozen
            schema._validate_record = __import__('unittest.mock', fromlist=['Mock']).Mock()
            assets = types.ModuleType('reliability.prior_transfer_assets')
            assets._no_gt = lambda row: None
            digest = handoff.hashlib.sha256(frozen).hexdigest()
            with patch.dict('sys.modules', {'reliability.utility_gt_firewall': firewall,
                  'reliability.g1_prior_transfer_confirmation': schema,
                  'reliability.prior_transfer_assets': assets}), \
                 patch.object(handoff, 'DIGEST', digest), patch.object(handoff, 'V3_DIGEST', digest), \
                 patch.object(handoff, 'clean_checkout'), patch.object(handoff, 'gpu_memory', return_value=400), \
                 patch.object(handoff, 'frozen_trees'), \
                 patch.object(handoff, 'validate_seed0_qualification', side_effect=
                    RuntimeError('prior qualification changed') if bad_predecessor and seed == 1 else None) as previous, \
                 patch.object(handoff, 'validate_seed1_qualification', create=True, side_effect=
                    RuntimeError('prior qualification changed') if bad_predecessor and seed == 2 else None) as previous1, \
                 patch.object(handoff.shutil, 'disk_usage', return_value=types.SimpleNamespace(
                    free=15*1024**3-1 if low_space else 16*1024**3)), \
                 patch.object(handoff.subprocess, 'check_output', return_value=''), \
                 patch.object(handoff.subprocess, 'Popen', return_value=types.SimpleNamespace(pid=123)) as start, \
                 contextlib.redirect_stdout(io.StringIO()):
                if bad_predecessor or low_space:
                    with self.assertRaisesRegex(RuntimeError,
                            'prior qualification changed' if bad_predecessor else '15 GiB'):
                        handoff.prepare(seed=seed)
                    start.assert_not_called()
                    self.assertFalse(Path(rows[seed]['view_dir']).parent.exists())
                else:
                    handoff.prepare(seed=seed)
                    self.assertEqual(previous.call_count, 2)
                    self.assertEqual(previous1.call_count, 2 if seed == 2 else 0)
                    self.assertEqual(start.call_args.args[0][3], f'--worker-seed{seed}')
                    record = json.loads((Path(rows[seed]['launcher_dir']) / 'launch_record.json').read_text())
                    self.assertEqual(record['training_argv'], rows[seed]['training_argv'])
                    self.assertEqual(record['seed'], seed)
                    self.assertIn(f'SEED={seed}', Path(rows[seed]['state_file']).read_text())
                    if seed == 1:
                        self.assertFalse(Path(rows[2]['run_dir']).exists())
                    self.assertFalse((root / 'probe').exists())
                    schema._validate_record.assert_called_once_with(confirmation,
                        require_targets_absent=False, verify_record_files=False)
            self.assertEqual(payload.read_bytes(), b'qualified seed0')
            if seed == 2:
                self.assertEqual((Path(rows[1]['run_dir']) / 'original').read_bytes(), b'qualified seed1')
            self.assertEqual(json.dumps(confirmation, sort_keys=True).encode(), frozen)

    def test_seed2_audit_uses_both_predecessors_and_existing_cli(self):
        with patch.object(handoff, 'clean_checkout'), \
             patch.object(handoff, 'read_digest', side_effect=[handoff.V3_DIGEST, 'recovery']), \
             patch.object(handoff, 'recovery_binding', return_value={'frozen': True}), \
             patch.object(handoff, 'validate_seed0_qualification') as previous0, \
             patch.object(handoff, 'validate_seed1_qualification', create=True) as previous1, \
             patch.object(handoff.subprocess, 'run') as run, \
             patch.object(handoff.subprocess, 'Popen') as start, \
             patch.dict(os.environ), patch.object(handoff, 'DIGEST', handoff.V3_DIGEST):
            handoff.audit(seed=2)
            previous0.assert_called_once_with({'frozen': True})
            previous1.assert_called_once_with({'frozen': True})
            self.assertEqual(run.call_args.args[0][-2:], ['--seed', '2'])
            self.assertIn('scripts/diagnostics/audit_prior_transfer_run.py', run.call_args.args[0])
            start.assert_not_called()

    def test_seed1_audit_reuses_existing_cli_without_training(self):
        with patch.object(handoff, 'clean_checkout'), \
             patch.object(handoff, 'read_digest', side_effect=[handoff.V3_DIGEST, 'recovery']), \
             patch.object(handoff, 'recovery_binding', return_value={'frozen': True}), \
             patch.object(handoff, 'validate_seed0_qualification') as previous, \
             patch.object(handoff.subprocess, 'run') as run, \
             patch.object(handoff.subprocess, 'Popen') as start, \
             patch.dict(os.environ), patch.object(handoff, 'DIGEST', handoff.V3_DIGEST):
            handoff.audit(seed=1)
            previous.assert_called_once_with({'frozen': True})
            self.assertEqual(run.call_args.args[0][-2:], ['--seed', '1'])
            self.assertIn('scripts/diagnostics/audit_prior_transfer_run.py', run.call_args.args[0])
            start.assert_not_called()

    def test_seed0_qualification_is_pinned_and_revalidates_recorded_payloads(self):
        self.check_qualification(seed=0)

    def test_seed1_qualification_is_pinned_and_revalidates_recorded_payloads(self):
        self.check_qualification(seed=1)

    def check_qualification(self, *, seed):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = dict(seed=seed, run_dir=str(root / 'run'), view_dir=str(root / 'view'),
                       launcher_dir=str(root / 'launch'), state_file=str(root / 'state'),
                       qualification_path=str(root / 'qualification'))
            for key in ('run_dir', 'view_dir', 'launcher_dir'):
                Path(row[key]).mkdir()
                (Path(row[key]) / 'payload').write_bytes(b'frozen')
            Path(row['state_file']).write_bytes(b'frozen')
            source, snapshot = root / 'source', root / 'snapshot'
            source.write_bytes(b'frozen'); snapshot.write_bytes(b'frozen')
            confirmation = dict(runs=[{}] * seed + [row], source_record=dict(path=str(source)),
                                snapshot_record=dict(path=str(snapshot)))
            files = [path for path in root.rglob('*') if path.is_file()]
            fingerprints = {str(path): dict(resolved_path=str(path.resolve()), bytes=6,
                sha256=handoff.hashlib.sha256(b'frozen').hexdigest()) for path in files}
            report = dict(outcome='QUALIFIED', gt_access='NONE', confirmation_sha256='frozen-confirmation',
                run_binding=dict(seed=seed, run_dir=row['run_dir'], view_dir=row['view_dir'],
                    repository_commit=handoff.COMMIT, snapshot_sha256=
                    '307b176e41111af403a565db94cfe8ada0a7d739361e1e380dcfaca08f49fc22',
                    evidence_version=4, qualification_path=row['qualification_path']),
                input_fingerprints=fingerprints)
            reads = []
            firewall = types.ModuleType('reliability.utility_gt_firewall')
            def read(identity, **kwargs):
                reads.append((identity, kwargs))
                if identity['path'] == row['qualification_path']:
                    self.assertEqual(identity['sha256'],
                        'fbad915581157be0c89e24ae13e605ddc9f1fe03a1495dc94dc44e61ce59a1c9' if seed == 0 else
                        'd0b3eae76beab3261097239d23d9f70b6e1bc96b66ebc099c1b34f4279391bf4')
                    return json.dumps(report).encode()
                payload = Path(identity['path']).read_bytes()
                if handoff.hashlib.sha256(payload).hexdigest() != identity['sha256']:
                    raise ValueError('payload changed')
                return payload
            firewall._read_verified = read
            firewall._reject_aliases = lambda *a, **kw: None
            assets = types.ModuleType('reliability.prior_transfer_assets')
            assets._file_inventory = lambda path: {'payload'}
            check = getattr(handoff, f'validate_seed{seed}_qualification', lambda *a: None)
            with patch.dict('sys.modules', {'reliability.utility_gt_firewall': firewall,
                                           'reliability.prior_transfer_assets': assets}), \
                 patch.object(handoff, 'DIGEST', 'frozen-confirmation'):
                check(confirmation)
                self.assertTrue(reads, 'predecessor must be read and pinned')
                report['run_binding']['seed'] = 1 - seed
                with self.assertRaises(RuntimeError):
                    check(confirmation)
                report['run_binding']['seed'] = seed
                Path(row['state_file']).write_bytes(b'changed')
                with self.assertRaises(ValueError):
                    check(confirmation)

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

    def run_worker(self, *, monitor_failure=False, wrong_binding=False, seed=0):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            launch = root / f'utility_prior_transfer_softv4_20261008_v3_seed{seed}.launch'
            launch.mkdir()
            run = root / 'run'
            run.mkdir()
            argv = ['fake-python', 'train.py', '--seed', str(seed)]
            row = dict(seed=seed, run_dir=str(run), view_dir=str(root / 'view'),
                       state_file=str(root / 'state'), launcher_dir=str(launch),
                       training_argv=argv)
            record = dict(row, schema_version=1, attempt=1, resumed=False,
                          replaces_completed_run=False, confirmation_sha256=handoff.DIGEST)
            if wrong_binding:
                record['training_argv'] = [*argv, '--unexpected']
            (launch / 'launch_record.json').write_text(json.dumps(record), encoding='utf-8')
            confirmation = {'runs': [{}] * seed + [row]}
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
                 patch.object(handoff, 'validate_seed0_qualification', create=True), \
                 patch.object(handoff, 'validate_seed1_qualification', create=True), \
                 patch.object(handoff, 'frozen_trees') as trees, \
                 patch.object(handoff, 'datetime', clock), \
                 patch.object(handoff.subprocess, 'Popen', return_value=process) as start, \
                 patch.object(handoff.time, 'sleep'), \
                 patch.object(handoff, 'gpu_memory', side_effect=
                              RuntimeError('monitor unavailable') if monitor_failure else [200, 400]), \
                 contextlib.redirect_stdout(io.StringIO()):
                if wrong_binding or monitor_failure:
                    with self.assertRaises(RuntimeError):
                        handoff.worker(launch, seed=seed) if seed else handoff.worker(launch)
                else:
                    handoff.worker(launch, seed=seed) if seed else handoff.worker(launch)
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
