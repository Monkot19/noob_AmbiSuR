"""User-operated seed0-only handoff; not a new training/evaluation engine.

Run from a transported pinned documentation artifact on AutoDL. The checkout
stays at the already qualified execution commit. Recovery is explicit, single-use,
and creates a fresh confirmation and paths; no resume or automatic retry.
"""
from datetime import datetime, timezone
from pathlib import Path
import json
import copy
import hashlib
import os
import re
import shlex
import shutil
import subprocess
import sys
import time

REPO = Path('/root/autodl-tmp/noob_AmbiSuR')
PYTHON = '/root/miniconda3/envs/ambisur/bin/python'
COMMIT = 'c701424c1b1f5a9006e6f19776769ee7bc8cb299'
ROOT = Path('/root/autodl-tmp/ambisur_diagnostics/Utility_Room/prior-transfer')
OLD_ID = 'utility_prior_transfer_softv4_20261008_v2'
NEW_ID = 'utility_prior_transfer_softv4_20261008_v3'
OLD_DIGEST = '51376355bb19434ebfd118090ea2c1b38371156a9dbaf4da51bb96aaf385020e'
CONFIRMATION = ROOT / (NEW_ID + '.confirmation.json')
DIGEST = None  # Supplied by the exclusive recovery publication, not an unchecked file.
GT = Path('/root/autodl-tmp/ambisur_data/gt/ScanNetpp/Utility_Room/mesh_aligned_0.05.ply')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def exclusive(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(value)


def runtime_environment(parent):
    env = dict(parent)
    for name in ('PYTHONPATH', 'PYTHONHOME', 'MKL_SERVICE_FORCE_INTEL'):
        env.pop(name, None)
    env.update(MKL_THREADING_LAYER='GNU', OMP_NUM_THREADS='1',
               PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    return env


def read_digest(path):
    from reliability.utility_gt_firewall import _read_bytes_guarded
    payload = _read_bytes_guarded(Path(path), (GT,))
    require(re.fullmatch(rb'[0-9a-f]{64}\n?', payload) is not None,
            'malformed detached digest')
    return payload.decode('ascii').strip()


def require_absent_targets(record):
    paths = list(record['probe_targets'].values())
    for row in record['runs']:
        paths.extend(row[key] for key in ('run_dir', 'view_dir', 'state_file',
            'launcher_dir', 'qualification_path'))
        paths.append(row['qualification_path'] + '.sha256')
    for value in paths:
        path = Path(value)  # Never resolve before the lexists/link checks.
        require(not os.path.lexists(path), 'target exists: ' + str(path))
        require(not any(parent.is_symlink() for parent in path.parents),
                'linked target ancestor: ' + str(path))


def recovery_record(old, stamp):
    new = copy.deepcopy(old)
    new.update(confirmation_id=NEW_ID, created_utc=stamp)
    for row in new['runs']:
        name = NEW_ID + '_seed' + str(row['seed'])
        row.update(
            run_dir=str(Path('/root/autodl-tmp/ambisur_runs/Utility_Room/prior-transfer-softv4-7k') / name),
            view_dir=str(Path('/root/autodl-tmp/ambisur_work/data_views') / name / 'colmap_undistorted'),
            state_file=str(ROOT / (name + '.env')), launcher_dir=str(ROOT / (name + '.launch')),
            qualification_path=str(ROOT / (name + '.qualification.json')))
        for flag, key in (('--source_path', 'view_dir'), ('--model_path', 'run_dir')):
            row['training_argv'][row['training_argv'].index(flag) + 1] = row[key]
    new['probe_targets'] = {
        'output_dir': str(ROOT / (NEW_ID + '.probe')),
        'staging_dir': str(ROOT / ('.' + NEW_ID + '.probe-staging')),
        'access_log_path': str(ROOT / (NEW_ID + '.first-gt-access.json'))}
    return new


def validate_startup_failure(row, read):
    run, launch = Path(row['run_dir']), Path(row['launcher_dir'])
    require(read(launch / 'exit_code.txt').strip() == b'2', 'not the failed startup attempt')
    log = read(run / 'train.log').decode('utf-8')
    require('MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1' in log,
            'missing diagnosed MKL failure')
    require('Training complete.' not in log and 'Training progress:' not in log,
            'attempt progressed beyond startup')
    require('train_return_code=1' in read(launch / 'launcher.log').decode('utf-8'),
            'wrong failed child return code')
    require(not list(run.glob('chkpnt*.pth')) and not (run / 'd0_evidence').exists(),
            'failed attempt has training evidence; stop')
    require(not (run / '.training_active').exists() and not (launch / '.training_active').exists(),
            'failed attempt still active')


def import_smoke():
    # Reproduce the actual parent imports then a Torch-first child, not train.py
    # (importing train.py itself would initialize CUDA and seed state).
    code = '''
import os, subprocess, sys
from reliability.g1_prior_transfer_confirmation import _validate_record
from reliability.utility_gt_firewall import _read_verified
from reliability.prior_transfer_assets import _no_gt
assert os.getenv('MKL_THREADING_LAYER') == 'GNU'
r = subprocess.run([sys.executable, '-B', '-c',
    "import os; assert os.getenv('MKL_THREADING_LAYER') == 'GNU'; import torch; import numpy; print('GNU_IMPORT_CHAIN=PASS', torch.__version__, numpy.__version__)"], timeout=45)
raise SystemExit(r.returncode)
'''
    subprocess.run([PYTHON, '-B', '-c', code], cwd=REPO,
                   env=runtime_environment(os.environ), timeout=60, check=True)


def failed_inventory(old):
    from reliability.utility_gt_firewall import _read_bytes_guarded, _reject_aliases
    row = old['runs'][0]
    def read(path):
        return _read_bytes_guarded(Path(path), (GT,))
    validate_startup_failure(row, read)
    paths = [Path(row['state_file'])]
    for key in ('run_dir', 'launcher_dir'):
        root = Path(row[key])
        require(root.is_dir() and not root.is_symlink(), 'bad failed attempt root')
        for path in root.rglob('*'):
            require(not path.is_symlink(), 'linked failure artifact')
            if path.is_file():
                paths.append(path)
    result = []
    for path in sorted(paths):
        _reject_aliases([{'path': str(path)}], protected=(GT,))
        payload = read(path)
        result.append({'path': str(path), 'sha256': hashlib.sha256(payload).hexdigest(),
                       'bytes': len(payload)})
    return result


def recovery_binding(digest):
    from reliability.utility_gt_firewall import _read_verified
    from reliability.g1_prior_transfer_confirmation import _validate_record
    old = json.loads(_read_verified({'path': str(ROOT / (OLD_ID + '.confirmation.json')),
        'sha256': OLD_DIGEST}, detached=True, protected=(GT,)))
    _validate_record(old, require_targets_absent=False, verify_record_files=False)
    new = json.loads(_read_verified({'path': str(CONFIRMATION), 'sha256': digest},
                                   detached=True, protected=(GT,)))
    require(new == recovery_record(old, new['created_utc']), 'recovery changed frozen contract')
    _validate_record(new, require_targets_absent=False, verify_record_files=False)
    receipt = json.loads(_read_verified({'path': str(ROOT / (NEW_ID + '.recovery.json')),
        'sha256': os.environ['UTILITY_F0_RECOVERY_SHA']}, detached=True, protected=(GT,)))
    require(receipt['old_confirmation_sha256'] == OLD_DIGEST and
            receipt['new_confirmation_sha256'] == digest and
            receipt['environment'] == {key: runtime_environment({})[key] for key in
                ('MKL_THREADING_LAYER', 'OMP_NUM_THREADS', 'PYTHONNOUSERSITE', 'PYTHONDONTWRITEBYTECODE')},
            'recovery receipt binding mismatch')
    require(receipt['failed_files'] == failed_inventory(old), 'failed attempt changed')
    # Preserve every old unauthorized target as absent, not merely the new ones.
    for row in old['runs'][1:]:
        for key in ('run_dir', 'view_dir', 'state_file', 'launcher_dir', 'qualification_path'):
            require(not os.path.lexists(row[key]), 'old unauthorized target exists')
    for path in old['probe_targets'].values():
        require(not os.path.lexists(path), 'old probe target exists')
    return new


def recover():
    global DIGEST
    clean_checkout()
    import_smoke()  # Before any publication or targets; no CUDA/data/GT reads.
    from reliability.utility_gt_firewall import _read_verified
    from reliability.g1_prior_transfer_confirmation import _validate_record, _canonical_bytes
    commands = subprocess.check_output(['ps', '-eo', 'args='], text=True)
    require(not re.search(r'\bpython[^\s]*\s+.*(?:train|estimate_colmap|launcher)\.py(?:\s|$)',
                          commands), 'training, DA3 or launcher is active')
    require(shutil.disk_usage('/root/autodl-tmp').free >= 15 * 1024**3,
            'less than 15 GiB free; preserve all targets and stop')
    old = json.loads(_read_verified({'path': str(ROOT / (OLD_ID + '.confirmation.json')),
        'sha256': OLD_DIGEST}, detached=True, protected=(GT,)))
    _validate_record(old, require_targets_absent=False, verify_record_files=False)
    require(old['repository'] == {'root': str(REPO), 'commit': COMMIT, 'clean': True},
            'wrong old repository binding')
    files = failed_inventory(old)
    new = recovery_record(old, datetime.now(timezone.utc).isoformat())
    require_absent_targets(new)
    _validate_record(new, require_targets_absent=True, verify_record_files=False)
    for row in new['runs']:
        for key in ('run_dir', 'view_dir', 'state_file', 'launcher_dir', 'qualification_path'):
            require(not os.path.lexists(row[key]), 'new target exists')
        require(not os.path.lexists(row['qualification_path'] + '.sha256'), 'qualification SHA exists')
    require(not os.path.lexists(CONFIRMATION) and
            not os.path.lexists(str(CONFIRMATION) + '.sha256'), 'new confirmation exists')
    receipt_path = ROOT / (NEW_ID + '.recovery.json')
    require(not os.path.lexists(receipt_path) and not os.path.lexists(str(receipt_path) + '.sha256'),
            'recovery receipt exists')
    # Reuse full source/snapshot guards before the canonical writer reopens record handles.
    for key in ('source_record', 'snapshot_record'):
        identity = {name: old[key][name] for name in ('path', 'sha256')}
        _read_verified(identity, protected=(GT,))
    frozen_trees(new, None)
    # Reuse canonical serializer/schema, but do not call the general writer's
    # unguarded reference reopens: every source/snapshot byte above is guarded.
    payload = _canonical_bytes(new)
    DIGEST = hashlib.sha256(payload).hexdigest()
    exclusive(CONFIRMATION, payload.decode('utf-8'))
    exclusive(str(CONFIRMATION) + '.sha256', DIGEST + '\n')
    receipt = {'schema_version': 1, 'attempt': 2, 'reason': 'MKL_STARTUP_ENVIRONMENT',
        'old_confirmation_sha256': OLD_DIGEST, 'new_confirmation_sha256': DIGEST,
        'failed_files': files, 'environment': runtime_environment({}),
        'resumed': False, 'seed_authorized': 0}
    payload = json.dumps(receipt, sort_keys=True, indent=2) + '\n'
    exclusive(receipt_path, payload)
    receipt_sha = hashlib.sha256(payload.encode()).hexdigest()
    exclusive(str(receipt_path) + '.sha256', receipt_sha + '\n')
    os.environ['UTILITY_F0_RECOVERY_SHA'] = receipt_sha
    recovery_binding(DIGEST)
    print('new_confirmation_sha256=' + DIGEST, flush=True)
    prepare()


def audit():
    global DIGEST
    clean_checkout()
    DIGEST = read_digest(str(CONFIRMATION) + '.sha256')
    os.environ['UTILITY_F0_RECOVERY_SHA'] = read_digest(
        ROOT / (NEW_ID + '.recovery.json.sha256'))
    recovery_binding(DIGEST)
    subprocess.run([PYTHON, '-B', 'scripts/diagnostics/audit_prior_transfer_run.py',
        '--confirmation', str(CONFIRMATION), '--confirmation-sha', DIGEST, '--seed', '0'],
        cwd=REPO, env=runtime_environment(os.environ), check=True)


def clean_checkout():
    require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO,
                                   text=True).strip() == COMMIT, 'wrong execution commit')
    require(not subprocess.check_output(['git', 'status', '--porcelain',
                                        '--untracked-files=all'], cwd=REPO,
                                       text=True).strip(), 'dirty checkout')
    subprocess.run(['git', 'diff', '--exit-code'], cwd=REPO, check=True)


def gpu_memory():
    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
        text=True, timeout=10)
    values = [int(value.strip()) for value in result.splitlines()]
    require(len(values) == 1 and values[0] >= 0, 'expected one monitored GPU')
    return values[0]


def verified_tree(root, entries, *, destination=None, private=False):
    """Thin transport guard; never follow an unlisted directory/link to read bytes."""
    from reliability.utility_gt_firewall import _read_verified, _reject_aliases
    from reliability.prior_transfer_assets import _file_inventory
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), 'invalid immutable root')
    require(not any(parent.is_symlink() for parent in root.parents),
            'symlink in immutable root parents')
    expected = {entry['path'] for entry in entries}
    for path in root.rglob('*'):
        require(not path.is_symlink(), 'symlink in immutable/private tree')
        if path.is_file():
            _reject_aliases([{'path': str(path)}], protected=(GT,))
    actual = _file_inventory(root)
    if private:
        actual.discard('sparse_da3_aligned/0/points3D.ply')
    require(actual == expected, 'frozen file inventory mismatch')
    for entry in entries:
        relative = Path(entry['path'])
        require(not relative.is_absolute() and '..' not in relative.parts,
                'unsafe manifest path')
        path = root / relative
        _reject_aliases([{'path': str(path)}], protected=(GT,))
        payload = _read_verified({'path': str(path), 'sha256': entry['sha256']},
                                 protected=(GT,))
        require(len(payload) == entry['bytes'], 'manifest size mismatch')
        if destination is not None:
            target = Path(destination) / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(payload)


def frozen_trees(confirmation, row, *, destination=None):
    from reliability.utility_gt_firewall import _read_verified
    records = {}
    for key in ('source_record', 'snapshot_record'):
        handle = {name: confirmation[key][name] for name in ('path', 'sha256')}
        records[key] = json.loads(_read_verified(handle, protected=(GT,)))
    source, snapshot = records['source_record'], records['snapshot_record']
    require(snapshot['snapshot_sha256'] ==
            '307b176e41111af403a565db94cfe8ada0a7d739361e1e380dcfaca08f49fc22',
            'frozen snapshot mismatch')
    entries = [*source['files'], *snapshot['derived_manifest']]
    require(len({entry['path'] for entry in entries}) == len(entries),
            'duplicate source/derived path')
    verified_tree(source['source_root'], source['files'])
    verified_tree(snapshot['snapshot_root'], entries, destination=destination)
    if row is not None:
        verified_tree(row['view_dir'], entries, private=True)


def prepare():
    clean_checkout()
    require(DIGEST is not None, 'missing explicit recovery digest')
    commands = subprocess.check_output(['ps', '-eo', 'args='], text=True)
    require(not re.search(r'\bpython[^\s]*\s+.*(?:train|estimate_colmap)\.py(?:\s|$)',
                          commands), 'training or DA3 is active')
    # Reuse canonical schema and protected I/O, without opening GT content.
    from reliability.g1_prior_transfer_confirmation import (
        _canonical_bytes, _validate_record)
    from reliability.utility_gt_firewall import _read_verified
    from reliability.prior_transfer_assets import _no_gt
    import hashlib

    confirmation = json.loads(_read_verified(
        {'path': str(CONFIRMATION), 'sha256': DIGEST}, detached=True, protected=(GT,)))
    require(hashlib.sha256(_canonical_bytes(confirmation)).hexdigest() == DIGEST,
            'noncanonical confirmation')
    _validate_record(confirmation, require_targets_absent=True, verify_record_files=False)
    require(confirmation['repository'] == {'root': str(REPO), 'commit': COMMIT,
                                           'clean': True}, 'repository binding mismatch')
    # No general loader's unguarded file reopens; known record bytes are guarded below.
    for item in confirmation['runs']:
        for key in ('run_dir', 'view_dir', 'state_file', 'launcher_dir', 'qualification_path'):
            require(not os.path.lexists(item[key]), 'preregistered target exists')
    for target in confirmation['probe_targets'].values():
        require(not os.path.lexists(target), 'probe target exists')
    row = confirmation['runs'][0]
    require(row['seed'] == 0 and row['training_argv'][0] == PYTHON,
            'seed0/interpreter mismatch')
    _no_gt(row)
    require(shutil.disk_usage('/root/autodl-tmp').free >= 15 * 1024**3,
            'less than 15 GiB free; stop without creating targets')
    gpu_memory()  # Qualify monitoring before launch, without CUDA inference.
    view, run, launch = (Path(row[key]) for key in ('view_dir', 'run_dir', 'launcher_dir'))
    # Full independent copy: baseline loader may write points3D.ply in aligned/0.
    # Never link that writable directory back into the frozen DA3 snapshot.
    view.parent.mkdir(parents=True, exist_ok=False)
    view.mkdir()
    frozen_trees(confirmation, row, destination=view)
    clean_checkout()
    # Preserve absent unauthorized targets, including all probe/firewall targets.
    for other in confirmation['runs'][1:]:
        for key in ('run_dir', 'view_dir', 'state_file', 'launcher_dir', 'qualification_path'):
            require(not os.path.lexists(other[key]), 'unauthorized target appeared')
    for target in confirmation['probe_targets'].values():
        require(not os.path.lexists(target), 'probe target appeared')
    run.mkdir(parents=True, exist_ok=False)
    launch.mkdir(parents=True, exist_ok=False)
    # attempt=1 is the existing auditor's first-launch-within-this-confirmation
    # contract. Cross-confirmation recovery attempt=2 lives in recovery.json.
    record = {'schema_version': 1, 'attempt': 1, 'resumed': False,
              'replaces_completed_run': False, 'confirmation_sha256': DIGEST,
              **{key: row[key] for key in ('seed', 'run_dir', 'view_dir', 'state_file',
                                         'launcher_dir', 'training_argv')}}
    exclusive(launch / 'launch_record.json', json.dumps(record, sort_keys=True) + '\n')
    state = {'RUN_DIR': str(run), 'VIEW_DIR': str(view), 'LAUNCH_DIR': str(launch),
             'SEED': '0', 'CONFIRMATION_SHA256': DIGEST}
    exclusive(row['state_file'], ''.join(key + '=' + shlex.quote(value) + '\n'
                                         for key, value in state.items()))
    exclusive(launch / 'confirmation_sha256.txt', DIGEST + '\n')
    worker = launch / 'launcher.py'
    exclusive(worker, Path(__file__).read_text(encoding='utf-8'))
    env = runtime_environment(os.environ)
    with (launch / 'launcher.log').open('x', encoding='utf-8') as log:
        process = subprocess.Popen([PYTHON, '-B', str(worker), '--worker', str(launch), DIGEST],
                                   cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print('F0_LAUNCH_DISPATCHED=YES')
    print('launcher_pid=' + str(process.pid))
    print('train_log=' + str(run / 'train.log'))
    print('launch_dir=' + str(launch))
    print('training_started=AWAIT_WORKER_RECEIPT')
    print('seed1_seed2_started=NO\ngt_content_accessed=NO\nC1_started=NO')


def worker(launch):
    require(launch == ROOT / (NEW_ID + '_seed0.launch'),
            'unexpected worker target')
    clean_checkout()
    from reliability.utility_gt_firewall import _read_bytes_guarded
    record = json.loads(_read_bytes_guarded(launch / 'launch_record.json', (GT,)))
    require(record['seed'] == 0 and record['confirmation_sha256'] == DIGEST,
            'worker binding mismatch')
    # Recheck canonical argv immediately before the subprocess, never reconstruct it.
    from reliability.utility_gt_firewall import _read_verified
    confirmation = json.loads(_read_verified(
        {'path': str(CONFIRMATION), 'sha256': DIGEST}, detached=True, protected=(GT,)))
    row = confirmation['runs'][0]
    require(all(record[key] == row[key] for key in
                ('seed', 'run_dir', 'view_dir', 'state_file', 'launcher_dir', 'training_argv')),
            'worker row mismatch')
    frozen_trees(confirmation, row)
    run = Path(record['run_dir'])
    exclusive(launch / 'launcher.pid', str(os.getpid()) + '\n')
    exclusive(run / '.training_active', 'seed0\n')
    exclusive(launch / '.training_active', 'seed0\n')
    start = datetime.now(timezone.utc)
    exclusive(launch / 'start_utc.txt', start.isoformat() + '\n')
    rc, peak, training = 2, 0, None
    try:
        with (run / 'train.log').open('x', encoding='utf-8') as log:
            env = runtime_environment(os.environ)
            exclusive(launch / 'threading_environment.json',
                      json.dumps({key: env[key] for key in runtime_environment({})}, sort_keys=True) + '\n')
            training = subprocess.Popen(record['training_argv'], cwd=REPO, env=env,
                                        stdin=subprocess.DEVNULL, stdout=log,
                                        stderr=subprocess.STDOUT)
            exclusive(launch / 'training.pid', str(training.pid) + '\n')
            while training.poll() is None:
                peak = max(peak, gpu_memory())
                time.sleep(1)
            rc = training.returncode
        print('train_return_code=' + str(rc), flush=True)
        require(rc == 0, 'training did not complete successfully')
        require(peak > 0, 'GPU telemetry missing')
        # No asset repair. Recheck immutable source/snapshot/private view after training.
        clean_checkout()
        frozen_trees(confirmation, row)
    except BaseException:
        rc = 2
        if training is not None and training.poll() is None:
            training.terminate()
            try:
                training.wait(timeout=20)
            except subprocess.TimeoutExpired:
                training.kill()
                training.wait()
        raise
    finally:
        end = datetime.now(timezone.utc)
        exclusive(launch / 'end_utc.txt', end.isoformat() + '\n')
        exclusive(launch / 'wall_seconds.txt', str(round((end-start).total_seconds())) + '\n')
        exclusive(launch / 'gpu_peak_mib.txt', str(peak) + '\n')
        exclusive(launch / 'exit_code.txt', str(rc) + '\n')
        for sentinel in (run / '.training_active', launch / '.training_active'):
            sentinel.unlink(missing_ok=True)  # Only the two markers created above.
    print('F0_TRAINING_COMPLETE=YES', flush=True)
    print('qualification_pending=YES', flush=True)


if __name__ == '__main__':
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    # Set before any NumPy-bearing project import, never mutate shell/packages.
    for name in ('MKL_SERVICE_FORCE_INTEL', 'PYTHONPATH', 'PYTHONHOME'):
        os.environ.pop(name, None)
    os.environ.update(runtime_environment(os.environ))
    if sys.argv[1:] == ['--audit']:
        audit()
    elif sys.argv[1:2] == ['--worker'] and len(sys.argv) == 4:
        DIGEST = sys.argv[3]
        recovery_binding(DIGEST)
        worker(Path(sys.argv[2]))
    else:
        require(len(sys.argv) == 1, 'no seed/protocol overrides allowed')
        recover()
