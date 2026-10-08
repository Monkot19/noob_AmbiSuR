"""User-operated seed0-only handoff; not a new training/evaluation engine.

Run from a transported pinned documentation artifact on AutoDL. The checkout
stays at the already qualified execution commit. No retry/resume is provided.
"""
from datetime import datetime, timezone
from pathlib import Path
import json
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
CONFIRMATION = ROOT / 'utility_prior_transfer_softv4_20261008_v2.confirmation.json'
DIGEST = '51376355bb19434ebfd118090ea2c1b38371156a9dbaf4da51bb96aaf385020e'
GT = Path('/root/autodl-tmp/ambisur_data/gt/ScanNetpp/Utility_Room/mesh_aligned_0.05.ply')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def exclusive(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(value)


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
    verified_tree(row['view_dir'], entries, private=True)


def prepare():
    clean_checkout()
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
    env = dict(os.environ)
    for name in ('PYTHONPATH', 'PYTHONHOME'):
        env.pop(name, None)
    env.update(OMP_NUM_THREADS='1', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    with (launch / 'launcher.log').open('x', encoding='utf-8') as log:
        process = subprocess.Popen([PYTHON, '-B', str(worker), '--worker', str(launch)],
                                   cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print('F0_LAUNCH_DISPATCHED=YES')
    print('launcher_pid=' + str(process.pid))
    print('train_log=' + str(run / 'train.log'))
    print('launch_dir=' + str(launch))
    print('training_started=AWAIT_WORKER_RECEIPT')
    print('seed1_seed2_started=NO\ngt_content_accessed=NO\nC1_started=NO')


def worker(launch):
    require(launch == ROOT / 'utility_prior_transfer_softv4_20261008_v2_seed0.launch',
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
            training = subprocess.Popen(record['training_argv'], cwd=REPO,
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
    if sys.argv[1:2] == ['--worker'] and len(sys.argv) == 3:
        worker(Path(sys.argv[2]))
    else:
        require(len(sys.argv) == 1, 'no seed/protocol overrides allowed')
        prepare()
