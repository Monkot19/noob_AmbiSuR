"""One approved offload operation. No recursive deletion or experiment launch."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
from datetime import datetime, timezone
from contextlib import ExitStack

ROOT = Path('/root/autodl-tmp/ambisur_diagnostics/Tool_Room/d0-g1')
RECEIPT = Path('/root/autodl-tmp/ambisur_diagnostics/storage-offload-20261008-v1.jsonl')
OUTPUTS = {
    'd0_g1_exploratory500_toolroom_seed0_20260917_v3':
        ('5e2b02903b0b3f44ea19f3a3dff6fab0f0486eba24f0e28e734e6f56a1bfc9aa', 76357832,
         'a3af25d6cea62396e0dd07f9387e20faeae7422cc190037e8e4b6c573b31d9aa'),
    'd0_g1_exploratory500_toolroom_seed0_20260917_v4':
        ('30d292db1156823f6d14b5052fbc1dd063fe976e9d5f41733fb9e2e035600c62', 76363660,
         '53bd6410d47dbd3baff0b310a98cf077114ea6f2b4075cf34c20a4e0f0f5303e'),
    'd0_g1_formal7k_v2_toolroom_seed0_20260918_v1':
        ('2f3d79004afd5fc3755ef06e8e04d275aa761c6e7ed2f52efbdcbb6e8cab4743', 998925215,
         'fe748fb5cf89c45d66dbf5136fd21e270d55fdccf02b38a902a22f96866a1696'),
    'd0_g1_softcal_v4_formal7k_toolroom_seed0_20260928_v1':
        ('d96ecd76ee537de516435220de7ca2f81f7158c9e2d06f5bd4299037c7e3801e', 1030801710,
         '86ad01a2499b97170fbc1e2e9ce4347900ad0ef646cfec4ded44fdb37205b43d'),
}
FIELDS = ('A', 'K', 'N', 'S', 'T_g', 'T_p', 'gt_distance', 'state')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def safe_path(root, relative):
    parts = PurePosixPath(relative).parts
    require(parts and not PurePosixPath(relative).is_absolute()
            and '..' not in parts and '\\' not in relative, 'unsafe relative path')
    candidate = root.joinpath(*parts)
    # Check every ancestor, including those above root, before opening anything.
    for component in (candidate, *candidate.parents):
        require(not component.is_symlink(), f'symlink rejected: {component}')
    require(candidate.resolve(strict=True).is_relative_to(root.resolve(strict=True)), 'path escape')
    info = candidate.lstat()
    require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, f'not an unlinked regular file: {candidate}')
    return candidate, identity(info)


def verified_bytes(root, entry, collect=False):
    path, before = safe_path(root, entry['path'])
    require(before[2] == entry['bytes'], f'size mismatch: {path}')
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_BINARY', 0)
    digest = hashlib.sha256()
    chunks = []
    with os.fdopen(os.open(path, flags), 'rb') as stream:
        require(identity(os.fstat(stream.fileno()))[:4] == before[:4], f'changed before read: {path}')
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
            if collect:
                chunks.append(chunk)
        require(identity(os.fstat(stream.fileno()))[:4] == before[:4], f'changed during read: {path}')
    require(safe_path(root, entry['path'])[1] == before, f'changed after read: {path}')
    require(digest.hexdigest() == entry['sha256'], f'SHA mismatch: {path}')
    return before, b''.join(chunks)


def build_plan(root):
    entries = []
    for name, (manifest_sha, archive_size, archive_sha) in OUTPUTS.items():
        relative = name + '/manifest.json'
        path, info = safe_path(root, relative)
        _, raw = verified_bytes(root, {'path': relative, 'bytes': info[2], 'sha256': manifest_sha}, True)
        manifest = json.loads(raw)
        entries.append({'path': name + '.tar.gz', 'bytes': archive_size, 'sha256': archive_sha})
        if name.startswith('d0_g1_formal7k_') or name.startswith('d0_g1_softcal_'):
            listed = {row['path']: row for row in manifest['files']}
            require(len(listed) == len(manifest['files']), 'duplicate manifest paths')
            for iteration in (3000, 7000):
                for field in FIELDS:
                    rel = f'iteration_{iteration:06d}/fields/{field}.ply'
                    row = listed[rel]
                    entries.append({'path': name + '/' + rel, 'bytes': row['bytes'], 'sha256': row['sha256']})
    require(len(entries) == 36 and sum(row['bytes'] for row in entries) == 4909886174,
            'approved inventory mismatch')
    return entries


def open_parent(root, relative, stack):
    if os.name != 'posix':  # Temporary Windows checks only; main requires Linux.
        return None
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory = os.open('/', flags)
    stack.callback(os.close, directory)
    for part in (root / relative).parent.parts[1:]:
        directory = os.open(part, flags, dir_fd=directory)
        stack.callback(os.close, directory)
    return directory


def require_no_open_writers(fingerprints):
    # Locks are advisory: require a quiescent operator boundary as well.
    # Do not start writers while this operation runs. This is not a sandbox
    # against a malicious root process ignoring locks and /proc admission.
    if sys.platform != 'linux':
        return
    identities = {tuple(info[:2]) for info in fingerprints}
    for process in Path('/proc').iterdir():
        if not process.name.isdigit() or process.name == str(os.getpid()):
            continue
        try:
            for descriptor in (process / 'fd').iterdir():
                try:
                    flags = (process / 'fdinfo' / descriptor.name).read_text()
                    access = int(re.search(r'^flags:\s+([0-7]+)$', flags, re.M)[1], 8)
                    if access & os.O_ACCMODE != os.O_RDONLY:
                        info = descriptor.stat()
                        require((info.st_dev, info.st_ino) not in identities,
                                f'target open for writing by PID {process.name}')
                except FileNotFoundError:
                    continue  # A descriptor closed during the inventory.
        except FileNotFoundError:
            continue  # A process exited during the inventory.


def remove_verified(root, entries, receipt):
    require(len({row['path'] for row in entries}) == len(entries), 'duplicate removal targets')
    # Validate the entire inventory before the first unlink or receipt creation.
    fingerprints = [verified_bytes(root, row)[0] for row in entries]
    for row, fingerprint in zip(entries, fingerprints):
        require(safe_path(root, row['path'])[1] == fingerprint, 'target changed after validation')
    for parent in receipt.parents:
        require(not parent.is_symlink(), 'receipt parent symlink rejected')
    before_free = shutil.disk_usage(root).free
    with ExitStack() as stack:
        handles = []
        for row, fingerprint in zip(entries, fingerprints):
            directory = open_parent(root, row['path'], stack)
            path = root / row['path']
            flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_BINARY', 0)
            descriptor = os.open(path.name if directory is not None else path,
                                 flags, dir_fd=directory)
            stream = stack.enter_context(os.fdopen(descriptor, 'rb'))
            if os.name == 'posix':
                import fcntl
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            require(identity(os.fstat(descriptor))[:4] == fingerprint[:4], 'changed while acquiring lock')
            handles.append((directory, stream))
        require_no_open_writers(fingerprints)
        journal = stack.enter_context(receipt.open('x', encoding='utf-8'))
        if os.name == 'posix':
            directory = os.open(receipt.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(directory)  # Persist receipt directory entry before unlink.
            finally:
                os.close(directory)
        def record(event, **values):
            journal.write(json.dumps({'event': event, 'utc': datetime.now(timezone.utc).isoformat(), **values}) + '\n')
            journal.flush()
            os.fsync(journal.fileno())

        record('plan', targets=entries, free_before=before_free,
               restore_root='D:/research_Space/output/ambisur_diagnostics/Tool_Room/d0-g1',
               server_formal_outputs='partial/offloaded; original manifests unchanged',
               original_gz_bytes_backed_up=False,
               concurrency='exclusive file locks; no concurrent writers permitted')
        removed = 0
        try:
            for row, fingerprint, (directory, stream) in zip(entries, fingerprints, handles):
                path, current = safe_path(root, row['path'])
                require(current == fingerprint, f'target changed; stop: {path}')
                require(identity(os.fstat(stream.fileno()))[:4] == fingerprint[:4],
                        f'locked file changed after path check: {path}')
                if directory is None:
                    stream.close()  # Windows forbids unlink of an open file.
                    path.unlink()  # Windows temporary tests only.
                else:
                    current = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
                    require(identity(current) == fingerprint, 'pinned directory entry changed')
                    os.unlink(path.name, dir_fd=directory)  # Never resolve ancestors again.
                    os.fsync(directory)
                removed += 1
                record('removed', **row)
            after_free = shutil.disk_usage(root).free
            record('complete', removed_count=removed, free_after=after_free,
                   actual_free_increase=after_free - before_free,
                   f0_storage_gate_pass=after_free >= 15 * 1024**3)
        except BaseException as error:
            record('failed', removed_count=removed, reason=str(error), automatic_retry=False)
            raise
    print(f'removed_count={removed}')
    print(f'free_bytes={after_free}')
    print(f'free_gib={after_free / 1024**3:.3f}')
    print(f'F0_STORAGE_GATE={"PASS" if after_free >= 15 * 1024**3 else "FAIL"}')
    print(f'offload_receipt={receipt}')


def main():
    require(sys.platform == 'linux', 'real cleanup requires Linux; no local result deletion')
    require(sys.argv[1:] == ['--execute-approved'], 'requires --execute-approved; no path overrides')
    require(not RECEIPT.exists() and not RECEIPT.is_symlink(), 'receipt exists; do not repeat cleanup')
    processes = subprocess.check_output(['ps', '-eo', 'args='], text=True)
    require(not any(re.search(r'\bpython(?:3(?:\.\d+)?)?\b', line)
                    and re.search(r'(?:train|evaluate[^ /]*|diagnose[^ /]*)\.py\b', line)
                    for line in processes.splitlines()), 'training/evaluator active; stop')
    plan = build_plan(ROOT)
    print('Validating 36 approved targets before removal...', flush=True)
    remove_verified(ROOT, plan, RECEIPT)
    print('STORAGE_CLEANUP=COMPLETE\ntraining_started=NO\ngt_content_accessed=NO')


if __name__ == '__main__':
    main()
