#!/usr/bin/env bash
# Authorized Gate E only. Run on AutoDL, not on Windows. No training or GT open.
set -euo pipefail
export OMP_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONHOME
cd /root/autodl-tmp/noob_AmbiSuR
test "$(git rev-parse HEAD)" = c701424c1b1f5a9006e6f19776769ee7bc8cb299
test -z "$(git status --porcelain --untracked-files=all)"
if ps -eo args= | grep -E '[p]ython[^ ]* .*([t]rain[.]py|[e]stimate_colmap[.]py)' >/dev/null; then
    echo 'STOP: training or DA3 inference is active'
    exit 2
fi
/root/miniconda3/envs/ambisur/bin/python -B - <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import json
import os

from reliability.g1_prior_transfer_confirmation import (
    build_prior_transfer_confirmation, load_prior_transfer_confirmation,
    validate_preregistration_targets, write_prior_transfer_confirmation,
)
from reliability.utility_gt_firewall import _read_verified
from reliability.utility_snapshot import _verify_source_manifest

repo = Path.cwd()
root = Path('/root/autodl-tmp/ambisur_diagnostics/Utility_Room/prior-transfer')
old_path = root / 'utility_prior_transfer_softv4_20261008_v1.confirmation.json'
old_sha = 'b991a1624f03060f0633b6b70eee16b763b4fd3fc1739daf1d956a29ea2f62c9'
gt_path = Path('/root/autodl-tmp/ambisur_data/gt/ScanNetpp/Utility_Room/mesh_aligned_0.05.ply')
protected = (gt_path,)

def record(path, digest):
    return json.loads(_read_verified({'path': str(path), 'sha256': digest}, protected=protected))

# GT is only an identity to protect from alias reads; never open or hash its bytes.
old = record(old_path, old_sha)
source_id = old['source_record']
snapshot_id = old['snapshot_record']
assert source_id == {
    'path': str(root / 'utility_room_source_147_v1.json'),
    'sha256': 'cd172697f1d5d831dd314dd1d85675777f6a95a7359735192a722cd883dd2b47',
}
assert snapshot_id == {
    'path': str(root / 'utility_room_da3_snapshot_147_v1.json'),
    'sha256': '522bad824116e9910636eb734efe2e222b2d5978abc3d384a0bf5f51e30cb143',
    'snapshot_sha256': '307b176e41111af403a565db94cfe8ada0a7d739361e1e380dcfaca08f49fc22',
}
source = record(source_id['path'], source_id['sha256'])
snapshot = record(snapshot_id['path'], snapshot_id['sha256'])
assert old['gt_mesh'] == {
    'path': str(gt_path), 'bytes': 24490351,
    'sha256': '213dbdfff9ba992000039533463e4fcd941708d9fd8fe077df53a8495b63cd75',
    'content_accessed': False,
}
assert old['repository']['commit'] == '8980d5849f26a6152d349235aa77c5c68ed1e692'
assert snapshot['snapshot_sha256'] == snapshot_id['snapshot_sha256']
assert snapshot['source_sha256'] == source['source_sha256']
assert snapshot['gt_access'] == 'NONE'
old = load_prior_transfer_confirmation(old_path, old_sha)
validate_preregistration_targets(old)
_verify_source_manifest(source)
snapshot_root = Path(snapshot['snapshot_root']).resolve(strict=True)
assert snapshot_root == Path('/root/autodl-tmp/ambisur_work/data_snapshots/utility_room_da3_147_20260930_v1/colmap_undistorted')
_verify_source_manifest(source, root_override=snapshot_root)
for image_root in (Path(source['source_root']) / 'images', snapshot_root / 'images'):
    assert {p.name for p in image_root.iterdir() if p.is_file()} == set(source['image_names'])
assert len(source['image_names']) == 147
derived = set()
for name in ('estimated_depths', 'estimated_confs', 'sparse_da3', 'sparse_da3_aligned'):
    derived.update(p.relative_to(snapshot_root).as_posix() for p in (snapshot_root / name).rglob('*') if p.is_file())
assert derived == {item['path'] for item in snapshot['derived_manifest']}
for item in snapshot['derived_manifest']:
    path = (snapshot_root / item['path']).resolve(strict=True)
    assert snapshot_root in path.parents
    _read_verified({'path': str(path), 'sha256': item['sha256']}, protected=protected)
    assert path.stat().st_size == item['bytes']

cid = 'utility_prior_transfer_softv4_20261008_v2'
output = root / (cid + '.confirmation.json')
assert not os.path.lexists(output) and not os.path.lexists(str(output) + '.sha256')
runs = []
for original in old['runs']:
    seed = original['seed']
    name = cid + '_seed' + str(seed)
    run_dir = Path('/root/autodl-tmp/ambisur_runs/Utility_Room/prior-transfer-softv4-7k') / name
    view_dir = Path('/root/autodl-tmp/ambisur_work/data_views') / name / 'colmap_undistorted'
    argv = list(original['training_argv'])
    assert argv[0] == '/root/miniconda3/envs/ambisur/bin/python'
    argv[argv.index('--source_path') + 1] = str(view_dir)
    argv[argv.index('--model_path') + 1] = str(run_dir)
    runs.append({
        'seed': seed, 'run_dir': str(run_dir), 'view_dir': str(view_dir),
        'state_file': str(root / (name + '.env')),
        'launcher_dir': str(root / (name + '.launch')),
        'qualification_path': str(root / (name + '.qualification.json')),
        'training_argv': argv,
    })
probe_targets = {
    'output_dir': str(root / (cid + '.probe')),
    'staging_dir': str(root / ('.' + cid + '.probe-staging')),
    'access_log_path': str(root / (cid + '.first-gt-access.json')),
}
for row in runs:
    for key in ('run_dir', 'view_dir', 'state_file', 'launcher_dir', 'qualification_path'):
        assert not os.path.lexists(row[key]), 'STOP: existing target ' + row[key]
    assert not os.path.lexists(row['qualification_path'] + '.sha256')
for path in probe_targets.values():
    assert not os.path.lexists(path), 'STOP: existing probe target ' + path
new = build_prior_transfer_confirmation(
    confirmation_id=cid, created_utc=datetime.now(timezone.utc).isoformat(),
    repository={'root': str(repo), 'commit': 'c701424c1b1f5a9006e6f19776769ee7bc8cb299', 'clean': True},
    source_record=source_id, snapshot_record=snapshot_id, gt_mesh=old['gt_mesh'],
    runs=runs, probe_targets=probe_targets,
)
for key in ('protocol', 'mesh_admission', 'accepted_geometry_releases', 'firewall'):
    assert new[key] == old[key], 'STOP: frozen contract changed: ' + key
published = write_prior_transfer_confirmation(new, output)
reloaded = load_prior_transfer_confirmation(output, published['sha256'])
assert reloaded == new
_read_verified({'path': str(output), 'sha256': published['sha256']}, detached=True, protected=protected)
validate_preregistration_targets(reloaded)
record(old_path, old_sha)
print(json.dumps({
    'confirmation_path': str(output), 'confirmation_sha256': published['sha256'],
    'execution_commit': new['repository']['commit'],
    'snapshot_sha256': snapshot_id['snapshot_sha256'],
    'runs': runs, 'probe_targets': probe_targets,
}, indent=2))
print('GATE_E_V2_CONFIRMATION=PASS')
print('old_confirmation_preserved=YES')
print('all_run_and_probe_targets_absent=YES')
print('training_started=NO\nda3_started=NO\nmesh_parsed=NO\ngt_evaluated=NO\nC1_started=NO')
PY
