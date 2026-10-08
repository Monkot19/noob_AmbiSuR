#!/usr/bin/env bash
# Already authorized F0 qualification; run only after launcher exit.
set -euo pipefail
export OMP_NUM_THREADS=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONHOME
cd /root/autodl-tmp/noob_AmbiSuR
test "$(git rev-parse HEAD)" = c701424c1b1f5a9006e6f19776769ee7bc8cb299
test -z "$(git status --porcelain --untracked-files=all)"
git diff --exit-code
/root/miniconda3/envs/ambisur/bin/python -B scripts/diagnostics/audit_prior_transfer_run.py \
  --confirmation /root/autodl-tmp/ambisur_diagnostics/Utility_Room/prior-transfer/utility_prior_transfer_softv4_20261008_v2.confirmation.json \
  --confirmation-sha 51376355bb19434ebfd118090ea2c1b38371156a9dbaf4da51bb96aaf385020e \
  --seed 0
git status --short --branch
echo 'UTILITY_GATE_F0_ASSET_QUALIFICATION=PASS'
echo 'seed1_seed2_started=NO'
echo 'gt_evaluated=NO'
echo 'C1_started=NO'
