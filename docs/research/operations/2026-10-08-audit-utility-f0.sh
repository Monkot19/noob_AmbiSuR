#!/usr/bin/env bash
# Already authorized F0 qualification; run only after launcher exit.
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_THREADING_LAYER=GNU PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONHOME MKL_SERVICE_FORCE_INTEL
cd /root/autodl-tmp/noob_AmbiSuR
test "$(git rev-parse HEAD)" = c701424c1b1f5a9006e6f19776769ee7bc8cb299
test -z "$(git status --porcelain --untracked-files=all)"
git diff --exit-code
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Sidecar reads must pass the protected reader too, before invoking the auditor.
/root/miniconda3/envs/ambisur/bin/python -B \
  "$HERE/2026-10-08-launch-utility-f0.py" --audit
git status --short --branch
echo 'UTILITY_GATE_F0_ASSET_QUALIFICATION=PASS'
echo 'seed1_seed2_started=NO'
echo 'gt_evaluated=NO'
echo 'C1_started=NO'
