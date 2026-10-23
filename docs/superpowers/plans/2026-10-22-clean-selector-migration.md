# Clean second-run selector migration

2026-10-22 user approves: new branch directly from c0-baseline, only selector code/tests and necessary config/log reuse; preserve the first-run branches and fixed baseline output. This is a bounded port of the approved algorithm, not a new method/spec review cycle. The old five-file boundary is amended only for lightweight seed/metadata and test package support.

Base d6f15c8891a53800d5e3100f95817a7dd7f98e2f; branch codex/second-run-selector in an independent managed worktree.

- [x] Adapt the existing unified test entry, remove old Core imports and observe missing-API RED.
- [x] Copy utils/candidate_selector.py byte-for-byte; port optional renderer tags and train reference/ALR helpers only onto baseline code.
- [x] Reuse seed-compatible safe_state and existing JSON/git/env metadata routines in utils/run_metadata.py without CoreConfig. Keep cfg_args/cfg_opts and original checkpoint tuples.
- [x] Local tests/compile/whitespace, exact baseline-boundary verification and one code review; native tests remain on AutoDL.

Code boundary: arguments/__init__.py, gaussian_renderer/__init__.py, train.py, utils/general_utils.py, utils/candidate_selector.py, utils/run_metadata.py, tests/test_candidate_selector.py, tests/__init__.py (package marker). No scene/CUDA/loss/geometry changes, reliability modules, GT, baseline rerun or server storage operations.

Keep the approved selector/train helper signatures. build_resolved_config(model,opt,pipeline), collect_run_identity(repo), write_run_metadata(directory,config,identity) reuse the existing serializer/git/env implementations without old Core schema/dispatch; training_path explicitly records baseline or candidate_selector.

Run python -B -m unittest tests.test_candidate_selector -v. Native skips are not acceptance; no dependency installation. Commit/push and server launch remain separate discussed batches.

## Local migration receipt

- Scoped unified unittest:23run,18CPU pass/5native skips; no GPU acceptance claimed. Bare discovery additionally fails importing gaussian_renderer (missing rasterizer) and scene (missing plyfile); no hidden failures or dependency installation.
- Eight code files compile, whitespace check passes; scene/CUDA/geometry/loss/environment sources match c0-baseline exactly. Selector SHA f5c4c2a51c496d2ab6eb4203c8e0745318d30cc352cd567d63e2cf3292deefeb matches the reviewed old-worktree module byte-for-byte.
- One stage review found no Critical/Important/required Minor; independently checked diff/hash/whitespace but did not run runtime tests. GPU behavior/training/benefit remain unjudged.
- Original D:/research_Space/noob_AmbiSuR dirty status and all dirty/untracked file hashes unchanged. No forced reset, delete, old-branch commit, baseline mutation, push or server operation.
- AutoDL command after discussed version handoff: python -B -m unittest tests.test_candidate_selector -v; all5native tests must execute. The old tests.gpu dispatch module is intentionally not ported. Installed wrapper and C extension must first be checked for stale one-run source/export.

## 2026-10-23 Version handoff batch

User approved local commit/push and exact server checkout/test instructions, and reports GPU server rented. Baseline extension compiled from the pinned base and import/export checks passed in /root/autodl-tmp/ambisur-c0-rasterizer.U365Ym/python; select it via per-process PYTHONPATH for all second-run commands. Preserve the original installed first-run package. Fresh local scoped validation remains18CPU pass/5native skips with syntax/whitespace/baseline-boundary checks passing. Use the same unittest suite on AutoDL with a one-shot runner that rejects skips, without creating another test script. Server GPU execution and30k are not part of the local success claim. Existing fixed baseline output stays read-only.
