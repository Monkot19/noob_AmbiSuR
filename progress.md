# AmbiSuR 新项目 Progress

## Session Log

### Project bootstrap
- **Status:** not_started
- [ ] 交接包已复制到仓库根目录
- [ ] 最终设计稿已核对
- [ ] baseline commit 已记录
- [ ] Git 分支与 tag 策略已建立
- [ ] Core implementation plan 已获用户批准

## Verification Log

| Date | Stage | Commit | Command | Result | Manifest |
|---|---|---|---|---|---|

## Cloud Runs

| Date | Scene | Stage | Commit | Seed | Result path | Decision |
|---|---|---|---|---:|---|---|

## Current Blocker

无。首次运行时由 Agent 更新。



## 2026-10-22 Clean second-run migration
User approves c0-baseline branch plus selector-only port and necessary seed/metadata reuse. Old records above are historical. The original dirty checkout stays unchanged. No baseline/GT/storage operation.
- Native managed worktree created from c0-baseline; branch codex/second-run-selector. Initial docs write needed its missing plans parent directory; created that parent and resumed, original checkout unaffected.
- Test-first migration RED observed at missing utils.run_metadata on clean c0 checkout, then ported the byte-identical selector and minimal baseline train/renderer changes. Initial local pass:18CPU and4native skipped.
- Added same-entry native environment check: installed rasterizer Python implementation must match baseline source and C extension must not expose old evidence export; stale editable installs cannot silently count as clean-source validation. AutoDL native tests remain unrun.
- Reused serializer/git/environment logging without CoreConfig/checkpoint/topology code; original scene/CUDA/loss/geometry/checkpoint/optimizer paths preserved. Metadata and seed tests live in the same test entry, no new framework.
- One changed baseline function header retained old trailing spaces; git diff --check exposed it, corrected that header only.
- Scoped unified test PASS23run/18CPU/5native skips; py_compile and diff check passed. Bare discovery also reports two known missing-dependency import errors (gaussian_renderer/rasterizer and scene/plyfile); not a full-suite pass. No local install or server execution.
- Single independent migration review found no Critical/Important/required Minor; native CUDA/logging/ALR/runtime/benefit remain unverified. Original dirty checkout status and hashes independently compared unchanged. Scene/CUDA/loss/geometry/environment match c0, selector module byte-identical.
- Updated only this new worktree's specs/plan/handoff and ledgers, with two algorithm provenance records. No commit/push, server cleanup, baseline overwrite or30k launch.

## 2026-10-23 Storage and native package handoff
- Applied using-superpowers/systematic-debugging to new server receipt; read current clean worktree handoff/config/source/ledgers. No code/test/branch/index changes.
- Verified local VSCode commit via installed code CLI and matched e4c7... to one of six server directories, enabling a precise five-directory cleanup proposal instead of deleting all VSCode files.
- Read baseline native setup/export source and official PyTorch extension docs. Server installed old evidence export confirmed by user receipt; build/install/provenance selection remains pending, no GPU test/training result inferred.
- Only these records updated; no local cleanup, remote connection, package install, commit/push or baseline mutation. Next user-operated commands can remove the five named obsolete VSCode versions with one interactive prompt and inspect compiler/toolchain without GPU.

### Follow-up server receipt
- User's interactive rm displayed its confirmation prompt; no affirmative response is present, and CLI size/df stayed unchanged. Deletion is not established. Reissue only the same five explicit old-version paths without the interactive prompt; retain e4c7... .
- Server nvcc12.8/g++11.4/Python3.10/Torch2.7.1+cu128/CUDA_HOME confirmed. This permits attempting a no-GPU build, not claiming native/GPU validation passed.
- Locally rechecked the peeled c0-baseline commit d6f15c8891a53800d5e3100f95817a7dd7f98e2f, native setup and tracked GLM header. Proposed pinned-source wheel/target on data disk, per-process Python path, preserving installed first-run extension. No build/install or server operation performed by agent.

### Successful cleanup/build; validation-command name error
- User receipt: system73%/8.3G available; VSCode CLI672M. Pinned c0 wheel built and installed in ambisur-c0-rasterizer.U365Ym/python, import selected that target. Agent-provided check stopped at AssertionError:markVisible.
- Applied systematic-debugging: fixed-commit ext.cpp binds mark_visible to C++markVisible; wrapper method markVisible calls _C.mark_visible. Corrected the earlier findings entry and prepared a single import/export recheck against the existing target. No rebuild, experiment-code edit, package reinstall, GPU test, commit/push or training performed.

### Corrected no-GPU import check passed (user receipt)
- User ran the corrected command successfully: wrapper and binary resolve inside /root/autodl-tmp/ambisur-c0-rasterizer.U365Ym/python; all three baseline exports are callable and old evidence export is absent. No-GPU build/import preparation is complete; CUDA rendering and five native selector tests are still pending.
- Local read-only status confirms codex/second-run-selector remains at c0 with the reviewed migration uncommitted; no commit/push or server checkout/training performed. Next discussed batch is local commit/push plus exact server checkout and per-process extension/test commands; postpone paid GPU startup until that handoff is ready.

### Version handoff authorized; fresh precommit verification
- User agreed to the commit/push/server-handoff batch and reported server rented. Applying executing-plans/verification-before-completion to the existing approved migration; no new implementation or algorithm decisions.
- Fresh unified test run:23 tests,18passed/5native skipped for missing local rasterizer;8 code files compile in memory. git diff --check and unchanged scene/submodules/geometry/loss/environment boundary passed; selector SHA unchanged. Native acceptance and30k remain pending.
- Scope of this commit is the8 reviewed code/test files plus approved specs/provenance/working records; original first-run checkout and fixed output remain untouched. Server test command will fail on any skipped test as well as errors/failures; reuse existing tests without adding a runner file.

### Version published for user-operated GPU validation
- Feature commit d979c070925a9dc9a5f5ce3a5563de56614fa192 pushed to origin/codex/second-run-selector. Fresh ls-remote matched local HEAD exactly; git status --short was empty before push and after verification. No pull request, force push, original-checkout mutation or server launch.
- This documentation-only follow-up marks version handoff complete. Server command will check clean status, fetch this branch, detach at the final exact handoff commit, print HEAD/diff, select the previously built baseline extension, assert visible CUDA, and execute the existing unified23-test suite rejecting any skip. User must return GPU result; no native pass, short real-data training or30k result claimed.

### 2026-10-23 User-returned GPU result:23/23 passed
- Read complete pasted receipt at C:/Users/DELL/.codex/attachments/c1cf6350-d080-4257-a259-7bd946b339f8/已粘贴的文本.txt. Exact3d85faf,RTX4090,independent baseline extension;23tests/2.875s/OK,no failures/errors/skips. No code fix/retest needed. Updated current handoff and milestones; no server launch or new method/diagnostic module.
- Read frozen baseline configuration/logs without writing its output: Tool Room406views,r2,eval=False,30k. Preparing one candidate run with seed0,budget4096 and original schedules/DA3; active real-data evidence first appears after7000. Proposed startup/active-phase checks can be observations within the same30k job, avoiding a separately interrupted/resumed smoke with changed total-iteration schedules or RNG state.
- Exact current server writable Tool Room data-view path is not supplied by this test receipt. Resolve that small operational input before final launch command; canonical data and baseline remain protected. Next training scope/command requires the user's batch discussion, no automatic30k authorization inferred from a test receipt.

### First30k handoff prepared from returned directory listing
- User returned the requested directories, including /root/autodl-tmp/ambisur_work/data_views/toolroom_uploaded_20260902/c0_candidate_r2_30k_seed0_20260902T082615Z/colmap_undistorted. Prepare launch for the discussed Tool Room/r2/seed0/30k/budget4096 scope; execution remains user-operated.
- Fresh real argument-parser check against read-only baseline cfg_args/cfg_opts passed: all existing fields match except intended input/output paths; optimization adds only seed/enable_candidate_selector/candidate_selector_budget. Keep original eval=False,DA3,ALR7001 and30k schedules,default7k/30k saving,no checkpoint resume.
- Protect both old private view and canonical data: new per-run thin view links images/sparse/estimated_depths/estimated_confs and copies sparse_da3_aligned with dereferenced independent files; require aligned points3D.bin/trans.json so loader cannot fall back to writing original sparse. Preserve split.json if present. Only the small writable subtree is copied; no new experiment/test framework.
- Proposed roots: data_views/second_run_selector/toolroom_r2_seed0_30k_v1/colmap_undistorted and ambisur_runs/Tool_Room/selector_v1/3d85faf_seed0_30k_v1. Refuse existing destinations; background train logs and exitcode stored in the new output. Use exact tested3d85faf code/independent rasterizer. No mesh/GT/evaluation or server training performed by agent. This documentation-only receipt does not require changing the tested server code version.
