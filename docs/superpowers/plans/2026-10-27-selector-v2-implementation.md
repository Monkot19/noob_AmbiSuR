# selector_V2 Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline; user explicitly authorized implementation and autonomous routine fixes while away. Do not request another plan approval.

**Goal:** V2=V1+periodic relocation of additionally selected Gaussians, ready for an AutoDL 7K→30K experiment.

**Architecture:** One existing selector core with compatible V1 wrapper and V2 target wrapper; one local Gaussian update method; action before fresh main reference render. Reuse original ALR, topology, metadata, native checkpoint and evaluator.

**Tech Stack:** Python3.10-compatible PyTorch; existing Windows CPU test runtime; AutoDL native GPU execution by user.

**Spec:** ../specs/2026-10-27-selector-v2-relocation-design.md and ../../research/2026-10-27-selector-v2-code-adaptation.md, both approved for implementation in latest user message.

## Global Constraints
- Work in D:/research_Space/noob_AmbiSuR on codex/second-run-selector-v2, based on ef333bb; preserve V1 branch, C worktree, baseline branch and all datasets/outputs.
- Budget16384/seed0/r2/DA3/total30000, existing native7K prefix; every100 iterations7100..29900, same NCC winners/targets, no extra loss/render/cache/GPU smoke.
- Five code files only: utils/candidate_selector.py, scene/gaussian_model.py, train.py, arguments/__init__.py, tests/test_candidate_selector.py. Documentation may change; metadata/renderer/CUDA/evaluator/checkpoint unchanged.
- New flag defaultsFalse, interval100; V1 default4096 and public three-tuple preserved. No dependency install.

## Review Focus
- Random accepted-ID order must stay paired with same evaluated camera-space target; nonidentity camera transform must invert correctly (Task1 test).
- V2/no-event/off/empty states must retain V1 labels/cost calls/global RNG and require no additional rendering (Task1 test).
- Invalid target, binding or state input must fail before any position/state write (Task1 test).
- Actual Adam Parameter identity, nonselected rows, global step and other groups must survive relocation (Task1 test).
- Current render must see relocated positions; no stale IDs across topology or position write after forward, and no extra global RNG usage (Task1 test and source/native boundary).

## Task1: Implement the approved five-file V2 increment

**Consumes:** Existing select_candidate_labels signature/three-tuple, exact c351 training helper, Gaussian Adam/stat arrays, original test fixture and AST native-method loader.

**Produces:** candidate_relocation_active(iteration,opt); select_candidate_labels_with_targets(same_arguments) → (labels,extra,stats,None|(ids,world_xyz)); GaussianModel.relocate_selected(ids,world_xyz) → int; opt.enable_candidate_relocation=False/candidate_relocation_interval=100.

- [x] Run existing unified test entry once to establish CPU/native boundary.
- [x] Add tests in tests/test_candidate_selector.py: scheduling/config/defaults; real V1/V2 producer equality and target/world-ID alignment; real selector→actual relocation→Adam step; no-op/input/state validation; training helper event/off ordering using actual producer/consumer and a render boundary double only where native renderer unavailable. Extend existing conditional native integration for real render/ALR when available.
- [x] Run new tests and observe missing-feature failures (RED), not fixture/import mistakes.
- [x] Move original selection body into private shared core; add thin V1/V2 wrappers and same-row world targets. Add local no_grad update using index_copy_/index_fill_ with all validation before mutation.
- [x] Add two CLI fields, early config validation and pre-render event insertion; add one moved-count log per event. Preserve all old loss/topology/restore paths.
- [x] Run unified test entry (GREEN), syntax compatibility/static diff checks; original protected producers/evaluator stay byte-identical.
- [x] Commit one feat for V2, record exact tests/skips and deviations.

## Task2: Publish reviewed version and 30K handoff
- [x] One fresh independent whole-branch review against spec/plan/donor-adaptation. Fix material findings locally with RED→GREEN; user authorized routine decisions, report any rulings.
- [x] Update existing handoff/docs and provide bash-heredoc AutoDL launch command with fresh private view/output, existing C0 extension and native7K/App, budget16384; no server training execution assumed.
- [x] Syntax-check shell handoff, final scoped tests if fixes changed code, git diff --check and clean status; confirm baseline/V1 branches unchanged.
- [x] Push new V2 branch and verify remote SHA, record fixed production commit separately from docs commit. No PR/merge or baseline training.

## Execution ledger
- 2026-10-27: User approved implementation and autonomous routine fixes. Created V2 branch at ef333bb, clean; C worktree remains detached3775135.
- Ruling: Execute approved plan immediately without another review prompt — explicit latest user authorization while away; no additional algorithm scope.
- Task1 evidence: original24 tests19CPU pass/5native skips. New5 CPU test methods first fail on absent relocation method/defaults. After implementation, real render_normal exposed a missing calibration method in the analytic camera; load actual Camera.get_calib_matrix_nerf into fixture, no production workaround. Unified30 tests24CPU pass/6native skips (2.450s); no GPU validation claimed.
- Task1 static: Python3.10 syntax all five files; normalized original selector core AST identical except optional target return; all existing GaussianModel methods and ALR AST identical. Renderer/CUDA/metadata/evaluator/extractor untouched.
- Task1 complete: commits ef333bb..a114ed5; unified30run/24CPU pass/6native skipped, original selector and baseline methods retained. Whole-branch independent review dispatched under requesting-code-review; no implementation delegation.
- Task2 review: fresh reviewer reran30 tests24pass/6native skips, no Critical/Important/Minor findings; no fix pass needed. Bash outer/inner/progress syntax and actual CLI parsing passed. V1 branch staysef333bb, baseline branchd6f15c8.
- Final: Ruling: efficacy/time not judged by local tests — await same-protocol30K, cost if wrong is an ineffective trial.
- Final: Ruling: keep approved DA3/ADC/full-active adaptation — this is not complete donor reproduction, cost if wrong is negative reconstruction gain.
- Final: Ruling: leave inherited ALR/Ray-Color/topology behaviors unchanged — preserve baseline contract, cost is retaining their limitations.
- Final: Ruling: native GPU not judged with missing extension — no native success claim; observe authorized30K, cost is possible runtime failure/relaunch.
- Final: Ruling: launch doc outside code review checked by main agent with bash-n/real CLI — actual server state awaits execution, cost is possible path/network startup failure.
- Task2 publication complete: new origin/codex/second-run-selector-v2 pushed; remote HEAD0afa978 matched local on verification. Fixed production commit a114ed5 is reachable; later publication receipt is docs-only. No PR/merge/server operation.
