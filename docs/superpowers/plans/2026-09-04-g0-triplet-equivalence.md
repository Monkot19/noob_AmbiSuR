# G0 Three-Run Feature-Off Equivalence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Do not dispatch subagents: the user requires single-agent local work and user-operated AutoDL commands. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved read-only three-run G0 comparator, replay it on the existing 500-iteration artifacts, then prepare a frozen Tool Room 8k baseline/baseline-repeat/E0 confirmation protocol.

**Architecture:** Keep the existing two-run strict comparator unchanged. Add pure standard-library envelope evaluation to the same diagnostics module, and add one AutoDL-only checkpoint extractor that lazily imports Torch, normalizes the three run artifacts into versioned JSON, and passes that report to the pure gate. Training, renderer/CUDA, gradients, optimizer behavior and topology code remain untouched.

**Tech Stack:** Python 3.10 standard library for gate/unit tests; PyTorch 2.7.1+cu128 only for AutoDL checkpoint loading; `unittest`; Git; one RTX 4090.

**Spec:** `docs/research/ambisur-reliability-routing-design.md` §13, “非确定 baseline 的数值等价合同（2026-09-04 已批准方案 A）”

## Global Constraints

- Modify only diagnostic tests/scripts and planning records; do not modify `train.py`, `arguments`, `reliability` method code, renderer/CUDA, loss, backward, optimizer or topology.
- Preserve `compare_runs(baseline, candidate, rtol=1e-5, atol=1e-7)` and its current CLI behavior.
- Use `factor=2.0`; do not expose a CLI override that could silently relax the approved gate.
- Exact invariants remain exact. A missing artifact, non-finite value, dtype/shape/Gaussian-count mismatch, optimizer-structure mismatch or zero-self non-exact field fails the gate.
- Numeric fields require both RMSE and MAE independently. Learned-output SHA, mismatch count and max-absolute difference are diagnostic only.
- Existing 500 runs are exploratory replay. Only a separately approved fresh 8k triplet can confirm G0.
- AutoDL remains user-operated: provide one auditable command at a time and analyze the returned full output before the next command.
- Do not create/move tags. Do not use GT or extract meshes for E0.

---

### Task 1: Pure Three-Run Gate

**Files:**
- Modify: `tests/test_compare_feature_off.py`
- Modify: `scripts/diagnostics/compare_feature_off.py`

**Interfaces:**
- Consumes: normalized pair statistics with keys `b1_b2`, `b1_e0`, `b2_e0`; each pair contains `exact`, `shape_equal`, `rmse`, `mean_abs`, `max_abs`, `mismatch_count`, and `element_count`.
- Produces: `evaluate_numeric_field(field_report, factor=2.0) -> dict`, `evaluate_scalar_triplet(name, b1, b2, e0, factor=2.0) -> dict`, and `evaluate_triplet_report(report, factor=2.0) -> dict`.
- `evaluate_triplet_report` returns `equivalent`, `factor`, `exact_failures`, `numeric_failures`, `numeric_results`, and `diagnostics`; its output metric keys are `rmse` and `mae` even though the normalized input pair statistic is named `mean_abs`. It never mutates its input.

- [x] **Step 1: Add focused failing tests**

Add tests that construct tiny normalized dictionaries and assert:

```python
self.assertTrue(evaluate_numeric_field(_field(self_rmse=1.0, e_rmse=2.0))["passed"])
self.assertFalse(evaluate_numeric_field(_field(self_rmse=1.0, e_rmse=2.000001))["passed"])
self.assertFalse(evaluate_numeric_field(_zero_self_but_not_exact())["passed"])
self.assertFalse(evaluate_numeric_field(_rmse_pass_mae_fail())["passed"])
self.assertEqual(evaluate_numeric_field(_b2_is_nearest())["metrics"]["rmse"]["nearest"], "b2")
```

Also test that missing/non-finite/shape-mismatched data fail, an exact invariant mismatch fails, learned PLY SHA differences remain in diagnostics without failing an otherwise valid report, and the four existing `compare_runs` tests still pass.

- [x] **Step 2: Observe RED locally**

Run:

```powershell
python -B -m unittest tests.test_compare_feature_off -v
```

Expected: import failure for the new three-run function names while the existing comparator tests remain collectable.

- [x] **Step 3: Implement the minimal pure gate**

For each of `rmse` and `mean_abs`, compute:

```python
self_distance = field_report["b1_b2"][metric]
candidate_pairs = {"b1": field_report["b1_e0"], "b2": field_report["b2_e0"]}
nearest = min(candidate_pairs, key=lambda key: candidate_pairs[key][metric])
candidate_distance = candidate_pairs[nearest][metric]
```

Reject absent values, booleans masquerading as numbers, negative distances and non-finite numbers. If `self_distance == 0`, require baseline-self exact and the selected candidate pair exact with distance zero; otherwise require `candidate_distance <= 2.0 * self_distance`. Require both metrics. Keep `max_abs`, mismatch count and hashes only in the returned diagnostics.

- [x] **Step 4: Observe GREEN and run the complete local suite**

Run:

```powershell
python -B -m unittest tests.test_compare_feature_off -v
python -B -m unittest discover -s tests -p 'test_*.py' -v
python -B -m py_compile scripts/diagnostics/compare_feature_off.py tests/test_compare_feature_off.py
git diff --check
```

Expected: all local non-GPU tests pass; compilation and diff checks return 0. Local discovery may continue to exclude `tests/gpu` when Torch is unavailable; record the exact count rather than claiming the AutoDL suite ran locally.

- [x] **Step 5: Commit Task 1**

```powershell
git add tests/test_compare_feature_off.py scripts/diagnostics/compare_feature_off.py
git commit -m "test: define G0 empirical envelope contract"
```

---

### Task 2: AutoDL Checkpoint Triplet Extractor

**Files:**
- Create: `scripts/diagnostics/audit_feature_off_triplet.py`
- Create: `tests/gpu/test_feature_off_triplet_audit.py`
- Modify: `tests/test_compare_feature_off.py`

**Interfaces:**
- CLI: `python -B scripts/diagnostics/audit_feature_off_triplet.py BASELINE_1 BASELINE_2 E0 --iteration 8000 --output REPORT.json`; `--exploratory` is allowed only for the historical 500 replay and forces final `g0_equivalent=false` even when every available numerical gate passes.
- Produces versioned JSON with top-level keys `schema_version=1`, `iteration`, `runs`, `exact_invariants`, `numeric_fields`, `scalar_metrics`, `diagnostics`, and `gate`.
- `load_checkpoint(run_dir, iteration)` requires `chkpnt<iteration>.pth` and validates the legacy outer tuple `(capture, iteration)` plus the 16-field `GaussianModel.capture()` tuple from `scene/gaussian_model.py::capture`.
- Capture names by index: `active_sh_degree`, `xyz`, `knn_f`, `features_dc`, `features_rest`, `scaling`, `rotation`, `opacity`, `max_radii2D`, `max_weight`, `xyz_gradient_accum`, `xyz_gradient_accum_abs`, `denom`, `denom_abs`, `optimizer`, `spatial_lr_scale`.

- [x] **Step 1: Write the AutoDL-oriented failing tests without creating fake CUDA state**

Use `unittest.mock` and small CPU Torch tensors to verify `tensor_pair_stats`, capture-length rejection, outer checkpoint-schema rejection, dtype/shape/count exact gates, optimizer group-name/hyperparameter/state-key/step extraction, and report exit code (`0` pass, `1` gate failure, `2` malformed/missing evidence). Use `@unittest.skipUnless(torch.cuda.is_available(), ...)` only for a one-test checkpoint load smoke; pure CPU Torch tests must not be skipped on AutoDL.

- [ ] **Step 2: Observe RED on AutoDL before implementation**

Run in the clean server checkout of the test commit:

```bash
/root/miniconda3/envs/ambisur/bin/python -B -m unittest tests.gpu.test_feature_off_triplet_audit -v
```

Expected: import failure for `scripts.diagnostics.audit_feature_off_triplet`; return code nonzero. Return the complete output before implementation proceeds.

- [ ] **Step 3: Implement extraction and normalization**

Load checkpoints onto CPU with `torch.load(..., map_location="cpu", weights_only=False)`. Accumulate pair distances in float64 and chunks so the 8k SH/optimizer tensors do not create three full float64 copies. Record direct-order tensor RMSE/MAE, exact, finite, shape, dtype, element count, mismatch count and max-absolute difference for all capture tensors and optimizer moments.

Exact gates:

- outer/capture schema, iteration, dtype/shape and Gaussian count;
- `active_sh_degree`, `spatial_lr_scale`;
- optimizer group order/names, non-`params` hyperparameters, state keys and step counters;
- `resolved_config.json` common model/optimization/pipeline values, allowing only the predeclared E0 `seed`/false Core metadata additions;
- per-role `g0_run_contract.json` written by the external launcher, containing role, exact expected commit, `dirty=false`, command, runtime, canonical dataset manifest SHA and private aligned-prior SHA. This is launcher provenance, not a training-code output;
- E0 `run_identity.json` must agree with its launcher contract. Baseline commits do not produce that E0-only file and must not be failed for its absence;
- exit code 0, no error/nonfinite log tokens, and required artifact presence.

Numeric gates: capture parameter/proxy tensors, optimizer `exp_avg`/`exp_avg_sq`, `app_model/iteration_<iteration>/app.pth::appear_ab`, and every predeclared evaluation scalar present in all three logs. If an 8k field is exact, it naturally enters the zero-self exact branch; do not maintain a 500-specific inactive-field allowlist.

- [ ] **Step 4: Run targeted and full AutoDL tests**

```bash
/root/miniconda3/envs/ambisur/bin/python -B -m unittest tests.gpu.test_feature_off_triplet_audit -v
/root/miniconda3/envs/ambisur/bin/python -B -m unittest discover -s tests -p 'test_*.py' -v
/root/miniconda3/envs/ambisur/bin/python -B -m py_compile scripts/diagnostics/audit_feature_off_triplet.py
git status --short --untracked-files=all
```

Expected: all tests pass and the worktree contains only the committed implementation under review.

- [ ] **Step 5: Commit Task 2**

```powershell
git add scripts/diagnostics/audit_feature_off_triplet.py tests/gpu/test_feature_off_triplet_audit.py tests/test_compare_feature_off.py
git commit -m "feat: add read-only three-run equivalence comparator"
```

---

### Task 3: Existing 500-Run Exploratory Replay

**Files:**
- Modify: `findings.md`
- Modify: `progress.md`

**Interfaces:**
- Inputs are read-only existing runs:
  - B1: `/root/autodl-tmp/ambisur_runs/Tool_Room/e0-paired-500/pair_20260903T090116Z/baseline_d6f15c88`
  - B2: `/root/autodl-tmp/ambisur_runs/Tool_Room/e0-baseline-self-repeat/baseline2_d6f15c88_20260903T094958Z`
  - E0: `/root/autodl-tmp/ambisur_runs/Tool_Room/e0-paired-500/pair_20260903T090116Z/e0_a2608215`
- Output: a new diagnostics JSON outside all three run directories; no artifact is edited.

- [ ] **Step 1: Server preflight and component suite**

Verify server branch/HEAD, clean status, Python 3.10.21/Torch 2.7.1, the three run directories, and all `chkpnt500.pth`, app weights, logs and safety files. Run the full suite before consuming results.

- [ ] **Step 2: Run the versioned comparator once**

Use `--iteration 500 --exploratory` and write to `/root/autodl-tmp/ambisur_diagnostics/e0-g0-a-500-replay-<commit8>.json`. Expected exploratory result is 64/64 numeric checks under the already approved rule while final `g0_equivalent` remains false; a different output is an implementation/debugging signal, not permission to alter the artifacts or gate. Historical launcher safety evidence may be cross-referenced, but newly introduced `g0_run_contract.json` files must not be fabricated inside old run directories.

- [ ] **Step 3: Reconcile against the original audit**

Require the same 32 recorded tensor/state fields, maximum ratio approximately 1.99 for `optimizer.opacity.exp_avg_sq` RMSE, and L1/PSNR nearest/self ratios approximately 0.293/0.174. Preserve the historical strict comparator FAIL and label this replay exploratory.

- [ ] **Step 4: Record and commit evidence**

```powershell
git add findings.md progress.md task_plan.md
git commit -m "docs: verify G0 comparator on 500-run evidence"
```

Stop if the replay differs; use `superpowers:systematic-debugging` before changing code.

---

### Task 4: Freeze and Run the Independent 8k Confirmation Triplet

**Files:**
- Modify before launch: `task_plan.md`
- Modify after audit: `findings.md`
- Modify after audit: `progress.md`
- Copy per run: `docs/research/experiment-manifest-template.md` into each new output directory and append the frozen G0 evidence list.

**Interfaces:**
- Dataset snapshot: Tool Room canonical manifest SHA `aad92aa2e0f0d072756b3a56c686d5c1d35f448811ce60ca4360c67dbc3ef255`.
- Baseline role SHA: `d6f15c8891a53800d5e3100f95817a7dd7f98e2f` for B1 and B2.
- E0 role SHA: `a26082154889ed539322425347af5a57a859a52f` for E.
- Fixed training protocol: Tool Room, `-r 2`, semantic seed 0, 8000 iterations, evaluation iterations `500 1000 5001 7001 8000`, save/checkpoint only at 8000, serial order B1 → B2 → E, unique private working view/output for every role.

- [ ] **Step 1: Obtain explicit experiment approval**

Present the final safety-gated shell command and estimated disk/time budget. Do not launch based only on approval of this implementation plan.

- [ ] **Step 2: Run one preflight command**

Require at least 35 GiB free, clean repository, exact local commit objects/tags, Python/Torch/CUDA/GPU match, 406 image/depth/conf basename match, canonical manifest match, and no pre-existing target directories. Do not fetch after the preflight begins; network failure must not contaminate the run decision.

- [ ] **Step 3: Launch B1, audit completion, then B2, then E**

Each role uses a private copy of `sparse_da3_aligned` and read-only links/copies for the other canonical inputs. Before launch, the wrapper writes a new `g0_run_contract.json` in that new output directory with role, exact commit, clean flag, command, runtime and input hashes. Record start/end UTC, exit code, peak GPU memory, environment, source hashes and full log. Never overlap roles on the GPU. Stop the sequence immediately on a failed role.

- [ ] **Step 4: Run the triplet comparator**

Require strict invariants plus every RMSE/MAE/scalar gate. Also require logs to reach 8000 and source-code boundary evidence for densify first eligible 600, trim 1000, Ray-Color 5001 and ALR 7001. Keep learned PLY/checkpoint hashes diagnostic only; keep canonical/prior hashes exact.

- [ ] **Step 5: Decide G0 without post-hoc edits**

- PASS only if every strict, numeric and safety gate passes.
- FAIL on count/shape/invariant/numeric/resource/error failure; preserve all outputs and stop before D0/C1.
- Do not create an E0 tag. Do not alter `c0-baseline`.
- Commit the result record as `docs: record G0 8k confirmation` and wait for explicit D0 authorization.

---

## Execution Order and Stop Points

1. Task 1 local RED → GREEN.
2. Task 2 AutoDL RED → local implementation → AutoDL GREEN.
3. Task 3 existing-artifact replay; debug any discrepancy before proceeding.
4. Request explicit 8k authorization.
5. Task 4 B1 → audit → B2 → audit → E → audit → triplet gate.

The user has already ruled out multi-agent execution. Use inline execution with review checkpoints after Tasks 1, 2 and 3.
