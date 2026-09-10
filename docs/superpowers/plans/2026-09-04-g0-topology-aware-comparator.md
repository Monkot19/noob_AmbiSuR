# G0 Topology-Aware Comparator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Do not dispatch subagents: the user requires single-agent local work and user-operated AutoDL commands. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Amend the read-only G0 triplet auditor so dynamic Gaussian topology is evaluated with fixed permutation-invariant summaries while provenance, schema, trailing shape, optimizer structure and safety remain exact.

**Architecture:** Keep the pure factor-2 envelope functions and legacy two-run comparator unchanged. Add one deterministic CPU-float64 Gaussian summary boundary to `audit_feature_off_triplet.py`, then make `build_report` emit count and summary scalar gates instead of exact leading dimensions or row-wise distances for Gaussian-indexed tensors. Fixed-shape app tensors continue through the existing direct RMSE/MAE path.

**Tech Stack:** Python 3.10, PyTorch 2.7.1 CPU tensor operations for the read-only audit, standard-library `unittest`, JSON, Git; AutoDL RTX 4090 runtime only for tests and later artifact consumption.

**Spec:** `docs/superpowers/specs/2026-09-04-g0-topology-aware-equivalence-design.md`; authoritative method contract: `docs/research/ambisur-reliability-routing-design.md` §13.

## Global Constraints

- Modify only `scripts/diagnostics/audit_feature_off_triplet.py`, its tests and planning/evidence documents. Do not modify training, renderer/CUDA, losses, gradients, topology, datasets or existing run artifacts.
- Preserve `compare_runs`, `tensor_pair_stats`, `evaluate_numeric_field`, `evaluate_scalar_triplet` and `evaluate_triplet_report` behavior for existing callers.
- Add explicit `--topology-aware`; default `build_report(..., topology_aware=False)` and schema-1 behavior remain unchanged. Only explicit topology-aware mode emits schema version 2.
- Keep factor `2.0` fixed and without a confirmation CLI override.
- Keep exact: data/prior hashes, effective config, normalized command contract, checkpoint fields/dtype, Gaussian trailing shapes, optimizer structure/hyperparameters/state keys/steps, fixed unupdated fields, required artifacts and log safety.
- Treat checkpoint Gaussian count and pre-topology log/PLY count as separate scalar fields. Never compare values from different save stages.
- Never pad, truncate, row-sort, nearest-neighbor match or align Gaussian tensors. Sorting one scalar channel solely to compute deterministic statistics is allowed and creates no row correspondence.
- Each Gaussian summary component is its own scalar gate: no averaging across channels, statistics or fields.
- Use CPU float64, canonical ascending scalar-value order, population standard deviation and quantiles `0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99` with the exact linear interpolation specified in the design.
- Existing B1/B2 artifacts stay read-only and retain their original exact-contract FAIL. E0 remains stopped until local and AutoDL GREEN evidence is reviewed.
- Do not create/move a tag or push/commit without the applicable user-approved execution step.

---

### Task 1: Lock the Gaussian summary contract with RED tests *(complete)*

**Files:**
- Modify: `tests/gpu/test_feature_off_triplet_audit.py`

**Interfaces:**
- Consumes after Task 2: `summarize_gaussian_tensor(tensor) -> dict` and `gaussian_summary_metrics(name, tensors_by_role) -> tuple[list[dict], dict]`.
- `summarize_gaussian_tensor` returns `dtype`, `trailing_shape`, `leading_count`, `values`, and `diagnostics`. `values` maps stable names such as `channel_000.mean`, `channel_001.q50`, and `row_l2.q99` to Python floats.
- `gaussian_summary_metrics` returns one scalar-metric record `{"name": ..., "b1": ..., "b2": ..., "e0": ...}` per summary component plus role-keyed raw diagnostics.

- [x] **Step 1: Add RED tests for exact statistics and naming**

Add a test using:

```python
tensor = torch.tensor(
    [[4.0, 0.0], [1.0, 3.0], [3.0, 1.0], [2.0, 2.0]],
    dtype=torch.float32,
)
summary = summarize_gaussian_tensor(tensor)
self.assertEqual(summary["dtype"], "torch.float32")
self.assertEqual(summary["trailing_shape"], [2])
self.assertEqual(summary["leading_count"], 4)
self.assertEqual(summary["values"]["channel_000.mean"], 2.5)
self.assertAlmostEqual(summary["values"]["channel_000.std"], math.sqrt(1.25))
self.assertEqual(summary["values"]["channel_000.q50"], 2.5)
self.assertEqual(summary["values"]["channel_000.q01"], 1.03)
self.assertIn("row_l2.q99", summary["values"])
```

- [x] **Step 2: Add RED tests for permutation invariance and validation**

Assert exact dictionary equality after row permutation. Assert `ValueError` with stable message fragments for a scalar tensor, leading dimension zero, NaN, and Inf:

```python
self.assertEqual(
    summarize_gaussian_tensor(tensor),
    summarize_gaussian_tensor(tensor[torch.tensor([2, 0, 3, 1])]),
)
for invalid, message in (
    (torch.tensor(1.0), "Gaussian dimension"),
    (torch.empty(0, 2), "empty"),
    (torch.tensor([[float("nan")]]), "finite"),
    (torch.tensor([[float("inf")]]), "finite"),
):
    with self.subTest(message=message):
        with self.assertRaisesRegex(ValueError, message):
            summarize_gaussian_tensor(invalid)
```

- [x] **Step 3: Add RED tests for independent scalar records**

Use three tensors with different leading dimensions and assert that `gaussian_summary_metrics("capture.xyz", tensors)` succeeds, creates separately named scalar entries, and retains counts only in diagnostics. Change only E0 channel 1 and assert the resulting metric list exposes the changed `capture.xyz.channel_001.*` entries rather than one averaged field.

- [x] **Step 4: Add RED tests for trailing-shape and dtype evidence**

Assert summaries preserve `[3]` versus `[4]` trailing shapes and `torch.float32` versus `torch.float64` dtypes for later exact gates; the helper must not cast this metadata away even though reductions use float64.

- [x] **Step 5: Commit and push the test-only RED checkpoint after user-approved execution**

```powershell
git add tests/gpu/test_feature_off_triplet_audit.py
git commit -m "test: define topology-aware G0 summaries"
git push origin research/core-routing
```

- [x] **Step 6: Observe the expected RED on AutoDL**

Run on a clean server checkout of the exact test commit:

```bash
/root/miniconda3/envs/ambisur/bin/python -B -m unittest \
  tests.gpu.test_feature_off_triplet_audit -v
```

Expected: only the newly added tests fail because `summarize_gaussian_tensor` and `gaussian_summary_metrics` do not exist; existing tests remain collectable. Return the full output before Task 2.

Observed on clean `research/core-routing@59ae1c9bd86f214e53fe46401c5cf5c927ab021a`: existing 6 tests passed and the new 6 tests errored only at importing the two absent helpers. The server worktree remained clean. This is the accepted TDD RED evidence for Task 2.

---

### Task 2: Implement deterministic permutation-invariant summaries *(complete)*

**Files:**
- Modify: `scripts/diagnostics/audit_feature_off_triplet.py`
- Test: `tests/gpu/test_feature_off_triplet_audit.py`

**Interfaces:**
- Produce constant `GAUSSIAN_QUANTILES`.
- Produce `summarize_gaussian_tensor(tensor) -> dict`.
- Produce `gaussian_summary_metrics(name, tensors_by_role) -> tuple[list[dict], dict]`.
- Keep summaries on CPU and return only JSON-safe Python values.

- [x] **Step 1: Add constants and canonical scalar summarization**

Implement the fixed contract without `torch.quantile` ambiguity:

```python
GAUSSIAN_QUANTILES = (
    ("q01", 0.01), ("q05", 0.05), ("q25", 0.25),
    ("q50", 0.50), ("q75", 0.75), ("q95", 0.95),
    ("q99", 0.99),
)

def _sorted_scalar_summary(values):
    values = values.detach().reshape(-1).cpu().to(torch.float64)
    if values.numel() == 0:
        raise ValueError("Gaussian summary cannot consume an empty tensor")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Gaussian summary requires finite values")
    ordered = torch.sort(values).values
    result = {
        "mean": float(ordered.mean().item()),
        "std": float(ordered.std(unbiased=False).item()),
    }
    last = ordered.numel() - 1
    for label, quantile in GAUSSIAN_QUANTILES:
        position = last * quantile
        lower = math.floor(position)
        upper = math.ceil(position)
        value = ((upper - position) * ordered[lower]
                 + (position - lower) * ordered[upper])
        result[label] = float(value.item())
    return result
```

- [x] **Step 2: Implement Tensor flattening, channel and row-norm summaries**

Require `tensor.ndim >= 1` and `shape[0] > 0`. Preserve original dtype/trailing shape; flatten to `[N, -1]` only after validation. Call `_sorted_scalar_summary` for every column and for `torch.linalg.vector_norm(flat64, dim=1)`. Use zero-padded `channel_{index:03d}` keys and `row_l2`.

- [x] **Step 3: Implement role-wise scalar records and diagnostics**

Require roles exactly `b1/b2/e0`. Preserve per-role dtype/trailing shape for the exact layer. When summary component keys match, emit one scalar record per sorted component name; when they do not, return no scalar records plus `summary_keys_equal=false` in diagnostics so `build_report` records a normal exact trailing-shape failure rather than attempting broadcast/alignment. Diagnostics must include per role: leading count, full shape, dtype, element count, min, max and a content SHA256 computed from contiguous CPU bytes in bounded chunks. Do not put count into the per-field summary values.

- [x] **Step 4: Run local collection/compile checks**

Local (Torch tests may skip; record the exact result):

```powershell
python -B -m unittest tests.test_compare_feature_off -v
python -B -m py_compile scripts/diagnostics/audit_feature_off_triplet.py tests/gpu/test_feature_off_triplet_audit.py
git diff --check
```

Expected locally: comparator tests pass, Torch-dependent tests either pass when Torch is present or report explicit skips, compilation/diff return 0.

Observed locally: all 12 Torch-dependent tests were explicitly skipped because the local Python has no Torch; the existing comparator suite passed 14/14, both files compiled, and `git diff --check` returned zero with line-ending warnings only. This is collection/static evidence, not Task 2 GREEN.

- [x] **Step 5: Commit and push the helper implementation candidate**

```powershell
git add scripts/diagnostics/audit_feature_off_triplet.py tests/gpu/test_feature_off_triplet_audit.py
git commit -m "feat: add topology-invariant Gaussian summaries"
git push origin research/core-routing
```

Candidate `a1643abc31e0dd423a363b9a5ca12d21ce90918a` was committed with only `scripts/diagnostics/audit_feature_off_triplet.py` and pushed to `origin/research/core-routing`. Planning/design changes remain uncommitted and were not mixed into the implementation commit.

- [x] **Step 6: Run focused AutoDL GREEN**

On the exact pushed implementation commit:

```bash
/root/miniconda3/envs/ambisur/bin/python -B -m unittest \
  tests.gpu.test_feature_off_triplet_audit -v
```

Expected: all focused GPU-module CPU-Torch tests pass; no dataset or training process is needed.

Observed on clean `a1643abc31e0dd423a363b9a5ca12d21ce90918a`: Python 3.10.21, Torch 2.7.1+cu128, CUDA available; all 12 focused tests passed in 0.042 s, return code was zero, and the server worktree remained clean.

---

### Task 3: Integrate topology-aware evidence into the versioned report *(complete)*

**Files:**
- Modify: `scripts/diagnostics/audit_feature_off_triplet.py` (`build_report`)
- Modify: `tests/gpu/test_feature_off_triplet_audit.py`
- Modify: `tests/test_compare_feature_off.py`

**Interfaces:**
- `build_report(...)` adds keyword-only `topology_aware=False`. The default returns the unchanged schema-1 report; `topology_aware=True` returns schema version 2.
- `exact_invariants` contains `capture.<name>.trailing_shape`, not full shape, and no exact `gaussian_count` or exact cross-run `run.final_points`.
- `scalar_metrics` contains `checkpoint.gaussian_count`, `run.final_points`, all Gaussian summary components and existing evaluation metrics.
- `numeric_fields` contains only fixed-shape application tensors; Gaussian capture and Adam moments are excluded from row-wise comparison.
- `diagnostics.gaussian_tensors` stores raw per-role Tensor diagnostics.

- [x] **Step 1: Add report-assembly RED tests with small synthetic tensors**

Extract a pure helper:

```python
build_topology_aware_tensor_evidence(
    captures, optimizer_tensors, app_tensors, snapshots, learned_ply
) -> dict
```

Test B1/B2/E0 counts `2/3/4` and require:

```python
self.assertNotIn("gaussian_count", exact_names)
self.assertNotIn("run.final_points", exact_names)
self.assertIn("capture.xyz.trailing_shape", exact_names)
self.assertIn("checkpoint.gaussian_count", scalar_names)
self.assertIn("run.final_points", scalar_names)
self.assertIn("capture.xyz.channel_000.mean", scalar_names)
self.assertEqual(fixed_numeric_names, ["appear_ab"])
```

Also require a per-role exact invariant `run.final_points_matches_ply` to be true only when each log count equals its own same-stage PLY vertex count.

- [x] **Step 2: Add RED tests for optimizer classification**

Assert `optimizer.xyz.exp_avg` and `exp_avg_sq` enter summary scalar metrics, `optimizer.xyz.step` does not, and an unknown tensor-state suffix raises `ValueError("unsupported optimizer tensor state")`. Optimizer structure and extracted step remain exact through the existing `normalize_optimizer` result.

- [x] **Step 3: Add RED tests preventing cross-statistic averaging**

Construct a report where all summary scalar components pass except one `q99` component above `2*d_B`. Assert `evaluate_triplet_report(report)["equivalent"]` is false and its scalar failure names that exact component.

- [x] **Step 4: Commit/push the integration tests and observe RED on AutoDL**

```powershell
git add tests/gpu/test_feature_off_triplet_audit.py tests/test_compare_feature_off.py
git commit -m "test: define topology-aware report assembly"
git push origin research/core-routing
```

Then run the exact test commit on AutoDL. Expected: new assembly/mode tests fail because `build_topology_aware_tensor_evidence` and `topology_aware` wiring are absent; all previously GREEN tests remain collectable.

Test-only commit `8c54a539df4de93e4b8dfe664ab150aa0a246966` was pushed with exactly `tests/gpu/test_feature_off_triplet_audit.py` and `tests/test_compare_feature_off.py`. Local non-Torch comparator tests passed 15/15 and both test files compiled; AutoDL RED observation is pending.

Observed on clean AutoDL `8c54a539df4de93e4b8dfe664ab150aa0a246966`: 31 tests ran; 27 existing/independent tests passed and exactly four new integration tests errored for the pre-registered missing assembly helper, `build_report(..., topology_aware=...)` keyword, or CLI flag. Return code was 1 and the server worktree remained clean.

- [x] **Step 5: Implement explicit report schema 2 integration**

In `build_report`:

1. replace exact checkpoint Gaussian count with scalar `checkpoint.gaussian_count`;
2. replace exact `run.final_points` with a scalar metric;
3. replace full Gaussian shapes with exact trailing shapes;
4. add per-role `run.final_points_matches_ply` exact evidence;
5. send all 13 capture tensors and only `exp_avg/exp_avg_sq` optimizer moments through `gaussian_summary_metrics`;
6. keep application tensors in `_tensor_field` direct RMSE/MAE;
7. retain optimizer structure, tensor keys, dtype, step and app full shape exact;
8. write raw Gaussian diagnostics under `diagnostics.gaussian_tensors`;
9. set `schema_version` to `2` only for `topology_aware=True`;
10. add CLI `--topology-aware`, pass it to `build_report`, and leave the omitted/default schema-1 path byte-for-byte semantically unchanged.

- [x] **Step 6: Run focused and complete local checks**

```powershell
python -B -m unittest tests.test_compare_feature_off -v
python -B -m unittest discover -s tests -p 'test_*.py' -v
python -B -m py_compile scripts/diagnostics/audit_feature_off_triplet.py tests/gpu/test_feature_off_triplet_audit.py tests/test_compare_feature_off.py
git diff --check
git status --short --untracked-files=all
```

Expected: all non-Torch local tests pass, Torch-dependent tests are reported as skips rather than falsely claimed as local GREEN, compilation/diff return 0, and only planned files are modified.

Observed locally: the dependency-free 32-test subset passed 32/32 and comparator tests passed 15/15; the three planned files compiled and `git diff --check` returned zero. Full discovery additionally found the already documented local-environment import gaps (`torch` in `test_feature_off_dispatch`, `numpy` in `test_seed_contract`) and reported the 16 Torch auditor tests as skips. Those environment gaps are not counted as GREEN; the exact AutoDL Python 3.10/Torch environment remains the full-suite gate.

- [x] **Step 7: Commit and push the integrated comparator candidate**

```powershell
git add scripts/diagnostics/audit_feature_off_triplet.py tests/gpu/test_feature_off_triplet_audit.py tests/test_compare_feature_off.py
git commit -m "fix: make G0 comparison topology-aware"
git push origin research/core-routing
```

Candidate `e781fef23f4f2adec5382808108e7e7e3331e11a` was committed and pushed with exactly `scripts/diagnostics/audit_feature_off_triplet.py`; training and method source remain unchanged.

- [x] **Step 8: Run the complete AutoDL component gate**

On the clean exact implementation commit:

```bash
/root/miniconda3/envs/ambisur/bin/python -B -m unittest \
  tests.gpu.test_feature_off_triplet_audit -v
/root/miniconda3/envs/ambisur/bin/python -B -m unittest discover \
  -s tests -p 'test_*.py' -v
/root/miniconda3/envs/ambisur/bin/python -B -m py_compile \
  scripts/diagnostics/audit_feature_off_triplet.py
git status --short --untracked-files=all
```

Expected: focused and full suites pass, compilation returns 0, and server status remains clean. Stop on any failure; do not launch E0.

Observed on clean AutoDL `e781fef23f4f2adec5382808108e7e7e3331e11a`: Python 3.10.21, NumPy 1.26.3 and Torch 2.7.1+cu128/CUDA 12.8 loaded successfully; the focused schema-2 suite passed 16/16 in 0.139 s and full repository discovery passed 54/54 in 0.449 s. `py_compile` and CLI help returned zero, `--topology-aware` was present, the post-test worktree was clean, and `training_started=NO`.

---

### Task 4: Freeze evidence and hand back to the existing 8k protocol *(in progress)*

**Files:**
- Modify: `docs/superpowers/plans/2026-09-04-g0-triplet-equivalence.md`
- Modify: `task_plan.md`
- Modify: `findings.md`
- Modify: `progress.md`

**Interfaces:**
- Record exact test commit, commands, counts, pass/fail totals and server clean status.
- Preserve B1/B2 paths and their original strict-count FAIL classification.
- Authorize no training by documentation alone; the next user-operated command remains the unchanged E0 role of the existing 8k protocol.

- [x] **Step 1: Run an immutable B1/B2 summary-only dry audit**

Use the existing CLI with B2 supplied only as the temporary third input, explicit `--topology-aware --exploratory`, and an output outside both run directories:

```bash
B1=/root/autodl-tmp/ambisur_runs/Tool_Room/g0-triplet-8k/g0_8k_r2_seed0_20260904_v1/b1_d6f15c88
B2=/root/autodl-tmp/ambisur_runs/Tool_Room/g0-triplet-8k/g0_8k_r2_seed0_20260904_v1/b2_d6f15c88
OUT=/root/autodl-tmp/ambisur_diagnostics/g0_8k_topology_aware_dry.json

test -x /usr/bin/time || { echo 'STOP: /usr/bin/time unavailable'; exit 2; }
sha256sum "$B1/chkpnt8000.pth" "$B1/point_cloud/iteration_8000/point_cloud.ply" \
  "$B2/chkpnt8000.pth" "$B2/point_cloud/iteration_8000/point_cloud.ply" > /tmp/g0_before.sha256
/usr/bin/time -v /root/miniconda3/envs/ambisur/bin/python -B \
  scripts/diagnostics/audit_feature_off_triplet.py "$B1" "$B2" "$B2" \
  --iteration 8000 --evaluation-iterations 500 1000 5001 7001 8000 \
  --topology-aware --exploratory --output "$OUT"
sha256sum -c /tmp/g0_before.sha256
```

The report must say `schema_version=2`, `exploratory=true`, `g0_equivalent=false`. Record `/usr/bin/time -v` wall time and maximum resident set size. This validates performance/schema only and cannot substitute E0.

AutoDL did not provide the external `/usr/bin/time` executable, so the executed command used Python 3.10 standard-library `time.perf_counter()` and Linux `resource.getrusage(RUSAGE_SELF).ru_maxrss` in the same audit process. The immutable-input and report gates were unchanged. Core audit and corrected post-validation both returned zero: schema 2, 91 exact invariants, 1 fixed numeric field, 1,938 scalar metrics, 25 Gaussian fields, 1,926 summary metrics, zero exact/numeric failures, `numerical_equivalent=true`, `exploratory=true`, and `g0_equivalent=false`. Wall time was 91.350 s, peak RSS 4,056,576 KiB (3961.5 MiB), report size 954,155 bytes, all six checkpoint/app/PLY after-hashes passed, and the server repository remained clean. A first outer wrapper return of 1 was traced only to corrupted validation-label text; the existing report was post-validated without rerunning the audit.

- [x] **Step 2: Review summary cardinality and output size without tuning results**

Require every planned capture/moment field and all nine statistics per channel plus row norm. Reject missing or duplicate scalar names. Do not remove summaries based on runtime or values; if resource usage is impractical, stop and return to the approved design rather than silently sampling.

Observed cardinality matched the frozen contract exactly: 13 capture fields plus 12 Adam moment fields, with 1,926 independently named summary metrics and no missing/duplicate summary names. No threshold, factor, statistic or field was changed after observing the values.

- [x] **Step 3: Update evidence documents and commit**

```powershell
git add docs/research/ambisur-reliability-routing-design.md \
  docs/superpowers/specs/2026-09-04-g0-topology-aware-equivalence-design.md \
  docs/superpowers/plans/2026-09-04-g0-topology-aware-comparator.md \
  docs/superpowers/plans/2026-09-04-g0-triplet-equivalence.md \
  task_plan.md findings.md progress.md
git commit -m "docs: freeze topology-aware G0 contract"
git push origin research/core-routing
```

- [ ] **Step 4: Apply the E0 launch gate** *(pending server sync to the documentation commit)*

Proceed to the existing E0 launcher only when all of the following are true: local planned suites pass, AutoDL focused/full suites pass, dry audit completes with bounded resources and no missing summary, server is clean at the exact pushed commit, B1/B2 artifacts/hashes are unchanged, and the user has reviewed the results. Otherwise keep `E0_ACTION=STOP`.

---

## Execution order and stopping conditions

1. Test-only RED commit and AutoDL observation.
2. Summary helper minimal GREEN.
3. Schema-2 report integration RED→GREEN and complete AutoDL verification.
4. Read-only B1/B2 dry audit for resource/schema validation.
5. Documentation evidence commit, then return to the already approved unchanged E0 8k launch protocol.

Any unexpected failure, non-finite statistic, missing field, duplicate metric name, unsupported optimizer tensor state, excessive audit memory/time, dirty worktree, artifact hash change or test regression stops execution. Use `superpowers:systematic-debugging`; do not weaken the contract or launch E0.
