# D0/G1 Component Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a diagnostic-only, iteration-7000 offline command that explains the failed G1 need/reliability signals without changing training, the frozen G1 gate, or any published formal artifact.

**Architecture:** A pure NumPy reporting module validates row-aligned snapshot, raw collector component, and exact GT-distance arrays and emits fixed quantile/risk tables. A separate CLI restores the existing checkpoint in a fresh process, recomputes current no-grad collector inputs over all training cameras, performs one exact full-mesh query, and atomically publishes a small JSON/CSV directory. Historical geometry stability is explicitly marked unavailable because the checkpoint contains post-refresh history and cannot reconstruct the pre-refresh comparison used at iteration 7000.

**Tech Stack:** Python 3.10, NumPy 1.26, PyTorch 2.7/CUDA 12.8, existing Open3D exact triangle query, standard-library `unittest`.

**Spec:** `docs/research/ambisur-reliability-routing-design.md` sections 4.2–4.3 and G1; approved 2026-09-18 failure-analysis boundary in this task.

## Global Constraints

- Training remains GT-free; this command runs only after training and never writes into the run, source tree, checkpoint, snapshot, or GT mesh.
- The formal G1 definition, 5 cm label, evaluation domain, AUROC thresholds, and 108-item formal package remain unchanged.
- The command evaluates iteration 7000 only and cannot emit or revise `g1_pass`.
- All finite Gaussian centers and all valid GT triangles remain in the exact point-to-triangle query; no opacity, scaling, visibility, frustum, AABB, or crop filter is permitted.
- Recomputed collector evidence runs under `torch.no_grad()` and must not update model parameters, gradients, optimizer state, topology, or checkpoint state.
- Historical stability at iteration 7000 is not reconstructable from the post-refresh checkpoint and must be reported as unavailable rather than recomputed against identical current centers/normals.
- Publication is atomic and non-overwriting; immutable input fingerprints must match before and after production.

---

### Task 1: Pure component report contract

**Files:**
- Create: `tests/test_g1_component_diagnostics.py`
- Create: `reliability/g1_component_diagnostics.py`

**Interfaces:**
- Consumes: row-aligned `snapshot: Mapping[str, np.ndarray]`, `components: Mapping[str, np.ndarray]`, and `distances: np.ndarray`.
- Produces: `build_component_report(snapshot, components, distances, *, iteration, metadata) -> dict` and `risk_bin_rows(report) -> tuple[dict, ...]`.

- [ ] **Step 1: Write the failing validation and schema tests**

```python
def test_component_report_is_diagnostic_only_and_uses_fixed_rows(self):
    report = build_component_report(
        snapshot=valid_snapshot(20),
        components=valid_components(20),
        distances=np.linspace(0.0, 0.10, 20),
        iteration=7000,
        metadata={"checkpoint_sha256": "a" * 64},
    )
    self.assertEqual(report["schema_version"], 1)
    self.assertTrue(report["diagnostic_only"])
    self.assertIsNone(report["g1_decision"])
    self.assertFalse(report["historical_geometry_stability_reconstructable"])
    self.assertEqual(report["point_count"], 20)

def test_component_report_rejects_shape_nonfinite_and_wrong_iteration(self):
    with self.assertRaisesRegex(ValueError, "iteration 7000"):
        build_component_report(..., iteration=3000, ...)
    with self.assertRaisesRegex(ValueError, "row count"):
        build_component_report(..., components={"view_count": np.ones(19)}, ...)
    with self.assertRaisesRegex(ValueError, "finite"):
        build_component_report(..., distances=np.array([np.nan] * 20), ...)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `python -B -m unittest tests.test_g1_component_diagnostics -v`

Expected: `ModuleNotFoundError: reliability.g1_component_diagnostics`.

- [ ] **Step 3: Implement fixed summaries without a decision path**

Implement these exact constants and fields:

```python
QUANTILES = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)
RISK_BINS = tuple(range(10))
SNAPSHOT_COMPONENTS = ("A", "S", "N", "T_p", "T_g", "K", "r_p", "r_g")
RAW_COMPONENTS = (
    "view_count", "s_count", "s_angle", "s_raw",
    "prior_confidence", "prior_multiview", "prior_support_views",
    "geometry_multiview", "geometry_depth_normal", "geometry_support_views",
    "pg_raw_k",
)
```

For every field, report count, min/max/mean, frozen quantiles, Pearson correlation with distance, tie-safe Spearman correlation, and ten equal-count risk bins containing count, score bounds, mean distance, and strict `distance > 0.05` rate. Report `S` fractions at `>=0.50/.75/.90/.95/.98/.99` and exact maximum fraction. Never produce a PASS/FAIL or alternate G1 gate.

- [ ] **Step 4: Run focused GREEN and static checks**

Run:

```bash
python -B -m unittest tests.test_g1_component_diagnostics -v
python -B -m py_compile reliability/g1_component_diagnostics.py tests/test_g1_component_diagnostics.py
git diff --check
```

Expected: all tests pass; no warnings or whitespace errors.

- [ ] **Step 5: Commit Task 1**

```bash
git add reliability/g1_component_diagnostics.py tests/test_g1_component_diagnostics.py
git commit -m "feat: add G1 component diagnostic summaries"
```

---

### Task 2: No-grad 7000 collector and atomic CLI

**Files:**
- Create: `tests/test_g1_component_diagnostic_cli.py`
- Create: `scripts/diagnostics/diagnose_d0_g1_components.py`
- Modify: `reliability/g1_component_diagnostics.py`

**Interfaces:**
- Consumes: `--run-dir`, `--source-root`, `--gt-mesh`, `--output-root`, `--confirmation-id`, `--expected-commit`, `--expected-dataset-sha`, and `--expected-gt-sha`.
- Produces atomically: `<output-root>/<confirmation-id>/report.json`, `risk_bins.csv`, `inputs.json`, and `manifest.json`.
- Produces exit code `0` only for a complete diagnostic publication; malformed input or mutation returns `2`. Metric values never change the exit code.

- [ ] **Step 1: Write failing collection and publication tests**

```python
def test_collect_components_exposes_s_terms_and_never_claims_stability(self):
    result = collect_component_arrays(fake_refresh_inputs())
    self.assertEqual(
        set(result),
        set(RAW_COMPONENTS),
    )
    self.assertNotIn("geometry_stability", result)

def test_cli_publishes_only_small_diagnostic_artifacts_atomically(self):
    exit_code, publication = run_diagnostic(args, dependencies=fakes)
    self.assertEqual(exit_code, 0)
    self.assertEqual(
        {path.name for path in publication["output_dir"].iterdir()},
        {"report.json", "risk_bins.csv", "inputs.json", "manifest.json"},
    )
    self.assertIsNone(json.loads((publication["output_dir"] / "report.json").read_text())["g1_decision"])

def test_cli_removes_staging_on_failure_or_input_mutation(self):
    with self.assertRaisesRegex(ValueError, "immutable input changed"):
        run_diagnostic(args, dependencies=mutating_fakes)
    self.assertFalse(target.exists())
    self.assertEqual(list(output_root.glob(f".{confirmation_id}.tmp-*")), [])
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `python -B -m unittest tests.test_g1_component_diagnostic_cli -v`

Expected: import failure for the missing diagnostic CLI/collector boundary.

- [ ] **Step 3: Implement the minimal orchestration**

The command must:

1. validate clean exact commit, canonical dataset SHA, GT SHA, completed run, iteration-7000 checkpoint/snapshot, and absent output path;
2. fingerprint checkpoint, snapshot, `resolved_config.json`, source tree, and GT mesh;
3. call existing `load_g1_iteration`, `load_valid_mesh`, and `closest_triangle_distances` exactly once for iteration 7000;
4. restore a fresh render runtime and call `D0EvidenceCollector` under `torch.no_grad()` without loading an optimizer step path;
5. compute `view_count/s_count/s_angle/s_raw`, prior and geometry current raw components, and current `pg_raw_k`; do not compute or label historical stability;
6. select the same finite row indices as the formal G1 join;
7. build the pure report, write the four fixed artifacts into a staging directory, verify exact manifest size/SHA, re-fingerprint inputs, and rename staging atomically;
8. free CUDA objects and exit without writing the run or source directory.

- [ ] **Step 4: Run CLI GREEN and regression tests**

Run:

```bash
python -B -m unittest tests.test_g1_component_diagnostics tests.test_g1_component_diagnostic_cli tests.test_g1_orchestration tests.test_g1_geometry tests.test_g1_offline_inputs -v
python -B -m py_compile reliability/g1_component_diagnostics.py scripts/diagnostics/diagnose_d0_g1_components.py tests/test_g1_component_diagnostic_cli.py
git diff --check
```

Expected: all available local tests pass; dependency-dependent tests skip explicitly rather than fail.

- [ ] **Step 5: Commit Task 2**

```bash
git add reliability/g1_component_diagnostics.py scripts/diagnostics/diagnose_d0_g1_components.py tests/test_g1_component_diagnostic_cli.py
git commit -m "feat: add lightweight D0 G1 component diagnostic"
```

---

### Task 3: Server qualification and one frozen diagnostic run

**Files:**
- Modify: `task_plan.md`
- Modify: `findings.md`
- Modify: `progress.md`

**Interfaces:**
- Consumes exact formal-v2 run `d0_formal_v2_r2_seed0_7k_20260918_v1` and the already approved Tool Room GT mesh.
- Produces one new diagnostic-only confirmation directory outside all run/data roots.

- [ ] **Step 1: Run focused server tests before real data**

Run the Task 1/2 test command plus existing CUDA collector, offline-input, geometry, and renderer boundary suites on the exact pushed commit. Require clean Git, no active `train.py`, Python 3.10.21, Torch 2.7.1+cu128, CUDA available, and unchanged dataset/GT hashes.

- [ ] **Step 2: Run one iteration-7000 diagnostic**

Use a new confirmation id under:

```text
/root/autodl-tmp/ambisur_diagnostics/Tool_Room/d0-g1-components/
```

The command may perform one exact full-mesh query and one 406-camera no-grad collector pass. It must not render publication figures, evaluate iteration 3000, create a tar archive, start training, or modify the formal G1 package.

- [ ] **Step 3: Audit the diagnostic output**

Verify four files only, manifest hashes, input fingerprints before/after, row count `1424279`, rejected centers `0`, `diagnostic_only=true`, `g1_decision=null`, `historical_geometry_stability_reconstructable=false`, and no active training/Git changes.

- [ ] **Step 4: Record evidence and choose exactly one follow-up specification**

Record:

- whether `S_count` or `S_angle` causes saturation;
- which current geometry component drives the wrong-direction `T_g/r_g` association;
- whether current support validity is the source of low `T_g` coverage;
- the fact that historical stability requires forward instrumentation in a future smoke if it remains unresolved.

Do not change formulas in this task. Present a separate approved specification for the single observation-sufficiency revision; retain geometry reliability as a separately tracked pre-C2 blocker.

- [ ] **Step 5: Commit the evidence record**

```bash
git add task_plan.md findings.md progress.md
git commit -m "docs: record D0 G1 component diagnosis"
```

---

## Plan Self-Review

- Spec coverage: the plan preserves the frozen G1 domain/gate, GT isolation, 7000-only decision, and exact-distance contract.
- Scope: this plan adds diagnostics only; it contains no revised `S`, `N`, reliability, arbitration, lifecycle, or training behavior.
- Type/interface consistency: Task 2 consumes Task 1 `RAW_COMPONENTS`, `build_component_report`, and `risk_bin_rows` exactly as named.
- Placeholder scan: no TBD/TODO or unspecified implementation step remains.
- Reviewer risk: post-refresh checkpoint history cannot recreate iteration-7000 stability; the plan makes that limitation a required report field and follow-up boundary.
