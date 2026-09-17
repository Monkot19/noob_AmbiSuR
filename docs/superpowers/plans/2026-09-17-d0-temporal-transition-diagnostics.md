# D0 Temporal Transition Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add exact topology-aware temporal stable-state diagnostics to D0, generate the frozen formal G1 timeline artifacts, and qualify a new immutable Tool Room `d0_formal_r2_seed0_7k_20260917_v2` run without changing training behavior.

**Architecture:** A focused no-gradient transition tracker owns lineage-aligned age and transition-count tensors and emits JSON-safe per-refresh summaries. `EvidenceAccumulator` updates and migrates that tracker beside the existing arbitration state, while the snapshot `.npz` inventory stays unchanged and `events.jsonl` becomes the sole persisted summary stream. The offline evaluator validates the complete seven-refresh timeline, creates the already-frozen formal artifact inventory, evaluates iterations 3000 and 7000, and atomically publishes only after input fingerprints remain unchanged.

**Tech Stack:** Python 3.10, PyTorch tensors under `torch.no_grad()`, NumPy `.npz`, JSON Lines, Matplotlib, Open3D, `unittest`, CUDA renderer, Git, AutoDL RTX 4090.

**Spec:** `docs/superpowers/specs/2026-09-17-d0-temporal-transition-diagnostics-design.md`

## Global Constraints

- `docs/research/ambisur-reliability-routing-design.md` remains the highest-priority method contract.
- Temporal transition means topology-aligned `stable(t-1) -> stable(t)` in the fixed order `Bypass, Consensus, Prior-led, Geometry-led, Abstain`.
- `TopologyChange.new_to_old` is the only lineage identity contract; no row-index, nearest-neighbour, geometric, or synthetic-ID matching is allowed.
- Clone/split children use `new_to_old=parent_index,is_new=True`; generic Evidence migration resets them, while temporal lineage migration inherits parent age/count. Unmapped new rows use `-1,True`; survivors use `old_index,False`; `-1,False` is invalid.
- The existing `iteration_*.npz` field inventory must remain byte-for-byte identical in names and meaning.
- Transition summaries belong only in `events.jsonl`; aligned age/count tensors belong only in versioned D0 runtime checkpoint state.
- All calculations run without gradients and must not change parameters, `.grad`, Adam state, optimizer-step count, densification proxy, clone/split/prune/trim decisions, lifecycle, or random-number consumption.
- Feature-off keeps the legacy training path and legacy tuple checkpoint schema.
- GT mesh is offline-only and must not enter training, D0 snapshots, events, caches, or checkpoints.
- Iteration 7000 is the sole formal G1 decision point; iteration 3000 is diagnostic-only.
- The completed v1 run is immutable engineering evidence and must not be overwritten, moved, deleted, or promoted.
- The official rerun path is `d0_formal_r2_seed0_7k_20260917_v2`; no stage tag is created until its D0 isolation and formal G1 gates pass.
- Every implementation task follows RED, minimal GREEN, focused regression, full relevant regression, and a separate commit.

## File Structure

- Create `reliability/transition_diagnostics.py`: pure tensor state, update rules, topology migration, checkpoint mapping, and JSON-safe summary construction.
- Modify `reliability/topology.py`: separate identity from reset semantics and provide explicit temporal-lineage migration without changing generic new-row reset behavior.
- Modify `scene/gaussian_model.py`: emit clone/split parent indices while preserving baseline topology actions and random-call order.
- Modify `reliability/evidence.py`: own the tracker, capture the pre-update stable state, update exactly once per refresh, migrate it, and version its checkpoint state.
- Modify `reliability/shadow.py`: expose the most recent transition summary without changing the snapshot return type.
- Modify `reliability/diagnostics.py`: append schema-2 event records while preserving the exact `.npz` inventory.
- Modify `train.py`: pass the D0-only transition summary to the writer; do not touch loss, backward, optimizer, or topology order.
- Create `reliability/g1_timeline.py`: validate seven event records and write the frozen state-proportion, transition, duration/jitter, and joint-coverage bundles.
- Modify `scripts/diagnostics/evaluate_d0_g1.py`: implement formal 3000/7000 orchestration and atomic 108-artifact publication.
- Extend focused tests under `tests/` and `tests/gpu/`; do not refactor unrelated modules.

---

### Task 1: Pure Temporal Transition Tracker

**Files:**
- Create: `reliability/transition_diagnostics.py`
- Create: `tests/test_transition_diagnostics.py`

**Interfaces:**
- Consumes: `ArbitrationState` integer values and `TopologyChange.new_to_old`.
- Produces: `TemporalTransitionDiagnostics(point_count: int, device=None)`, `update(previous_stable: torch.Tensor, current_stable: torch.Tensor) -> dict`, `on_topology_change(change: TopologyChange) -> None`, `state_dict() -> dict`, and `load_state_dict(state: dict) -> None`.
- Produces public tensors `stable_age_refreshes: torch.int64[P]` and `stable_transition_count: torch.int64[P]`.

- [ ] **Step 1: Write hand-checked RED tests for first refresh, unchanged rows, changed rows, and absent-state means**

```python
def test_first_refresh_uses_initialized_bypass_as_previous_state(self):
    tracker = TemporalTransitionDiagnostics(3, device="cpu")
    summary = tracker.update(
        torch.tensor([0, 0, 0], dtype=torch.int8),
        torch.tensor([0, 2, 4], dtype=torch.int8),
    )
    self.assertEqual(summary["transition_count_matrix"][0], [1, 0, 1, 0, 1])
    self.assertEqual(tracker.stable_age_refreshes.tolist(), [1, 1, 1])
    self.assertEqual(tracker.stable_transition_count.tolist(), [0, 1, 1])
    self.assertEqual(summary["jitter_count"], 2)
    self.assertAlmostEqual(summary["jitter_rate"], 2.0 / 3.0)
    self.assertIsNone(summary["mean_stable_age_refreshes_by_state"]["Consensus"])
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `python -B -m unittest tests.test_transition_diagnostics -v`

Expected: import failure for `reliability.transition_diagnostics`.

- [ ] **Step 3: Implement the minimal no-gradient tracker and strict validation**

```python
@torch.no_grad()
def update(self, previous_stable, current_stable):
    previous = self._validate_states(previous_stable, "previous_stable")
    current = self._validate_states(current_stable, "current_stable")
    changed = current != previous
    self.stable_age_refreshes = torch.where(
        changed,
        torch.ones_like(self.stable_age_refreshes),
        self.stable_age_refreshes + 1,
    )
    self.stable_transition_count.add_(changed.to(torch.int64))
    counts = torch.bincount(previous.to(torch.int64) * 5 + current.to(torch.int64), minlength=25).reshape(5, 5)
    fractions = counts.to(torch.float64) / float(self.point_count)
    return self._summary(current, counts, fractions, changed)
```

Reject empty populations, wrong shapes/dtypes/devices, values outside `[0, 4]`, non-integer matrices on restore, and non-finite derived fractions. Serialize absent per-state means as JSON `null` (`None` in Python), never NaN.

- [ ] **Step 4: Add RED/GREEN tests for topology migration and checkpoint round-trip**

```python
def test_topology_migration_inherits_mapped_parent_and_resets_unmapped_rows(self):
    tracker = TemporalTransitionDiagnostics(2, device="cpu")
    tracker.stable_age_refreshes.copy_(torch.tensor([3, 7]))
    tracker.stable_transition_count.copy_(torch.tensor([1, 4]))
    tracker.on_topology_change(TopologyChange(
        new_to_old=torch.tensor([1, 1, -1, 0]),
        is_new=torch.tensor([False, True, True, False]),
    ))
    self.assertEqual(tracker.stable_age_refreshes.tolist(), [7, 7, 0, 3])
    self.assertEqual(tracker.stable_transition_count.tolist(), [4, 4, 0, 1])
```

Run: `python -B -m unittest tests.test_transition_diagnostics tests.test_topology_migration -v`

Expected: all tests pass, including mapped clone inheritance and unmapped-row zero initialization. This test intentionally uses `new_to_old=1,is_new=True` for a clone child.

- [ ] **Step 5: Verify detachment and commit**

Run: `python -B -m unittest tests.test_transition_diagnostics -v`

Verify every returned tensor/state tensor has `requires_grad == False`, JSON serialization succeeds with `allow_nan=False`, and updating does not mutate either input.

Commit:

```bash
git add reliability/transition_diagnostics.py tests/test_transition_diagnostics.py
git commit -m "feat: add temporal D0 transition tracker"
```

---

### Task 2: Evidence, Topology, and Checkpoint Integration

**Files:**
- Modify: `reliability/topology.py` (`TopologyChange.__post_init__`, `migrate_tensor`, new `migrate_lineage_tensor`, append/compose helpers)
- Modify: `scene/gaussian_model.py` (`densification_postfix`, `densify_and_clone`, `densify_and_split`)
- Modify: `reliability/evidence.py` (`EvidenceAccumulator.__init__`, `refresh`, `on_topology_change`, `state_dict`, `load_state_dict`)
- Modify: `tests/test_evidence_accumulator_state.py`
- Modify: `tests/test_d0_shadow_runtime.py`
- Modify: `tests/gpu/test_evidence_accumulator_cuda.py`

**Interfaces:**
- Consumes: `TemporalTransitionDiagnostics` from Task 1.
- Produces: `EvidenceAccumulator.latest_transition_diagnostics: dict | None` and evidence checkpoint schema version `3` containing `temporal_transition_diagnostics`.
- Preserves: `EvidenceSnapshot` fields and `refresh(inputs) -> EvidenceSnapshot`.

- [ ] **Step 0: Write RED tests for approved identity/reset separation**

```python
def test_mapped_new_child_resets_generic_state_but_inherits_lineage_state(self):
    change = TopologyChange(
        new_to_old=torch.tensor([0, 0, -1]),
        is_new=torch.tensor([False, True, True]),
    )
    source = torch.tensor([7])
    self.assertEqual(migrate_tensor(source, change, fill_value=0).tolist(), [7, 0, 0])
    self.assertEqual(migrate_lineage_tensor(source, change, fill_value=0).tolist(), [7, 7, 0])
```

Add GaussianModel mapping tests proving clone parents use selected old indices, split parents use `selected_indices.repeat(N)` in the exact child row order, composition preserves `is_new=True`, and feature-off action/random order is unchanged.

Run: `python -B -m unittest tests.test_topology_migration tests.test_topology_composition tests.test_gaussian_topology_mapping -v`

Expected: RED because `TopologyChange` rejects mapped new children and `migrate_lineage_tensor` does not exist.

- [ ] **Step 1: Write RED tests that force exactly one update from pre-update stable state**

```python
def test_refresh_records_temporal_stable_transition_not_candidate_disagreement(self):
    accumulator = EvidenceAccumulator(2, cfg=self.cfg, device="cpu")
    first = accumulator.refresh(self.inputs())
    self.assertEqual(first.stable.tolist(), [0, 0])
    self.assertEqual(
        accumulator.latest_transition_diagnostics["transition_count_matrix"][0][0],
        2,
    )
    accumulator.refresh(self.inputs())
    third = accumulator.refresh(self.inputs())
    self.assertEqual(third.stable.tolist(), [2, 2])
    self.assertEqual(
        accumulator.latest_transition_diagnostics["transition_count_matrix"][0][2],
        2,
    )
```

Run: `python -B -m unittest tests.test_evidence_accumulator_state -v`

Expected: failure because `latest_transition_diagnostics` and checkpoint schema 3 do not exist.

- [ ] **Step 2: Integrate tracker after the sole arbitration update**

```python
previous_stable = self.arbitration.stable_state.clone()
candidate = candidate_state(snapshot, self.cfg)
stable = self.arbitration.update(candidate)
self.latest_transition_diagnostics = self.transition_diagnostics.update(
    previous_stable, stable
)
```

Instantiate the tracker beside `self.arbitration`. Do not add another `candidate_state` or `arbitration.update` call and do not change `EvidenceSnapshot`.

- [ ] **Step 3: Add RED/GREEN tests for topology lineage and resume equivalence**

The test uses `new_to_old=[1, 1, -1, 0]` so a mapped child inherits parent age/count, an unmapped new row starts at zero, and survivors retain history. A second accumulator restored from `state_dict()` must emit exactly the same next summary and tensors as the uninterrupted accumulator.

Run: `python -B -m unittest tests.test_evidence_accumulator_state tests.test_d0_shadow_runtime -v`

Expected: all topology and round-trip assertions pass with `EvidenceAccumulator.STATE_VERSION == 3`.

- [ ] **Step 4: Prove refresh isolation on CPU and CUDA**

Extend existing isolation tests to capture parameters, `.grad`, Adam state, densification proxy, RNG state, and topology mapping before and after refresh. Assert unchanged values and exactly one optimizer step remains owned by the training loop.

Run CPU: `python -B -m unittest tests.test_evidence_accumulator_state tests.test_d0_shadow_runtime -v`

Run server CUDA: `python -B -m unittest tests.gpu.test_evidence_accumulator_cuda -v`

Expected: all tests pass; transition tensors reside on the accumulator device and remain detached.

- [ ] **Step 5: Run the reliability regression and commit**

Run: `python -B -m unittest tests.test_reliability_evidence tests.test_arbitration tests.test_topology_migration tests.test_topology_composition tests.test_gaussian_topology_mapping tests.test_evidence_accumulator_state tests.test_d0_shadow_runtime -v`

Commit:

```bash
git add reliability/topology.py scene/gaussian_model.py reliability/evidence.py tests/test_topology_migration.py tests/test_topology_composition.py tests/test_gaussian_topology_mapping.py tests/test_evidence_accumulator_state.py tests/test_d0_shadow_runtime.py tests/gpu/test_evidence_accumulator_cuda.py
git commit -m "feat: persist topology-aware D0 transitions"
```

---

### Task 3: Schema-2 Events and Training Wiring

**Files:**
- Modify: `reliability/shadow.py`
- Modify: `reliability/diagnostics.py`
- Modify: `train.py`
- Modify: `tests/test_d0_diagnostics.py`
- Modify: `tests/test_d0_shadow_runtime.py`
- Modify: `tests/gpu/test_feature_off_dispatch.py`

**Interfaces:**
- Consumes: `EvidenceAccumulator.latest_transition_diagnostics`.
- Produces: `D0ShadowRuntime.latest_transition_diagnostics` and `write_snapshot(output_directory, iteration, snapshot, *, transition_diagnostics)`.
- Produces event schema version `2` with the exact approved summary keys; preserves the 15-name `.npz` inventory.

- [ ] **Step 1: Write RED tests for exact event schema and unchanged NPZ fields**

```python
expected_event_keys = {
    "schema_version", "iteration", "point_count", "joint_valid_count",
    "transition_count_matrix", "transition_fraction_matrix",
    "jitter_count", "jitter_rate",
    "mean_stable_age_refreshes", "mean_stable_age_refreshes_by_state",
    "mean_stable_transition_count", "mean_stable_transition_count_by_state",
}
self.assertEqual(set(events[0]), expected_event_keys)
self.assertEqual(events[0]["schema_version"], 2)
self.assertAlmostEqual(sum(map(sum, events[0]["transition_fraction_matrix"])), 1.0)
self.assertEqual(set(arrays.files), set(SNAPSHOT_FIELDS))
```

Also assert JSON contains neither `gt` nor `mesh`, duplicate iterations fail closed, wrong matrix shape/count/fraction sum fails before writing, and no partial event line remains after failure.

Run: `python -B -m unittest tests.test_d0_diagnostics tests.test_d0_shadow_runtime -v`

Expected: failures for the missing keyword and schema mismatch.

- [ ] **Step 2: Add runtime exposure and atomic event validation**

After a due refresh, copy the accumulator summary to `D0ShadowRuntime.latest_transition_diagnostics`; retain it across checkpoint round-trip. In `write_snapshot`, validate summary totals against `snapshot.stable.shape[0]`, serialize with `json.dumps(..., allow_nan=False)`, create the `.npz` exclusively, and append one newline-terminated event only after validation succeeds.

- [ ] **Step 3: Wire `train.py` without touching optimization order**

```python
snapshot = shadow_runtime.maybe_refresh(iteration, build_inputs)
if snapshot is not None:
    write_snapshot(
        scene.model_path,
        iteration,
        snapshot,
        transition_diagnostics=shadow_runtime.latest_transition_diagnostics,
    )
```

The call remains at the existing D0 location. No loss, `backward`, `optimizer.step`, densification, trim, or checkpoint ordering changes are permitted.

- [ ] **Step 4: Verify feature-off and legacy checkpoint contracts**

Run: `python -B -m unittest tests.gpu.test_feature_off_dispatch tests.test_core_runtime tests.test_d0_diagnostics tests.test_d0_shadow_runtime -v`

Expected: feature-off selects the legacy path, writes no D0 event, and uses the original tuple checkpoint schema; shadow mode writes schema-2 events.

- [ ] **Step 5: Run full CPU suite and commit**

Run: `python -B -m unittest discover -s tests -p 'test_*.py' -v`

Commit:

```bash
git add reliability/shadow.py reliability/diagnostics.py train.py tests/test_d0_diagnostics.py tests/test_d0_shadow_runtime.py tests/gpu/test_feature_off_dispatch.py
git commit -m "feat: record temporal D0 event diagnostics"
```

---

### Task 4: Formal Timeline Artifacts and 3000/7000 Orchestration

**Files:**
- Create: `reliability/g1_timeline.py`
- Modify: `scripts/diagnostics/evaluate_d0_g1.py`
- Modify: `tests/test_g1_visualization.py`
- Modify: `tests/test_g1_orchestration.py`
- Modify: `tests/test_g1_publication.py`
- Create: `tests/test_g1_timeline.py`

**Interfaces:**
- Consumes: seven schema-2 `events.jsonl` records at iterations `(1000, 2000, 3000, 4000, 5000, 6000, 7000)` and seven unchanged snapshots.
- Produces: `load_d0_timeline(run_directory: Path) -> D0Timeline`, `write_timeline_artifacts(timeline: D0Timeline, output_directory: Path) -> tuple[Path, ...]`.
- Extends: `run_evaluator(args)` formal branch to evaluate `(3000, 7000)` and publish the frozen `required_artifacts()` inventory.

- [ ] **Step 1: Write RED tests for strict timeline loading**

Tests must reject missing/duplicate/out-of-order iterations, schema other than 2, matrix totals unequal to `point_count`, fractions that are non-finite or do not sum to one, negative age/count values, and any event key containing `gt` or `mesh`. A valid fixture must retain matrices exactly and never compare snapshot row `i` across refreshes.

Run: `python -B -m unittest tests.test_g1_timeline -v`

Expected: import failure for `reliability.g1_timeline`.

- [ ] **Step 2: Implement the three frozen deterministic timeline bundles**

Create exactly the three bundles already reserved by `required_artifacts()`, each as `.png`, `.svg`, `.pdf`, `.csv`, and `.json`:

```text
timeline/state_proportion
timeline/state_transition
timeline/joint_coverage
```

Use the fixed state palette/order and refresh iteration on the x-axis. `state_transition.*` is a deterministic multi-panel/source bundle containing the 5x5 previous-row/current-column matrices plus event-provided duration, cumulative-transition, and jitter series; duration and jitter do not create additional artifact names. `state_proportion.*` uses current snapshots and `joint_coverage.*` uses current snapshot/event coverage. Never derive temporal transitions from adjacent `.npz` row positions. Assert that the complete formal inventory remains exactly 108 paths.

Run: `python -B -m unittest tests.test_g1_timeline tests.test_g1_visualization -v`

Expected: deterministic artifact names, CSV column contracts, JSON metadata, and finite plotted values pass.

- [ ] **Step 3: Write RED formal orchestration tests**

Mock only expensive mesh-distance/render calls. Assert the formal branch:

```python
self.assertEqual(evaluated_iterations, [3000, 7000])
self.assertEqual(report["decision_iteration"], 7000)
self.assertFalse(report["exploratory"])
self.assertEqual(len(manifest["files"]), 108)
```

Also assert missing timeline inputs, input mutation, GT hash mismatch, existing target directory/archive, or a failed 7000 evaluation prevents publication and cleans staging.

Run: `python -B -m unittest tests.test_g1_orchestration tests.test_g1_publication -v`

Expected: failure at the current formal `NotImplementedError`.

- [ ] **Step 4: Implement minimal formal evaluator branch**

Load and fingerprint checkpoints at 3000/7000, snapshots 1000–7000, schema-2 events, resolved config, source tree, and GT mesh. Evaluate full finite Gaussian centers against the complete valid GT mesh at 3000 and 7000; render the frozen fields/views for both iterations; create metric figures for 7000; create all timeline bundles; derive the formal pass/fail solely from the frozen 7000 G1 gate; re-fingerprint inputs; atomically publish directory, `.tar.gz`, and `.tar.gz.sha256`.

- [ ] **Step 5: Run evaluator regressions and commit**

Run: `python -B -m unittest tests.test_g1_geometry tests.test_g1_metrics tests.test_g1_visualization tests.test_g1_timeline tests.test_g1_orchestration tests.test_g1_publication tests.gpu.test_g1_override_render -v`

Commit:

```bash
git add reliability/g1_timeline.py scripts/diagnostics/evaluate_d0_g1.py tests/test_g1_timeline.py tests/test_g1_visualization.py tests/test_g1_orchestration.py tests/test_g1_publication.py
git commit -m "feat: publish formal D0 temporal evaluation"
```

---

### Task 5: Server Qualification, v2 Formal Run, and D0 Gate

**Files:**
- Modify after evidence is returned: `task_plan.md`
- Modify after evidence is returned: `findings.md`
- Modify after evidence is returned: `progress.md`
- No method-source edits are permitted during server qualification.

**Interfaces:**
- Consumes: exact pushed commit from Tasks 1–4, Tool Room dataset hash `aad92aa2e0f0d072756b3a56c686d5c1d35f448811ce60ca4360c67dbc3ef255`, prior hash `69a21ab8756f43834a5357f27ca6cf40c6b7b15695e0e8cade1914fe70956977`, and GT hash `31547a31069f736792d4b13fff76c73483f59239dbbabe1971e115f9ab17171d`.
- Produces: short multi-refresh smoke evidence, immutable v2 run, formal G1 report/archive, D0 isolation conclusion, and the go/no-go decision for C1.

- [ ] **Step 1: Run server component and CUDA gates at the exact commit**

```bash
python -B -m unittest discover -s tests -p 'test_*.py' -v
python -B -m unittest tests.gpu.test_evidence_accumulator_cuda tests.gpu.test_renderer_evidence_adapter tests.gpu.test_g1_override_render -v
```

Require a clean `research/core-routing` worktree, exact expected HEAD, no active `train.py`, Python 3.10, CUDA available, and no test failure.

- [ ] **Step 2: Run a new 500-iteration multi-refresh shadow smoke**

Use a new output path, seed 0, resolution 2, refresh interval 100, checkpoint at 500, and no GT path in the command. Require five schema-2 events, unchanged `.npz` field inventory, exact transition totals, finite fractions, topology-migrated state at changing Gaussian counts, checkpoint round-trip, one optimizer step per training iteration, zero errors/non-finite tokens/GT references, and unchanged dataset/prior hashes.

- [ ] **Step 3: Audit feature-off equivalence and D0 isolation**

Run the existing feature-off component suite and compare the shadow smoke's training invariants. The audit must explicitly confirm no parameter/gradient/Adam/densification/topology/lifecycle action consumed transition diagnostics and no additional optimizer step occurred.

- [ ] **Step 4: Launch immutable formal v2 training**

Use a new private data view and exact path:

```text
/root/autodl-tmp/ambisur_runs/Tool_Room/d0-formal-7k/d0_formal_r2_seed0_7k_20260917_v2
```

Frozen command parameters: resolution 2, seed 0, 7000 iterations, D0 shadow enabled, refresh interval 1000, evaluation at 1000–7000, checkpoints at 3000 and 7000, and no GT argument. Require exit code 0, seven snapshots/events, both checkpoints, clean logs, immutable input hashes, and restored clean branch state.

- [ ] **Step 5: Run formal G1 and publish the downloadable archive**

Evaluate the v2 run with the full GT mesh and the approved full finite-center domain. Require formal 108-artifact inventory, iteration-7000 gate decision, exact manifest, archive checksum verification using the checksum file's directory, and report the absolute server paths for the output directory, `.tar.gz`, and `.tar.gz.sha256`.

- [ ] **Step 6: Record evidence and decide C1 readiness**

Write only observed facts as facts and retain unvalidated interpretations as hypotheses. C1 may start only if the short smoke, D0 isolation audit, formal timeline validation, and G1 gate all pass. If coverage or G1 fails, report the frozen failure without changing thresholds, evaluation domain, or joint-validity contract.

Commit:

```bash
git add task_plan.md findings.md progress.md
git commit -m "docs: record formal D0 temporal gate"
```

Do not create a tag in this task. Tag creation requires a separate explicit approval after the recorded D0/G1 evidence is reviewed.

---

## Final Verification Checklist

- [ ] `git status --short` is empty before every server run and before each push.
- [ ] The `.npz` field set is unchanged and event schema is exactly version 2.
- [ ] Transition matrices are 5x5, non-negative, total to current point count, and fractions sum to one.
- [ ] First-refresh, hysteresis, clone/split inheritance, pruning, resume, and absent-state semantics are covered by tests.
- [ ] Feature-off retains the legacy path/checkpoint and emits no transition diagnostics.
- [ ] D0 introduces no gradient, optimizer, densification, topology, lifecycle, RNG, or GT influence.
- [ ] Formal evaluator uses iterations 3000/7000 and only iteration 7000 decides G1.
- [ ] v1 remains untouched; v2 uses a new immutable directory.
- [ ] Formal output contains the frozen 108 artifacts and a verified archive checksum.
- [ ] `task_plan.md`, `findings.md`, and `progress.md` cite exact commits, paths, hashes, and observed results.
