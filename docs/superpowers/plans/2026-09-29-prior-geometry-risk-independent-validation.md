# Prior/Geometry Risk Independent Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Diagnose whether historical `T_g/r_g` violates the already approved reliability semantics and, only if a formula-derived implementation defect is confirmed and repaired, build the version-5 Utility Room multi-seed assets and compact nested `M0 -> M1 -> M2` read-only validation path.

**Architecture:** The plan is a sequential, fail-closed pipeline. Real collector/reprojection plus synthetic/state conformance tests and a GT-free Tool Room audit make the first decision; a green existing implementation produces `NO_SEMANTIC_REPAIR_JUSTIFIED` and stops, while an exact semantic RED permits only the minimal root-cause repair and Evidence version 5. Utility source/DA3 admission freezes one snapshot, one canonical confirmation is written before any run/probe target exists, three GT-free seed runs reload it, and the final nested evaluator reloads the same SHA before any GT query.

**Tech Stack:** Python 3.10, PyTorch 2.7.1+cu128, NumPy 1.26.3, standard-library `unittest`, existing COLMAP loaders, Pillow, Open3D full-mesh distance query, JSON/CSV/SHA256, Git, DA3 preprocessing, AutoDL RTX 4090.

**Spec:** `docs/superpowers/specs/2026-09-29-prior-geometry-risk-independent-validation-design.md`

## Global Constraints

- Task 1 is a diagnosis, not a presumption of a bug. Production `T_g/r_g` code and `EvidenceAccumulator.STATE_VERSION` remain unchanged until an approved-formula test fails for the expected semantic reason.
- If the current implementation satisfies every formula, validity, temporal, topology, and state test, freeze `NO_SEMANTIC_REPAIR_JUSTIFIED` and stop Tasks 4–13. Do not change formulas, signs, thresholds, or the test to manufacture RED.
- A confirmed repair may restore only the approved `E_g,mv`, `E_g,dn`, `E_g,stab`, geometric-mean `T_g`, and valid-domain `r_g` semantics. No new component, learned mapping, GT-selected sign, or retuned constant is allowed.
- Evidence version 5 is conditional on an actual stored-semantic repair. Version 4 remains the immutable identity of old assets; no legacy checkpoint can be resumed or relabelled as version 5.
- Utility GT is inaccessible to diagnosis, repair, source audit, DA3, and all three D0 runs. Before the final probe confirmation, only GT existence, byte size, and SHA256 may be recorded; no distance, label, overlay, or metric may be produced.
- Utility upload admission requires exactly 147 images and 147 registered COLMAP image records, exact case-sensitive basename equality, only `PINHOLE`/`SIMPLE_PINHOLE` cameras, finite poses/points, matching image dimensions, and no `split.json`.
- DA3 requires a hash-pinned preprocessing confirmation before launch. It runs once into a new target, produces one immutable derived snapshot, and seeds 0/1/2 reuse that exact snapshot. Regeneration creates a different snapshot ID.
- Immediately after that DA3 snapshot is frozen—and while every planned seed run/view/state path and probe target is still absent—write one canonical dual-risk confirmation. Seed launches and the final evaluator must reload that same confirmation and detached SHA; it may not be regenerated after seeing run or GT results.
- Utility runs are seeds 0/1/2, resolution 2, 7,000 iterations, refreshes 1,000–7,000, checkpoints 3,000/7,000, GT-free, sequential, and admitted only with Evidence version 5.
- The only formal models are `M0=[A,1-S]`, `M1=M0+[1-r_p]`, and `M2=M1+[1-r_g]`, evaluated on identical finite `V_p & V_g` rows with full-mesh `distance>0.05 m` labels.
- Preserve the exact five-slab OOF, float64 IRLS, fold-local preprocessing, column-residual threshold `>1e-8`, fixed-origin 0.5 m paired voxel bootstrap, 2,000 replicates, and `SeedSequence([20260928, iteration, training_seed])` contract.
- This plan does not authorize DA3, Utility training, Utility GT evaluation, C1, a new `N`, or unrelated dataset downloads. Every server/data stage in Task 13 requires its own explicit user authorization.
- Follow strict RED → verify RED → minimal GREEN → focused/full regression → commit. A wrong-kind failure is fixed before production code is touched.

## File Structure

- Create `tests/test_geometry_reliability_semantics.py`: approved-formula combination, refresh-time, topology, and no-grad conformance tests against current production code.
- Modify `tests/test_reprojection_reliability.py`: direct `reprojection_validity_and_errors` monotonicity and validity coverage for the real `E_g,mv` input boundary.
- Modify `tests/test_d0_collector.py`: end-to-end `D0EvidenceCollector` coverage for the real `E_g,mv`/`E_g,dn` score, validity, support, and transport boundary.
- Create `reliability/geometry_reliability_audit.py`: pure report schema and invariant checks for synthetic cases plus existing D0 state; no GT or predictive metric.
- Create `scripts/diagnostics/audit_geometry_reliability_semantics.py`: read-only Tool Room admission, state inspection, compact atomic JSON publication, and explicit semantic outcome.
- Create `tests/test_geometry_reliability_audit.py`: audit schema, parser, no-GT boundary, mutation, and stop-outcome tests.
- Conditionally modify only the production file(s) named by the exact semantic RED: `reliability/collector.py` for collector assembly/validity/support, `reliability/evidence.py` for reprojection/combination/state semantics, and the existing renderer transport path only if a focused transport RED locates the defect there. A confirmed stored-semantic repair also bumps `EvidenceAccumulator.STATE_VERSION` from 4 to 5 without changing snapshot fields.
- Conditionally modify `tests/test_reprojection_reliability.py`, `tests/test_d0_collector.py`, `tests/test_geometry_reliability_semantics.py`, `tests/test_evidence_accumulator_state.py`, `tests/test_d0_shadow_runtime.py`, `tests/test_topology_migration.py`, and `tests/gpu/test_evidence_accumulator_cuda.py`: root-cause boundary, version-5, chronology, topology, resume, and real-CUDA repair coverage.
- Create `reliability/utility_da3.py`: Utility source audit, preprocessing confirmation, derived-snapshot audit, deterministic manifests, and detached SHA validation.
- Create `scripts/diagnostics/prepare_utility_da3_confirmation.py`: read-only 147-way source audit and atomic preprocessing confirmation; it has no GT argument and never invokes DA3.
- Create `scripts/diagnostics/finalize_utility_da3_snapshot.py`: post-DA3 derived-array/model/transformation audit and immutable snapshot record; it has no GT argument.
- Create `tests/test_utility_da3.py` and `tests/test_utility_da3_cli.py`: source, camera, basename, finite, hash, no-overwrite, regeneration, and no-GT tests.
- Create `reliability/g1_dual_risk_confirmation.py`: pre-run version-5 three-seed/probe confirmation schema, detached SHA, absent-target preregistration checks, and later completed-run evaluator admission; old `g1_confirmation.py` stays version-4 compatible.
- Create `tests/test_g1_dual_risk_confirmation.py`: exact constants, three-seed commands, absent targets, version/hash/mutation, and chronology tests.
- Create `reliability/g1_dual_risk.py`: common-domain construction, nested OOF models, direction checks, seed-level metrics, macro paired bootstrap, report validation, and exhaustive outcomes.
- Create `tests/test_g1_dual_risk.py`: hand-checked statistical and decision tests.
- Create `scripts/diagnostics/probe_g1_dual_risk.py`: strict three-run admission, full-mesh queries, optional fixed alignment audit, compact atomic publication, CLI, and exit codes.
- Create `tests/test_g1_dual_risk_cli.py`: parser, version-5/run/hash binding, same-domain orchestration, no-render/no-training, mutation, and publication tests.
- Update `task_plan.md`, `findings.md`, and `progress.md` only at explicit result/freezing boundaries; never rewrite a prior failed or stopped outcome.

## Review Focus

- False-positive repair: `test_green_semantics_forces_no_repair_outcome` must prove that an already conforming current implementation cannot trigger a version bump or proceed to Utility work (Tasks 1–3).
- Collector/reprojection inversion: `test_collector_geometry_scores_decrease_with_error_and_preserve_valid_support_semantics` must pass controlled errors through the real `D0EvidenceCollector`/reprojection boundary and prove that larger valid errors cannot raise `E_g,mv` or `E_g,dn` (Tasks 1 and 4).
- History/topology contamination: `test_mapped_new_child_resets_geometry_history_while_survivor_keeps_previous_refresh` must distinguish evidence reset from temporal diagnostic lineage inheritance (Tasks 1 and 4).
- Utility preprocessing admission: `test_source_audit_rejects_fisheye_case_mismatch_and_unregistered_image` must reject renamed `OPENCV_FISHEYE`, case-only mismatches, extras, and missing records, while `test_preprocessing_cli_has_no_gt_surface_and_detects_source_change` proves confirmation/finalization cannot read GT or bless changed/regenerated assets (Tasks 6–7).
- Pseudoreplication or macro-CI overclaim: `test_macro_bootstrap_averages_fixed_seed_paired_voxel_gains_without_pooling_rows_or_resampling_seeds` must pin the run-level/scene-level interpretation and exact bootstrap construction (Tasks 9–10).

---

### Task 1: Freeze the Geometry Semantic Diagnosis Contract

**Files:**
- Create: `tests/test_geometry_reliability_semantics.py`
- Modify: `tests/test_reprojection_reliability.py`
- Modify: `tests/test_d0_collector.py`
- Read only: `reliability/collector.py`
- Read only: `reliability/evidence.py`
- Read only: `reliability/topology.py`

**Interfaces:**
- Consumes: `reprojection_validity_and_errors`, `reproject_depth_normal_maps`, `D0EvidenceCollector`, `combine_geometry_reliability`, `compute_geometry_stability`, `EvidenceAccumulator.refresh`, `EvidenceAccumulator.on_topology_change`, `TopologyChange`, and existing renderer/`EvidenceRefreshInputs` fixtures.
- Produces: an approved-formula conformance suite whose result is either a specific semantic RED or `NO_SEMANTIC_REPAIR_JUSTIFIED`; it produces no implementation fix.

- [ ] **Step 1: Add real reprojection-boundary monotonicity and validity tests**

In `tests/test_reprojection_reliability.py`, add controlled valid depth/normal pairs at increasing error and invalid pairs covering non-finite, non-positive, out-of-frame, foreground-occluded, and zero-normal inputs. Assert that valid depth/normal errors are non-decreasing with the physical mismatch, invalid rows contribute zero error with `valid=False`, and no invalid row can be reinterpreted as high reliability or support.

- [ ] **Step 2: Add real collector-boundary score and support tests**

In `tests/test_d0_collector.py`, add `test_collector_geometry_scores_decrease_with_error_and_preserve_valid_support_semantics`. Drive controlled two-pass renders through the real `D0EvidenceCollector` and its `reproject_depth_normal_maps` call. Assert that increasing a still-valid multi-view depth/normal mismatch cannot increase `geometry_multiview` (`E_g,mv`); increasing the primitive-versus-depth-normal angular mismatch cannot increase `geometry_depth_normal` (`E_g,dn`); `geometry_support_views` counts only source views with positive transported valid reprojection count; and the `E_g,dn` denominator includes only finite, nonzero normals with `alpha>=0.5`. Keep score, validity, and support assertions separate so a zero-filled invalid value cannot pass as reliable evidence.

- [ ] **Step 3: Add direct component and stability monotonicity tests**

Add `test_geometry_components_are_reliability_monotone` and `test_stability_decreases_only_with_movement_or_rotation`. Use mathematically hand-checked values to assert that increasing any valid reliability component cannot decrease `T_g`, and increasing normalized movement or rotation cannot increase `E_g,stab`. Check exact values only for exactly representable anchors and dtype-appropriate tolerance otherwise.

- [ ] **Step 4: Add refresh chronology and validity tests**

Add `test_stability_uses_the_previous_valid_refresh_before_overwrite`, `test_first_history_is_unknown_then_becomes_valid`, and `test_invalid_geometry_observation_preserves_ema_but_makes_r_g_unusable`. Assert `V_g=False` means unknown, `r_g=0` only as a gated output, invalid rows do not update `t_g_ema`, and current centers/normals are copied only after stability is computed.

- [ ] **Step 5: Add topology and isolation tests**

Add `test_mapped_new_child_resets_geometry_history_while_survivor_keeps_previous_refresh` and `test_geometry_refresh_is_detached_and_does_not_change_training_state`. Assert survivors follow `new_to_old`; any `is_new=True` row resets geometry evidence/history even with a mapped parent; temporal diagnostic lineage may still inherit separately; no parameter grad, optimizer field, densification proxy, or topology decision changes.

- [ ] **Step 6: Run the diagnosis on the unmodified implementation**

Run:

```bash
python -B -m unittest \
  tests.test_reprojection_reliability \
  tests.test_d0_collector \
  tests.test_geometry_reliability_semantics -v
```

Expected branch:

- If every test passes, record `SEMANTIC_TEST_OUTCOME=NO_SEMANTIC_REPAIR_JUSTIFIED`; do not edit any production file, do not bump version, and continue only to Task 2 so the read-only conclusion can be frozen.
- If a test fails, it must fail on an explicit approved-formula value, collector/reprojection boundary, validity/support contract, or state transition. Record the exact test, expected value/state, actual value/state, and implicated production function/boundary as `SEMANTIC_TEST_OUTCOME=DEFECT_CANDIDATE`. Syntax, fixture, tolerance, device, or unrelated failures are not defect evidence.

- [ ] **Step 7: Commit the diagnostic tests without production changes**

```bash
git add tests/test_reprojection_reliability.py tests/test_d0_collector.py \
  tests/test_geometry_reliability_semantics.py
git commit -m "test: specify geometry reliability semantics"
```

### Task 2: Build the Read-Only Semantic Audit and Outcome Schema

**Files:**
- Create: `tests/test_geometry_reliability_audit.py`
- Create: `reliability/geometry_reliability_audit.py`
- Create: `scripts/diagnostics/audit_geometry_reliability_semantics.py`

**Interfaces:**
- Consumes: Task 1 conformance results, existing version-4 Tool Room checkpoint/snapshot state, `load_g1_iteration`, immutable-input fingerprint helpers, and current raw component collector without any GT dependency.
- Produces: `build_geometry_semantic_report(...) -> dict`, `validate_geometry_semantic_report(report) -> None`, `audit_exit_code(report) -> int`, and `run_audit(args, dependencies=None) -> tuple[int, dict]` with outcomes `SEMANTIC_DEFECT_CONFIRMED`, `NO_SEMANTIC_REPAIR_JUSTIFIED`, or `INCONCLUSIVE`.

- [ ] **Step 1: Write RED tests for exhaustive outcome semantics**

Hand-build formula-pass, explicit-formula-failure, and damaged-input records. Add `test_green_semantics_forces_no_repair_outcome` to prove that a fully conforming current implementation produces `NO_SEMANTIC_REPAIR_JUSTIFIED`, cannot authorize a version bump, and cannot proceed to Utility work. Assert that only a named approved-formula/state mismatch can produce `SEMANTIC_DEFECT_CONFIRMED`; direction against GT, correlation, or a requested manual override is rejected. Map outcomes to exit codes 1, 0, and 2 respectively so a defect is a deliberate stop requiring repair, not an accidental success.

- [ ] **Step 2: Write RED tests for the no-GT Tool audit boundary**

Assert the parser requires repository, Tool run, source, output, diagnostic ID, exact commit, dataset SHA, and prior SHA. Assert it exposes no GT path, mesh, candidate, sign, threshold, or formula override. With fake inputs, require version-4 checkpoint/snapshot joins, finite/range checks, `r_g == V_g*T_g`, current-valid agreement, state tensor shapes, immutable before/after fingerprints, and an output path outside run/source.

- [ ] **Step 3: Verify RED and commit tests**

Run: `python -B -m unittest tests.test_geometry_reliability_audit -v`

Expected: import failure for `reliability.geometry_reliability_audit` or the new CLI, with no unrelated failure.

Commit: `test: specify geometry semantic audit`

- [ ] **Step 4: Implement the pure report and read-only CLI GREEN**

Implement immutable JSON-safe report validation and compact atomic publication of exactly `inputs.json`, `report.json`, and `manifest.json`. The CLI may load raw current geometry components and version-4 state, but may not load a mesh or compute an error label. It must clean staging on failure and reject overwrite/input mutation.

- [ ] **Step 5: Verify GREEN and commit**

Run:

```bash
python -B -m unittest \
  tests.test_geometry_reliability_semantics \
  tests.test_geometry_reliability_audit \
  tests.test_evidence_accumulator_state \
  tests.test_d0_shadow_runtime -v
```

Expected: all pass; no GT import or training launch occurs.

Commit: `feat: add read-only geometry semantic audit`

### Task 3: Run and Freeze the Tool Room No-GT Diagnosis

**Files:**
- Update after result: `task_plan.md`
- Update after result: `findings.md`
- Update after result: `progress.md`
- Do not modify: `reliability/evidence.py`

**Interfaces:**
- Consumes: exact reviewed Task 1–2 commit and the existing immutable Tool Room version-4 3,000/7,000 assets/source/prior.
- Produces: one hash-verified no-GT diagnostic directory and one frozen branch decision.

- [ ] **Step 1: Stop for explicit server-diagnostic authorization**

Present the exact commit, clean status, focused/full test counts, command, existing Tool input paths/hashes, absent output path, and `GT_ARGUMENT_PRESENT=NO`. Do not run the command until the user authorizes this stage.

- [ ] **Step 2: Qualify the exact commit on AutoDL**

Run focused tests, full discovery, `py_compile`, clean-tree/no-training checks, and a CUDA boundary test. Any failure returns to a new RED/GREEN commit before the real diagnostic.

- [ ] **Step 3: Execute the one-time no-GT audit and verify its manifest**

Run the CLI once against the existing Tool Room assets. Verify output hashes, no GT path/reference, no training process, no source/run mutation, exact commit, and exact outcome.

- [ ] **Step 4: Apply the hard branch gate**

- If Task 1 conformance and the real audit are green, freeze `NO_SEMANTIC_REPAIR_JUSTIFIED`, update the three planning files, commit `docs: freeze geometry semantic stop`, and stop the entire plan. Tasks 4–13 are forbidden.
- If an approved-formula test has a reproducible RED and the audit names the same implementation defect, freeze `SEMANTIC_DEFECT_CONFIRMED` with the test/function evidence and proceed to Task 4.
- If inputs or computation prevent the decision, freeze `INCONCLUSIVE`; repair only the measurement defect under the unchanged contract, then rerun Task 3. Do not touch production geometry semantics.

### Task 4: Conditionally Repair the Confirmed Defect and Advance to Version 5

**Condition:** Execute only after Task 3 freezes `SEMANTIC_DEFECT_CONFIRMED`. Otherwise this task is out of scope.

**Files:**
- Conditionally modify: the exact existing production file(s) located by the Task 1/Task 3 RED. Expected boundaries are `reliability/collector.py` for `E_g,mv`/`E_g,dn` assembly, validity, and support; `reliability/evidence.py` for reprojection, combination, history, and the required version bump; or the existing renderer evidence-transport path only when its own focused transport test is RED. Do not edit every listed boundary by default.
- Modify after any confirmed stored-semantic repair: `reliability/evidence.py` (`EvidenceAccumulator.STATE_VERSION` only if the root-cause repair is elsewhere).
- Modify: `tests/test_reprojection_reliability.py`
- Modify: `tests/test_d0_collector.py`
- Modify: `tests/test_geometry_reliability_semantics.py`
- Modify: `tests/test_evidence_accumulator_state.py`
- Modify: `tests/test_d0_shadow_runtime.py`
- Modify: `tests/test_topology_migration.py`
- Modify: `tests/gpu/test_evidence_accumulator_cuda.py`

**Interfaces:**
- Consumes: the exact failing approved-formula/collector/reprojection/transport test, failure trace, and named production function or boundary from Task 3.
- Produces: the minimal formula-preserving repair using the existing serialized fields, `EvidenceAccumulator.STATE_VERSION = 5`, strict version-4 load rejection, and unchanged snapshot field inventory.

- [ ] **Step 1: Re-run the exact semantic RED before editing code**

Run the single failing test and capture its expected failure. If it now passes, fails differently, or requires changing a formula/constant, stop and return to Task 3.

- [ ] **Step 2: Extend RED coverage around the confirmed root cause**

Add the smallest adjacent boundary cases needed to distinguish the root cause from alternative implementations. If the failure is upstream, preserve the real collector/reprojection score, validity, support, and transport assertions rather than replacing them with a post-combination mock. Add version tests asserting new state writes version 5, version 4 cannot load/resume, and old version-4 assets remain readable only through explicitly historical diagnostic paths.

- [ ] **Step 3: Implement the minimal GREEN**

Modify only the production boundary implicated by the RED and set `EvidenceAccumulator.STATE_VERSION = 5`. The repair may land in collector/reprojection assembly, evidence combination/history, or renderer evidence transport according to the observed failure; the plan does not predetermine `reliability/evidence.py` as the root cause. Keep `D0ShadowRuntime.STATE_VERSION = 2`, snapshot fields, arbitration, gradients, topology actions, constants, and approved component formulas unchanged. If the repair needs a new serialized field or runtime-envelope meaning, stop for a spec amendment instead of silently bumping another schema.

- [ ] **Step 4: Run focused and full GREEN**

Run:

```bash
python -B -m unittest \
  tests.test_reprojection_reliability \
  tests.test_d0_collector \
  tests.test_geometry_reliability_semantics \
  tests.test_evidence_accumulator_state \
  tests.test_d0_shadow_runtime \
  tests.test_topology_migration \
  tests.gpu.test_evidence_accumulator_cuda -v
python -B -m unittest discover -s tests -v
```

Expected: all tests pass; historical version-4 formal report/confirmation tests retain their old identity and no old checkpoint is admitted as version 5.

- [ ] **Step 5: Commit the repair**

Stage only the production path(s) named in Task 3 plus the exact test files changed in Steps 1–2, then inspect the staged inventory:

```bash
git diff --name-only
git diff --cached --name-only
git commit -m "fix: restore geometry reliability semantics"
```

This is not permission to stage every candidate boundary or any unrelated production file.

### Task 5: Freeze Version-5 Repair Qualification Without Utility GT

**Files:**
- Modify: `tests/test_geometry_reliability_audit.py`
- Modify: `reliability/geometry_reliability_audit.py`
- Modify: `scripts/diagnostics/audit_geometry_reliability_semantics.py`
- Update after qualification: `task_plan.md`, `findings.md`, `progress.md`

**Interfaces:**
- Consumes: Task 4 version-5 state and the Task 2 audit interface.
- Produces: strict version-5 post-repair qualification, a formula-only before/after explanation, and a frozen exact repair commit; still no predictive claim.

- [ ] **Step 1: Add RED tests for post-repair qualification**

Assert the audit now requires version 5 for repaired fixtures, rejects mixed 4/5 state, proves the exact formerly failing semantic case is corrected, and preserves no-GT/no-training/immutable-input behavior. The report must distinguish `semantic_correct=True` from `predictive_useful=None`.

- [ ] **Step 2: Implement and verify GREEN**

Run focused/full CPU tests, real CUDA tests, `py_compile`, `git diff --check`, and clean/no-training checks. Repeat the Tool engineering audit only in no-GT mode. Expected: repaired formula/state chronology passes, source/run inputs remain unchanged, and no GT-derived field exists.

- [ ] **Step 3: Request code review and freeze the repair**

Review the Task 1–5 diff against the approved spec. Fix Critical/Important findings through new RED/GREEN commits. Then record exact commit, schema versions, formulas/constants, test outputs, and audit manifest in the planning files.

Commit: `docs: freeze version five geometry qualification`

- [ ] **Step 4: Stop for Utility-preparation implementation approval**

Do not inspect the Utility upload or implement/run preprocessing commands merely because version 5 qualified. Present the frozen repair evidence and request the next stage explicitly.

### Task 6: TDD the 147-Way Source Audit and Preprocessing Confirmation

**Condition:** Task 5 must be frozen and the user must authorize Utility-preparation implementation. This task does not run DA3.

**Files:**
- Create: `tests/test_utility_da3.py`
- Create: `tests/test_utility_da3_cli.py`
- Create: `reliability/utility_da3.py`
- Create: `scripts/diagnostics/prepare_utility_da3_confirmation.py`

**Interfaces:**
- Produces: `audit_utility_source(source_root: Path, expected_count: int = 147) -> dict`, `build_preprocessing_confirmation(...) -> dict`, `write_preprocessing_confirmation(...) -> dict`, `load_preprocessing_confirmation(path: Path, expected_sha256: str) -> dict`, and `run_preflight(args) -> tuple[int, dict]`.
- The CLI requires explicit `max_points`, `ransac_thresh`, DA3 checkpoint SHA, code commit, source SHA, command/output paths, and confirmation ID; it has no defaults for the two preprocessing values and no GT argument.

- [ ] **Step 1: Write source-audit RED tests**

Create small text-COLMAP fixtures and real image headers. Assert exact 147-way behavior through parameterized small-count fixtures: supported cameras, image dimensions, registered/disk basename identity, case sensitivity, valid camera IDs, finite poses/points, no duplicate/extra/missing images, and absent `split.json`. Add `test_source_audit_rejects_fisheye_case_mismatch_and_unregistered_image` from Review Focus.

- [ ] **Step 2: Write confirmation/CLI RED tests**

Assert deterministic manifest ordering and canonical JSON/detached SHA, safe IDs/paths, absent derived target, no overwrite, source re-fingerprint before publication, explicit preprocessing parameters, and exact normalized DA3 command. Add `test_preprocessing_cli_has_no_gt_surface_and_detects_source_change`; reject any `--gt-*` argument and any attempt to inspect the GT directory.

- [ ] **Step 3: Verify RED and commit tests**

Run: `python -B -m unittest tests.test_utility_da3 tests.test_utility_da3_cli -v`

Expected: missing-module/import failures only.

Commit: `test: specify Utility DA3 preflight`

- [ ] **Step 4: Implement minimal source audit and confirmation GREEN**

Use existing COLMAP text readers and Pillow rather than a new parser. Hash only `images/` and `sparse/0/`; reject pre-existing derived target paths. Write only the confirmation JSON and detached SHA outside source. Do not invoke a subprocess or DA3.

- [ ] **Step 5: Verify and commit GREEN**

Run focused tests plus dataset-reader regressions and `py_compile`. Expected: all pass and no fixture accesses a GT path.

Commit: `feat: add Utility DA3 preflight confirmation`

### Task 7: TDD the One-Time DA3 Snapshot Finalizer

**Files:**
- Modify: `tests/test_utility_da3.py`
- Modify: `tests/test_utility_da3_cli.py`
- Modify: `reliability/utility_da3.py`
- Create: `scripts/diagnostics/finalize_utility_da3_snapshot.py`

**Interfaces:**
- Produces: `audit_da3_snapshot(source_root: Path, confirmation: dict) -> dict`, `write_da3_snapshot_record(...) -> dict`, and `run_finalize(args) -> tuple[int, dict]`.
- Consumes: the exact preprocessing confirmation and derived directories produced by the separately authorized DA3 run.

- [ ] **Step 1: Write RED tests for derived artifacts**

Assert exactly one depth and confidence `.npy` per registered full image filename, finite numeric arrays with image-compatible shapes, loadable `sparse_da3/0`, loadable `sparse_da3_aligned/0`, finite positive `trans.json.scale`, complete aligned model files, unchanged source manifest, and deterministic derived manifest/snapshot ID.

- [ ] **Step 2: Write RED tests for regeneration and no-GT behavior**

Assert a different derived byte, command, DA3 checkpoint, or source hash creates a different snapshot ID; it cannot overwrite the old record. The finalizer parser has no GT argument, never reads the GT root, and rejects a derived tree created before its pinned confirmation chronology.

- [ ] **Step 3: Verify RED, implement GREEN, and run regression**

Run RED first, implement only the audit/finalizer, then run:

```bash
python -B -m unittest \
  tests.test_utility_da3 \
  tests.test_utility_da3_cli \
  tests.test_g1_confirmation -v
```

Expected GREEN: exact counts/hashes are enforced without invoking DA3 or GT.

- [ ] **Step 4: Commit**

Commit RED: `test: specify Utility DA3 snapshot finalization`

Commit GREEN: `feat: add Utility DA3 snapshot finalizer`

### Task 8: TDD the Version-5 Three-Seed Confirmation Contract

**Files:**
- Create: `tests/test_g1_dual_risk_confirmation.py`
- Create: `reliability/g1_dual_risk_confirmation.py`

**Interfaces:**
- Produces: `build_dual_risk_confirmation(...) -> dict`, `write_dual_risk_confirmation(...) -> dict`, `load_dual_risk_confirmation(path: Path, expected_sha256: str) -> dict`, `validate_dual_risk_preregistration(record, planned_targets) -> dict`, and `validate_dual_risk_admission(record, runs, probe_targets) -> dict`.
- Consumes: exact repair commit, Evidence version 5, the already frozen DA3 snapshot, GT file identity only as immutable SHA/size/path metadata, seeds 0/1/2 planned run commands and absent run/view/state paths, absent probe targets, and the immutable statistical constants from the spec.

- [ ] **Step 1: Write the exact confirmation RED contract**

Assert the record binds three and only three seeds `{0,1,2}`, unique absent run/view/state paths, identical dataset snapshot ID, resolution 2, 7,000 iterations, refresh 1,000, checkpoints/evaluation 3,000 and 7,000, no GT in training commands, Evidence version 5, exact probe models/gates/bootstrap, and absent diagnostic targets. Add `test_confirmation_is_written_after_snapshot_and_before_all_run_view_state_and_probe_targets` and require the confirmation timestamp/detached SHA to precede every later target creation.

- [ ] **Step 2: Write chronology, mutation, and legacy rejection tests**

Reject version 4, mixed commits/snapshots/configs, duplicate/missing seeds, existing targets at preregistration, changed source/DA3/GT hashes, changed confirmation bytes, and a confirmation timestamp after any run/view/state/probe target or first Utility GT-derived artifact. Prove that completed runs are later admitted by reloading the identical confirmation SHA without rewriting the record. Preserve old `g1_confirmation.py` and its version-4 tests unchanged.

- [ ] **Step 3: Verify RED, implement GREEN, and commit**

Run: `python -B -m unittest tests.test_g1_dual_risk_confirmation tests.test_g1_confirmation -v`

Expected RED: missing new module. Implement canonical validation/write/load without adding tunable CLI gates, rerun to GREEN, then commit:

- `test: specify dual risk confirmation`
- `feat: add dual risk confirmation`

### Task 9: Freeze the Pure Nested Statistical RED Contract

**Files:**
- Create: `tests/test_g1_dual_risk.py`

**Interfaces:**
- Consumes: generic metric/solver primitives from `reliability.g1_complementarity` where their existing behavior exactly matches the approved contract; do not alter the frozen Tool v2 report path.
- Produces: failing tests for `DualRiskConfig`, `DualRiskDomain`, `build_dual_risk_domain`, `crossfit_nested_models`, `paired_nested_voxel_bootstrap`, `build_dual_risk_report`, `validate_dual_risk_report`, and `dual_risk_exit_code`.

- [ ] **Step 1: Specify the common domain and raw directions**

Use finite/non-finite joined fixtures with independent `V_p/V_g`. Assert the denominator is all finite centers, selection is strict `V_p & V_g`, M0/M1/M2 use identical original rows and labels, separate channel coverage is reported, and opacity/scaling/frustum/crop fields cannot affect selection. Assert invalid channels are excluded, not encoded as risk 1.

- [ ] **Step 2: Specify nested fold-local fitting**

Use hand-checked data where M1 adds information over M0 and M2 adds information over M1. Assert identical five-slab folds, training-only standardization/weights, pooled OOF gains, five paired deltas per increment, both column-residual checks, finite solver diagnostics, and no interaction/candidate selection.

- [ ] **Step 3: Specify fixed-seed macro bootstrap**

Add `test_macro_bootstrap_averages_fixed_seed_paired_voxel_gains_without_pooling_rows_or_resampling_seeds`. Use three seeds with different row counts and voxel maps. Assert seed-local paired resampling with `SeedSequence([20260928, iteration, seed])`, no row pooling, no seed resampling, no refit, replicate-wise arithmetic macro mean, 2,000 values, and percentile endpoints.

- [ ] **Step 4: Specify gates and exhaustive outcomes**

Hand-build pass, valid-negative, and invalid reports. Assert every Section 7 gate, 3,000/7,000 role, per-seed non-negative gains, macro thresholds, coverage, raw direction, and column residual. Outcomes/exit codes are exactly `DUAL_RISK_EVIDENCE_FEASIBLE/0`, `NO_STABLE_GEOMETRY_COMPLEMENT/1`, and `INCONCLUSIVE/2`; all C1/causal/cross-scene claims remain false/null.

- [ ] **Step 5: Verify RED and commit tests**

Run: `python -B -m unittest tests.test_g1_dual_risk -v`

Expected: import failure for `reliability.g1_dual_risk`.

Commit: `test: specify nested dual risk statistics`

### Task 10: Implement the Pure Nested Engine GREEN

**Files:**
- Create: `reliability/g1_dual_risk.py`
- Test: `tests/test_g1_dual_risk.py`
- Do not modify: `reliability/g1_complementarity.py`

**Interfaces:**
- Consumes: Task 9 tests and stable generic primitives imported from the frozen prior probe.
- Produces: the exact pure interfaces named in Task 9 and flat row adapters for seed/fold, risk-bin, and bootstrap CSVs.

- [ ] **Step 1: Implement validated common-domain and nested OOF primitives**

Keep report arrays float64 and original-row identity int64. Construct M0/M1/M2 once per seed/iteration, reuse the same folds, and compute the two pooled AUROC gains from complete OOF predictions.

- [ ] **Step 2: Implement seed-local bootstrap and macro assembly**

Group each seed's rows independently by fixed-origin voxels, preserve paired model multiplicities, and assemble macro replicate `b` only from the three seed-specific replicate-`b` gains. Report per-seed intervals diagnostically but gate the registered macro interval.

- [ ] **Step 3: Implement exact validation and decisions**

Reject unknown/missing fields, changed constants, non-finite values, wrong seed/iteration inventory, altered model sequence, invalid report claims, or wrong bootstrap length. Do not add CLI-settable thresholds.

- [ ] **Step 4: Run GREEN and regression**

Run:

```bash
python -B -m unittest \
  tests.test_g1_dual_risk \
  tests.test_g1_complementarity \
  tests.test_g1_metrics -v
```

Expected: all pass and the frozen Tool v2 engine/report tests are byte-semantically unchanged.

- [ ] **Step 5: Commit**

Commit: `feat: add nested dual risk statistics`

### Task 11: Freeze Version-5 CLI and Compact Publication RED

**Files:**
- Create: `tests/test_g1_dual_risk_cli.py`

**Interfaces:**
- Consumes: Task 8 confirmation, Task 10 pure engine, `load_g1_iteration(..., expected_evidence_version=5)`, `load_valid_mesh`, `closest_triangle_distances`, existing fingerprint/manifest utilities, and only the fixed three-camera GT alignment helper.
- Produces: failing tests for `run_probe(args, dependencies=None) -> tuple[int, dict]` and `build_parser()` in `scripts.diagnostics.probe_g1_dual_risk`.

- [ ] **Step 1: Specify strict three-run/version-5 admission**

Assert exact clean diagnostic commit and reload the same canonical confirmation path/SHA written before training. Bind its three predeclared targets to the three completed run identities, same DA3 snapshot, expected seeds/configs/hashes, version-5 checkpoint/snapshot pairs at 3,000/7,000, and absent output. Reject a replacement or post-training confirmation even when its fields otherwise match, and expose no extra run/candidate/model/threshold arguments.

- [ ] **Step 2: Specify full-domain orchestration**

With six fake joined assets, assert distance is queried for every finite center before `V_p & V_g` selection, row maps remain strict, all three models receive identical rows, and no training/collector/evidence refresh path is called. Mesh validation rejects only non-finite/degenerate triangles and cannot crop or transform GT.

- [ ] **Step 3: Specify compact atomic output**

Require `inputs.json`, `report.json`, `seed_folds.csv`, `risk_bins.csv`, `bootstrap.csv`, `alignment.json`, up to three fixed overlay PNGs, and `manifest.json`. Assert no PLY, field render, checkpoint/snapshot copy, 1 GB archive, or training artifact. Verify deterministic ordering, byte/SHA manifest, no overwrite, staging cleanup, and immutable re-fingerprinting.

- [ ] **Step 4: Specify three-state CLI behavior and GT chronology**

Assert metric-negative reports are still valid atomic publications with exit 1; safely characterizable input/model failures publish `INCONCLUSIVE` with exit 2; unsafe path, pre-confirmation GT artifact, overwrite, or input mutation publishes nothing. Any GT-derived output timestamp preceding the confirmation is rejected.

- [ ] **Step 5: Verify RED and commit**

Run: `python -B -m unittest tests.test_g1_dual_risk_cli -v`

Expected: import failure for `scripts.diagnostics.probe_g1_dual_risk`.

Commit: `test: specify dual risk probe publication`

### Task 12: Implement the Version-5 CLI and Compact Publication GREEN

**Files:**
- Create: `scripts/diagnostics/probe_g1_dual_risk.py`
- Test: `tests/test_g1_dual_risk_cli.py`

**Interfaces:**
- Consumes: all Task 8/10/11 interfaces.
- Produces: strict admission, six full-mesh evaluations, three fixed alignment overlays, compact atomic output, manifest, concise console JSON, and exact exit codes without training.

- [ ] **Step 1: Implement request/admission and immutable fingerprints**

Resolve safe paths, reload the exact pre-training confirmation and detached SHA, require its commit/hashes and completed three runs at the predeclared targets, fingerprint every consumed asset, and reject a rewritten/later confirmation, version 4, or mixed state before any GT query.

- [ ] **Step 2: Implement six iteration evaluations and macro report**

Load seeds 0/1/2 at 3,000/7,000, query the full valid mesh for all finite centers, build strict common domains, compute nested seed reports, then assemble the frozen macro gates. Generate only the fixed alignment audit after confirmation admission.

- [ ] **Step 3: Implement atomic compact publication**

Write to a unique staging directory, validate all JSON/CSV/PNG inventory, re-fingerprint inputs, build the manifest, and atomically rename. Do not archive unless a later explicit transport request asks for the already validated compact files.

- [ ] **Step 4: Run GREEN and full relevant regression**

Run:

```bash
python -B -m unittest \
  tests.test_g1_dual_risk_cli \
  tests.test_g1_dual_risk \
  tests.test_g1_dual_risk_confirmation \
  tests.test_g1_geometry \
  tests.test_g1_visualization \
  tests.test_g1_complementarity_cli -v
python -B -m py_compile \
  reliability/g1_dual_risk.py \
  reliability/g1_dual_risk_confirmation.py \
  scripts/diagnostics/probe_g1_dual_risk.py
```

Expected: all pass; no old formal or Tool v2 publication contract changes.

- [ ] **Step 5: Commit**

Commit: `feat: add compact dual risk probe`

### Task 13: Whole-Branch Review and Separately Authorized Execution Gates

**Files:**
- Update only after each completed gate: `task_plan.md`, `findings.md`, `progress.md`
- No experiment artifact is committed to Git.

**Interfaces:**
- Consumes: all conditional implementation commits that remain reachable after the Task 3 branch gate.
- Produces: reviewed code plus a sequence of user-operated server commands; it does not collapse their approvals.

- [ ] **Step 1: Run local whole-branch verification**

Run `python -B -m unittest discover -s tests -v`, all relevant GPU tests where hardware is available, `py_compile` for every new/modified Python file, `git diff --check`, and `git status --short`. Record failures by exact name; do not claim completion from a focused subset.

- [ ] **Step 2: Request whole-branch code review**

Review from the pre-plan base commit through HEAD against the approved spec and this plan. Resolve all Critical/Important findings through new RED/GREEN commits; rerun Step 1 afterward.

- [ ] **Step 3: Freeze the exact implementation commit**

Push only after clean/full verification. Record the 40-character SHA, schema versions, test counts, and no-GT/no-training status. This freeze still does not authorize server data actions.

- [ ] **Step 4: Enforce six distinct server approvals**

Request and execute these in order; never combine an unapproved later gate into an earlier command:

1. **Tool semantic diagnostic authorization** — Task 3 only; no GT.
2. **Utility source-audit/preprocessing-confirmation authorization** — read-only 147-way audit and confirmation only; no DA3 and no GT content.
3. **One-time DA3 authorization** — exact confirmed command/new target, followed by Task 7 snapshot finalization; no GT.
4. **Canonical dual-risk confirmation authorization** — after the DA3 snapshot is frozen and while all three seed run/view/state paths plus probe targets are still absent, bind their planned commands/paths, the snapshot, GT SHA metadata, exact code/config/statistical gates, and write one canonical record with detached SHA; still no training or GT metric.
5. **Utility version-5 training authorization** — reload that confirmation, launch seeds 0/1/2 sequentially at its predeclared targets using the same snapshot, and remain GT-free; completion/deep audit of all three runs is required before proceeding.
6. **Utility GT probe authorization** — the evaluator reloads the same pre-training confirmation path/SHA, binds its predeclared targets to the completed runs, rejects any replacement/post-training confirmation, then performs one compact read-only GT probe and manifest/result audit; no C1.

- [ ] **Step 5: Apply the final scientific stop rule**

Freeze exactly one outcome. `DUAL_RISK_EVIDENCE_FEASIBLE` authorizes only a new architecture specification; `NO_STABLE_GEOMETRY_COMPLEMENT` stops the dual-risk route; `INCONCLUSIVE` allows only same-contract measurement repair. None starts C1 automatically.

## Execution Handoff Boundary

Approval of this document authorizes only implementation according to its TDD tasks after the user separately selects an execution approach. It does not authorize any server diagnostic, data preprocessing, DA3 command, Utility training, GT access, or C1. The first executable activity after implementation approval is Task 1's local/current-code semantic conformance suite; every later external stage remains stopped at Task 13's explicit gates.
