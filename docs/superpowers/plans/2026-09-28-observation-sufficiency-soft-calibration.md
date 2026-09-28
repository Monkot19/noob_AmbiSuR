# Observation-Sufficiency Soft Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. The user selected current-task/native execution; do not dispatch subagents unless the user explicitly changes that choice.

**Goal:** Replace the hard-capped observation-count and angular-sufficiency calibrations with the approved fixed half-saturation functions, preserve every other Core/G1 contract, and admit a new Tool Room formal result only through a version-4, preregistered identity gate.

**Architecture:** `reliability.evidence` remains the sole owner of the fixed `S_count`, `S_angle`, `S`, and evidence-state semantics. A new pure `reliability.g1_confirmation` module owns canonical preregistration records and run-identity validation; `offline_g1` exposes the checkpoint evidence version, and the evaluator performs formal admission before mesh queries, renders, metrics, or atomic publication. Experiments remain user-operated: code qualification precedes a 500-step no-topology smoke, which precedes a separately authorized, preregistered 7k run.

**Tech Stack:** Python 3.10, PyTorch 2.7.1+cu128, NumPy 1.26.3, standard-library `unittest`, CUDA 12.8 custom rasterizer, JSON/SHA256, Git, AutoDL RTX 4090.

**Spec:** `docs/superpowers/specs/2026-09-28-observation-sufficiency-soft-calibration-design.md`

## Global Constraints

- Implement only `S_count=M/(M+5)` and `S_angle=D/(D+D_c)`, where `D_c=(1-cos(30 degrees))/2`; preserve `S=sqrt(S_count*S_angle)` and `N=1-S(1-A)`.
- Do not change `A`, EMA coefficients, refresh schedules, prior/geometry reliability, arbitration, routing, topology rules, G1 labels, G1 thresholds, the full finite-center/full valid-mesh domain, or Supporting features.
- Keep the historical `T_g/r_g` reversal isolated as a separate pre-C2 investigation; do not change or retune it under this plan.
- `EvidenceAccumulator.STATE_VERSION` changes exactly from 3 to 4; version 3 is rejected for training resume and for new-calibration formal admission.
- Training remains GT-free. GT mesh appears only in the post-training evaluator and never in run config, run identity, checkpoint, snapshot, cache, or training command.
- Formal G1 metrics remain `AUROC(N)>0.60` and gain over `max(A,1-S)>=0.03`; iteration 7000 is the only decision point and iteration 3000 is diagnostic-only.
- A Tool Room r2/seed-0 PASS authorizes only the written C1 specification and implementation plan. Utility Room and the E3 multi-seed protocol remain mandatory for scientific/generalization claims.
- The 500-step smoke has refreshes at 100/200/300/400/500 and ends before the first real densification at 600; it must never be cited as real-topology evidence.
- Existing baseline and formal-v2 assets are immutable. Every smoke, formal run, report, archive, and confirmation uses a new path and confirmation ID.
- Preserve RED and GREEN evidence in separate commits. Each server command requires a clean exact commit, no active training, fixed input hashes, and user-operated execution.

## File Structure

- Modify `reliability/evidence.py`: fixed half-saturation constants/functions, finite argument validation, and evidence state version 4.
- Modify `tests/test_reliability_evidence.py`: mathematical anchors, monotonicity, no plateau, composition, validation, chunking, dtype, and no-grad CPU contracts.
- Modify `tests/test_evidence_accumulator_state.py`: version-4 round trip, version-3 rejection, and unchanged topology reset/temporal-lineage behavior.
- Modify `tests/gpu/test_evidence_accumulator_cuda.py`: CUDA dtype/device/no-grad/chunk equivalence and finite soft-calibration coverage.
- Create `reliability/g1_confirmation.py`: canonical confirmation construction/loading and exact formal run-identity/config/hash validation.
- Create `tests/test_g1_confirmation.py`: confirmation serialization, detached SHA, absent-target, mutation, metadata, and protocol tests.
- Modify `reliability/offline_g1.py`: expose and optionally require the nested checkpoint evidence version while retaining exploratory readability.
- Modify `tests/test_g1_offline_inputs.py`: version-4 formal load and version-3 exploratory/formal separation.
- Modify `scripts/diagnostics/evaluate_d0_g1.py`: require a confirmation contract for formal mode, preload admitted 3000/7000 inputs, and keep exploratory mode decision-null.
- Modify `tests/test_g1_publication.py` and `tests/test_g1_orchestration.py`: CLI/formal admission, fail-before-publication, and immutable-input coverage.
- Update `task_plan.md`, `findings.md`, and `progress.md` after each review/experiment gate; create no stage tag until the frozen formal result passes.

## Review Focus

- Nonfinite or boundary `k_c/theta_c` inputs: reject NaN/Inf, nonpositive `k_c`, and `theta_c` outside the open `(0,180)` interval before tensor computation (Task 1).
- Mixed formal checkpoint semantics: reject if either 3000 or 7000 is not evidence version 4, even when snapshots and row counts otherwise agree (Tasks 3–5).
- Forged or late confirmation: reject a wrong detached SHA, mutated JSON, pre-existing target, mismatched ID/path, or record created without absence proofs before any formal publication (Tasks 3–5).
- Hidden run drift: reject dirty/wrong commits, changed argv/seed/resolution/Core flags/iteration schedules, and before/after dataset or prior hashes that differ from the confirmation (Tasks 3–5).
- False scientific promotion: keep version-3/exploratory decisions null and report a Tool Room seed-0 PASS only as authorization to draft C1, never as E3 completion (Tasks 5 and 9).

---

### Task 1: Freeze the Formula and State-Version RED Contract

**Files:**
- Modify: `tests/test_reliability_evidence.py`
- Modify: `tests/test_evidence_accumulator_state.py`
- Modify: `tests/gpu/test_evidence_accumulator_cuda.py`

**Interfaces:**
- Consumes: existing `compute_observation_sufficiency(...) -> ObservationSufficiency` and `EvidenceAccumulator.state_dict/load_state_dict`.
- Produces: failing tests for the approved formulas and version 4; no production behavior changes.

- [ ] **Step 1: Add CPU mathematical-anchor and validation tests**

Add focused tests that construct camera directions with hand-checked `M` and `D`, then assert with dtype-appropriate tolerance:

```python
S_count(M=0, 5, 10) == (0.0, 0.5, 2.0 / 3.0)
S_angle(D=0, D_c) == (0.0, 0.5)
S == sqrt(S_count * S_angle)
N == 1.0 - S * (1.0 - A)
```

Also assert strictly increasing representative count/dispersion inputs do not collapse to the former plateau, outputs remain finite in `[0,1)`, empty point domains remain empty, and invalid `k_c/theta_c_degrees` values fail closed.

- [ ] **Step 2: Add state-version and migration tests**

Change the round-trip expectation to version 4, clone a valid state with `version=3` and assert `load_state_dict` raises `unsupported evidence state version`, and retain the existing survivor/new-row reset plus temporal-lineage inheritance assertions.

- [ ] **Step 3: Add CUDA soft-calibration contract tests**

On CUDA, compare `chunk_size=2` with an unchunked call for the same CPU hit matrix and CUDA centers, assert identical `M_obs`, dtype/device preservation, finite outputs, `requires_grad=False`, and no finite plateau at the old caps. Add a small CUDA `EvidenceAccumulator` topology fixture that proves surviving evidence rows migrate, new/mapped-child evidence rows reset, temporal lineage follows the existing parent contract, and the resulting state still serializes as version 4.

- [ ] **Step 4: Run RED and commit only tests**

Run:

```bash
python -B -m unittest tests.test_reliability_evidence tests.test_evidence_accumulator_state -v
```

Expected: failures identify the old hard caps and state version 3, not fixture/import errors. On AutoDL also run the named CUDA class and preserve its expected RED.

Commit:

```bash
git add tests/test_reliability_evidence.py tests/test_evidence_accumulator_state.py tests/gpu/test_evidence_accumulator_cuda.py
git commit -m "test: specify soft sufficiency calibration"
```

### Task 2: Implement the Fixed Soft Calibration and Version 4

**Files:**
- Modify: `reliability/evidence.py`
- Test: files from Task 1

**Interfaces:**
- Consumes: Task 1 tests.
- Produces: `OBSERVATION_COUNT_HALF_SATURATION=5.0`, `OBSERVATION_ANGLE_HALF_SATURATION_DEGREES=30.0`, updated `compute_observation_sufficiency`, and `EvidenceAccumulator.STATE_VERSION=4`.

- [ ] **Step 1: Implement minimal formula and argument validation**

Keep the public signature and chunked detached computation. Validate scalar constants with `math.isfinite`; compute score tensors on the Gaussian device/dtype using:

```python
count_score = count_float / (count_float + k_c)
angle_score = dispersion / (dispersion + D_c)
```

Do not add configuration flags, quantile scaling, learned parameters, scene-dependent values, or terminal clamps to 1.

- [ ] **Step 2: Increment only the nested evidence version**

Set `EvidenceAccumulator.STATE_VERSION=4`. Do not change `D0ShadowRuntime.STATE_VERSION`, checkpoint schema version, event schema, snapshot fields, or topology migration behavior.

- [ ] **Step 3: Run focused GREEN and regressions**

Run:

```bash
python -B -m unittest \
  tests.test_reliability_evidence \
  tests.test_evidence_accumulator_state \
  tests.test_d0_shadow_runtime \
  tests.test_topology_migration \
  tests.test_topology_composition -v
python -B -m py_compile reliability/evidence.py
```

Expected: all pass; existing topology and temporal assertions remain unchanged.

- [ ] **Step 4: Commit the production GREEN**

```bash
git add reliability/evidence.py
git commit -m "feat: soften observation sufficiency calibration"
```

### Task 3: Freeze Confirmation and Formal-Admission RED Tests

**Files:**
- Create: `tests/test_g1_confirmation.py`
- Modify: `tests/test_g1_offline_inputs.py`
- Modify: `tests/test_g1_publication.py`
- Modify: `tests/test_g1_orchestration.py`

**Interfaces:**
- Consumes: version-4 state from Task 2 and existing formal evaluator CLI.
- Produces: failing tests for `build_confirmation_record`, `write_confirmation_record`, `load_confirmation_record`, `validate_formal_admission`, `load_g1_iteration(..., expected_evidence_version=...)`, and the two new formal CLI arguments.

- [ ] **Step 1: Define the canonical record fixture**

The fixture is a JSON object with `schema_version=1`, unique `confirmation_id`, UTC preflight time, formula strings/constants, exact 40-character formula commit, tokenized training argv, Tool Room scene, seed 0, resolution 2, iteration 7000, refresh 1000, training evaluation iterations 1000–7000, G1 iterations/checkpoints `(3000,7000)`, save iteration 7000, dataset/prior/offline-GT SHA256, resolved run/view/report/output/archive paths with all absence flags true, exact resolved-config expectations, and unchanged G1 thresholds. The GT entry is evaluation-only and never appears in training argv/config.

- [ ] **Step 2: Add canonical-write and tamper RED tests**

Assert canonical sorted JSON plus trailing newline produces a deterministic digest and detached `.sha256`; reject an existing record/sidecar or any target path that already exists; reject malformed SHA, mutation after hashing, unsafe confirmation IDs, missing fields, changed constants, or changed gates.

- [ ] **Step 3: Add checkpoint-version and run-identity RED tests**

Create small versioned Core checkpoint fixtures at 3000/7000. Assert formal loading requires nested evidence version 4 in both and rejects `(3,4)`, `(4,3)`, `(3,3)`, missing evidence, or wrong checkpoint iteration. Assert default/explicit exploratory loading can still read a structurally valid version-3 asset without producing a formal decision.

- [ ] **Step 4: Add fail-before-publication orchestration RED tests**

Assert formal `run_evaluator` requires `--confirmation-contract` and `--expected-confirmation-sha`, validates the record before calling the expensive producer, and leaves no output/archive/SHA/staging on any mismatch in commit, dirty state, argv, seed, resolution, Core flags, schedule, dataset/prior before/after hashes, confirmation ID, or resolved paths. Add the confirmation, run identity, prior-hash records, and both checkpoints to immutable fingerprints.

- [ ] **Step 5: Run RED and commit only tests**

Run:

```bash
python -B -m unittest \
  tests.test_g1_confirmation \
  tests.test_g1_offline_inputs \
  tests.test_g1_publication \
  tests.test_g1_orchestration -v
```

Expected: failures are missing confirmation interfaces/new CLI arguments and missing version enforcement.

Commit:

```bash
git add tests/test_g1_confirmation.py tests/test_g1_offline_inputs.py tests/test_g1_publication.py tests/test_g1_orchestration.py
git commit -m "test: specify soft calibration formal admission"
```

### Task 4: Implement Canonical Confirmation and Versioned Offline Loading

**Files:**
- Create: `reliability/g1_confirmation.py`
- Modify: `reliability/offline_g1.py`
- Test: `tests/test_g1_confirmation.py`
- Test: `tests/test_g1_offline_inputs.py`

**Interfaces:**
- Consumes: canonical fixture contract from Task 3.
- Produces:
  - `build_confirmation_record(...) -> dict`
  - `write_confirmation_record(record: Mapping, path: Path) -> tuple[Path, Path, str]`
  - `load_confirmation_record(path: Path, expected_sha256: str) -> dict`
  - `validate_formal_admission(record: Mapping, *, run_dir: Path, confirmation_id: str, evaluator_commit: str, dataset_sha256: str, prior_sha256: str, gt_sha256: str) -> None`
  - `load_g1_iteration(run_dir, iteration, *, expected_evidence_version: int | None = None) -> G1IterationInputs`

- [ ] **Step 1: Implement canonical record construction/writing/loading**

Use only standard-library JSON, `hashlib`, datetime/path validation, exclusive creation, and a detached SHA sidecar. Validate all fixed formula/gate/protocol values and absence proofs; do not infer or rewrite target paths after record creation.

- [ ] **Step 2: Implement exact metadata admission**

Read `run_identity.json`, `resolved_config.json`, `dataset_manifest_before/after.sha256`, and `aligned_prior_before/after.sha256`. Require clean exact commit, token-for-token argv, seed, expected resolved config fields, Core-only shadow features, exact paths/schedules, and hashes equal to the record. Reject GT references in training argv/config.

- [ ] **Step 3: Expose nested evidence-version validation**

In `load_g1_iteration`, read `core_state.evidence.version`; when `expected_evidence_version=4`, reject any other/missing value before returning joined inputs. With `None`, preserve existing read-only diagnostic behavior.

- [ ] **Step 4: Run primitive GREEN and commit**

```bash
python -B -m unittest tests.test_g1_confirmation tests.test_g1_offline_inputs -v
python -B -m py_compile reliability/g1_confirmation.py reliability/offline_g1.py
git add reliability/g1_confirmation.py reliability/offline_g1.py
git commit -m "feat: add soft calibration confirmation contract"
```

### Task 5: Enforce Formal Admission in the Evaluator

**Files:**
- Modify: `scripts/diagnostics/evaluate_d0_g1.py`
- Test: `tests/test_g1_publication.py`
- Test: `tests/test_g1_orchestration.py`

**Interfaces:**
- Consumes: Task 4 confirmation and `load_g1_iteration(..., expected_evidence_version=4)`.
- Produces: formal-only CLI arguments `--confirmation-contract`, `--expected-confirmation-sha`, and `--expected-prior-sha`; pre-admitted joined inputs for iterations 3000/7000.

- [ ] **Step 1: Add formal-only parser/admission boundary**

Formal mode requires all three new arguments. Exploratory mode may omit the confirmation but must retain `g1_evaluable=null` and `g1_pass=null`. Load and validate the confirmation plus metadata, including the already-required GT SHA, before creating staging or invoking mesh/renders/metrics.

- [ ] **Step 2: Preload exactly the admitted formal checkpoints**

Load iterations 3000 and 7000 with `expected_evidence_version=4`, require the key set exactly equals `FORMAL_ITERATIONS`, then pass those joined inputs into `produce_formal_3000_7000` so the metric path consumes the same admitted checkpoint/snapshot joins without a second metric load.

- [ ] **Step 3: Expand immutable fingerprints and report provenance**

Fingerprint the confirmation JSON, detached SHA sidecar, run identity, resolved config, dataset/prior before/after hashes, checkpoints, snapshots/events, source root, and GT mesh. Record confirmation digest/evidence version in `inputs.json`; preserve the 108-file artifact inventory and atomic cleanup semantics.

- [ ] **Step 4: Run formal evaluator GREEN and commit**

```bash
python -B -m unittest \
  tests.test_g1_confirmation \
  tests.test_g1_offline_inputs \
  tests.test_g1_publication \
  tests.test_g1_orchestration -v
python -B -m py_compile scripts/diagnostics/evaluate_d0_g1.py
git add scripts/diagnostics/evaluate_d0_g1.py
git commit -m "feat: enforce preregistered formal G1 admission"
```

### Task 6: Full Regression, CUDA Qualification, and Review Gate

**Files:**
- Modify: `task_plan.md`
- Modify: `findings.md`
- Modify: `progress.md`

**Interfaces:**
- Consumes: Tasks 1–5 exact commits.
- Produces: one clean pushed implementation commit chain qualified for smoke, not permission to train.

- [ ] **Step 1: Run local static and available full regressions**

Run `git diff --check`, `py_compile` for every changed Python file, focused suites, and `python -B -m unittest discover -s tests -v`. Record dependency-based skips separately; do not call a skipped GPU contract passed.

- [ ] **Step 2: Push the clean branch and qualify on AutoDL**

On the exact clean commit run the focused CPU suites, full discovery, and CUDA evidence tests. Require zero failures, expected CUDA tests actually executed, no active training, and unchanged canonical input hashes.

- [ ] **Step 3: Request code review and resolve findings with TDD**

Use `superpowers:requesting-code-review`. Any behavioral defect returns to RED/GREEN; do not fold unrelated `T_g/r_g` work into this branch.

- [ ] **Step 4: Verify completion evidence and stop for smoke authorization**

Use `superpowers:verification-before-completion`, update the planning files, and ask the user for explicit authorization of the new 500-step smoke. Do not launch it merely because tests pass.

Commit planning evidence with `docs: record soft calibration qualification`.

### Task 7: Run and Audit the 500-Step Version-4 Smoke

**Files:**
- Modify: `task_plan.md`
- Modify: `findings.md`
- Modify: `progress.md`

**Interfaces:**
- Consumes: user authorization and Task 6 exact commit.
- Produces: a new Tool Room r2/seed-0/500 run with refresh interval 100 and version-4 state evidence; no formal conclusion.

- [ ] **Step 1: Freeze a no-training preflight**

Require clean exact commit, 406 image/depth/conf files, canonical dataset/prior SHA, RTX 4090 runtime, at least 10 GiB free, no active training, and absent run/view/state paths.

- [ ] **Step 2: Launch one private-view GT-free smoke**

Use `--core_shadow_mode --d0_refresh_interval 100 --iterations 500 --seed 0 -r 2`, evaluations/refreshes at 100–500, save/checkpoint at 500, and no GT argument or GT path. Record command, commit, hashes, PID, wall time, and GPU peak.

- [ ] **Step 3: Deep-audit the five refreshes and checkpoint**

Require exit 0, one completion marker, five snapshots/events, finite tensors, evidence version 4, exact checkpoint/snapshot joins, refresh count 5/last 500, no GT references, immutable input hashes, optimizer step 499, clean Git, and no training process. Explicitly record `real_topology_evidence=NO` because first densification is 600.

- [ ] **Step 4: Record smoke evidence and stop**

Commit only documentation (`docs: record soft calibration smoke`). A PASS authorizes preregistration preparation, not formal training.

### Task 8: Freeze Formal-7k Confirmation and Request Training Authorization

**Files:**
- Create: one immutable server-side `.confirmation.json` plus detached `.sha256` under the diagnostics root (not committed to Git).
- Modify: `task_plan.md`
- Modify: `findings.md`
- Modify: `progress.md`

**Interfaces:**
- Consumes: Task 7 PASS and exact qualified formula commit.
- Produces: a pre-run confirmation ID, JSON path, detached SHA, absent target paths, and exact user-visible training command; no run/view directory and no training process.

- [ ] **Step 1: Generate the record through production helpers**

Freeze exact commit/formulas/constants, tokenized argv, Tool Room/seed0/r2, 7000 iterations, refresh/evaluation 1000–7000, checkpoints/G1 points 3000/7000, save 7000, dataset/prior hashes, offline GT mesh hash, resolved-config expectations, G1 gates, and fresh run/view/report/output/archive paths. The GT path/hash belongs only to the offline-evaluation section of the record and is not an input to the training launcher.

- [ ] **Step 2: Verify chronology and digest**

Require every target absent before the record is written; recompute the canonical JSON digest and verify the detached sidecar. Print both paths and SHA. Do not create the run/view or start training in this step.

- [ ] **Step 3: Review the exact server launch/audit command**

The command must create a private view, preserve canonical input hashes, launch only the confirmed argv, write before/after dataset/prior hashes, and restore the branch without touching GT.

- [ ] **Step 4: Stop for explicit formal-7k authorization**

The approved implementation plan is not itself authorization to spend GPU time on the formal run.

### Task 9: Run Formal-7k, Enforce Admission, and Apply the Frozen Decision

**Files:**
- Modify: `task_plan.md`
- Modify: `findings.md`
- Modify: `progress.md`

**Interfaces:**
- Consumes: explicit Task 8 authorization, confirmation JSON/digest, qualified code commit, canonical Tool Room inputs, and offline GT only after training.
- Produces: immutable formal D0/G1 bundle or a fail-closed diagnostic; never silently advances C1.

- [x] **Step 1: Launch and audit the confirmed 7k run**

Require exit 0, one completion marker, seven refresh snapshots/events, checkpoints 3000/7000 with evidence version 4, before/after input hashes unchanged, zero GT references, finite state, correct optimizer schedule, and real changing point counts with exact checkpoint/snapshot row alignment.

- [x] **Step 2: Run formal G1 through the confirmation gate**

Invoke the evaluator with the exact confirmation JSON and detached SHA, expected commit/dataset/prior/GT hashes, iterations 3000/7000, and a fresh confirmation-bound output. Any admission mismatch exits malformed before metric publication.

- [x] **Step 3: Audit publication and decision**

Verify 108 artifacts, manifest/archive/SHA, immutable-input before/after fingerprints, full finite-center/full valid-triangle domain, no staging residue, diagnostic-only iteration 3000, and decision-only iteration 7000.

- [x] **Step 4: Apply the unchanged stop rule**

If both frozen G1 inequalities pass, report only `C1_WRITTEN_SPEC_AND_PLAN_AUTHORIZED`; do not implement C1 until those documents are separately approved. If either fails, record the fixed operationalization as failed, stop C1, and do not search more Tool Room constants/transforms. In both cases, do not claim E3/generalization; Utility Room and multi-seed obligations remain.

- [x] **Step 5: Verify, review, and record**

Use `superpowers:verification-before-completion` and `superpowers:requesting-code-review`; commit documentation with the observed result. Create no stage tag unless all tag prerequisites in the root project plan are met.

**Observed result:** Formal admission, D0 integrity, full-domain evaluation, atomic publication, archive SHA, and the 108-file local manifest audit passed. The decision iteration returned `AUROC(N)=0.547598`, best component `A=0.545626`, and gain `0.001972`; both frozen inequalities failed. C1 is stopped, no stage tag is created, and no further Tool Room transform/constant search is authorized. The frozen failure anatomy and the diagnostic-only independent-evidence feasibility gate are recorded in `docs/research/2026-09-28-softcal-v4-g1-failure-anatomy.md`.

## Plan Self-Review Record

- Spec coverage: fixed formulas/constants, state version 4, training-resume rejection, formal admission, confirmation chronology/SHA, 500-smoke evidence boundary, synthetic and real topology evidence, unchanged G1, stop rule, and E3 limitations each map to a named task.
- Step scan: each checkbox is one test, implementation, verification, commit, or explicit authorization action; no implementation task contains an unresolved algorithm choice.
- Type consistency: Task 3 RED names exactly the interfaces Task 4 produces; Task 5 consumes the same `expected_evidence_version=4` loader and confirmation validator; Tasks 8–9 consume the canonical record/digest generated by Task 4.
- Review Focus: all five listed failure classes have explicit negative tests or experiment gates in their owning tasks.
- Proportion: code bodies are limited to the two approved formulas and contract assertions; the plan specifies signatures and evidence rather than transcribing implementation.
- Error log: two discovery `rg` commands initially used malformed grouped patterns under PowerShell; subsequent discovery used fixed-string/separate patterns. No repository file was changed by those failed reads.
