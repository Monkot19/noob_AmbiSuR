# Prior-Risk Utility Transfer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fail-closed, read-only Utility Room seeds-0/1/2 replication pipeline that tests whether frozen `1-r_p` adds stable conditional predictive information beyond `[A,1-S]`, without modifying production training or authorizing routing/C1.

**Architecture:** Three isolated layers keep chronology auditable: Utility source/DA3 snapshot admission, pre-training canonical confirmation plus GT-free run qualification, and a compact post-firewall multi-seed evaluator. The statistical layer reuses only frozen generic primitives from the Tool Room complementarity probe and owns new three-seed aggregation/decision semantics; the evaluator cannot parse Utility GT until both the prior-transfer confirmation and an immutable geometry candidate-or-termination release have been reloaded and verified.

**Tech Stack:** Python 3.10, NumPy 1.26.3, PyTorch 2.7.1+cu128, Open3D 0.18.0, Pillow, standard-library `unittest`, existing COLMAP readers, JSON/CSV/SHA256, Git, Bash launchers, one AutoDL RTX 4090 24 GB.

**Spec:** `docs/superpowers/specs/2026-09-30-prior-risk-utility-transfer-design.md`

## Global Constraints

- This plan implements only the approved prior-risk transfer line. `T_g`, `r_g`, `1-r_g`, `K`, `Delta`, temporal features, alternate transforms, interactions, and learned gates never enter a model or decision.
- Models are exactly `M0=[A,1-S]` and `M1=[A,1-S,1-r_p]`; production formulas, Evidence v4, training losses, gradients, topology, checkpoints, and routing remain unchanged.
- Utility training is GT-free. Before the shared GT firewall releases, GT access is limited to path existence, byte size, and SHA256; no mesh parser, sampler, query, render, summary, or GT-derived preprocessing is permitted.
- Utility GT content may be accessed only after reloading both the canonical prior-transfer confirmation and either an approved Geometry Stage G-C candidate confirmation or a detached-SHA `NO_ACTION_SPECIFIC_SIGNAL` termination confirmation that permanently prevents Utility feedback from reopening geometry research.
- The uploaded source has exactly 147 images and a one-to-one text-COLMAP registration. Loader arrays are exactly 147 `estimated_depths/<complete image filename>.npy` and 147 `estimated_confs/<complete image filename>.npy`; depth `.jpg` previews are allowed but are manifest-tracked and not counted as training arrays.
- DA3 runs once from a reviewed preprocessing confirmation. Seeds 0/1/2 reuse one immutable snapshot; regeneration, repair, realignment, or argument change creates a new snapshot and invalidates the old experiment confirmation.
- The canonical experiment confirmation is created after snapshot freeze while every seed run/view/state/launcher path and every probe target is absent. It is never rewritten after a run starts.
- Three runs are fixed at Utility Room, resolution 2, seeds 0/1/2, 7,000 iterations, refresh 1,000, evaluation 1,000–7,000, checkpoints 3,000/7,000, shadow diagnostics only, Evidence v4.
- Iteration 7,000 is the only performance decision. Iteration 3,000 contributes only raw-risk direction-stability gates and descriptive model metrics.
- Evaluation uses every finite center with `V_p=True`, full valid-mesh distance, strict `distance>0.05 m`, and no opacity/scaling/visibility/frustum/AABB/state/crop/distance filter. M0 and M1 use identical rows and folds.
- Cross-fitting, solver, numerical-independence, direction, voxel bootstrap, seed aggregation, thresholds, and three outcomes remain exactly as approved; no CLI flag may tune them.
- GT mesh identity/alignment/coverage failure returns `INCONCLUSIVE`; no realignment, rescaling, crop, AABB filter, component selection, or domain repair is allowed.
- Every external phase requires a separate explicit authorization. Approval of this plan does not authorize implementation, server source audit, DA3, confirmation creation, training, GT access/evaluation, five-state arbitration, or C1.
- Preserve test-only RED and minimal GREEN in separate commits. Every implementation task ends clean and reviewable; do not combine unrelated modules.

## Frozen Mesh-Admission Constants

The canonical confirmation binds these values before any Utility GT parse:

- coordinate transform: exact `4x4` identity; world unit: meter;
- sparse-point sample: up to 50,000 finite registered `points3D` rows selected by ascending SHA256 of `(point_id, xyz_bytes)`; use all if fewer;
- sparse-point-to-mesh requirements: median distance `<=0.05 m`, 90th percentile `<=0.15 m`, and fraction at `<=0.10 m` `>=0.80`;
- camera coverage rays: fixed full-frame `8 x 6` pixel-center grid for every one of 147 registered cameras;
- coverage requirements: aggregate ray-hit fraction `>=0.80`, at least 90% of cameras have per-camera hit fraction `>=0.50`, and every camera has finite origin/directions;
- all finite, non-degenerate mesh triangles remain in the evaluation mesh; admission statistics never select evaluation rows or triangles.

## File Structure

- Create `reliability/utility_snapshot.py`: pure source audit, canonical manifests, preprocessing-confirmation validation, and post-DA3 snapshot admission.
- Create `scripts/diagnostics/audit_utility_source.py`: read-only source CLI with no GT argument.
- Create `scripts/diagnostics/prepare_utility_da3_confirmation.py`: canonical pre-DA3 record; never invokes DA3.
- Create `scripts/diagnostics/finalize_utility_da3_snapshot.py`: post-DA3 audit and immutable snapshot record.
- Create `reliability/g1_prior_transfer.py`: strict domain, per-seed OOF comparison, multi-seed bootstrap, direction gates, schema, and exhaustive decision.
- Create `reliability/g1_prior_transfer_confirmation.py`: post-snapshot/pre-target confirmation plus run and publication target binding.
- Create `scripts/diagnostics/create_g1_prior_transfer_confirmation.py`: canonical confirmation CLI; GT metadata only.
- Create `reliability/prior_transfer_assets.py`: GT-free per-seed completion/timeline/checkpoint/topology/optimizer qualification.
- Create `scripts/diagnostics/audit_prior_transfer_run.py`: one-run qualification CLI.
- Create `reliability/utility_gt_firewall.py`: geometry-release admission and frozen mesh alignment/coverage audit.
- Create `scripts/diagnostics/evaluate_g1_prior_transfer.py`: three-run admission, first-GT-access ordering, six joined evaluations, compact atomic publication, and exit codes.
- Create matching focused tests under `tests/`; do not modify production `train.py`, renderer/CUDA, evidence, arbitration, or lifecycle files.

## Review Focus

- **Filename/camera drift:** reject case-only differences, duplicate registrations, unsupported camera models, non-finite poses/points, or arrays named without the full image extension (Tasks 1–2).
- **Chronology leakage:** reject a confirmation written after any run/view/state/probe target, a rewritten confirmation, or any GT parser call before both firewall releases (Tasks 4, 7, 9).
- **Row/model asymmetry:** prove distances are queried for all finite centers before `V_p` selection and that M0/M1 share rows, folds, preprocessing, weights, and OOF inventory (Task 3 and Task 9).
- **Spatial/seed pseudoreplication:** prove per-seed fixed-origin voxel resampling, replicate-wise three-seed macro averaging, no row pooling, no seed resampling, and no refitting (Task 3).
- **Plausible but invalid assets:** reject lazy-Adam misunderstandings, mixed Evidence versions, absent topology change, bad mesh alignment/coverage, classless folds/replicates, solver non-convergence, mutation, overwrite, or partial publication (Tasks 5–10).

---

### Task 1: Utility Source Audit Contract

**Files:**
- Create: `tests/test_utility_snapshot.py`
- Create: `tests/test_utility_source_cli.py`
- Create: `reliability/utility_snapshot.py`
- Create: `scripts/diagnostics/audit_utility_source.py`

**Interfaces:**
- Produces: `UtilitySourceAudit`, `audit_utility_source(source_root: Path, expected_count: int = 147) -> UtilitySourceAudit`, `source_manifest(audit: UtilitySourceAudit) -> dict`, and CLI `run_audit(args) -> tuple[int, dict]`.
- Consumes: existing text-COLMAP readers and Pillow image headers; no GT path or DA3 output.

- [ ] **Step 1: Write the failing source-audit tests**

Use parameterized small fixtures plus one 147-name inventory fixture. Assert exact disk/registration basename and case equality, unique image records, referenced camera IDs, `PINHOLE`/`SIMPLE_PINHOLE` only, image-header dimensions equal COLMAP dimensions, finite normalized quaternions/translations/points, and deterministic manifest ordering. Include negative tests for fisheye, case drift, unregistered/extra/missing images, duplicate names, bad camera IDs, non-finite poses/points, unreadable images, and a pre-existing derived directory.

- [ ] **Step 2: Write the failing CLI/no-GT tests**

Assert required `--source-root`, `--expected-count`, `--output`, and source SHA fields; reject unsafe/non-absolute paths and overwrite. Assert parser/help contains no GT/mesh option, mocks prove no access outside `images/` and `sparse/0/`, input re-fingerprinting detects mutation, and canonical JSON plus detached SHA are written atomically.

- [ ] **Step 3: Run RED and commit tests only**

Run: `python -B -m unittest tests.test_utility_snapshot tests.test_utility_source_cli -v`

Expected: import failures for `reliability.utility_snapshot` and/or `scripts.diagnostics.audit_utility_source`, with fixtures themselves loadable.

Commit: `test: specify Utility source admission`

- [ ] **Step 4: Implement the minimal source audit and CLI**

Implement the exact interfaces above. Hash only canonical uploaded source files, use no path guesses, and write outside the source tree. Do not create a snapshot, import DA3, or inspect GT.

- [ ] **Step 5: Run GREEN and commit**

Run the Task 1 tests plus relevant COLMAP/dataset-reader tests and `py_compile`. Expected: all pass, deterministic bytes on repeat, source fixture unchanged.

Commit: `feat: add Utility source admission`

### Task 2: DA3 Preprocessing Confirmation and Snapshot Finalizer

**Files:**
- Modify: `tests/test_utility_snapshot.py`
- Create: `tests/test_utility_da3_cli.py`
- Modify: `reliability/utility_snapshot.py`
- Create: `scripts/diagnostics/prepare_utility_da3_confirmation.py`
- Create: `scripts/diagnostics/finalize_utility_da3_snapshot.py`

**Interfaces:**
- Produces: `build_da3_confirmation(...) -> dict`, `write_da3_confirmation(record, path: Path) -> dict`, `load_da3_confirmation(path: Path, expected_sha256: str) -> dict`, `audit_da3_snapshot(snapshot_root: Path, confirmation: Mapping) -> dict`, and `write_snapshot_record(...) -> dict`.
- Consumes: Task 1 source audit/manifest, explicit repository commit/environment/DA3 checkpoint SHA, explicit `max_points` and `ransac_thresh`, absent staging/final targets, and the exact normalized command using `scripts/run_da3_single.sh`.

- [ ] **Step 1: Write RED tests for pre-DA3 chronology**

Assert the confirmation binds clean exact commit, DA3 model/checkpoint bytes, Python/Torch/CUDA/environment, source manifest, explicit numeric arguments, normalized command, staging/final paths, and target absence. Reject defaults for preprocessing arguments, source-tree output, existing target, malformed SHA, dirty identity, overwrite, changed source, or a command containing GT.

- [ ] **Step 2: Write RED tests for post-DA3 admission**

Assert exactly one finite numeric depth and confidence array named `<complete image filename>.npy` per 147 registered images; shapes must match the registered image or the documented DA3 downsample shape with exact scale metadata. Require loadable finite `sparse_da3/0`, loadable `sparse_da3_aligned/0`, finite positive alignment scale/transform, unchanged source manifest, and a complete derived manifest containing previews and every other derived byte. Reject missing/extra/mis-cased arrays, NaN/Inf, wrong shapes, invalid models, mutation, partial staging, and regeneration under the old ID.

- [ ] **Step 3: Write RED CLI/atomicity tests and commit**

Assert neither CLI invokes DA3 implicitly. The prepare CLI writes confirmation only; the finalizer consumes an already completed derived tree and atomically writes a new snapshot record/detached SHA without overwrite. Both parsers have no GT argument and clean staging after failure.

Run: `python -B -m unittest tests.test_utility_snapshot tests.test_utility_da3_cli -v`

Expected: new interface failures only while Task 1 remains green.

Commit: `test: specify Utility DA3 snapshot contract`

- [ ] **Step 4: Implement confirmation/finalization GREEN**

Keep command execution outside the module. Snapshot identity is SHA256 over the canonical record containing source, environment, command, alignment metadata, and complete derived manifest; any byte/argument change yields a new ID.

- [ ] **Step 5: Verify and commit**

Run Task 1–2 suites, dataset-reader regressions, and static compile. Expected: exact `.npy` loader inventory and preview-inclusive manifest pass without reading GT.

Commit: `feat: freeze Utility DA3 snapshots`

### Task 3: Pure Three-Seed Prior-Transfer Statistics

**Files:**
- Create: `tests/test_g1_prior_transfer.py`
- Create: `reliability/g1_prior_transfer.py`
- Do not modify: `reliability/g1_complementarity.py`

**Interfaces:**
- Consumes: `G1IterationInputs`, full-finite-center distances, and stable generic functions from `reliability.g1_complementarity` only where semantics match exactly.
- Produces: `PriorTransferConfig`, `PriorTransferDomain`, `build_transfer_domain`, `crossfit_transfer_models`, `raw_prior_direction`, `paired_seed_voxel_bootstrap`, `build_transfer_report`, `validate_transfer_report`, flat CSV adapters, and `transfer_exit_code`.

- [ ] **Step 1: Write RED tests for the common domain**

Assert strict snapshot/checkpoint row identity; denominator all finite centers; selection finite and `V_p=True`; coverage exact; distance queried before selection; fixed `distance>0.05`; identical M0/M1 rows/folds. Prove opacity, scaling, visibility, frustum, AABB, state, crop, and distance cannot filter. Reject non-boolean validity, row permutations, non-finite selected evidence, or misaligned distances.

- [ ] **Step 2: Write RED tests for folds, solver, and direction**

Freeze five equal-count longest-axis slabs with x/y/z tie order and original-row tie breaks, fold-local mean/population-scale/balanced weights, float64 Newton/IRLS constants from the spec, and strict column residual `>1e-8`. Assert 3000 contributes only highest-vs-lowest quintile and Spearman direction gates; its AUROC/AUPRC gains are descriptive and cannot change the outcome. Assert 7000 requires every seed’s separation `>=0.05`.

- [ ] **Step 3: Write RED tests for seed-local paired bootstrap**

Use three seeds with different row/voxel counts. Assert fixed-origin 0.5 m voxels, `SeedSequence([20260930, iteration, training_seed])`, exactly 2,000 replicates, paired M0/M1 multiplicities, no refit, no row pooling, no seed resampling, and replicate-wise arithmetic mean across the three seed gains. Classless/non-finite replicates are `INCONCLUSIVE`, never redrawn.

- [ ] **Step 4: Write RED tests for exhaustive decisions**

Assert exact outcomes/exit codes: `PRIOR_RISK_TRANSFER_SUPPORTED/0`, `NO_CROSS_SCENE_REPLICATION/1`, `INCONCLUSIVE/2`. PASS requires per-seed 7000 coverage `>=0.80`, all direction/independence/solver gates, every seed 7000 gain non-negative, three-seed mean gain `>=0.02`, and macro interval lower bound strictly `>0.005`. Assert report claims cannot authorize C1, routing, causal/statistical independence, or broad generalization.

- [ ] **Step 5: Run RED and commit tests**

Run: `python -B -m unittest tests.test_g1_prior_transfer -v`

Expected: import failure for the new module only.

Commit: `test: specify Utility prior transfer statistics`

- [ ] **Step 6: Implement GREEN, regress frozen Tool probe, and commit**

Implement the interfaces without adding tunable parameters or changing `g1_complementarity.py`. Run:

```bash
python -B -m unittest \
  tests.test_g1_prior_transfer \
  tests.test_g1_complementarity \
  tests.test_g1_metrics -v
```

Expected: all pass; frozen Tool v2 semantics unchanged.

Commit: `feat: add Utility prior transfer statistics`

### Task 4: Canonical Post-Snapshot, Pre-Target Confirmation

**Files:**
- Create: `tests/test_g1_prior_transfer_confirmation.py`
- Create: `tests/test_g1_prior_transfer_confirmation_cli.py`
- Create: `reliability/g1_prior_transfer_confirmation.py`
- Create: `scripts/diagnostics/create_g1_prior_transfer_confirmation.py`

**Interfaces:**
- Produces: `build_prior_transfer_confirmation(...) -> dict`, `write_prior_transfer_confirmation(...) -> dict`, `load_prior_transfer_confirmation(path: Path, expected_sha256: str) -> dict`, `validate_preregistration_targets(record) -> None`, and `validate_completed_run_binding(record, run_records) -> None`.
- Consumes: exact code commit/clean status, Task 2 snapshot record/SHA, GT path-size-SHA metadata only, seeds/commands/config, frozen statistics/mesh-admission constants, planned run/view/state/launcher/probe paths, and shared firewall policy.

- [ ] **Step 1: Write RED tests for exact record content**

Require exactly seeds `{0,1,2}`, resolution 2, iterations 7000, refresh 1000, evaluations 1000–7000, checkpoints 3000/7000, Evidence v4, shadow-only feature, no GT in training argv, identical snapshot, exact models/solver/folds/bootstrap/gates/outcomes, the frozen mesh-admission constants, and both accepted geometry-release types.

- [ ] **Step 2: Write RED tests for chronology and target absence**

Require the snapshot record to predate confirmation; all three run/view/state/launcher paths and every probe/staging/publication target absent at build and write time. Reject missing/duplicate seeds, shared mutable run paths, pre-existing targets, confirmation overwrite, changed source/snapshot/GT metadata, dirty/wrong commit, a training command containing GT, and any confirmation created after a target.

- [ ] **Step 3: Write RED tests for immutable later admission**

Assert the evaluator later reloads the exact same bytes/SHA and binds completed runs at the predeclared paths without rewriting the record. A structurally equivalent replacement or later confirmation is rejected. Parser exposes no formula, model, fold, solver, bootstrap, threshold, seed, or iteration overrides.

- [ ] **Step 4: Run RED and commit tests**

Run: `python -B -m unittest tests.test_g1_prior_transfer_confirmation tests.test_g1_prior_transfer_confirmation_cli -v`

Expected: missing-module/script failures only.

Commit: `test: specify Utility transfer confirmation`

- [ ] **Step 5: Implement GREEN and commit**

Use canonical JSON, detached SHA, strict field inventory, absolute paths, and no-overwrite atomic write. GT handling reads metadata supplied by the caller; it must not open or parse mesh content.

Run focused tests plus existing `tests.test_g1_confirmation`. Expected: all pass and old formal confirmation remains unchanged.

Commit: `feat: add Utility transfer confirmation`

### Task 5: GT-Free Per-Seed Asset Qualification

**Files:**
- Create: `tests/test_prior_transfer_assets.py`
- Create: `tests/test_prior_transfer_assets_cli.py`
- Create: `reliability/prior_transfer_assets.py`
- Create: `scripts/diagnostics/audit_prior_transfer_run.py`

**Interfaces:**
- Produces: `qualify_prior_transfer_run(run_dir: Path, seed: int, confirmation: Mapping) -> dict` and CLI `run_audit(args) -> tuple[int, dict]`.
- Consumes: one completed predeclared run, its view/state/launcher records, the original confirmation, existing timeline/offline-join/checkpoint parsers, source/snapshot fingerprints, and process/log metadata.

- [ ] **Step 1: Write RED completion and identity tests**

Assert exact seed/path/commit/config/snapshot binding; exit 0; one completion marker; absent launcher/training/sentinel; unchanged source/prior manifests; no GT path/token; no error/non-finite token; seven ordered refreshes; and no replacement/rerun of a scientifically completed seed.

- [ ] **Step 2: Write RED checkpoint/topology/optimizer tests**

Require Evidence v4 at 3000/7000, strict checkpoint/snapshot row joins, finite centers, real topology change across the seven counts, refresh counts/iterations, temporal state round trip, and lazy Adam reconciliation that permits a dormant group with no state while requiring every active group at the schedule-derived step.

- [ ] **Step 3: Write RED parser/no-GT/publication tests**

Parser has no GT/mesh argument. Audit writes only a canonical compact qualification JSON/SHA outside the run, never changes the run/view/snapshot, and treats malformed/mutated assets as failure rather than repairing them.

- [ ] **Step 4: Run RED, implement GREEN, verify, and commit separately**

RED commit: `test: specify Utility transfer run qualification`

GREEN tests include existing timeline/offline/checkpoint/topology regressions and static compile.

GREEN commit: `feat: qualify Utility transfer runs`

### Task 6: Shared Geometry-Release Firewall

**Files:**
- Create: `tests/test_utility_gt_firewall.py`
- Create: `reliability/utility_gt_firewall.py`

**Interfaces:**
- Produces: `load_geometry_release(path: Path, expected_sha256: str) -> dict`, `validate_geometry_release(record: Mapping) -> str`, and `authorize_first_gt_access(prior_record, geometry_record, *, access_log_path: Path) -> dict`.
- Consumes: Task 4 prior confirmation and exactly one immutable geometry release.

- [ ] **Step 1: Write RED candidate-release tests**

Accept only a Stage G-C record that binds an approved action, cluster unit, outcome/horizon, unique formula/constants, validity/state/topology, thresholds, metrics, coverage, compute budget, stop rules, exact commit, and detached SHA. Reject draft G-A/G-B documents, missing fields, altered bytes, or a release created after any Utility GT-derived artifact.

- [ ] **Step 2: Write RED terminal-release tests**

Accept only canonical outcome `NO_ACTION_SPECIFIC_SIGNAL` with exact stage/evidence/spec/commit identity, `candidate_admitted_to_utility=false`, and `utility_cannot_reopen=true`. Explicitly reject the earlier `NO_SEMANTIC_REPAIR_JUSTIFIED` audit as insufficient by itself, any alternate outcome, or any record allowing restoration after Utility feedback.

- [ ] **Step 3: Write RED ordering/log tests**

Mock the GT parser and prove it is unreachable until both records reload and hash-verify. First access writes one immutable log binding both identities before parser invocation; pre-existing conflicting log, later record, mutation, or absent release fails closed.

- [ ] **Step 4: Run RED, implement GREEN, and commit separately**

RED commit: `test: specify Utility GT firewall`

GREEN commit after focused confirmation/hash/atomicity tests: `feat: enforce Utility GT firewall`

### Task 7: Fail-Closed Utility Mesh Admission

**Files:**
- Modify: `tests/test_utility_gt_firewall.py`
- Modify: `reliability/utility_gt_firewall.py`

**Interfaces:**
- Produces: `audit_utility_mesh(mesh_path: Path, source_root: Path, confirmation: Mapping) -> UtilityMeshAdmission`.
- Consumes: a successful Task 6 access token, exact mesh identity, full validated triangles, source COLMAP cameras/points, and the confirmation-frozen constants.

- [ ] **Step 1: Write RED identity/surface tests**

Reject wrong path/size/SHA, unloadable mesh, non-finite vertices, or a mesh with no finite non-degenerate triangles. Assert the valid surface retains every finite non-degenerate triangle and applies exact identity/no scale/no transform.

- [ ] **Step 2: Write RED alignment/coverage tests**

Use hand-built meshes, sparse points, and cameras to verify deterministic SHA-minhash point sampling, exact point-distance thresholds, 8x6 pixel-center rays, aggregate/per-camera hit gates, and all 147 camera inventory. Prove admission summaries do not return row-selection masks or transformed/cropped meshes.

- [ ] **Step 3: Write RED fail-closed outcome tests**

Any identity/alignment/coverage failure yields `INCONCLUSIVE` before model fitting/distance evaluation. Assert no repair path can realign, rescale, crop, AABB-filter, select a component, or change the sample domain.

- [ ] **Step 4: Run RED, implement GREEN, and commit separately**

RED commit: `test: specify Utility mesh admission`

GREEN regression includes `tests.test_g1_geometry` and full-mesh boundary tests.

GREEN commit: `feat: add Utility mesh admission`

### Task 8: Three-Run Evaluator and Compact Atomic Publication

**Files:**
- Create: `tests/test_g1_prior_transfer_cli.py`
- Create: `scripts/diagnostics/evaluate_g1_prior_transfer.py`

**Interfaces:**
- Produces: `run_evaluator(args, dependencies=None) -> tuple[int, dict]` and `build_parser()`.
- Consumes: Tasks 3–7, six `load_g1_iteration(..., expected_evidence_version=4)` results, existing exact point-to-triangle query, immutable fingerprints, and manifest helpers.

- [ ] **Step 1: Write RED request and admission tests**

Require exact diagnostic commit, prior confirmation path/SHA, three qualification records, one geometry release path/SHA, source/snapshot/GT identities, output root, and safe diagnostic ID. Parser exposes no seed/model/candidate/iteration/fold/solver/bootstrap/threshold/crop/transform override. Reject output under any immutable input or an existing target.

- [ ] **Step 2: Write RED call-order and full-domain tests**

With dependency spies, require: reload prior confirmation; bind all three completed runs; reload geometry release; write/verify first-access log; admit mesh; load six assets; query mesh distance for every finite center; only then apply `V_p=True`; run identical M0/M1 rows/folds. Prove no training, collector, evidence refresh, render, or production write is called.

- [ ] **Step 3: Write RED multi-seed report tests**

Assert per-seed 3000 direction gates and descriptive metrics, per-seed 7000 primary results, replicate-wise macro bootstrap, exact decision gates, solver/fold diagnostics, coverage, mesh admission, provenance, and null/false C1/causal/independence/generalization claims. Damaged but safely characterizable inputs publish `INCONCLUSIVE`; unsafe path/mutation publishes nothing.

- [ ] **Step 4: Write RED compact publication tests**

Require exactly `inputs.json`, `mesh_admission.json`, `report.json`, `seed_folds.csv`, `risk_bins.csv`, `bootstrap.csv`, and `manifest.json`. Verify canonical ordering, file byte counts/SHA, no overwrite, staging cleanup, and before/after fingerprints. Forbid PNG, PLY, checkpoint/snapshot copies, training outputs, and archive creation.

- [ ] **Step 5: Run RED and commit tests**

Run: `python -B -m unittest tests.test_g1_prior_transfer_cli -v`

Expected: import failure for the evaluator script only.

Commit: `test: specify Utility prior transfer publication`

- [ ] **Step 6: Implement GREEN and commit**

Implement strict orchestration and atomic publication without adding fallbacks. Run Task 3–8 tests plus existing confirmation, geometry, offline-input, complementarity, and publication regressions; run `py_compile`.

Commit: `feat: publish Utility prior transfer probe`

### Task 9: Whole-Branch Verification and Review

**Files:**
- Modify after verified implementation only: `task_plan.md`
- Modify after verified implementation only: `findings.md`
- Modify after verified implementation only: `progress.md`

**Interfaces:**
- Consumes: Tasks 1–8.
- Produces: one reviewed exact implementation commit eligible for later server qualification; no data or scientific result.

- [ ] **Step 1: Run local full verification**

Run full `unittest discover`, all focused suites, every relevant explicit CUDA test when hardware exists, `py_compile`, parser `--help`, `git diff --check`, and clean status. Record exact skips/failures; focused tests cannot substitute for discovery.

- [ ] **Step 2: Review the whole branch**

Use `superpowers:requesting-code-review` against the approved spec and this plan. Fix every Critical/Important finding with a new RED/GREEN commit and rerun Step 1. Confirm no production method file changed.

- [ ] **Step 3: Freeze exact qualification identity**

Record the 40-character commit, complete test inventory, schemas, parser surfaces, and no-data/no-training/no-GT status. Prepare but do not execute a server qualification command.

Commit: `docs: freeze Utility transfer implementation`

### Task 10: Separately Authorized Server Phases

**Files:**
- No experiment artifact is committed to Git.
- Update the three planning files only after each authorized gate completes.

**Interfaces:**
- Consumes: the exact reviewed Task 9 commit.
- Produces: user-operated, one-phase-at-a-time commands and immutable audit records; approval of this plan grants none of these executions.

- [ ] **Gate A — Exact-commit server qualification**

Request authorization, then verify detached exact commit, clean worktree, no training, Python/NumPy/Torch/CUDA/Open3D/Pillow, focused/full/CUDA tests, compile, and parser surfaces. Stop for review.

- [ ] **Gate B — Source-upload audit**

Request authorization, then execute only Task 1 against the uploaded 147-image source. GT access is path/size/SHA metadata only and may be recorded by an outer shell without passing GT to the audit CLI. Publish source record/SHA; stop for review.

- [ ] **Gate C — DA3 preprocessing confirmation**

Request authorization, choose and review explicit `max_points`, `ransac_thresh`, DA3 checkpoint identity, environment, command, and absent new snapshot target. Write/reload the confirmation only; do not run DA3. Stop for review.

- [ ] **Gate D — One-time DA3 and snapshot admission**

Request authorization, reload Gate C, create the private preprocessing tree, run exactly its command once, then finalize the snapshot. Any failure stops; no repair in place. Publish snapshot record/SHA; stop for review.

- [ ] **Gate E — Canonical transfer confirmation**

While all seed run/view/state/launcher and probe targets remain absent, request authorization and write the Task 4 confirmation. Reload bytes/SHA and reprove target absence. No training or GT parse; stop for review.

- [ ] **Gates F0/F1/F2 — Three GT-free seeds**

For each seed in order 0, 1, 2: request its launch authorization, reload the same confirmation/snapshot, create only that seed’s predeclared private view/run/state, train once, complete Task 5 qualification, and stop for review before the next seed. A completed scientific seed is never replaced.

- [ ] **Gate G — Geometry release admission**

Request authorization to create or admit exactly one immutable Stage G-C candidate release or terminal `NO_ACTION_SPECIFIC_SIGNAL` confirmation. The earlier `NO_SEMANTIC_REPAIR_JUSTIFIED` record alone does not release GT. Stop for review.

- [ ] **Gate H — First Utility GT access and probe**

Only after all three assets and Gate G qualify, request explicit GT-probe authorization. Reload the unchanged transfer confirmation and geometry release, write the first-access log, run mesh admission, then execute the compact evaluator once. Stop immediately with `INCONCLUSIVE` on mesh identity/alignment/coverage failure; do not repair the domain.

- [ ] **Gate I — Publication audit and conclusion freeze**

Verify all seven manifest files and exact outcome. Freeze one of `PRIOR_RISK_TRANSFER_SUPPORTED`, `NO_CROSS_SCENE_REPLICATION`, or `INCONCLUSIVE` without changing a candidate, seed, threshold, split, solver, preprocessing snapshot, or mesh. A positive result authorizes only a new written hypothesis or another independent-scene protocol—not five-state routing, a new scalar `N`, prior-gradient increase, production change, or C1.

## Execution Handoff Boundary

This plan is complete only when reviewed and explicitly approved. Such approval would authorize implementation Tasks 1–9 under TDD, not Task 10 server execution. Every Task 10 gate remains a separate user decision, and no earlier gate carries authority into the next.
