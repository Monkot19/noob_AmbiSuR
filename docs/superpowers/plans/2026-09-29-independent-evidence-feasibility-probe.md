# Independent-Evidence Feasibility Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. The user selected current-task/native TDD execution; do not dispatch subagents unless the user explicitly changes that choice.

**Goal:** Build one read-only, single-candidate, spatially blocked diagnostic that determines whether `1-r_p` adds stable predictive information beyond `A` and `1-S` in the frozen version-4 Tool Room assets.

**Architecture:** A new pure NumPy module owns domain construction, spatial folds, deterministic logistic fitting, paired OOF metrics, voxel bootstrap, direction checks, numerical-independence checks, and the exhaustive three-state decision. A separate CLI admits the existing formal run, computes full-mesh labels through the established offline G1 loader, and atomically publishes only compact JSON/CSV artifacts. Training, formal G1, evidence state, routing, rendering, and checkpoint formats remain untouched.

**Tech Stack:** Python 3.10, NumPy 1.26.3, standard-library `unittest`, existing Torch checkpoint loader, existing Open3D full-mesh distance query, JSON/CSV/SHA256, Git, AutoDL RTX 4090 for the final read-only real-asset run.

**Spec:** `docs/research/2026-09-28-softcal-v4-g1-failure-anatomy.md`, Section 6

## Global Constraints

- The sole decision candidate is raw risk `1-r_p`; `1-K` and temporal transition count are descriptive-only and cannot affect the decision.
- Evaluate iterations 3000 and 7000 separately; 7000 is primary and 3000 is direction-stability evidence only.
- The comparison domain is every finite checkpoint centre with strict snapshot/checkpoint same-row join and `V_p=True`; baseline and augmented models use identical rows and labels.
- Apply no opacity, scaling, visibility, frustum, AABB, or manual-crop filter; the label remains full-valid-mesh distance `>0.05 m`.
- Use deterministic five-fold spatial slabs, training-fold-only class weights and standardization, float64 damped Newton/IRLS, and no hyperparameter selection from GT.
- Primary gain is pooled OOF AUROC difference, not mean fold AUROC. Paired fold deltas remain descriptive.
- Bootstrap uses fixed-origin 0.5 m voxels, paired OOF predictions, NumPy PCG64 seed `20260928`, 2,000 replicates, and a percentile 95% interval without model refitting.
- Outcomes are exactly `INDEPENDENT_EVIDENCE_FEASIBLE`, `NO_CLEAR_COMPLEMENT`, or `INCONCLUSIVE`; exit codes are respectively 0, 1, and 2.
- Existing checkpoints, snapshots, confirmation, source data, GT mesh, and formal reports are read-only. The probe starts no training and cannot authorize C1 or claim causal or cross-scene validity.
- Preserve RED and GREEN evidence in separate commits. Do not run the real Tool Room probe until implementation review, exact-commit qualification, clean-tree/input-hash gates, and explicit execution authorization.

## File Structure

- Create `reliability/g1_complementarity.py`: immutable configuration, domain/fold construction, deterministic solver, cross-fitting, bootstrap, direction and column-space diagnostics, schema validation, and three-state report assembly.
- Create `tests/test_g1_complementarity.py`: hand-checked unit tests for every statistical and decision contract.
- Create `scripts/diagnostics/probe_g1_prior_complementarity.py`: formal-run admission, immutable input binding, full-mesh orchestration, atomic compact publication, CLI, and exit codes.
- Create `tests/test_g1_complementarity_cli.py`: parser, provenance/admission, publication, mutation, no-overwrite, no-training, and failure-mode tests.
- Update `task_plan.md`, `findings.md`, and `progress.md` only after implementation/review and after the separately authorized real probe.

## Review Focus

- Row-domain asymmetry or leakage: prove baseline and augmented predictions cover the exact same finite/`V_p=True` rows, and reject wrong joins or validation-derived preprocessing (Tasks 1–2).
- Spatial pseudoreplication: prove deterministic slab membership and fixed-origin voxel grouping, including negative world coordinates and tied centre coordinates (Tasks 1–2).
- Numerically plausible but invalid fitting: reject classless folds, zero/non-finite scales, singular/non-convergent solves, rejected line-search steps, and non-finite predictions as `INCONCLUSIVE` (Tasks 1–4).
- Miscomputed primary evidence: distinguish pooled OOF AUROC gain from mean fold gain and prove paired voxel resampling uses identical multiplicities without refitting (Tasks 1–2).
- Scientific overclaim or partial publication: enforce exact three-state semantics, null C1/causal/generalization claims, immutable-input re-fingerprinting, no overwrite, and atomic cleanup (Tasks 3–6).

---

### Task 1: Freeze the Pure Statistical RED Contract

**Files:**
- Create: `tests/test_g1_complementarity.py`

**Interfaces:**
- Consumes: `G1IterationInputs` from `reliability.offline_g1` and `binary_curves` from `reliability.g1_metrics`.
- Produces: failing behavioral tests for the future `ProbeConfig`, `build_probe_domain`, `make_spatial_folds`, `raw_risk_direction`, `column_independence`, `fit_logistic`, `crossfit_comparison`, and `paired_voxel_bootstrap` interfaces.

- [ ] **Step 1: Specify strict domain construction and coverage**

Use hand-built joined fixtures containing finite/non-finite centres, non-contiguous original row indices, both `V_p` values, and distinct snapshot values. Assert only finite joined `V_p=True` rows survive, original-row identity is preserved, distances and labels stay aligned, coverage uses all finite centres as the denominator, and opacity/scale-like fixture fields cannot alter selection. Assert malformed row maps, shapes, dtypes, fields, and non-finite selected inputs fail closed.

- [ ] **Step 2: Specify deterministic spatial folds, direction, and column independence**

Assert five equal-count slabs use the retained centres' longest AABB axis with coordinate/original-row tie breaking; every row appears in exactly one validation fold. Cover negative coordinates and ties. Assert raw `1-r_p` quintiles, tie-aware Spearman, and 7000 separation use raw candidate order. Assert the fold-local projection residual detects an exactly dependent candidate and admits a hand-checked independent column at the frozen strict `1e-8` threshold.

- [ ] **Step 3: Specify the frozen solver and paired OOF comparison**

Use a separable-but-regularized literal fixture and assert float64 finite probabilities, deterministic repeatability, fold-local means/scales/weights, the frozen optimizer settings, and matching baseline/augmented validation indices. Add classless, zero-scale, non-finite, Hessian/line-search, and non-convergence cases that raise the dedicated inconclusive error rather than changing solver settings.

- [ ] **Step 4: Specify pooled gain and paired voxel bootstrap**

Use literal OOF predictions where pooled AUROC gain differs from the arithmetic mean of fold gains. Assert the primary value is pooled. Use a small fixed-origin voxel fixture, including negative coordinates, and assert PCG64 seed `20260928`, 2,000 paired replicates, unchanged prediction inputs, same voxel multiplicities for both models, no refits, percentile endpoints, and classless/non-finite replicate failure.

- [ ] **Step 5: Run RED and commit tests only**

Run:

```bash
python -B -m unittest tests.test_g1_complementarity -v
```

Expected: import failure for `reliability.g1_complementarity`; no fixture syntax or unrelated failure.

Commit: `test: specify prior complementarity statistics`

### Task 2: Implement the Pure Statistical Engine GREEN

**Files:**
- Create: `reliability/g1_complementarity.py`
- Test: `tests/test_g1_complementarity.py`

**Interfaces:**
- Consumes: the Task 1 test contract, existing joined G1 inputs, and existing tie-safe `binary_curves`.
- Produces: immutable `ProbeConfig`/`ProbeDomain`/fit-result structures and the pure statistical functions named in Task 1.

- [ ] **Step 1: Implement validated domain and spatial primitives**

Implement strict same-row selection, coverage, fixed label, deterministic five-slab membership, raw quintile/Spearman summaries, fixed-origin voxel IDs, and fold-local relative column residual. Keep all report-facing scalars JSON-safe and all model arrays float64.

- [ ] **Step 2: Implement deterministic damped Newton/IRLS**

Implement weighted mean logistic loss with intercept-exempt slope L2 `1e-4`, zero initialization, exact dense Hessian solve, 100-iteration cap, gradient infinity tolerance `1e-8`, Armijo `1e-4`, shrink `0.5`, and minimum step `2^-20`. Raise `ProbeInconclusiveError` for every frozen invalid-computation condition; provide no fallback.

- [ ] **Step 3: Implement cross-fitting and bootstrap**

Fit baseline `[A,1-S]` and augmented `[A,1-S,1-r_p]` on identical training rows per fold; store every validation prediction exactly once. Compute pooled OOF AUROC/AUPRC and paired fold deltas. Bootstrap existing paired OOF predictions by fixed-origin voxel with PCG64 seed `20260928`, exactly 2,000 replicates, and percentile `[2.5,97.5]`.

- [ ] **Step 4: Run GREEN and focused regression**

Run:

```bash
python -B -m unittest tests.test_g1_complementarity tests.test_g1_metrics tests.test_g1_offline_inputs -v
```

Expected: all pass, with no warnings or skips caused by the new module.

- [ ] **Step 5: Commit implementation**

Commit: `feat: add prior complementarity statistics`

### Task 3: Freeze Report Schema and Three-State RED Contract

**Files:**
- Modify: `tests/test_g1_complementarity.py`

**Interfaces:**
- Consumes: Task 2 per-iteration outputs.
- Produces: failing tests for `build_probe_report`, `validate_probe_report`, `probe_exit_code`, `fold_rows`, `risk_bin_rows`, and `bootstrap_rows`.

- [ ] **Step 1: Specify exact JSON decision schema**

Assert the schema records diagnostic-only scope, sole candidate, fixed configuration, sample-domain/coverage evidence, both iterations, pooled and fold metrics, independence residuals, direction results, bootstrap distribution/interval, provenance, failed gates, and inconclusive reasons. Assert C1 authorization, causal claim, and cross-scene claim are always false/null.

- [ ] **Step 2: Specify exhaustive decisions and exit codes**

Hand-build literal passing, valid-negative, and invalid-computation iteration summaries. Assert all gates are conjunctive; outcomes and exit codes are exactly feasible/0, no-clear-complement/1, and inconclusive/2. Assert 3000 cannot substitute for 7000 and a failure cannot be hidden by a better fold.

- [ ] **Step 3: Specify flat CSV rows and fail-closed validation**

Assert deterministic fold, 20-bin raw-risk, and 2,000-replicate bootstrap row inventories. Reject unknown/missing fields, altered constants, extra candidates, wrong iteration roles, non-finite metrics, wrong bootstrap length/seed/interval, and any report that implies training or C1 authorization.

- [ ] **Step 4: Run RED and commit tests only**

Run: `python -B -m unittest tests.test_g1_complementarity -v`

Expected: failures identify missing report/schema functions while Task 1–2 tests remain green.

Commit: `test: specify complementarity probe decisions`

### Task 4: Implement Report Assembly and Decision GREEN

**Files:**
- Modify: `reliability/g1_complementarity.py`
- Test: `tests/test_g1_complementarity.py`

**Interfaces:**
- Consumes: Task 2 outputs and Task 3 schema tests.
- Produces: validated JSON-safe reports, deterministic CSV-row adapters, and exhaustive status/exit-code mapping.

- [ ] **Step 1: Implement report construction and conjunctive gates**

Encode coverage `>=0.80` at 7000, both direction checks, all-fold relative residual `>1e-8`, 7000 pooled gain `>=0.02`, 95% lower bound `>0.005`, 3000 pooled gain `>=0`, and validity/finite requirements without adding tunable CLI arguments.

- [ ] **Step 2: Implement validation, CSV adapters, and exit codes**

Validate exact fields/constants/types/counts before publication. Preserve all paired bootstrap values for audit, keep fold deltas diagnostic, and map only the three frozen outcomes to 0/1/2.

- [ ] **Step 3: Run GREEN and regression**

Run:

```bash
python -B -m unittest tests.test_g1_complementarity tests.test_g1_metrics -v
```

Expected: all pass.

- [ ] **Step 4: Commit implementation**

Commit: `feat: add complementarity probe decision report`

### Task 5: Freeze CLI Admission and Atomic-Publication RED Contract

**Files:**
- Create: `tests/test_g1_complementarity_cli.py`

**Interfaces:**
- Consumes: Task 4 report functions; existing `load_confirmation_record`, formal run identity/hash records, `load_g1_iteration(..., expected_evidence_version=4)`, `load_valid_mesh`, `closest_triangle_distances`, `fingerprint_inputs`, `build_manifest`, and `assert_inputs_unchanged`.
- Produces: failing tests for `run_probe(args, dependencies=None)` and `build_parser()` in the future CLI.

- [ ] **Step 1: Specify parser and formal asset admission**

Freeze required args for run/source/GT/confirmation/output paths, diagnostic ID, diagnostic commit, confirmation SHA, and dataset/prior/GT hashes. Assert there are no candidate, model, fold, solver, threshold, bootstrap, crop, or iteration override flags. Require clean exact diagnostic code identity and validate the run against the original version-4 confirmation/run identity while keeping the new diagnostic output path distinct.

- [ ] **Step 2: Specify real boundary orchestration with controlled dependencies**

Use complete fake joined 3000/7000 inputs and a fake validated mesh/distance query. Assert both checkpoints require Evidence v4, distances are computed for every finite centre before `V_p` selection, strict row maps reach the pure engine, and no training/render/collector path is called.

- [ ] **Step 3: Specify atomic compact publication**

Require exactly `inputs.json`, `report.json`, `folds.csv`, `risk_bins.csv`, `bootstrap.csv`, and `manifest.json`; verify deterministic ordering and manifest hashes. Assert no archive, PNG, PLY, checkpoint, snapshot copy, or training artifact is produced.

- [ ] **Step 4: Specify no-overwrite, mutation, and three-state CLI behavior**

Assert pre-existing targets and unsafe requests fail without publication, while post-read immutable-input mutation removes staging because no result can be trusted. A readable but damaged/mismatched formal asset, checkpoint Evidence version, row join, fold, or numerical computation becomes an atomically published `INCONCLUSIVE` report before avoidable downstream work. Statuses map to exit 0/1/2, and both valid-negative and safely characterizable inconclusive reports are published.

- [ ] **Step 5: Run RED and commit tests only**

Run: `python -B -m unittest tests.test_g1_complementarity_cli -v`

Expected: import failure for `scripts.diagnostics.probe_g1_prior_complementarity`.

Commit: `test: specify complementarity probe publication`

### Task 6: Implement Read-Only CLI and Atomic Publication GREEN

**Files:**
- Create: `scripts/diagnostics/probe_g1_prior_complementarity.py`
- Test: `tests/test_g1_complementarity_cli.py`

**Interfaces:**
- Consumes: Tasks 2/4 pure engine/report and the existing formal admission/offline geometry/publication helpers.
- Produces: direct script/module entrypoint, compact atomic report directory, manifest, stdout summary, and frozen exit status.

- [ ] **Step 1: Implement request validation and admission**

Resolve safe paths/ID, reject outputs inside immutable inputs, require completed formal run and both 3000/7000 assets, load the canonical confirmation by detached SHA, verify formula/run/config/input hashes and Evidence v4, require exact clean diagnostic commit, and fingerprint every consumed input before computation.

- [ ] **Step 2: Implement full-domain orchestration**

Load both joined iterations, query full valid GT mesh for all finite centres, build the strict `V_p=True` domains, compute iteration reports, and assemble the three-state diagnostic report. Convert declared damaged-input, fold, solver, metric, and bootstrap conditions into a validated `INCONCLUSIVE` report whenever the failure can be safely characterized. Unsafe request paths, overwrite, or input mutation during execution remain fail-closed operational errors with no publication.

- [ ] **Step 3: Implement compact atomic publication and CLI**

Write validated JSON/CSV artifacts to a unique staging directory, re-fingerprint inputs, build/validate the manifest, atomically rename only after all checks pass, print a concise JSON summary, and exit according to the report outcome. Never start training or write under run/source/GT paths.

- [ ] **Step 4: Run GREEN and focused regression**

Run:

```bash
python -B -m unittest tests.test_g1_complementarity_cli tests.test_g1_complementarity tests.test_g1_confirmation tests.test_g1_offline_inputs tests.test_g1_geometry -v
python -B -m py_compile reliability/g1_complementarity.py scripts/diagnostics/probe_g1_prior_complementarity.py tests/test_g1_complementarity.py tests/test_g1_complementarity_cli.py
```

Expected: all pass and compile cleanly.

- [ ] **Step 5: Commit implementation**

Commit: `feat: publish prior complementarity probe`

### Task 7: Whole-Branch Qualification and Real-Probe Gate

**Files:**
- Modify: `task_plan.md`
- Modify: `findings.md`
- Modify: `progress.md`

**Interfaces:**
- Consumes: Tasks 1–6 and the approved Section 6 contract.
- Produces: reviewed exact implementation commit and a separate, user-authorized server command; it does not itself produce a scientific result.

- [ ] **Step 1: Run local full verification**

Run all repository unit tests, explicit relevant GPU tests only where the environment supports them, static compile, `git diff --check`, clean-worktree audit, and parser help. Record every skip/failure by name; do not call the branch complete on focused tests alone.

- [ ] **Step 2: Perform one whole-branch review**

Use `superpowers:requesting-code-review` against the approved spec and this plan. Resolve Critical/Important findings with one TDD fix pass; record deferred minors and rulings. Re-run the full verification suite after fixes.

- [ ] **Step 3: Freeze exact server qualification**

Commit only implementation/review fixes, then prepare a user-operated AutoDL command that checks exact commit, clean detached checkout, no active training, Python/NumPy/Torch/Open3D versions, focused/full tests, static compile, immutable confirmation/data/prior/GT hashes, and absent diagnostic output target. Do not push or execute externally without authorization.

- [ ] **Step 4: Run the real probe only after explicit authorization**

On the exact qualified commit, evaluate existing formal iteration-3000/7000 assets once and publish only the compact bundle. The command must report domain counts, coverage, direction checks, numerical-independence residuals, pooled OOF metrics, fold deltas, bootstrap interval, exact outcome, manifest hashes, no-training status, and clean Git state.

- [ ] **Step 5: Freeze the result without reinterpretation**

Update the three planning files with the exact report identity and outcome. `NO_CLEAR_COMPLEMENT` stops this direction; `INCONCLUSIVE` permits only same-contract measurement repair; `INDEPENDENT_EVIDENCE_FEASIBLE` permits only a new written hypothesis and independent-scene validation plan. None authorizes C1 directly.
