# Geometry Stage G-A TDD Implementation Plan — Review Draft

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task, only after separate approval. Steps use checkbox (`- [ ]`) syntax. Current authorization is document writing only; no implementation or experiment execution.

**Goal:** Build an isolated, single-cluster paired synthetic harness testing the one approved action-specific `Q_g`, not a production reliability/routing module.

**Architecture:** Pure cluster/action/outcome/metric functions feed a branch-private synthetic renderer/optimizer adapter. Existing topology mappings, semantic residual equations and tie-safe binary metrics are reused rather than rewritten. Reference surfaces are available only to the outcome evaluator, never to action/score construction.

**Tech Stack:** Python 3.10, unittest, NumPy, PyTorch/CUDA and existing Gaussian renderer/topology; RTX 4090 qualification. Arithmetic follows the spec's float64 contract; no environment modification is implicit.

**Spec:** `docs/superpowers/specs/2026-09-30-geometry-stage-ga-exact-subspecification.md`; parent `docs/superpowers/specs/2026-09-30-geometry-update-authorization-design.md`.

**Status:** Planning authorized on 2026-10-08. **Not implementation-ready:** Gate 0 below requires a reviewed specification addendum. No approved formula, action, constant or gate is changed by this draft. An approved sub-specification's stale “Draft” heading is not silently rewritten here.

## Gate 0 — Specification consistency, before any implementation

These are document-level issues, not observed model failures. They cannot produce a termination release or justify opening Utility GT.

1. **Certain coverage contradiction (§§3.2,9.1–9.2,10.3).** Twelve azimuths spaced 30 degrees apart yield at most one camera in a 20-degree azimuth sector. The ill-conditioned family therefore cannot satisfy eight retained cameras or four per partition. Its validity coverage is zero under these rules, while every family's positive gate requires at least 0.60. The author must approve a consistent family/support/coverage contract; this plan selects no relaxation.
2. **Planar cluster feasibility (§§2.1,9.1).** An ordinary single-layer 0.02-m planar lattice contains 21 centers within 0.05 m of an interior lattice anchor: 5 on the center column, 10 on the adjacent columns, 6 on the outer columns. It cannot supply 32. Rotating the grid does not change distances. Explicit surface/family/anchor layouts are required; no denser grid, larger radius, duplicate rows or substitute surface is silently introduced.
3. **Exact fixture and horizon closure (§§2.3,6,9).** Freeze the family-to-surface mapping, anchor indices and initial row layout, surface triangulation/cylinder extent, canonical camera names/cluster-key encoding, seeds, eight-camera replay schedule, horizon loss, optimizer groups/hyperparameters and step-4 topology thresholds. Otherwise two engineers can implement different interventions while claiming the same specification. Hidden defaults from the 7k production run are not a freeze.
4. **Differentiable measurement closure (§§3–5).** Define the residual weight normalization denominator and per-camera objective whose translation derivative is `g_v`. Production `collector.py` detaches maps and marks reprojection `no_grad`; it cannot directly supply the exact autograd Jacobian. Approve a research-only differentiable semantic adapter with parity tests and an explicit renderer dtype boundary, or stop if the exact contract cannot be met. No finite-difference action, detached zero Jacobian, altered residual or production patch is an allowed fallback.

- [ ] Obtain one explicit addendum resolving all four items before Task 1 code, fixture generation or outcome inspection.
- [ ] Bind approved charter/sub-specification/addendum SHA256 in every future synthetic confirmation. Until resolved, the tasks below are a bounded task map for review, not executable permission or a complete scientific freeze.

## Global Constraints

- Evidence remains v4. Exclude `T_g`, `r_g`, `1-r_g`, geometry history, Utility data/GT, Tool Room GT and post-action values from action, `Q_g`, validity and selection.
- Exactly one cluster and sequential branches per pair; no batching or concurrent branch execution. Full-scene renders retain non-target interactions.
- 32 immutable pre-action rows, radius 0.05 m, guard distance 0.10 m; finite-only selection with distance/row-index tie breaks. Do not resolve Gate 0 by changing these values.
- Stop-gradient pre-action masks, weights and component inventories; invalid pixels are neither residual zero nor support.
- `epsilon=1e-8`, `tau_d=0.05`, `tau_n=tau_dn=0.10`; support `Z>1e-4`; at most 16 views, at least 8 with 4 per partition and 4 internal directed edges per partition.
- Exact shared-xyz Gauss–Newton action: `H=J.T@J/n+1e-3 I`, `b=J.T@r/n`, raw `-solve(H,b)`, clipped at 0.01 m. At least 256 components/all three families, finite tensors, minimum eigenvalue strictly above `1e-8`.
- Sole score: geometric mean of direction, virtual improvement, view support, conditioning and magnitude; five unchanged equations from spec §5. `V_Q=False -> null/unknown/unauthorized`; sole tail `Q_g>=0.50`.
- Eight completed Adam steps after injection; at most one topology event after step 4; immutable source and private model/optimizer/RNG/schedule per branch.
- 128 fixed pairs (8 families ×16), at most 2,048 optimizer steps. One valid execution is single-use; no candidate/threshold/action/horizon/suite search.
- <=4,096 Gaussians, 512×512 renders, <=12 GiB allocated GPU memory, <=30 seconds/pair, <=60 minutes total excluding one-time import/compile, <=5 GiB output.
- Approval of this plan must not imply synthetic execution, Tool Room intervention, Utility GT release, production changes, five-state or C1.

## Review Focus

1. A null/unsupported residual or zero derivative must not silently become authorization — Tasks 2–3.
2. Cloned tensor/optimizer containers must not share storage with source or the other branch — Task 4.
3. Split counts, disappeared ancestors and non-target descendants must not alter ancestor weighting or hide spillover — Task 5.
4. Family omissions, invalid cases or classless replicates must not be repaired by denominator changes/redraw — Task 6.
5. An engineering exception or incomplete budget run must not manufacture a scientific terminal release — Tasks 7–8.

## File map and reuse

All new implementation below is **future research-only code**, not created by this planning turn:

- `research/geometry_ga/contracts.py`: immutable record types/approved configuration and schema validation.
- `research/geometry_ga/clusters.py`: deterministic rows, regions and camera partition.
- `research/geometry_ga/residuals.py`: isolated differentiable adapter, frozen masks/weights and production semantic parity.
- `research/geometry_ga/action.py`: exact translation solve and unique score.
- `research/geometry_ga/fixtures.py`: approved canonical fixture manifest, no selection from outcomes.
- `research/geometry_ga/branches.py`: private restore/replay and existing topology calls.
- `research/geometry_ga/outcomes.py`: reference-only ancestor outcomes and eight safety gates.
- `research/geometry_ga/report.py`: metrics, family bootstrap, budget and conclusion.
- `scripts/diagnostics/evaluate_geometry_ga.py`: separate approval/confirmation admission, sequential execution, atomic publication.
- Corresponding `tests/test_geometry_ga_{clusters,residuals,action,branches,outcomes,report,cli}.py` and `tests/gpu/test_geometry_ga_backend.py`.

Reuse `reliability/topology.py::TopologyChange/compose_topology_changes`, production clone/split/prune in `scene/gaussian_model.py`, residual equations in `reliability/evidence.py` and `collector.py`, mesh distance helpers in `offline_g1.py`, tie-safe `g1_metrics.py::binary_curves`, and existing manifest conventions. Do not modify frozen complementarity, production training, Evidence or CUDA code to make a research test pass.

## Common task cycle and review

For each Task 1–7, after Gate 0 and explicit implementation approval: write the named tests first; run the task's `python -B -m unittest ... -v` command and record the intended assertion/import RED; commit `test: ...`; implement only the named interface; rerun GREEN plus affected existing regressions; compile changed files; diff-check; commit the independently testable module. No fixture outcomes or predictive experiment belong to a unit-test qualification. Backend qualification is separately approved, not inferred from a mocked test.

### Task 1 — Canonical units and immutable records

**Files:** contracts.py, clusters.py, tests/test_geometry_ga_clusters.py.

**Interfaces:** `build_cluster(centers, original_rows, anchor_row) -> dict`; `partition_views(fixture_id, cluster_key, support, edges) -> dict`. Records carry rows/key/regions and explicit validity reasons, never an imputed score. Contracts fix `PairInput`, `PreActionMeasurements`, `ActionRecord`, `BranchOutcome`, `PairOutcome` schemas; no stage-specific field may reference real GT or telemetry.

- [ ] RED `test_exact_cluster_membership_and_frozen_regions`: assert exactly32 distinct rows, original-row ties, radius boundary, finite exclusion, duplicate-key lowest-anchor retention, immutable guard/far partition.
- [ ] RED: camera hash ordering/encoding follows the approved addendum; cap16, alternating P/H, no cross-partition edge, support threshold strictly `>1e-4`, insufficient cameras invalid.
- [ ] GREEN: implement the deterministic rules; invalid clusters remain in the frozen inventory/coverage denominator. Reject missing addendum identity rather than guessing layouts.
- [ ] Verify `tests.test_geometry_ga_clusters`; commit `feat: add research cluster contracts`.

### Task 2 — Fixed differentiable internal residual boundary

**Files:** residuals.py, tests/test_geometry_ga_residuals.py, GPU backend test.

**Interfaces:** `freeze_residual_inventory(pair_input, partition) -> dict`; `residual_vector(delta, frozen_inventory, render_fn) -> tensor`; `translation_jacobian(residual_fn) -> (r,J)`. Runtime action inputs exclude reference surfaces/RGB and external priors by schema and call signature.

- [ ] RED `test_residual_semantic_parity_and_gradient_path`: same pre-action depth/normal errors and validity as production; sign invariance, foreground occlusion and invalid support; scalar normalization and `g_v` follow the approved addendum.
- [ ] RED: changing visibility cannot remove a frozen component; each partition has >=256 components/all families; invalid values cannot pass through as zero.
- [ ] RED: a hand-derived differentiable test has its exact nonzero Jacobian; gradients reach only the shared translation variable. Source state and fixed weights have no gradients/writes.
- [ ] GREEN: implement the isolated approved adapter, without removing production detach/no-grad. A missing CUDA derivative/dtype capability stops backend qualification; do not substitute finite differences in the action.
- [ ] Verify `tests.test_geometry_ga_residuals` and existing `tests.test_reprojection_reliability tests.test_d0_collector`; GPU parity remains a separately approved command. Commit `feat: add isolated geometry action residuals`.

### Task 3 — Sole action and sole pre-action Q_g

**Files:** action.py, tests/test_geometry_ga_action.py.

**Interfaces:** `propose_translation(r,J) -> dict`; `score_authorization(measurements, action) -> dict`. Consume Task 2 measurements; produce raw/clipped delta, component validity and either a finite Q or null.

- [ ] RED: diagonal hand-checked solve, `n` normalization, regularization1e-3, norm cap0.01, no optimizer advance, non-member/other-parameter byte identity. Missing family, failed solve, nonfinite or insufficient components invalid; zero raw step does not invent benefit.
- [ ] RED: spec §5 exact five formulas, `D_30=(1-cos30°)/2`, virtual scale0.10, conditioning scale0.05, q_mag exponential, minimum combined/P/H support. Increasing virtual harm or disagreement cannot increase its component, other inputs held fixed.
- [ ] RED `test_five_component_anchor_and_unknown_validity`: five components equal0.5 give Q0.5 and authorize; any invalid component gives null; valid zero component gives Q0; no transform/reweight/fallback. Pin the scalar anchor, without inventing an alternative score:

```python
q = (0.5 * 0.5 * 0.5 * 0.5 * 0.5) ** (1 / 5)
self.assertAlmostEqual(q, 0.5)
self.assertTrue(q >= 0.50)
```
- [ ] GREEN: exact autograd solve/score, float64 and existing dtype-appropriate tolerance. Verify `tests.test_geometry_ga_action`; commit `feat: implement frozen research geometry action`.

### Task 4 — Fixed fixtures and private paired replay

**Files:** fixtures.py, branches.py, tests/test_geometry_ga_branches.py; GPU backend test.

**Interfaces:** `build_fixture_manifest(approved_contract) -> dict`; `restore_branch(pair_input) -> branch`; `run_branch(branch, action, schedule) -> dict`. Manifest fixes every anchor, surface, perturbation, schedule and random seed before any Y; Task 1 defines record schemas.

- [ ] RED `test_canonical_inventory_and_independent_restore`: exact8×16 inventory and approved surface/anchor assignment; variant rotation22.5 degrees; topology variants four each survivor/clone/split/prune. No output-derived selection, omission or replacement.
- [ ] RED: model/Adam/topology/RNG/schedule tensors restored independently and byte-identically; no shared storage; injection precedes first render and does not count as a step; exactly8 completed steps, topology only after4.
- [ ] RED: source mutation, replay mismatch and unsupported mapping invalidate pair; full scene retained; target/non-target states may diverge indirectly under identical rules and are measured, not forced equal after horizon.
- [ ] GREEN: reuse production topology operations via an isolated harness; no change to train.py or Gaussian production methods. Verify `tests.test_geometry_ga_branches` plus topology migration/composition; commit `feat: add private synthetic paired replay`.

### Task 5 — Ancestor-balanced benefit, safety and spillover

**Files:** outcomes.py, tests/test_geometry_ga_outcomes.py.

**Interfaces:** `lineage_errors(ancestors, mapping, centers, reference) -> dict`; `evaluate_pair(control, treated, frozen_regions) -> dict`. Only this boundary receives synthetic reference surfaces/RGB. No action/score recomputation from outcomes.

- [ ] RED `test_ancestor_weighting_and_disappearance`: exact triangle distances; per-ancestor RMS descendant distance capped0.10 m, disappeared=0.10 m; equal ancestor weight, not descendant weight; invalid/unmapped child returns measurement failure.
- [ ] RED: tau=control−treated; benefit `>=max(0.001,0.05*Y_control)` at step8 only; static correctness or best intermediate step cannot substitute.
- [ ] RED: eight independent safety tests: finite/cap, p95 increase<=0.001, no treatment-only disappearance, RGB increase<=max(0.002,0.01*control), guard<=max(0.001,0.02*control), far<=max(0.0005,0.01*control), injection isolation, legitimate topology/Adam integrity. Boundary equality passes; a finite unsafe action is negative even with positive mean benefit.
- [ ] GREEN: evaluate exact frozen regions through descendants; an empty protected region needs the reviewed Gate 0 contract, not an invented average. Verify `tests.test_geometry_ga_outcomes` and existing geometry tests; commit `feat: evaluate ancestor geometry action outcomes`.

### Task 6 — Frozen uncertainty and all decision gates

**Files:** report.py, tests/test_geometry_ga_report.py.

**Interfaces:** `summarize_pairs(pair_rows, manifest, budget) -> dict`; outcome exactly `GEOMETRY_AUTHORIZATION_HYPOTHESIS_READY`, `NO_ACTION_SPECIFIC_SIGNAL` or `INCONCLUSIVE`.

- [ ] RED `test_complete_denominators_and_outcome_precedence`: denominator128 and per-family16 including invalids; AUROC/AUPRC valid-only with existing tie-safe metrics. Missing classes/nonfinite measurement are INCONCLUSIVE; no authorized rows is valid-negative, not vacuous PASS.
- [ ] RED: exactly2000 family-stratified paired row resamples, PCG64 seed20260930, 16 replacement draws/family, percentiles2.5/97.5; no branch replay/refit/redraw. Classless/nonfinite replicate is INCONCLUSIVE.
- [ ] RED: test every positive boundary independently: coverage>=0.80/family>=0.60, AUROC>=0.80/lower>0.70, AUPRC>=prevalence+0.20, precision>=0.90/lower>=0.80, recall>=0.50, unsafe rate<=0.05/upper<=0.10, all semantic tests, coherent-offset all authorized, no specified negative-family authorization, all budgets.
- [ ] RED: otherwise valid measured budget violation or metric failure is NO_ACTION_SPECIFIC_SIGNAL; corrupt/uncomputable pair or infrastructure failure is INCONCLUSIVE and cannot create a scientific terminal release.
- [ ] GREEN: deterministic report plus exhaustive failed-gate/reason inventory; verify `tests.test_geometry_ga_report`; commit `feat: summarize frozen geometry action study`.

### Task 7 — One-shot confirmation and compact publication adapter

**Files:** evaluate_geometry_ga.py, tests/test_geometry_ga_cli.py.

**Interfaces:** `build_parser()`, `run_evaluator(args, dependencies) -> (exit_code, dict)`. Requires approved contract/addendum path/SHA, exact clean code commit, fixture manifest path/SHA, explicit execution-confirmation path/SHA and absent output target. The execution confirmation is created only at the later authorized server gate.

- [ ] RED `test_admission_precedes_branch_and_forbids_overrides`: parser has no candidate/threshold/action/horizon/seed/fixture-selection/real-data override. Unapproved/absent confirmation, wrong hashes, existing target, source-overlap or Utility/Tool Room asset reference fail before any branch run.
- [ ] RED: one pair at a time and control/treated sequential; budget timing includes action/Jacobian/render/outcome, memory includes the full harness. Single-use identity blocks a second complete valid execution.
- [ ] RED: publish only canonical inputs, fixture manifest, pair outcomes, metrics/bootstrap, logs and SHA manifest; no galleries/checkpoints. Exclusive publication, mutation detection, no overwrite, bounded outputs. Report metadata explicitly denies routing/C1/Utility authorization.
- [ ] RED: negative report may contain termination evidence but must not silently create/admit a Utility firewall release; that canonical record still requires the separately approved Gate G schema/process. Inconclusive engineering failure cannot masquerade as terminal evidence.
- [ ] GREEN: adapter only; verify `tests.test_geometry_ga_cli`; commit `feat: add gated geometry synthetic evaluator`.

### Task 8 — Code qualification, not a predictive execution

- [ ] After separate implementation approval and Tasks1–7, use one fresh whole-branch review against spec/addendum. Fix Critical/Important findings with observed RED/GREEN; do not iterate reviews for nonblocking wording.
- [ ] Run all new CPU tests plus relevant production residual/topology/geometry regressions, full unittest discovery, compile/help/diff checks. Record dependency failures/skips honestly; no local environment repair by default.
- [ ] Request exact-commit AutoDL backend qualification separately. Tiny hand-derived backend fixtures must not instantiate the128-family predictive suite, compute real Tool Room outcomes or read Utility GT.
- [ ] Freeze qualified code/test identities; one code qualification suffices for subsequent documentation-only receipts. Commit `docs: freeze geometry G-A code qualification`.

### Task 9 — Future individually approved synthetic stages

- [ ] **S-A confirmation only:** after Gate0 and code qualification, request permission to create the canonical fixture/execution confirmation before outputs; freeze exact command, manifest, code, renderer environment, budgets and full specification/addendum hashes. No branch execution.
- [ ] **S-B single-use execution:** request separate permission to run the128 paired suite once. No Tool Room/Utility data, no production write, no fallback candidate. Stop on contract/infrastructure failure; do not call an incomplete run a valid negative.
- [ ] **S-C audit/freeze:** verify all artifacts and gates, then freeze exact outcome. Positive authorizes only a request to write G-B Tool Room specification. Valid-negative may support a separately approved canonical termination confirmation; it is not itself an automatic GT release. INCONCLUSIVE permits only same-contract engineering repair with approval.

## Self-review / handoff

Spec coverage maps §§2–3 to Tasks1–2; §§4–5 to Task3; §§6,9 to Task4; §§7–8 to Task5; §§10–11 to Task6; §§11–12 to Tasks7–9. Each Review Focus item has an owning RED test. Existing shared modules are regression dependencies, not alternative statistical cores.

Gate0 makes the unresolved scientific decisions explicit instead of supplying new constants. This draft does not claim those omissions have been resolved or that a valid128-pair experiment is currently implementable. Review the blocked plan and approve an addendum before any implementation authorization; current inline execution preference is preserved, with no new agent delegation requested.
