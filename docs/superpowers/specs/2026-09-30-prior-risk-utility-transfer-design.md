# Prior-Risk Utility Room Multi-Seed Transfer Design

**Status:** Draft for formal review. The in-chat design summary was confirmed on 2026-09-30. This document authorizes neither implementation nor experiment execution.

**Scope:** Define one read-only, independent-scene replication test for the already frozen candidate `1-r_p`. This line does not modify the production method and is independent of the geometry-update authorization hypothesis.

## 1. Frozen scientific claim

The frozen Tool Room seed-0 result is:

> Current Tool Room evidence indicates that `1-r_p` is more worthy of independent-scene validation than further recombination of `A/S`.

The Utility Room experiment asks only whether that conditional predictive contribution reproduces across a new scene and three prespecified training seeds.

The candidate meaning is fixed:

- `r_p` is usable external-prior reliability only when `V_p=True`;
- `1-r_p` is an external-prior warning against casual Bypass;
- low `r_p` can never strengthen prior supervision or increase a prior gradient;
- `V_p=False` means unknown and is excluded, not encoded as maximum risk;
- the candidate is not Gaussian self-reliability, causal utility, routing permission, or a replacement for geometry authorization.

Current `T_g/r_g` are internal geometry-consistency telemetry only. They are excluded from every model, gate, candidate inventory, and claim in this specification. Evidence remains version 4 and the production formula is unchanged.

## 2. Utility source and preprocessing snapshot

### 2.1 Uploaded source contract

The canonical source upload is:

```text
/root/autodl-tmp/ambisur_data/source/ScanNetpp/Utility_Room/colmap_undistorted/
├── images/                     # exactly 147 images
└── sparse/0/
    ├── cameras.txt
    ├── images.txt
    └── points3D.txt
```

The physically separate GT asset is:

```text
/root/autodl-tmp/ambisur_data/gt/ScanNetpp/Utility_Room/mesh_aligned_0.05.ply
```

Before DA3, the source audit must verify exactly 147 image files and 147 registered COLMAP image records with one-to-one basename/case correspondence, finite poses and points, correct image dimensions, referenced camera IDs, and loader-supported `PINHOLE` or `SIMPLE_PINHOLE` camera models. The uploaded source tree and GT file receive separate SHA256 identities.

During source audit and preprocessing, Utility GT access is limited to path existence, byte size, and SHA256. The mesh may not be parsed, rendered, queried, sampled, summarized, or used to choose preprocessing.

### 2.2 One frozen DA3 snapshot

The repository DA3 pipeline derives exactly one canonical Utility snapshot:

```text
Utility_Room/colmap_undistorted/
├── estimated_depths/           # exactly 147 loader-consumed .npy arrays; .jpg previews allowed
├── estimated_confs/            # exactly 147 loader-consumed .npy arrays
├── sparse_da3/0/
└── sparse_da3_aligned/0/
```

The loader consumes `estimated_depths/<complete image filename>.npy` and `estimated_confs/<complete image filename>.npy`, where `<complete image filename>` includes the original image extension and case. Postprocessing admission therefore requires exactly 147 such depth `.npy` files and exactly 147 such confidence `.npy` files, each in one-to-one correspondence with the 147 registered image filenames. Depth `.jpg` previews may coexist in `estimated_depths`; they are not counted as training arrays but every preview and every other derived file remains part of the complete derived manifest.

Before DA3 starts, a preprocessing confirmation must bind the exact code commit, DA3 model/checkpoint checksum, environment, command, constants, source manifest, and absent output target. Postprocessing admission additionally requires finite supported array shapes, loadable raw/aligned models, finite alignment metadata, an unchanged source manifest, and a complete derived SHA256 manifest.

DA3 runs once. Seeds 0/1/2 reuse the identical frozen snapshot. Any regeneration, repaired array, realignment, or parameter change creates a new snapshot identity and invalidates the earlier experiment confirmation.

Training uses private per-seed working views so loader writes cannot mutate the canonical snapshot. GT remains outside every working view, command, configuration, cache, checkpoint, and log.

No other AmbiSuR benchmark dataset is required for this replication.

## 3. Shared Utility GT information firewall

This transfer line is independent of the geometry-authorization hypothesis, but both use Utility as an untouched independent scene. The first Utility GT access requires exactly one of two immutable releases:

- **candidate release:** a geometry Stage G-C confirmation binds the action, intervention unit, target, one candidate formula, validity/state contract, thresholds, metrics, coverage, compute budget, and stop rules before any Utility GT-derived value is produced; or
- **terminal release:** a canonical geometry termination confirmation and detached SHA bind the exact repository/specification identity, `NO_ACTION_SPECIFIC_SIGNAL`, the stage and evidence supporting termination, the fact that no geometry candidate is admitted to Utility, and the permanent rule that Utility results cannot reopen or revise this terminated geometry branch.

In both cases:

- source audit, DA3 preprocessing, and GT-free D0 training may occur earlier only under separate approvals and without parsing Utility GT;
- the prior-transfer confirmation and the selected geometry release must both be reloaded and hash-verified immediately before first Utility GT access;
- the first Utility GT access must be logged with both identities;
- the `1-r_p` outcome cannot change the frozen geometry hypothesis, and the geometry outcome cannot change this candidate, domain, model, or gate.

This sequencing prevents Utility Room from becoming a second development scene.

## 4. GT-free version-4 D0 assets

The replication uses three new GT-free shadow runs:

- scene: Utility Room;
- seeds: 0, 1, and 2;
- resolution: `-r 2`;
- iterations: 7,000;
- D0 refresh interval: 1,000;
- evaluation iterations: 1,000 through 7,000 in steps of 1,000;
- checkpoints: 3,000 and 7,000;
- decision iteration: 7,000;
- iteration 3,000: direction-stability diagnostic only;
- Evidence schema: version 4;
- enabled feature: shadow diagnostics only;
- GT available to training: no.

Each seed has unique run, view, launcher, state, and output paths. Seeds are not substituted, omitted, or rerun after seeing GT. A failed infrastructure run may be repeated only from the same immutable confirmation and must retain an audit trail; a scientifically completed seed cannot be replaced.

Training-asset qualification requires a clean exact commit, normal exit, one completion marker, no active process/sentinel, unchanged source/prior hashes, no GT reference, no error/non-finite token, complete seven-refresh timeline, real topology changes, strict snapshot/checkpoint row joins, Evidence v4 at iterations 3,000 and 7,000, and reconciled lazy optimizer state.

Asset qualification is GT-free and cannot decide transfer success.

## 5. Frozen read-only probe

### 5.1 Models and common sample domain

The only two models are:

```text
M0 = [A, 1-S]
M1 = [A, 1-S, 1-r_p]
```

No `r_g`, `1-r_g`, `K`, `Delta`, temporal feature, validity indicator, interaction, nonlinear candidate, or alternative transform is evaluated.

For each seed and iteration separately, the evaluator:

1. strictly joins checkpoint centers and snapshot rows;
2. retains every finite checkpoint center with `V_p=True`;
3. evaluates the complete valid Utility GT triangle mesh for exactly those rows;
4. uses identical rows and folds for M0 and M1.

It rejects only non-finite centers and non-finite or degenerate GT triangles. It does not filter by opacity, scaling, visibility count, view frustum, AABB, state, manual crop, or distance. The label is fixed as:

```text
high_error = distance_to_complete_valid_GT_mesh > 0.05 m.
```

Coverage is:

```text
count(finite centers with V_p=True) / count(all finite centers).
```

### 5.2 GT mesh fail-closed admission

The first formal mesh access begins with a preregistered admission audit, before any prediction metric is computed. It must verify:

- exact GT mesh path, byte size, and SHA256 against the canonical confirmation;
- a loadable triangle surface with the frozen finite/non-degenerate-triangle rules;
- the frozen coordinate-frame identity and absence of any evaluation-time transform;
- the preregistered scene-alignment and coverage checks against the frozen Utility source/camera reconstruction.

The exact alignment/coverage audit algorithm and thresholds must be written into the canonical confirmation before the mesh is parsed. They may inspect coverage for admission but may not select or filter the evaluation rows.

If identity, coordinate alignment, or scene coverage is abnormal, the entire probe returns `INCONCLUSIVE` and stops before model fitting. The evaluator must not repair or reinterpret the asset by realignment, rescaling, cropping, AABB filtering, component selection, manual masking, or changing the evaluation domain. A corrected mesh is a new immutable asset requiring a new confirmation and separate approval.

### 5.3 Deterministic spatial cross-fitting

Each seed/iteration uses deterministic five-fold spatial cross-fitting. Among eligible finite centers, choose the world-coordinate axis with greatest range; resolve equal ranges in `x,y,z` order. Sort by that coordinate with original checkpoint row index as the tie breaker, then split into five equal-count contiguous slabs.

M0 and M1 use the same training and validation rows. All means, population standard deviations, balanced class weights, and numeric-column checks are calculated on the current training folds only. Validation rows contribute to none of them.

The deterministic float64 weighted logistic regression is frozen as:

- balanced class weights computed from the current training folds;
- unpenalized intercept;
- slope L2 penalty `1e-4`;
- zero initialization;
- exact dense Newton/IRLS solve;
- maximum 100 iterations;
- gradient infinity-norm tolerance `1e-8`;
- Armijo constant `1e-4`;
- backtracking factor `0.5`;
- minimum step `2^-20`.

A classless fold, zero or non-finite scale, non-finite prediction, failed solve, or failed convergence makes the corresponding formal result `INCONCLUSIVE`; there is no alternate split or solver fallback.

After fold-local standardization, `1-r_p` must add a numerically independent column relative to intercept, `A`, and `1-S`. Its relative residual from that numeric column space must be strictly greater than `1e-8` in every training fold. This is numeric design independence, not statistical independence.

### 5.4 Primary metrics and direction checks

All out-of-fold predictions are pooled once per seed/iteration. The primary increment is:

```text
Delta_AUROC(seed, iteration) = AUROC_OOF(M1) - AUROC_OOF(M0).
```

It is not the average of the five fold AUROCs. AUPRC, marginal raw-risk metrics, five paired fold deltas, solver diagnostics, class counts, class weights, and preprocessing values are reported but do not replace the pooled primary quantity.

Iteration 7,000 is the only performance decision point. Iteration 3,000 retains the preregistered raw-risk direction-stability checks below and reports all model metrics descriptively, but no 3,000 AUROC/AUPRC gain is a PASS/FAIL gate.

On eligible rows, raw `1-r_p` is sorted with original row index as the tie breaker and divided into deterministic equal-count quintiles. At both 3,000 and 7,000, for every seed:

- the highest-risk quintile high-error rate must not be below the lowest-risk quintile rate;
- tie-aware Spearman correlation between `1-r_p` and GT distance must be non-negative.

At 7,000, every seed's highest-minus-lowest quintile high-error-rate difference must be at least `0.05`.

### 5.5 Paired voxel bootstrap

After OOF predictions are fixed, validation rows are grouped into fixed-origin `[0,0,0]`, `0.5 m x 0.5 m x 0.5 m` world-coordinate voxels. For each seed/iteration, run exactly 2,000 paired replicates with NumPy `PCG64` initialized by:

```text
SeedSequence([20260930, iteration, training_seed]).
```

Each replicate samples voxels with replacement and includes all member rows with sampled-voxel multiplicity. M0 and M1 use the same sampled voxels. Models are not refit.

For replicate `b`, the Utility macro gain is the arithmetic mean of the three seed-level paired gains at index `b`. The 95% interval is the `[2.5%, 97.5%]` percentile interval across the 2,000 macro replicates. A classless or non-finite replicate makes the result `INCONCLUSIVE`; it is not silently redrawn.

This interval represents within-Utility spatial uncertainty for the fixed three seeds. It is not a seed-population interval or a cross-scene generalization interval. Seed-level gains remain visible and independently gated.

## 6. Frozen decision gates

`PRIOR_RISK_TRANSFER_SUPPORTED` requires all of the following:

- 7,000-step coverage is at least `0.80` for each seed;
- every direction condition in Section 5.4 passes;
- every fold passes numerical-column, class, finite-output, and solver contracts;
- every seed's 7,000 `Delta_AUROC` is non-negative;
- the arithmetic mean of the three 7,000 seed gains is at least `0.02`;
- the macro paired-bootstrap 95% lower bound at 7,000 is strictly greater than `0.005`.

`NO_CROSS_SCENE_REPLICATION` is returned when the evaluation is valid but any coverage, direction, independence, performance, or uncertainty gate fails.

`INCONCLUSIVE` is returned only when corrupted/mutated input, incomplete formal admission, invalid folds, solver failure, non-finite values, invalid bootstrap replicates, or infrastructure failure prevents the frozen comparison from being computed. It cannot trigger candidate, threshold, model, split, or seed replacement.

## 7. Confirmation and approval chronology

After the one-time DA3 snapshot is frozen, but while all three seeds' run/view/state paths and all probe targets are still absent, create one canonical confirmation. It must bind:

- exact repository commit and clean status;
- source, aligned-prior, DA3 snapshot, GT-file, and confirmation SHA256 identities;
- seed list and exact training commands/configuration;
- Evidence v4 and checkpoints 3,000/7,000;
- the sample domain, GT label, models, fold algorithm, solver, bootstrap, thresholds, and exhaustive outcomes;
- every run/view/state/probe output path and its required absence;
- the shared Utility GT information firewall.

Only after confirmation review and separate authorization may GT-free seeds run. Only after all three assets independently qualify, the same confirmation is reloaded, exact runs are bound, one of the two geometry releases in Section 3 is reloaded and verified, and a separate GT-probe authorization is granted may Utility GT be parsed.

The required authorization sequence is:

1. source-upload audit;
2. DA3 preprocessing;
3. DA3 snapshot admission;
4. canonical confirmation creation;
5. three GT-free seed runs;
6. per-seed asset qualification;
7. verify the immutable geometry Stage G-C or terminal release;
8. first Utility GT probe and mesh admission;
9. publication audit and conclusion freeze.

No step inherits authority from this design document.

## 8. Output and claim boundary

The probe publishes compact, atomic, checksummed artifacts only: canonical inputs/provenance JSON, report JSON, fold CSV, direction/risk-bin CSV, bootstrap summary or compact replicate data, and manifest. It does not generate field renders, overlays, PLY files, or a gigabyte archive unless separately requested after the decision is frozen.

A positive result means only:

> On Utility Room under three prespecified seeds, `1-r_p` reproduced stable conditional predictive information beyond `[A,1-S]` on the valid prior-evidence domain.

It does not establish causality, statistical independence, broad cross-scene generalization, reconstruction benefit, or routing safety. It can authorize a written prior-warning/abstention hypothesis or another independent-scene confirmation. It cannot authorize a new scalar `N`, five-state arbitration, geometry authorization, prior-gradient increase, production method changes, or C1.

## 9. Current authorization boundary

This document currently authorizes only formal review of the independent transfer protocol. It does not authorize:

- an implementation plan or code change;
- server upload audit or preprocessing;
- DA3 execution;
- Utility training;
- Utility GT access or probe execution;
- production routing, five-state arbitration, or C1.

Every later phase requires an approved implementation plan and an explicit execution authorization.
