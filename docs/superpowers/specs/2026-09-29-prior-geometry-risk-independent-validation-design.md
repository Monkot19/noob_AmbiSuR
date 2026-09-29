# Prior/Geometry Risk Semantics and Independent Validation Design

**Status:** Draft for formal review; the in-chat design summary and its two amendments were approved on 2026-09-29. This document is not an implementation or experiment authorization.

**Scope:** Define the semantic division between prior risk and geometry risk, the fail-closed boundary for repairing historical `T_g/r_g`, and the preregistered Utility Room / multi-seed read-only validation that must precede any new routing architecture or C1 training.

## 1. Decision and frozen evidence boundary

The version-4 Tool Room seed-0 probe is retained exactly as a positive feasibility result for one candidate:

> Current Tool Room evidence indicates that `1-r_p` is more worthy of independent-scene validation than further recombination of `A/S`.

The result does not establish statistical independence, causal training benefit, routing benefit, reconstruction improvement, or cross-scene generalization. It does not authorize a new scalar `N`, C1, a new Tool Room training search, or a multi-candidate sweep. It authorizes only the written hypothesis and independent validation protocol in this document.

The next scientific question is not whether a more favorable transformation can be found. It is whether two already defined evidence channels have distinct conditional value on independent data:

- external-prior risk: `1-r_p`;
- internal-geometry risk: `1-r_g`, but only after the historical `T_g/r_g` semantic defect has been diagnosed and, if justified by the approved formulas, repaired.

## 2. Semantic division of responsibility

| Quantity | Frozen meaning | Permitted future role | Forbidden interpretation |
|---|---|---|---|
| `A` | appearance/representation ambiguity | difficulty context | reliability or permission by itself |
| `S` | observation sufficiency | difficulty context | a substitute for prior or geometry reliability |
| `1-r_p` | external-prior risk when `V_p=True` | warning against casual Bypass; prior-evidence authorization input | Gaussian self-reliability; a reason to increase prior gradient |
| `1-r_g` | internal-geometry risk when `V_g=True` | internal-evidence authorization input | an external-prior score; a value when `V_g=False` |
| `K` | external/internal agreement when joint evidence is valid | agreement/conflict evidence | standalone reliability |
| `Delta=r_p-r_g` | relative evidence authority | direction of a valid-channel disagreement | a valid score when either required channel is unknown |

`V_p=False` and `V_g=False` mean **unknown**, not maximal risk. No model may encode an invalid channel as `r=0` and then silently interpret `1-r=1` as maximum risk. Rows outside the declared validity domain are excluded from the corresponding formal incremental comparison; no missing-risk imputation or validity-indicator shortcut is allowed.

Low `r_p` must never increase external-prior gradient. If a later architecture is authorized, `1-r_p` can veto Bypass or force abstention/extra validation, but cannot convert weak prior evidence into stronger prior supervision. This monotonic safety rule is architectural, not tunable.

The present work must not collapse `A`, `S`, `r_p`, and `r_g` into another scalar `N`. A future dual-risk architecture is considered only after independent validation and must keep difficulty context, evidence authorization, agreement/conflict, and five-state decisions inspectable as separate quantities.

## 3. Pre-C2 `T_g/r_g` repair boundary

### 3.1 Approved formula semantics

The approved design already fixes the direction of every internal-geometry component:

- higher `E_g,mv` means stronger cross-view geometry consistency;
- higher `E_g,dn` means stronger primitive/depth-normal agreement;
- higher `E_g,stab` means greater center/normal stability between consecutive evidence refreshes;
- `T_g=(E_g,mv E_g,dn E_g,stab)^(1/3)` is therefore reliability, not risk;
- `r_g=V_g T_g` is usable reliability only while `V_g=True`;
- `1-r_g` is the corresponding raw risk only on that valid domain.

The repair may restore the implementation to those approved semantics. It may not replace a component, negate a score, retune a threshold, change an exponent, introduce a learned mapping, or choose a sign because it improves GT association.

If the implementation already satisfies the formulas and state-time contract, the observed historical reversal is a scientific limitation rather than a repairable bug. The required outcome is then `NO_SEMANTIC_REPAIR_JUSTIFIED`; the project must stop this geometry branch instead of changing the formula under the name of a repair.

### 3.2 Evidence permitted during repair

The repair phase may use only:

1. mathematically hand-checked synthetic inputs;
2. CPU/CUDA component tests whose expected direction follows directly from the formulas;
3. temporal-state, validity, topology-migration, checkpoint, and resume tests;
4. read-only Tool Room engineering diagnostics over existing checkpoints, snapshots, logs, current component outputs, and camera/neighbor contracts.

Existing frozen Tool Room GT reports may be cited only as the historical trigger for the investigation. No new Tool Room GT query, candidate comparison, sign search, threshold search, or formula selection is permitted during repair.

Utility GT must remain unseen by the repair process. Before the repair commit, evidence schema, qualification tests, nested-probe schema, thresholds, and confirmation format are frozen, the process may record only the Utility GT file's existence, byte size, and SHA256. It may not query mesh distances, construct labels, render overlays, or inspect any Utility metric. This chronology prevents Utility Room from becoming another development scene.

### 3.3 Temporal and topology contract

The repaired implementation must satisfy all of the following:

- History compares the current Gaussian center/normal with the immediately previous **valid evidence refresh** after the prior state has been migrated to the current Gaussian domain.
- Previous history is read before it is overwritten. First history has `v_hist=False`; current values cannot compare with themselves.
- Invalid component observations do not update the corresponding EMA and are not replaced by numeric zero.
- `V_g=False` prevents `r_g` from entering arbitration or evaluation, even if a historical EMA tensor exists for logging.
- A surviving Gaussian follows the existing `new_to_old` mapping. A newly created clone/split row remains `is_new=True` and resets geometry evidence/history even if temporal diagnostic lineage separately inherits its parent's state.
- A new Gaussian cannot become geometry-valid until the required new-semantic history has actually been observed.
- Prune, clone, split, checkpoint round-trip, and resume preserve row alignment and version identity.
- D0 remains `no_grad`, and the repair cannot change gradients, optimizer state, densification proxies, topology decisions, or feature-off behavior.

Every monotonicity expectation is derived from the formula. Examples include: increasing cross-view inconsistency cannot increase `E_g,mv`; increasing normal disagreement cannot increase `E_g,dn`; increasing normalized motion or rotation cannot increase `E_g,stab`; with the other valid components fixed, increasing any component cannot decrease `T_g` or `r_g`.

### 3.4 Version and completion gate

Any change to stored `T_g/r_g`, geometry-history, validity, or EMA semantics creates **Evidence state version 5**. Version-4 checkpoints remain readable for historical diagnosis only and cannot be resumed into or admitted as version-5 evidence. Formal loading must require version 5 at iterations 3,000 and 7,000 and reject legacy, mixed, or missing versions.

The repair is complete only after:

- synthetic direction and boundary tests pass on CPU and real CUDA where the production boundary is CUDA-backed;
- validity, invalid-history isolation, topology migration, checkpoint/resume, and feature-off tests pass;
- a GT-free Tool Room read-only replay/engineering diagnostic demonstrates the expected state chronology without changing an existing artifact;
- the exact repair commit, schema versions, formulas, constants, and qualification outputs are frozen;
- the independent Utility confirmation record and probe schema are frozen before any Utility GT-derived result is produced.

Passing these gates establishes semantic correctness, not predictive usefulness.

## 4. Utility Room source and DA3 data contract

### 4.1 What the user uploads

Only the source assets and physically separate GT asset need to be uploaded:

```text
/root/autodl-tmp/ambisur_data/
├── source/ScanNetpp/Utility_Room/colmap_undistorted/
│   ├── images/                         # exactly 147 uploaded images
│   └── sparse/0/
│       ├── cameras.txt
│       ├── images.txt
│       └── points3D.txt
└── gt/ScanNetpp/Utility_Room/
    └── mesh_aligned_0.05.ply
```

The uploaded `sparse/0` must be the genuinely undistorted camera model, not the original `OPENCV_FISHEYE` model copied under a new directory name. `cameras.txt` must contain only loader-supported `PINHOLE` or `SIMPLE_PINHOLE` cameras. All 147 image basenames and case must match the 147 registered records in `images.txt`; camera dimensions must match the uploaded files; all poses and points must be finite. A mismatch stops preprocessing.

No `split.json` is used, and training does not pass `--eval`: all 147 cameras follow the same all-camera D0 protocol previously used for Tool Room. A future appearance holdout would be a separate protocol and cannot be introduced silently here.

### 4.2 What AmbiSuR/DA3 derives on the server

The following are **derived canonical assets**, not user-upload requirements:

```text
Utility_Room/colmap_undistorted/
├── estimated_depths/<image_filename>.npy
├── estimated_confs/<image_filename>.npy
├── sparse_da3/0/
└── sparse_da3_aligned/0/
    ├── cameras.bin
    ├── images.bin
    ├── points3D.bin
    └── trans.json
```

They are produced once from the uploaded Utility source by the repository DA3 pipeline (`scripts/run_da3.sh` / its single-scene equivalent), audited, hashed, and frozen as one Utility dataset snapshot. Seeds 0/1/2 reuse that identical snapshot; DA3 is not rerun per seed.

Before DA3 runs, a preprocessing confirmation must bind the exact repository commit, DA3 model/checkpoint identity and checksum, Python/CUDA environment, command, `max_points`, `ransac_thresh`, input manifest, and new output path. The preferred rule is to recover and reuse the verifiable Tool Room ScanNet++ DA3 arguments. If no verifiable record exists, one pair of values must be chosen from script semantics and resource constraints **before** DA3 and without opening Utility GT, then frozen for every seed. Utility metrics may never be used to choose or revise preprocessing arguments.

Post-DA3 admission requires exactly 147 depth arrays and 147 confidence arrays with exact image-filename correspondence, finite supported shapes, a loadable raw DA3 model, a loadable aligned model, finite `trans.json.scale`, a complete SHA256 manifest, and an unchanged source manifest. Any regeneration creates a new dataset snapshot identity and invalidates an earlier experiment confirmation.

Training never points directly at the canonical upload because the loader writes `points3D.ply`. Each seed receives a private working view: image/source/derived arrays may be read-only links, while `sparse_da3_aligned` is copied to the private view. GT remains outside every working view, training configuration, cache, checkpoint, and log.

No DTU, Tanks and Temples, Mip-NeRF 360, or other AmbiSuR benchmark data need be downloaded for this phase. Those datasets remain later generalization assets and cannot replace the preregistered Utility test.

## 5. New Utility version-5 D0 assets

Old Tool Room or Utility checkpoints cannot acquire corrected geometry history retrospectively. After the version-5 repair and protocol are frozen, Utility Room requires three new GT-free D0 shadow runs:

- scene: Utility Room;
- seeds: 0, 1, and 2;
- resolution: `-r 2`;
- iterations: 7,000;
- D0 refreshes: 1,000, 2,000, 3,000, 4,000, 5,000, 6,000, 7,000;
- checkpoints: 3,000 and 7,000;
- decision iteration: 7,000; iteration 3,000 is direction-stability only;
- evidence schema: version 5;
- GT available to training: no.

Each seed uses a unique run, working-view, launcher, state-file, and output path. The exact training commit, clean status, command, seed, runtime, source/DA3 manifests, and before/after hashes are recorded. Runs do not overlap on the GPU. A failed seed is not replaced, reseeded, resumed under changed semantics, or omitted from the macro result.

Completion admission requires normal exit, exactly one training-complete marker, no active process/sentinel, no error/non-finite/GT reference, unchanged input hashes, complete seven-refresh timeline, real topology changes, strict snapshot/checkpoint joins, version-5 state at both checkpoints, and valid optimizer lazy-state reconciliation. This is training-asset qualification only; it does not inspect GT or decide feasibility.

## 6. Frozen nested read-only probe

### 6.1 Models and common sample domain

The only models are:

```text
M0 = [A, 1-S]
M1 = [A, 1-S, 1-r_p]
M2 = [A, 1-S, 1-r_p, 1-r_g]
```

The primary question is the prespecified increment `M2-M1`. `M1-M0` is the independent-scene transfer check for the already frozen prior-risk observation. No alternative candidates, interactions, nonlinear transformations, temporal features, validity indicators, `K`, `Delta`, or model-selection sweep are allowed.

For each seed and iteration separately, the evaluator:

1. strictly joins checkpoint centers and snapshot rows;
2. evaluates the complete valid Utility GT triangle mesh for every finite center;
3. rejects only non-finite centers and non-finite/degenerate GT triangles;
4. selects the common formal domain `V_p & V_g`;
5. evaluates `M0`, `M1`, and `M2` on exactly those same rows.

It does not filter by opacity, scaling, visibility count, view frustum, AABB, manual crop, state, or distance. The binary label remains `distance_to_GT_mesh > 0.05 m`. Coverage is

```text
count(finite centers with V_p & V_g) / count(all finite centers).
```

The report also gives the separate `V_p` and `V_g` coverage so a common-domain result cannot hide a collapsing channel.

### 6.2 Spatial cross-fitting and numerical contract

Each seed/iteration uses deterministic five-fold spatial cross-fitting. The fold axis is the finite common-domain center coordinate with greatest range; rows are sorted by that coordinate, ties are broken by original checkpoint row index, and the ordered rows are split into five equal-count contiguous slabs. The same folds and rows are used by all three models.

All preprocessing is fold-local. Means, population scales, class weights, and numerical-column checks are computed from the current training folds only. Validation rows do not contribute to fitting or preprocessing. A zero/non-finite scale, classless training or validation fold, non-finite prediction, or solver failure makes the result `INCONCLUSIVE`; it does not trigger another split or fallback.

The model is the already qualified deterministic float64 weighted logistic regression:

- balanced class weights computed from the current training folds;
- intercept unpenalized;
- slope L2 penalty `1e-4`;
- zero initialization;
- exact dense Newton/IRLS solve;
- maximum 100 iterations;
- gradient infinity-norm tolerance `1e-8`;
- Armijo constant `1e-4`;
- backtracking factor `0.5`;
- minimum step `2^-20`.

For `M1-M0`, the fold-local standardized `1-r_p` column must have relative residual strictly greater than `1e-8` outside the numeric column space of intercept, `A`, and `1-S`. For `M2-M1`, standardized `1-r_g` must pass the same test against the `M1` design. This is numerical column independence, not a claim of statistical or causal independence.

Each model's out-of-fold predictions are pooled once per seed/iteration. The primary gain is pooled AUROC difference, not the arithmetic mean of fold AUROCs:

```text
Delta_prior(seed, iteration) = AUROC_OOF(M1) - AUROC_OOF(M0)
Delta_geometry(seed, iteration) = AUROC_OOF(M2) - AUROC_OOF(M1).
```

AUPRC, marginal raw-risk AUROC/AUPRC, five paired fold gains, calibration summaries, class counts, weights, standardization values, convergence diagnostics, and numerical residuals are reported but do not replace the frozen primary quantities.

### 6.3 Raw-risk direction checks

On the common `V_p & V_g` rows, raw `1-r_p` and raw `1-r_g` are assessed separately. For each risk, rows are sorted by risk with original row index as the tie breaker and divided into deterministic equal-count quintiles.

At both 3,000 and 7,000, for every seed:

- the highest-risk quintile's high-error rate must be no lower than the lowest-risk quintile's;
- tie-aware Spearman correlation between raw risk and GT distance must be non-negative.

At 7,000, the across-seed arithmetic mean of the three highest-minus-lowest quintile high-error-rate differences must be at least `0.05` for each risk. Quintiles are formed from raw risk, never from fitted predictions.

### 6.4 Paired voxel bootstrap and multi-seed summary

Gaussian rows are technical/spatial subsamples; they are not independent experimental replicates. Training seed is an algorithmic run-level replicate within one scene, not an independent scene replicate. For cross-scene generalization this protocol still has `n_scene=1`; for stochastic run stability it has three prespecified seeds. The report must keep seed-level results visible and must not pool all Gaussian rows across seeds to manufacture a larger `n`.

After OOF predictions are fixed, each seed/iteration groups validation rows into fixed-origin `[0,0,0]`, `0.5 m x 0.5 m x 0.5 m` world-coordinate voxels. It runs exactly 2,000 paired bootstrap replicates using NumPy `PCG64` initialized by `SeedSequence([20260928, iteration, training_seed])`. Each replicate samples that seed's voxels with replacement and includes all member rows with sampled-voxel multiplicity. The same sampled voxels are used for M0, M1, and M2; no model is refit.

For replicate index `b`, the macro gain is the arithmetic mean of the three seed-specific paired gains at index `b`. The 95% interval is the `[2.5%,97.5%]` percentile interval of those 2,000 macro gains. A classless or non-finite replicate makes the protocol `INCONCLUSIVE`; it is not silently redrawn. This interval describes within-Utility spatial uncertainty of the mean across the **fixed** three seeds. It does not resample seeds and is not a seed-population or cross-scene generalization interval. Seed-to-seed stability is assessed separately by the three reported gains and their all-non-negative gate.

## 7. Preregistered gates and exhaustive outcomes

### 7.1 Prior-risk transfer check (`M1-M0`)

The frozen Tool Room result transfers to Utility only if all of the following hold:

- common-domain coverage is at least `0.80` for every seed at iteration 7,000;
- raw `1-r_p` passes every direction condition in Section 6.3;
- every fold passes numerical column independence and solver validity;
- each seed's 7,000 `Delta_prior` is non-negative;
- the arithmetic mean of the three seed-level 7,000 `Delta_prior` values is at least `0.02`;
- the macro paired-bootstrap 95% lower bound at 7,000 is strictly greater than `0.005`;
- the arithmetic mean of the three seed-level 3,000 `Delta_prior` values is non-negative.

### 7.2 Geometry-risk increment (`M2-M1`)

The geometry channel is incrementally feasible only if the same validity, coverage, direction, numerical-independence, and solver conditions hold for `1-r_g`, and:

- each seed's 7,000 `Delta_geometry` is non-negative;
- the arithmetic mean of the three seed-level 7,000 `Delta_geometry` values is at least `0.02`;
- the macro paired-bootstrap 95% lower bound at 7,000 is strictly greater than `0.005`;
- the arithmetic mean of the three seed-level 3,000 `Delta_geometry` values is non-negative.

These are planned comparisons, not a family of post-hoc candidates. No multiple-candidate correction is substituted because the protocol has exactly one nested sequence and two prespecified increments; neither increment may be dropped after inspection.

### 7.3 Outcomes

The final outcomes are exhaustive:

- `DUAL_RISK_EVIDENCE_FEASIBLE`: both Sections 7.1 and 7.2 pass and every required computation is valid.
- `NO_STABLE_GEOMETRY_COMPLEMENT`: evaluation completes validly, but any prior-transfer, geometry-increment, coverage, direction, numerical-independence, performance, or uncertainty gate fails.
- `INCONCLUSIVE`: corrupted/mismatched assets, failed admission, GT coverage/alignment failure, classless fold/bootstrap replicate, solver non-convergence, non-finite value, incomplete seed, or another contract failure prevents a valid result.

`INCONCLUSIVE` permits only same-contract measurement diagnosis and correction. It does not permit a new candidate, split, solver, threshold, seed replacement, crop, or preprocessing choice. `NO_STABLE_GEOMETRY_COMPLEMENT` stops the dual-risk route; it cannot be converted to a pass by returning to Tool Room.

`DUAL_RISK_EVIDENCE_FEASIBLE` authorizes only a new written dual-risk architecture specification. It still does not authorize C1 training. That later specification may use `A/S` as difficulty context, `r_p/r_g` as external/internal evidence authorization, and `K/Delta` for agreement/conflict before producing five states. Its gradients and lifecycle effects require separate approval and TDD.

## 8. Chronology, confirmation, and leakage prevention

Before the first Utility GT-derived number, image, distance array, or label is produced, a canonical confirmation record and detached SHA256 must freeze:

- exact repair/probe commit and clean Git state;
- Evidence version 5 and every other serialized schema version;
- approved formulas, constants, validity rules, and no-gradient isolation;
- Utility source, DA3-derived, aligned-prior, and GT-mesh SHA256 identities;
- DA3 provenance and exact preprocessing command;
- seed 0/1/2 training commands, resolution, refresh/checkpoint iterations, and absent output targets;
- the exact `M0/M1/M2` sequence, common domain, GT label, fold construction, solver, standardization, class weighting, numerical threshold, voxel bootstrap, seed, replicate count, gates, and outcomes;
- compact output inventory and absent diagnostic targets;
- the confirmation record's own SHA256.

The evaluator must verify this record, version-5 checkpoints at 3,000/7,000, run identities, commands, source/prior/GT hashes, snapshot joins, and input fingerprints before computing metrics. Input mutation during evaluation invalidates publication.

GT mesh admission uses the complete aligned mesh. Only non-finite or degenerate triangles are rejected. Coordinate provenance, bounds, camera/frustum coverage, units, and a small fixed three-camera alignment audit are checked after protocol freeze and before metrics. An alignment or coverage defect stops as `INCONCLUSIVE`; the evaluator cannot crop, move, scale, or substitute the mesh. The fixed audit may produce at most three overlay PNGs plus JSON metadata and is not a model-visualization bundle.

## 9. Output and resource contract

The nested probe is lightweight and read-only. It publishes atomically to a new diagnostics path:

- confirmation/input/provenance JSON;
- one report JSON;
- flat seed/iteration/fold CSV;
- raw-risk quintile/bin CSV;
- paired-bootstrap CSV;
- manifest JSON;
- at most three fixed GT-alignment overlay PNGs and their metadata.

It does not render evidence fields, export colored Gaussian PLYs, copy checkpoints/snapshots, build a 1 GB archive, or start training. A small compressed archive is optional only if it contains exactly the compact manifest-listed outputs; the uncompressed directory plus verified manifest is sufficient for review.

## 10. TDD and review contract

Implementation must proceed in independent RED/GREEN units:

1. synthetic `E_g,mv/E_g,dn/E_g,stab/T_g/r_g` direction and validity tests;
2. refresh chronology, invalid-history isolation, topology mapping, new-child reset, checkpoint/resume, and version-4 rejection tests;
3. feature-off/no-grad/no-topology-effect regression tests and real CUDA transport tests;
4. Utility source/DA3 manifest, camera-model, basename, shape, finite, and immutable-snapshot admission tests;
5. common-domain `M0/M1/M2` input tests and strict version-5 formal admission;
6. nested fold-local cross-fitting, numerical-column, macro seed, paired voxel-bootstrap, and exhaustive outcome tests;
7. compact atomic publication, no-overwrite, input-mutation, and no-render/no-training tests.

No production behavior is changed before the relevant RED test is observed. A stage is not complete until focused tests, full regression, static checks, clean worktree, and server CPU/CUDA qualification pass on an exact commit. The repair and probe each receive independent code review before any Utility GT access.

## 11. Stop conditions

Stop and report without reinterpretation if any of the following occurs:

- no approved-formula semantic defect can be demonstrated in `T_g/r_g`;
- repairing the defect would require a GT-selected sign, threshold, transformation, or new component;
- Utility source is not truly undistorted `PINHOLE/SIMPLE_PINHOLE`, or 147-way names/poses do not align;
- DA3 provenance/arguments cannot be frozen without consulting Utility GT;
- any version-5 run is incomplete, contaminated by GT, uses a changed dataset snapshot, or fails state/topology/checkpoint admission;
- common `V_p & V_g` coverage, folds, solver, bootstrap, or GT alignment is invalid;
- the frozen performance/direction gates fail.

The study must not download unrelated AmbiSuR benchmark datasets, search another Tool Room calibration, merge risks into a new `N`, or begin C1 merely to avoid a negative independent result.
