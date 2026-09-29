# Soft-calibrated observation sufficiency: formal G1 failure anatomy

**Date:** 2026-09-28
**Status:** frozen formal result; C1 no-go; revised feasibility contract awaiting approval
**Scope:** Tool Room, resolution 2, seed 0, 7,000 iterations, diagnostic iteration 3,000, decision iteration 7,000

## 1. Decision

The preregistered soft-calibrated version-4 operationalization failed the unchanged formal G1 metric gate. The run, admission, evaluation domain, immutable-input binding, and publication package all passed their integrity gates; the evaluator exited with code 1 solely because the scientific metric gate failed.

Therefore:

- `FORMAL_G1=FAIL_METRIC_GATE`;
- `C1_GO_NO_GO=NO_GO`;
- C1, C2, Supporting, lifecycle, a 30k continuation, and a stage tag are not authorized;
- no further Tool Room search over observation-sufficiency constants, transforms, or combinations is permitted under this operationalization;
- the negative result rejects the fixed formula, not the broader question of whether a genuinely complementary evidence source could support reliability routing.

## 2. Frozen identity and integrity evidence

| Item | Frozen value |
|---|---|
| Formula/training commit | `815ccb8700e4aad4a2dbecf5959d0b9c54a17126` |
| Confirmation ID | `d0_g1_softcal_v4_formal7k_toolroom_seed0_20260928_v1` |
| Confirmation SHA256 | `074ee091cf71e21b7a9f49256bf3bbdbc4b83a62babba557ee714d9fa42f0bf0` |
| Evidence version | `4` at both 3,000 and 7,000 |
| Dataset SHA256 | `aad92aa2e0f0d072756b3a56c686d5c1d35f448811ce60ca4360c67dbc3ef255` |
| Aligned-prior SHA256 | `69a21ab8756f43834a5357f27ca6cf40c6b7b15695e0e8cade1914fe70956977` |
| GT-mesh SHA256 | `31547a31069f736792d4b13fff76c73483f59239dbbabe1971e115f9ab17171d` |
| Archive SHA256 | `86ad01a2499b97170fbc1e2e9ce4347900ad0ef646cfec4ded44fdb37205b43d` |
| Local `report.json` SHA256 | `238fdd1cb05bc03adfa24272dd830c7dca4f10659589df6a79e459f83b259991` |
| Immutable fingerprints | 20, checkpoint/snapshot binding PASS |
| Evaluation domain | all finite Gaussian centres; full valid GT mesh; no opacity, scale, visibility, frustum, AABB, or manual crop filter |
| Mesh | 2,526,537 vertices; 5,096,267 triangles; 0 rejected triangles |
| Publication | 108 manifest files; 109 archive members; 60 PNG; 16 PLY |
| Local bundle verification | 108/108 files matched manifest byte counts and SHA256 |

The formal training completed once with exit code 0, unchanged before/after dataset and prior hashes, no GT references in training, seven refreshes, real topology changes, exact checkpoint/snapshot row alignment, and valid versioned state. Optimizer-step discrepancies reported by an initial audit were reconciled as expected Adam lazy-state/topology replacement behaviour; no training asset was changed or rerun.

## 3. Frozen metrics

The primary label remains strict centre-to-full-mesh distance `> 0.05 m`. Iteration 3,000 is diagnostic only; iteration 7,000 is the sole decision point.

| Iteration | Role | Centres | Prevalence | AUROC(N) | AUPRC(N) | AUROC(A) | AUROC(1-S) | Best component | Gain over best | Decision |
|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| 3,000 | early diagnostic | 1,093,102 | 0.262649 | 0.525454 | 0.289615 | 0.498263 | 0.563420 | `1-S` | -0.037965 | diagnostic fail |
| 7,000 | primary gate | 1,417,059 | 0.187807 | 0.547598 | 0.220698 | 0.545626 | 0.530889 | `A` | +0.001972 | formal fail |

The unchanged requirements were `AUROC(N) > 0.60` and gain over `max(A, 1-S) >= 0.03`. At 7,000, the shortfalls were 0.052402 AUROC and 0.028028 gain. There was no prevalence collapse and the result was evaluable.

Relative to the earlier version-3 formal run, version 4 improved the 7,000-step AUROC by approximately 0.004687, AUPRC by 0.003101, and gain by 0.009905. This comparison is descriptive rather than a paired causal estimate because the two GPU runs ended with different stochastic topology populations. The improvement is too small to change the frozen decision.

## 4. Failure anatomy

### 4.1 The engineering and admission chain did not fail

The negative decision cannot be attributed to a malformed run, stale evidence version, wrong checkpoint, GT leakage, changed input, incomplete mesh, filtered evaluation domain, publication corruption, or evaluator crash. Formal admission, immutable-input binding, mesh validation, atomic publication, and local manifest verification all passed.

### 4.2 Soft calibration fixed saturation, not predictive complementarity

Version 4 changed the observation-sufficiency distribution as designed and avoided the old hard saturation. At iteration 7,000, `S` ranged from 0 to 0.897230 with mean 0.646383, while `N` ranged from 0.117179 to 1 with mean 0.599758. The failure is therefore not that the new formula remained numerically constant.

However, the final ROC/PR curves show `N` tracking `A` closely. The combination contributes only +0.001972 AUROC beyond `A`, far below the preregistered +0.03 complementarity requirement. At 3,000, combining with `A` actually degrades the stronger `1-S` signal by 0.037965 AUROC.

### 4.3 The N risk ordering is weak and non-monotone

For the intended `retain_low` direction at 7,000, the lowest 5% of `N` has mean GT distance 0.060836 m, worse than the full-set 0.045841 m. The high-error rate is 0.165639 at 5% coverage and 0.187807 at full coverage. Mean-distance risk has 14 monotonicity violations and `N` priority has Spearman correlation -0.082359 with distance. This is not a reliable low-risk ranking.

### 4.4 The five-state partition contains a coarse risk signal but is degenerate

At 7,000:

| Stable state | Count | Fraction | Mean distance (m) | High-error rate |
|---|---:|---:|---:|---:|
| Bypass | 777,284 | 0.548519 | 0.042448 | 0.154007 |
| Consensus | 0 | 0 | n/a | n/a |
| Prior-led | 325,903 | 0.229985 | 0.039064 | 0.159959 |
| Geometry-led | 22 | 0.000016 | 0.067789 | 0.272727 |
| Abstain | 313,850 | 0.221480 | 0.061279 | 0.300427 |

`Abstain` is enriched for high error, so the diagnostic system is not information-free. But Consensus is empty and Geometry-led is negligible. The state space is therefore too degenerate to justify activating the planned routing chain, and the state result cannot override the failed primary N gate.

### 4.5 Prior reliability is the only current feasibility lead

The external prior-reliability score `r_p` comes from DA3 confidence and multiview consistency rather than the formulas that define `A` or `S`. This provenance does not prove finite-sample numerical independence; that question is reserved for the fold-local column-space test in Section 6.3. Its frozen 7,000-step `retain_high` risk rows are directionally useful:

| Retained highest-r_p coverage | Mean distance (m) | High-error rate |
|---:|---:|---:|
| 5% | 0.028981 | 0.112091 |
| 20% | 0.030422 | 0.107497 |
| 50% | 0.035277 | 0.126151 |
| 100% | 0.045841 | 0.187807 |

Its distance-priority Spearman correlation is -0.173008 and its mean-distance/high-error risk curves have only 1/1 monotonicity violation. The direction is also consistent at iteration 3,000, where the correlation is -0.137450 and all 1,093,102 evaluated rows are valid. This is clear marginal risk stratification, but it does **not** yet establish conditional complementarity over `A` and `1-S`, statistical independence from the trained representation, or causal benefit from applying prior supervision.

Historical `r_g` points in the wrong direction: its highest-reliability 20% has a 0.272559 high-error rate versus 0.205674 over its full valid set, and its priority-distance Spearman is +0.099097. It remains a separate pre-C2 blocker and is not a candidate for immediate promotion.

## 5. Scientific interpretation boundary

The supported conclusion is:

> The fixed version-4 soft observation-sufficiency formula is numerically healthy but does not provide sufficient or complementary error prediction on the preregistered Tool Room seed-0 formal gate. It must not advance to C1.

The result does not prove that all reliability routing is futile. It leaves open a narrower hypothesis: an evidence source that is not algebraically derived from SH complexity and view sufficiency may add conditional information. The present bundle gives `r_p` enough marginal evidence to justify one short, read-only feasibility probe, not a new training run.

## 6. Short independent-evidence feasibility gate

### 6.1 Scope

The next phase is diagnostic-only and uses existing version-4 checkpoints/snapshots plus the frozen GT mesh. It starts no training, renders no 54-view publication bundle, changes no formula, and cannot produce a C1 PASS. It may produce only `INDEPENDENT_EVIDENCE_FEASIBLE`, `INCONCLUSIVE`, or `NO_CLEAR_COMPLEMENT`.

### 6.2 Frozen candidate set

To prevent another same-scene search, the decision candidate is fixed before computation:

1. `1-r_p`: external-prior unreliability risk; the only candidate allowed to produce `INDEPENDENT_EVIDENCE_FEASIBLE`.

Joint-valid `1-K` and temporal transition count may be reported only as descriptive context. They cannot trigger a feasibility PASS in this probe. Promoting either would require a new preregistration before looking at its conditional result.

`r_g/T_g` is excluded until the already-recorded direction/history defect is resolved independently. No polynomial, logarithmic, threshold, learned neural score, or hand-picked spatial crop may be introduced.

### 6.3 Test of complementarity

At iterations 3,000 and 7,000, construct the comparison domain independently from that iteration's checkpoint and snapshot:

- begin with every checkpoint Gaussian centre and reject only non-finite centres;
- require a strict same-row snapshot/checkpoint join and then retain only rows with `V_p=True`;
- use exactly these same rows and labels for the baseline and augmented models;
- apply no opacity, scaling, visibility, frustum, AABB, or manual-crop filter;
- retain the frozen binary label `centre-to-full-valid-GT-mesh distance > 0.05 m`;
- define candidate coverage as `count(V_p=True) / count(all finite checkpoint centres)` and report it separately at both iterations.

Compare a baseline model using only `A` and `1-S` against the same model plus `1-r_p`. Use spatially blocked five-fold cross-fitting so nearby Gaussians do not appear in both train and validation folds. Each iteration is split into five equal-count contiguous slabs along the longest finite-centre AABB axis, with centre coordinate and original row index as deterministic sort keys. Both models use identical folds and training/validation rows.

The model is a CPU float64 logistic regression with an intercept, class-balanced binary cross-entropy, slope-only L2 penalty `1e-4`, and zero initialization. Class weights are computed from the current four training folds only as `n_train/(2*n_class)`. Feature means and population standard deviations are also computed from those training folds only and then applied unchanged to the held-out fold. A missing class, zero/non-finite training-fold standard deviation, non-finite intermediate/output, or invalid held-out prediction makes the probe `INCONCLUSIVE`; validation rows may not contribute to weights, standardization, fitting, or convergence decisions.

The frozen solver is deterministic damped Newton/IRLS in float64 with an exact dense Hessian solve, at most 100 iterations, gradient-infinity-norm tolerance `1e-8`, Armijo constant `1e-4`, backtracking factor `0.5`, and minimum accepted step `2^-20`. Failure to meet the convergence tolerance, solve the Hessian system, or find an accepted step is `INCONCLUSIVE`; no fallback solver, altered tolerance, or replacement fold split is allowed. The same solver contract is used for baseline and augmented models.

The primary performance values are computed after concatenating every held-out prediction exactly once. In particular,

`pooled_OOF_gain = AUROC(all augmented OOF predictions) - AUROC(all baseline OOF predictions)`.

It is not the arithmetic mean of the five fold AUROCs. The five paired fold-level AUROC deltas are still reported as diagnostics, along with pooled OOF AUROC and AUPRC for both models.

Before fitting each augmented training fold, standardize `1-r_p` using that fold's training statistics and project it onto the numerical column space of the training-fold baseline design `[intercept, standardized A, standardized (1-S)]`. Define the relative residual as `||candidate - projection||_2 / max(||candidate||_2, float64 tiny)`. The augmented column is numerically independent only if this ratio is strictly greater than `1e-8` in every fold. Otherwise the computable result is `NO_CLEAR_COMPLEMENT`. This finite-sample linear-column test replaces any claim that `1-r_p` is provably not a deterministic function of `A`, `S`, or `N`; predictive complementarity is decided only by OOF performance.

For spatial uncertainty, reuse the already-fitted paired OOF predictions. Assign every retained row to the fixed-world-origin voxel

`(floor(x/0.5), floor(y/0.5), floor(z/0.5))`,

with coordinates in metres and origin `(0,0,0)`. Draw 2,000 paired cluster-bootstrap replicates with NumPy `PCG64` seed `20260928`, sampling voxels with replacement and including all member rows with the sampled voxel multiplicity. Baseline and augmented predictions always use the same sampled voxels. Do not refit either model inside a replicate. The reported 95% interval is the percentile interval at `[2.5%, 97.5%]` of paired pooled AUROC gain; its lower endpoint is the gate value. A replicate without both label classes or with a non-finite AUROC makes the run `INCONCLUSIVE` rather than triggering resampling or a different interval. This interval describes within-Tool-Room spatial stability only, not cross-scene generalization.

For the raw-direction check, sort the retained `V_p=True` rows by raw `1-r_p`, breaking ties by original checkpoint row index, and form deterministic equal-count quintiles. At both iterations, the highest-risk quintile's high-error rate must be no lower than the lowest-risk quintile's, and the tie-aware Spearman correlation between raw `1-r_p` and GT distance must be non-negative. At iteration 7,000, the highest-minus-lowest quintile high-error-rate difference must additionally be at least `0.05`. Quintiles are never formed from augmented-model predictions.

The feasibility report must include:

- candidate validity coverage and fixed quantiles;
- marginal AUROC/AUPRC and 20 equal-count risk bins;
- pooled-OOF baseline and augmented AUROC/AUPRC and their pooled AUROC gain;
- the five paired fold-level AUROC deltas and the frozen paired voxel-bootstrap interval;
- raw-risk top-versus-bottom quintile high-error-rate difference and Spearman correlation;
- every fold's numerical-independence residual, convergence status, class counts, weights, and standardization statistics;
- results at both iterations, without selecting the better iteration after inspection.

This is exploratory mechanism triage. GT may be used only by this offline diagnostic, never by training or checkpoint state.

The implementation must encode the full domain, fold, solver, numerical-independence, bootstrap, metric, threshold, and three-outcome contracts in tests and the output JSON schema before the real report is computed. The publication is a small atomic JSON/CSV bundle; the existing checkpoints, snapshots, GT mesh, and formal G1 publication remain read-only.

### 6.4 Continue/stop rule

`INDEPENDENT_EVIDENCE_FEASIBLE` requires `1-r_p` to satisfy all of the following:

- at least 80% valid coverage at iteration 7,000;
- the frozen raw-risk direction conditions hold at both 3,000 and 7,000, including at least `0.05` top-minus-bottom quintile separation at 7,000;
- the augmented column passes the fold-local numerical-independence test in every fold;
- iteration-7,000 pooled-OOF AUROC gain over the `A,1-S` baseline is at least `0.02` and the paired voxel-bootstrap 95% percentile lower bound is greater than `0.005`;
- iteration-3,000 pooled-OOF AUROC gain is non-negative;
- every required model, metric, fold, and bootstrap replicate is valid and finite.

The three outcomes are exhaustive:

- `INDEPENDENT_EVIDENCE_FEASIBLE`: every frozen coverage, direction, numerical-independence, performance, and uncertainty gate passes;
- `NO_CLEAR_COMPLEMENT`: the evaluation completes validly, but one or more of those gates fails;
- `INCONCLUSIVE`: corrupted or mismatched inputs, a classless fold or bootstrap replicate, solver non-convergence, non-finite values, or another contract failure prevents a valid calculation. It is not a PASS and does not authorize changing the model, folds, candidate, preprocessing, seed, or uncertainty method.

`NO_CLEAR_COMPLEMENT` stops all new training for this direction and triggers reconsideration of the paper-level contribution structure. `INCONCLUSIVE` permits only diagnosis and correction of the measurement failure under the unchanged frozen contract. `INDEPENDENT_EVIDENCE_FEASIBLE` authorizes only a new written hypothesis/specification and an independent-scene validation protocol; it does not validate causal benefit, authorize C1, or authorize a Tool Room training rerun.

Accordingly, this is a single-candidate, read-only, spatially blocked conditional-complementarity diagnostic. It asks only whether `1-r_p` contributes stable predictive information beyond `A` and `1-S`; it is neither a method-effect experiment nor a cross-scene generalization test.

## 7. Frozen `1-r_p` complementarity result

### 7.1 Measurement chronology and integrity

The first real probe publication (`...20260929_v1`) returned `INCONCLUSIVE` because the report validator incorrectly required independently constructed five-way quintiles and twenty-way risk bins to have matching grouped endpoint counts. This was a measurement-contract defect, not a scientific result. The frozen candidate, domain, folds, solver, bootstrap, thresholds, and inputs were unchanged. Commit `4388d2f1ed232ec5a7599d7ae2e22a7c2f21e752` repaired only that validator by checking each deterministic partition against its own `divmod` inventory. AutoDL then passed 75 focused tests and 291 full-discovery tests before the v2 rerun.

The corrected v2 publication is internally complete: evaluator exit code 0, outcome `INDEPENDENT_EVIDENCE_FEASIBLE`, no failed gates, no inconclusive reasons, and all five manifest-listed payload files match their recorded byte counts and SHA256 digests. It binds diagnostic commit `4388d2f1ed232ec5a7599d7ae2e22a7c2f21e752`, formula commit `815ccb8700e4aad4a2dbecf5959d0b9c54a17126`, formal confirmation SHA256 `074ee091cf71e21b7a9f49256bf3bbdbc4b83a62babba557ee714d9fa42f0bf0`, and the previously frozen dataset, prior, GT-mesh, and run-identity hashes. No training was started or modified.

### 7.2 Frozen measurements

| Iteration | Role | Eligible / finite centres | Marginal AUROC (`1-r_p`) | Baseline AUROC | Augmented AUROC | Pooled OOF gain | 95% voxel-bootstrap gain |
|---:|---|---:|---:|---:|---:|---:|---:|
| 3,000 | direction stability | 1,093,102 / 1,093,102 | 0.589713 | 0.518032 | 0.572746 | 0.054714 | [0.035258, 0.074344] |
| 7,000 | primary | 1,417,059 / 1,417,059 | 0.623265 | 0.519882 | 0.609023 | 0.089141 | [0.067379, 0.108420] |

Coverage is 1.0 at both iterations. At 3,000, the raw-risk top-minus-bottom quintile error-rate separation is 0.154474 and Spearman correlation with GT distance is 0.137450. At 7,000, the corresponding values are 0.170999 and 0.173008. Every fold-local numerical-independence residual is above 0.95. The 7,000 fold gains are all positive (`0.122770`, `0.089374`, `0.126887`, `0.057999`, `0.000704`); one 3,000 diagnostic fold is negative, but the preregistered pooled gain and paired spatial-bootstrap lower bound remain positive and the 3,000 pooled-gain gate passes.

### 7.3 Frozen conclusion and authority boundary

The supported conclusion is:

> Within the frozen Tool Room seed-0 assets, `1-r_p` contains stable conditional predictive information about centre-to-GT-mesh error beyond the `[A, 1-S]` baseline. It is therefore a viable independent evidence source for a new hypothesis.

This result does **not** demonstrate causal training benefit, routing benefit, reconstruction improvement, or cross-scene generalization. It does not rescue the rejected `N` formula, does not authorize C1, and does not authorize another Tool Room training search. The only authorized continuation is to write a new prior-aware hypothesis and a preregistered independent-scene validation protocol, including Utility Room and multiple seeds, before deciding whether any new training investment is justified.

## 8. Artifact locations

- Server formal directory: `/root/autodl-tmp/ambisur_diagnostics/Tool_Room/d0-g1/d0_g1_softcal_v4_formal7k_toolroom_seed0_20260928_v1`
- Server archive: `/root/autodl-tmp/ambisur_diagnostics/Tool_Room/d0-g1/d0_g1_softcal_v4_formal7k_toolroom_seed0_20260928_v1.tar.gz`
- Local read-only copy: `D:\research_Space\output\ambisur_diagnostics\Tool_Room\d0-g1\d0_g1_softcal_v4_formal7k_toolroom_seed0_20260928_v1`
- Server complementarity-v2 directory: `/root/autodl-tmp/ambisur_diagnostics/Tool_Room/d0-g1-complementarity/d0_g1_prior_complementarity_softcal_v4_toolroom_seed0_20260929_v2`

The uncompressed local folder is sufficient for the present audit because every manifest-listed file passed byte-count and SHA256 verification. The server archive and detached SHA remain the canonical transport/long-term archival pair.
