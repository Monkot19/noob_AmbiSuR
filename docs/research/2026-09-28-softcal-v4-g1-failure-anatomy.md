# Soft-calibrated observation sufficiency: formal G1 failure anatomy

**Date:** 2026-09-28
**Status:** frozen formal result; C1 no-go
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

The external prior-reliability score `r_p` is not a deterministic function of `A` or `S`; it comes from DA3 confidence and multiview consistency. Its frozen 7,000-step `retain_high` risk rows are directionally useful:

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

At iterations 3,000 and 7,000, compare a baseline model using only `A` and `1-S` against the same model plus `1-r_p`. Use spatially blocked five-fold cross-fitting so nearby Gaussians do not appear in both train and validation folds. Each iteration is split into five equal-count contiguous slabs along the longest checkpoint-centre AABB axis, with centre coordinate and original row index as deterministic sort keys. The model is a CPU float64 logistic regression with an intercept, fold-local feature standardization, class-balanced binary cross-entropy, slope-only L2 penalty `1e-4`, zero initialization, and a fixed deterministic optimizer budget; no hyperparameter is selected from GT results. The same fitting procedure is used for baseline and augmented models.

The feasibility report must include:

- candidate validity coverage and fixed quantiles;
- marginal AUROC/AUPRC and 20 equal-count risk bins;
- cross-fitted baseline and augmented AUROC/AUPRC;
- paired fold-level deltas and a fixed-seed spatial-block bootstrap interval computed from 0.5 m validation voxels;
- top-versus-bottom quintile high-error-rate difference;
- results at both iterations, without selecting the better iteration after inspection.

This is exploratory mechanism triage. GT may be used only by this offline diagnostic, never by training or checkpoint state.

### 6.4 Continue/stop rule

`INDEPENDENT_EVIDENCE_FEASIBLE` requires `1-r_p` to satisfy all of the following:

- at least 80% valid coverage at iteration 7,000;
- the prespecified risk direction agrees at both 3,000 and 7,000;
- iteration-7,000 cross-fitted AUROC gain over the `A,1-S` baseline is at least 0.02 and the spatial-block bootstrap lower bound is greater than 0.005;
- iteration-3,000 gain is non-negative;
- top-versus-bottom risk-quintile high-error separation is at least 0.05 at 7,000;
- the signal is not a deterministic function of `A`, `S`, or `N`.

If no candidate passes, record `NO_CLEAR_COMPLEMENT`, stop all new training for this direction, and reconsider the paper-level contribution structure. If a candidate passes, the only authorized next action is a new written hypothesis/specification with an independent validation scene; it does not authorize C1 or a Tool Room training rerun.

## 7. Artifact locations

- Server formal directory: `/root/autodl-tmp/ambisur_diagnostics/Tool_Room/d0-g1/d0_g1_softcal_v4_formal7k_toolroom_seed0_20260928_v1`
- Server archive: `/root/autodl-tmp/ambisur_diagnostics/Tool_Room/d0-g1/d0_g1_softcal_v4_formal7k_toolroom_seed0_20260928_v1.tar.gz`
- Local read-only copy: `D:\research_Space\output\ambisur_diagnostics\Tool_Room\d0-g1\d0_g1_softcal_v4_formal7k_toolroom_seed0_20260928_v1`

The uncompressed local folder is sufficient for the present audit because every manifest-listed file passed byte-count and SHA256 verification. The server archive and detached SHA remain the canonical transport/long-term archival pair.
