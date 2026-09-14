# G0 Behavioral Feature-Off Equivalence Design

**Status:** architectural specification confirmed by user on 2026-09-14; schema-3 comparator component implemented and AutoDL focused/full tests passed at `301e69781861c32e61100251e5ebe150940a4e7c`. The historical schema-2 FAIL remains valid; pre-launch contract and unseen E0 confirmation have not run, so G0 is not yet passed.

**Authority:** `docs/research/ambisur-reliability-routing-design.md` §13 is the highest-priority method specification. This document gives the implementation-facing contract for its 2026-09-11 behavioral-equivalence amendment.

## Trigger and preserved evidence

The formal Tool Room 8k schema-2 triplet audit completed safely and produced a valid report. All 105 exact invariants passed. The fixed-shape application Tensor, evaluation metrics, pre-topology count and post-topology count also passed. However, 343 of 1,926 Gaussian capture and Adam-moment scalar summaries exceeded the per-summary `2*d_B` envelope, so the triplet remains FAIL under the contract that was frozen before that E0 run.

The 343 failures comprise 122 capture summaries and 221 optimizer-moment summaries. Many ratios are dominated by baseline self-distances between `1e-10` and `1e-19`; the maximum ratio is not a near-boundary event. The evidence supports changing the role of internal numerical summaries, not increasing factor `2.0` after seeing results.

The original report and its FAIL decision are immutable historical evidence. This amendment does not rewrite that result, delete any field, modify training, or authorize D0/C1.

## Objective

G0 answers one engineering question: when every Core feature is off, does the E0 integration preserve observable baseline training semantics and quality within measured baseline self-repeatability?

G0 is not intended to require two stochastic densification/pruning trajectories to reproduce every internal Gaussian parameter distribution or Adam moment independently. Internal state remains valuable for diagnosing divergence and catching structural corruption, so it is retained in full but split into structural hard evidence and numerical diagnostic evidence.

## Three evidence layers

### 1. Exact and provenance hard gates

The following must pass without numerical tolerance:

- role-specific expected commit and clean worktree state;
- a pre-launch confirmation contract whose identity, role paths, normalized commands and content hash were frozen while the new E0 output path was absent;
- canonical dataset and aligned-prior SHA256;
- semantic seed, resolution, iteration schedule and common effective training configuration;
- all Core flags false and resolved `training_path="legacy"`;
- required checkpoint, PLY, application model, configuration, log, launcher-contract and E0 metadata artifacts;
- checkpoint outer schema, capture field names and Tensor dtypes;
- every Gaussian-indexed Tensor trailing shape;
- optimizer group order/names, hyperparameters, state keys and step counters;
- internal-summary field set, stable names, expected cardinality and uniqueness.

Role commit values are checked against role-specific expectations: B1/B2 use the frozen baseline commit and E0 uses the frozen E0 commit. Different output and private-view paths are allowed only where declared by the launcher contract.

### 2. Safety and observable-numeric hard gates

The safety layer requires:

- successful process exit and one training-complete marker;
- evaluation records at iterations `500`, `1000`, `5001`, `7001` and `8000`;
- zero Traceback, RuntimeError, CUDA error, device-side assertion, AssertionError and non-finite log tokens;
- no missing, empty or non-finite consumed Tensor;
- unchanged immutable input and run-artifact hashes across the read-only audit;
- no extra optimizer step, repeated backward injection or abnormal sustained resource growth.

For a distance `D`, retain the frozen baseline envelope:

```text
d_B = D(B1, B2)
d_E = min(D(E0, B1), D(E0, B2))

d_B == 0  -> the selected E0/baseline pair must be exact and d_E == 0
d_B > 0   -> d_E <= 2 * d_B
```

This remains a hard gate for only the following observable numerical evidence:

- L1 and PSNR at every required evaluation iteration;
- pre-topology count from the final progress/PLY save stage;
- post-topology Gaussian count from the checkpoint stage;
- fixed-shape, non-Gaussian-indexed application Tensors such as `appear_ab`, using direct element-order RMSE and MAE.

Counts from different save stages must not be mixed. Scalar metrics use absolute distance. RMSE and MAE are independent gates. All three pairwise distances, the nearest baseline and factor `2.0` remain recorded. Learned-output hashes are diagnostic and input hashes remain hard.

### 3. Internal numerical diagnostic layer

The existing 13 Gaussian capture fields and 12 Gaussian-indexed Adam `exp_avg/exp_avg_sq` fields continue to produce the complete frozen set of 1,926 permutation-invariant scalar summaries:

- per semantic channel and per-row L2 norm;
- mean and population standard deviation;
- quantiles `0.01`, `0.05`, `0.25`, `0.50`, `0.75`, `0.95`, `0.99`;
- CPU float64 canonical reductions and the previously frozen quantile interpolation.

Every summary retains B1/B2/E0 values, self-distance, candidate distance, nearest baseline, ratio, absolute distance and whether it exceeds the diagnostic `2*d_B` reference. Diagnostic outliers never directly make `g0_equivalent=false`.

This does not make the layer optional. Missing or duplicate names, wrong cardinality, dtype/trailing-shape mismatch, unsupported optimizer state, empty Tensor or non-finite value remains a hard failure. No implementation may omit an outlier, average failures into a pass, or pad, truncate, reorder, match or align Gaussian rows.

## Formal decision

Let:

- `H_exact` be the conjunction of all exact/provenance hard gates;
- `H_safe` be the conjunction of all safety/completeness hard gates;
- `H_behavior` be the conjunction of the observable numerical envelope gates;
- `H_internal_structure` be the conjunction of internal field/cardinality/schema/dtype/trailing-shape/finite gates.

Then schema-3 behavioral G0 is:

```text
G0_behavioral = H_exact and H_safe and H_behavior and H_internal_structure
```

The number and magnitude of internal numerical diagnostic outliers are reported but do not appear in this Boolean expression.

## Report and CLI contract

Backward compatibility is mandatory:

- default mode remains schema version 1;
- explicit `--topology-aware` alone retains the historical schema-2 all-summary-hard behavior;
- explicit `--topology-aware --behavioral-g0` selects schema version 3;
- `--behavioral-g0` without `--topology-aware` is rejected before artifact loading.

A non-exploratory schema-3 invocation additionally requires `--confirmation-contract PATH`. The JSON contract is created and hashed by the read-only preflight before training and contains `contract_version`, a unique `confirmation_id`, the resolved B1/B2/new-E0 directories, role commits, dataset/prior hashes, semantic seed, resolution, iteration/evaluation schedules and normalized role commands. Its preflight record must state that the new E0 directory was absent. The auditor checks the supplied paths and launcher evidence against this contract. Schema-3 exploratory replay does not require a confirmation contract but can never set `g0_equivalent=true`. The already observed E0 has no eligible pre-launch contract and therefore cannot be promoted by changing a CLI flag.

Schema 3 separates evidence into hard and diagnostic collections. The exact JSON names may follow existing project conventions, but the report must expose at least:

- exact/safety hard failures;
- hard fixed-Tensor and observable-scalar results/failures;
- internal diagnostic summary results and outliers;
- diagnostic outlier counts by capture/optimizer family, field and statistic;
- `hard_equivalent`, `g0_equivalent`, `audit_completed`, `exploratory`, factor and exit code.

`g0_equivalent` must equal the conjunction of hard gates in confirmation mode. An exploratory or retrospective invocation can never set it true.

## Independent confirmation protocol

The first E0 8k result was observed before this amendment and is retrospective evidence only. It cannot confirm schema 3.

After this specification, tests and comparator implementation are frozen:

1. retain the existing immutable B1/B2 baseline runs and their launcher contracts;
2. run a no-training preflight that selects a unique confirmation id and new private-view/output paths, proves the new E0 output path is absent, writes the confirmation contract above and records its SHA256 in the approved execution record;
3. launch one new, previously unseen E0 run from exact commit `a26082154889ed539322425347af5a57a859a52f`;
4. use Tool Room, resolution 2, semantic seed 0, 8,000 iterations, the same runtime/data/prior snapshot and required evaluation boundaries;
5. use the frozen new private input view and output directory; do not overwrite or mutate any prior run;
6. run the schema-3 comparator once against frozen B1/B2 and the unseen E0, supplying the frozen confirmation contract;
7. preserve all input/run hashes and the complete diagnostic layer regardless of PASS or FAIL.

If every hard gate passes, the unseen triplet establishes revised G0 and permits requesting D0 implementation. Any hard failure preserves the result and stops. Neither outcome authorizes changing the contract after the run.

## Required tests

Tests must be written before implementation and cover:

- schema-1 and schema-2 outputs and decisions remain unchanged;
- behavioral mode requires topology-aware mode and emits schema 3;
- non-exploratory schema 3 requires a valid confirmation contract, rejects path/commit/command/hash mismatches and rejects a contract that did not record the E0 path as absent;
- a synthetic report with hard gates passing and internal numerical outliers yields `g0_equivalent=true` while retaining every outlier;
- evaluation, count or fixed application Tensor envelope failure yields `g0_equivalent=false`;
- exact/provenance, optimizer structure/step, dtype or trailing-shape mismatch yields failure;
- missing, duplicate, empty or non-finite internal summary input yields failure;
- internal outlier counts are complete and grouped deterministically;
- retrospective/exploratory reports cannot produce confirmation PASS;
- factor stays exactly `2.0` and has no confirmation CLI override;
- no padding, truncation, row matching or result-dependent field deletion path exists;
- the current schema-2 report replays as its original FAIL and as schema-3 retrospective diagnostics without becoming a confirmation PASS.

AutoDL verification must run the focused comparator tests, full repository suite, compilation/CLI checks and a read-only dry audit before the unseen E0 launch is requested.

## Scope and stop conditions

Allowed implementation files are limited to the read-only G0 comparator, its tests and planning/evidence documents. Training, renderer/CUDA, losses, gradients, topology, datasets, prior artifacts, old run directories and tags remain out of scope.

Stop and report if schema-1/schema-2 behavior changes, a hard field must be demoted to make a result pass, diagnostic summaries cannot be reproduced completely, artifacts change during audit, or the unseen E0 fails any hard gate. Do not begin D0/C1 until the unseen schema-3 confirmation passes and the user separately approves the next phase.
