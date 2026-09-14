# G0 Topology-Aware Feature-Off Equivalence Design

**Status:** historical schema-2 acceptance contract; its 2026-09-10 formal FAIL remains valid. For future G0 acceptance, its per-summary numeric hard gate is superseded by the approved behavioral-equivalence amendment in `2026-09-11-g0-behavioral-equivalence-design.md`. Summary construction and diagnostics remain active.

**Authority:** `docs/research/ambisur-reliability-routing-design.md` §13 remains the highest-priority method specification. This file records the implementation-facing form of its 2026-09-04 topology-aware amendment.

## Problem and evidence

The 8k exact-baseline B1/B2 runs used identical effective arguments, optimization options, normalized commands, data/prior hashes, runtime declarations and optimizer structure/steps. Both completed without logged errors or non-finite values. Nevertheless, the first logged Gaussian-count difference appears at iteration 600, the baseline's first eligible densification: 206,147 versus 206,146. At iteration 8000, pre-topology counts differ by 7,596 (0.506%) and post-topology checkpoint counts differ by 6,032 (0.443%).

The original confirmation attempt therefore remains FAIL under its frozen exact-count/shape contract. The evidence establishes baseline topology divergence in this runtime, not the exact low-level CUDA operator that caused it. E0 has not run, so no E0 evidence was used to select this amendment.

## Objective

G0 must determine whether E0 with every Core feature off introduces deviation beyond exact-baseline self-repeatability. It must not require two stochastic trajectories with dynamic densification/pruning to retain identical Gaussian identities or cardinality.

This amendment changes only the read-only G0 comparator. It does not change training, renderer/CUDA, losses, gradients, topology actions, the frozen factor `2.0`, C1's same-state gradient oracle, or the original run artifacts. The new report mode is selected explicitly with `--topology-aware`; the default schema-1 path remains unchanged for the 500 exploratory replay and existing callers. Confirmation at 8k must use the explicit mode and emits schema version 2.

## Exact layer

The following remain exact across B1, B2 and E0, subject only to predeclared role commit and private/output path differences:

- canonical data and aligned-prior hashes;
- common effective configuration and normalized command;
- role commit, clean state, runtime and feature-off legacy dispatch metadata;
- checkpoint schema, field set and dtype;
- every Gaussian-indexed Tensor's trailing shape (`shape[1:]`);
- optimizer group order/names, hyperparameters, state keys and step counters;
- fixed-shape unupdated fields;
- required artifacts, exit status, log completion and zero error/non-finite tokens.

A missing value, empty Gaussian Tensor, non-finite value, dtype/trailing-shape mismatch or structural mismatch fails G0. Dynamic Gaussian count and only the corresponding leading dimension are excluded from the exact layer.

## Numerical layer

For any scalar or fixed-shape learned Tensor field and distance `D`, define:

```text
d_B = D(B1, B2)
d_E = min(D(E0, B1), D(E0, B2))
```

The frozen condition remains:

```text
d_B == 0  -> candidate pair must be exact with d_E == 0
d_B > 0   -> d_E <= 2 * d_B
```

Fixed-shape learned Tensors, including `app_model` tensors, retain direct element-order RMSE and MAE gates. Predeclared loss/evaluation metrics use absolute distance.

Pre-topology log/PLY count and post-topology checkpoint count are separate named scalar gates using absolute distance. Their save stages must never be mixed.

## Gaussian-indexed summaries

For a Gaussian-indexed Tensor with shape `[N, d1, ..., dk]`, flatten only the trailing dimensions to `[N, D]`. Do not sort or match rows. For every semantic trailing channel and for the per-row L2 norm, compute:

```text
mean
population standard deviation (unbiased=False)
quantiles: 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99
```

Every `field/channel-or-row-norm/statistic` is a separately named scalar gate using absolute distance and the frozen envelope. The comparator may not average across channels, statistics or fields to hide a failure. Each scalar sample vector is converted to CPU float64 and sorted before all reductions, so mean, population standard deviation (`unbiased=False`) and quantiles use the same canonical order and are invariant to Gaussian-row permutation. This value sorting is internal to a statistic and never creates Gaussian correspondence. For sorted values `z[0]...z[n-1]`, quantile interpolation is fixed as `a=(n-1)q`, `l=floor(a)`, `u=ceil(a)`, `Q=(u-a)z[l]+(a-l)z[u]`, with the sole order statistic used when `l==u`.

This applies to Gaussian parameters, densification proxy/state Tensors and Gaussian-indexed Adam `exp_avg`/`exp_avg_sq`. Optimizer structure and steps remain in the exact layer. Raw Tensor hashes, element counts, min/max and learned PLY hashes are diagnostics only.

## Prohibited normalizations

The comparator must not:

- pad, truncate, reorder, nearest-neighbor match or otherwise align Gaussian rows;
- remove a failed field or summary after observing E0;
- average failures across fields/statistics;
- change factor `2.0` or choose a new baseline reference after results;
- reinterpret the original exact-contract run as a PASS.

## Artifact reuse and decision

The existing B1/B2 are allowed as the baseline envelope because E0 was stopped before launch and the summary contract is frozen before any E0 artifact exists. After comparator RED→GREEN tests and server component verification, launch only the already-planned E0 role with the unchanged 8k command, input snapshot and private-view discipline.

G0 passes only when every exact invariant, count gate, summary component, fixed-shape numerical gate, evaluation scalar and safety/resource gate passes. Any failure preserves all artifacts and stops D0/C1. A PASS authorizes requesting D0 execution; it does not itself authorize D0, C1, a tag, or a method-source change.

## Required tests

- differing leading Gaussian dimensions are accepted for summary construction, while dtype/trailing-shape mismatches fail;
- row permutation leaves every summary exactly unchanged;
- mean/std and every frozen quantile are computed in float64 with the fixed definition;
- each channel/statistic is gated independently, so one failed component cannot be averaged away;
- pre/post counts are separate scalar envelope gates, including boundary, over-boundary and zero-self cases;
- empty/non-finite inputs fail;
- Gaussian-indexed optimizer moments use summaries, while optimizer structure/steps remain exact;
- fixed-shape app tensors retain direct RMSE/MAE;
- padding, truncation or row-matching code paths do not exist;
- existing 500 exploratory behavior and legacy two-run comparator remain unchanged;
- confirmation output records all three distances, nearest baseline, factor and original-run provenance.

## Scope

Files allowed in the implementation phase are limited to the G0 comparator, its CPU/AutoDL tests and planning/evidence documents. Training and method source, renderer/CUDA, datasets, baseline results, Git tags and server dependencies are outside scope.
