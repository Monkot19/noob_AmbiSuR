# D0 Temporal State-Transition Diagnostics Design

**Status:** architectural specification confirmed by the user on 2026-09-17. Implementation has not started.

**Authority:** `docs/research/ambisur-reliability-routing-design.md` remains the highest-priority method specification. This document defines the implementation-facing clarification for the formal D0 state-transition, duration, and jitter diagnostics.

## Trigger

The first Tool Room r2/seed-0/7000 D0 shadow run completed safely at commit `1eba8db36c3672bbc24832811469dd63fc4d825b`: exit code 0, seven no-GT snapshots, checkpoints at iterations 3000 and 7000, no logged error/nonfinite/GT reference, 628 seconds wall time, and 8,792 MiB peak GPU memory. That run revealed that the existing snapshots contain only the current per-Gaussian state.

Between refreshes, baseline densification and pruning change Gaussian count and row order. Consequently, comparing adjacent `.npz` rows cannot recover temporal state transitions. Such a comparison would be an unapproved approximation. The completed v1 run remains immutable engineering evidence but cannot satisfy the formal transition/duration/jitter reporting contract.

## Frozen meaning

At evidence refresh `t`, a temporal transition is the per-lineage pair

```text
stable(t-1) -> stable(t)
```

where `stable(t-1)` has already been migrated through every intervening topology change into the current Gaussian row domain. The transition matrix is a fixed `5 x 5` matrix in the approved state order:

```text
Bypass, Consensus, Prior-led, Geometry-led, Abstain
```

Rows are previous stable states and columns are refreshed stable states. Both integer counts and fractions over the current Gaussian population must be reported.

This definition is not candidate-to-stable disagreement within one refresh. It is a temporal stable-state transition.

## Topology lineage semantics

The existing `TopologyChange.new_to_old` mapping is the sole identity contract. Before each evidence refresh:

- surviving rows retain their prior state history;
- clone/split children inherit the mapped parent lineage history;
- pruned rows disappear from the current domain;
- no offline row-index matching, nearest-neighbour matching, geometry matching, or persistent synthetic ID is introduced.

This makes the transition diagnostic exact with respect to the topology mapping already used by D0 state migration and does not alter topology decisions.

## Diagnostic state

The shadow runtime maintains three no-gradient diagnostic tensors aligned to the current Gaussian rows:

- `stable_age_refreshes`: consecutive evidence-refresh count in the current stable state;
- `stable_transition_count`: cumulative number of stable-state changes along the mapped lineage;
- the previous stable state already held by the arbitration state machine before the current update.

At each refresh:

1. Copy the topology-aligned previous stable state.
2. Compute the new candidate and update the existing hysteretic stable state exactly once.
3. Set `changed = new_stable != previous_stable`.
4. Update `stable_age_refreshes = 1` for changed rows and `+1` for unchanged rows.
5. Increment `stable_transition_count` only for changed rows.
6. Build the `5 x 5` temporal transition matrix from the previous/new stable pair.

At the first refresh, the initialized Bypass state is the valid previous state. Thus the first matrix describes initial Bypass-to-refreshed-state behavior rather than fabricating an unavailable pre-training observation.

## Persisted output

The existing `iteration_*.npz` field inventory and row-join contract remain unchanged. Per-refresh summary diagnostics are appended to the corresponding `events.jsonl` record:

- `transition_count_matrix`: `5 x 5` non-negative integers;
- `transition_fraction_matrix`: `5 x 5` finite fractions summing to 1 for a non-empty population;
- `jitter_count` and `jitter_rate`, where jitter is `changed` at this refresh;
- overall and per-state mean `stable_age_refreshes`;
- overall and per-state mean `stable_transition_count`;
- the existing iteration, point count, and joint-valid count.

The aligned diagnostic tensors must be included in the versioned D0 runtime checkpoint state so resume and topology migration preserve their semantics. Feature-off checkpoints retain the legacy tuple schema and contain none of these fields.

## Training isolation

All new calculations execute under the existing D0 `torch.no_grad()` refresh path. They must not:

- write model parameters or `.grad`;
- add a backward pass or optimizer step;
- alter Adam state;
- change the frozen baseline total-gradient densification proxy;
- gate clone, split, prune, multi-view trim, or any lifecycle action;
- consume GT or write GT-derived data into training artifacts.

D0 remains observation-only.

## Formal evaluator contract

The formal evaluator reads the seven enhanced event records for iterations 1000–7000. It generates the already frozen timeline artifacts:

- state proportion;
- temporal state-transition matrices/counts;
- current stable-state duration in refresh units;
- jitter rate;
- joint coverage.

Iteration 7000 remains the sole G1 gate. These diagnostics do not change the AUROC threshold, relative-gain threshold, prevalence stop gate, risk-coverage direction, GT domain, or any C1–C6 method behavior.

## Verification and rerun policy

Implementation must follow TDD and verify:

- hand-checked temporal matrices, age updates, transition counts, and first-refresh behavior;
- topology clone/split/prune migration;
- checkpoint round-trip and resume equivalence;
- unchanged `.npz` field inventory;
- exact event schema and finite/summing fractions;
- feature-off absence and D0 gradient/optimizer/topology isolation;
- a short multi-refresh GPU smoke before the formal rerun.

The official rerun uses a new immutable output path ending in `d0_formal_r2_seed0_7k_20260917_v2`. The v1 run is never overwritten, moved, deleted, or retroactively promoted. No tag is created, and C1 remains blocked until the v2 D0 isolation audit and formal G1 gate complete.
