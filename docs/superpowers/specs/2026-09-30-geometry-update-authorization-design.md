# Geometry-Update Authorization Hypothesis Design

**Status:** Draft for formal review. The in-chat design summary was confirmed on 2026-09-30. This document authorizes neither implementation nor experiment execution.

**Scope:** Define a new, action-specific hypothesis for predicting when a geometry update is beneficial and safe. This specification is independent of the frozen prior-risk transfer line. It supersedes every future-facing use of `r_g` as a geometry-risk or routing-authorization quantity in `2026-09-29-prior-geometry-risk-independent-validation-design.md`.

## 1. Frozen starting point

Task 3 concluded `NO_SEMANTIC_REPAIR_JUSTIFIED` on the approved formula, synthetic contracts, production collector/reprojection boundaries, version-4 state at iterations 3,000 and 7,000, and a Tool Room no-GT read-only audit.

The resulting boundary is final for this research line:

- `T_g/r_g` conform to their approved implementation semantics;
- that conformance does **not** establish external geometric correctness or safe geometry-update authorization;
- historical GT direction anomalies show that the construct is unsuitable for that responsibility;
- `T_g/r_g` are archived only as **internal geometry-consistency telemetry**;
- `1-r_g` is not a risk candidate, label, feature, threshold source, or routing input;
- Evidence remains version 4; no version-5 migration is authorized;
- no sign flip, complement, threshold change, or GT-driven reinterpretation is allowed;
- the original five-state arbitration and C1 remain blocked.

The new question is deliberately different:

> Before applying one precisely specified geometry update, can a GT-free quantity predict whether that update will be beneficial and safe for the affected Gaussian under a frozen finite-horizon evaluation?

This is an action-authorization problem, not a static geometry-correctness classifier.

## 2. Scientific estimand

### 2.1 Frozen action

Before any empirical target is generated, one clean geometry update action `a` must be specified completely. Its implementation, affected parameter set, magnitude/clipping, support requirements, source observations, optimizer interaction, and invalid-row behavior must be frozen.

The first admissible action is a bounded update derived only from the existing depth/normal and multiview geometry observations. It may not add Ray-Color, a new appearance loss, learned confidence, GT, or any Supporting-stage capability.

For Gaussian `i` at state `x_t`, compare two paired branches:

```text
a = 0: do not apply the candidate geometry update
a = 1: apply exactly the frozen candidate geometry update
```

The branches must share the same initial state, camera/view schedule, RNG state, optimizer state, topology schedule, training inputs, and non-geometry operations. Any implementation detail that differs beyond the action makes the pair invalid.

### 2.2 Benefit and safety target

Let `Y_i(a; H)` be a preregistered finite-horizon geometry error after branch `a`, evaluated at horizon `H`. Define the paired treatment effect

```text
tau_i(H) = Y_i(0; H) - Y_i(1; H).
```

Positive `tau_i` means the update reduces the frozen geometry error. Authorization requires both:

1. **benefit:** `tau_i(H)` exceeds a preregistered practical-effect margin;
2. **safety:** the branch passes preregistered finite-state, bounded-update, protected-appearance, optimizer, and topology constraints.

The target is the effect of this one action in this one state and horizon. It is not a claim that the Gaussian is globally correct, reliable, or permanently safe.

The exact geometry error, horizon `H`, benefit margin, and safety tolerances must be chosen from mathematical semantics and synthetic calibration before Tool Room target inspection. They may not be selected for favorable Tool Room or Utility metrics.

## 3. Candidate authorization signal

The future candidate is provisionally named `Q_g`. The name carries no validity until the staged gates below pass.

`Q_g` must use only information available **before** the candidate update and must be GT-free. Its only admissible component families are:

- cross-view agreement of the proposed update direction and scale;
- a bounded virtual-step estimate of held-out residual improvement;
- valid supporting-view count and spatial distribution;
- local numerical conditioning and update-magnitude diagnostics.

Every component must have an explicit validity predicate. `V_Q=False` means **unknown and unauthorized**, never maximal risk or permission by default. Invalid values may be logged separately but cannot be numerically imputed into `Q_g`.

The following are forbidden inputs or shortcuts:

- `r_g`, `1-r_g`, or a renamed transformation of current geometry telemetry;
- Utility GT, Tool Room GT labels, post-action measurements, or future state;
- opacity, scaling, visibility, or crop filters selected after viewing target performance;
- a sweep over many candidate transforms followed by reporting the best one;
- a learned black-box gate before a mechanistic candidate survives the synthetic and Tool Room stages.

The preferred initial mechanism is paired intervention plus a bounded virtual look-ahead. A static “wrong geometry” classifier is rejected because static wrongness does not identify the effect of applying a particular update. End-to-end learned authorization is deferred because it would obscure the action, validity, and monotonic safety contracts at this stage.

## 4. State and training-isolation contract

Any prototype must initially be diagnostic-only:

- `no_grad` and detached from production parameters;
- no optimizer-state writes;
- no densification/pruning/topology influence;
- no training-loss or gradient scaling;
- no checkpoint field reused from `T_g/r_g`;
- feature-off behavior remains baseline equivalent.

If a later approved implementation needs persistent history, it must receive a new, separately named state contract and schema. It may not silently reuse Evidence v4 geometry history or be called Evidence v5 merely because Task 3 once considered that number.

The temporal contract must specify when a pre-action feature is sampled, when the action occurs, how the finite horizon is measured, how topology migration maps paired rows, and when a row becomes invalid. Newly created rows cannot inherit authorization merely from a parent.

## 5. Staged development and information firewall

### Stage G-A: analytic and synthetic contract

Use hand-checked synthetic scenes and controlled perturbations to establish:

- exact action direction and bounded magnitude;
- benefit/safety target semantics;
- validity/support behavior;
- monotonic expectations;
- temporal ordering and topology mapping;
- degenerate, unsupported, non-finite, and contradictory cases.

The synthetic generator must include cases where geometry is wrong but the update is harmful, and cases where geometry is already good but a small update is beneficial. This prevents `Q_g` from collapsing into a static error score.

### Stage G-B: Tool Room development domain

Only after Stage G-A is frozen may Tool Room be used as the development scene. Tool Room may generate paired, short-horizon intervention targets under the frozen action. Spatial and seed-level development/holdout partitions must be fixed before target inspection.

Tool Room can be used to reject the hypothesis or choose between already specified formula components only within a preregistered development allocation. The final Tool Room holdout is single-use. No result may be rescued through a sign flip, threshold search, alternate horizon, alternate action, or post-hoc row filter.

### Stage G-C: pre-Utility freeze

Before any Utility GT-derived value is read, the following must be immutable and checksummed:

- exact repository commit and clean status;
- action implementation and branch-pairing procedure;
- target equation, horizon, benefit margin, and safety constraints;
- one `Q_g` formula and all constants;
- component and combined validity predicates;
- temporal/topology/state contract;
- sample domain and spatial split;
- metric, uncertainty, and stop rules;
- output schema and confirmation record.

Before this freeze, Utility GT handling is limited to path existence, byte size, and SHA256. No mesh parsing, distance query, label, overlay, summary, or GT-guided preprocessing is allowed. This firewall is shared with the prior-risk Utility transfer specification: Utility GT stays unopened until this geometry contract has reached Stage G-C, even though the two hypotheses remain analytically independent.

Passing Stage G-C authorizes only a later request for independent Utility validation. It does not authorize routing or C1.

## 6. Preregistered stop rules

The geometry-authorization branch stops with `NO_ACTION_SPECIFIC_SIGNAL` if any of the following is established under a valid evaluation:

- the candidate predicts static geometry error but not the paired update effect;
- synthetic benefit direction or safety monotonicity fails;
- validity or support semantics require an arbitrary imputation;
- the Tool Room holdout effect is directionally inconsistent or practically negligible;
- safety violations concentrate in the authorized tail;
- usable authorization coverage falls below the preregistered minimum;
- a useful result requires `r_g`, GT, post-action leakage, alternate signs, candidate sweeps, or threshold changes;
- compute or memory overhead violates the preregistered budget.

The result is `INCONCLUSIVE` rather than a failure when corrupted inputs, invalid paired branches, missing classes/effects, non-finite outputs, infrastructure failure, or an unfrozen contract prevents the estimand from being computed. An inconclusive result cannot trigger a fallback candidate.

The only positive development outcome is `GEOMETRY_AUTHORIZATION_HYPOTHESIS_READY`. It means the one frozen, action-specific candidate is coherent enough for independently approved Utility validation. It is not a causal claim beyond the paired intervention, a production authorization, a five-state authorization, or C1 approval.

## 7. Relationship to the prior-risk line

The two research lines answer different questions:

| Line | Quantity | Question | Forbidden inference |
|---|---|---|---|
| Prior-risk transfer | `1-r_p` on `V_p=True` | Does an external-prior warning add conditional predictive information across scene/seed? | permission to strengthen prior gradients or replace geometry authorization |
| Geometry-update authorization | future `Q_g` on `V_Q=True` | Is one specified geometry action beneficial and safe in the current state? | static correctness, general reliability, or reuse of `r_g` |

Neither result can substitute for the other. A positive prior-risk transfer result alone cannot reconstruct the missing geometry-authorization axis. A positive `Q_g` result would still require a later written architecture and new G0/G1/C1 approvals before it can affect training.

## 8. Current authorization boundary

This document currently authorizes only formal review of the hypothesis design. It does not authorize:

- implementation or TDD planning;
- synthetic generation or Tool Room intervention runs;
- DA3 or Utility preprocessing;
- Utility training or GT evaluation;
- Evidence-version changes;
- production routing, five-state arbitration, or C1.

Each later stage requires a separate approved implementation plan and an explicit execution authorization.
