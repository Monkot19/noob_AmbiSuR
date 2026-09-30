# Geometry-Update Authorization Hypothesis Design

**Status:** Stage G-A research charter, draft for formal review. The in-chat design summary was confirmed on 2026-09-30. This is not a complete implementation specification and authorizes neither implementation planning nor experiment execution.

**Scope:** Bound the analytic and synthetic Stage G-A needed to define a new, action-specific hypothesis for predicting when a geometry update is beneficial and safe. This charter is independent of the frozen prior-risk transfer line. It supersedes every future-facing use of `r_g` as a geometry-risk or routing-authorization quantity in `2026-09-29-prior-geometry-risk-independent-validation-design.md`.

## 1. Frozen starting point

Task 3 concluded `NO_SEMANTIC_REPAIR_JUSTIFIED` on the approved formula, synthetic contracts, production collector/reprojection boundaries, version-4 state at iterations 3,000 and 7,000, and a Tool Room no-GT read-only audit.

The resulting boundary is final for this research line:

- `T_g/r_g` conform to their approved implementation semantics;
- that conformance does **not** establish external geometric correctness or safe geometry-update authorization;
- current Tool Room evidence has not validated external geometric correctness or safe routing authorization for this construct, so it does not hold that responsibility under the current protocol;
- `T_g/r_g` are archived only as **internal geometry-consistency telemetry**;
- `1-r_g` is not a risk candidate, label, feature, threshold source, or routing input;
- Evidence remains version 4; no version-5 migration is authorized;
- no sign flip, complement, threshold change, or GT-driven reinterpretation is allowed;
- the original five-state arbitration and C1 remain blocked.

The new question is deliberately different:

> Before applying one precisely specified geometry update to one preregistered spatial intervention unit, can a GT-free quantity predict whether that update will be beneficial and safe under a frozen finite-horizon evaluation?

This is an action-authorization problem, not a static geometry-correctness classifier.

## 2. Scientific estimand

### 2.1 Paired intervention unit

The causal/diagnostic unit is one **pre-action spatial cluster** `C`, not one Gaussian and not an arbitrary simultaneous set of candidates. Stage G-A must define one deterministic, GT-free cluster-construction rule from the pre-action state, including its coordinate frame, neighborhood rule, membership tie breaks, minimum/maximum size, guard region, validity, and behavior near scene boundaries. Until that exact rule is approved, no paired target may be generated.

Stage G-A forbids batching: every branch pair treats exactly one valid cluster. Spatial separation or a guard distance cannot establish absence of cross-cluster interference because rendering and optimizer coupling are global. Any future batching proposal requires a separate approved no-interference contract and synthetic validation; it cannot rely on distance alone.

Before any empirical target is generated, one clean cluster-level geometry update action `a_C` must also be specified completely. Its implementation, affected parameter set, per-member update, aggregation, magnitude/clipping, support requirements, source observations, optimizer interaction, and invalid-row behavior must be frozen.

The first admissible action is a bounded update derived only from the existing depth/normal and multiview geometry observations. It may not add Ray-Color, a new appearance loss, learned confidence, GT, or any Supporting-stage capability.

For cluster `C` at state `x_t`, compare two paired branches:

```text
a_C = 0: do not apply the candidate geometry update to members of C
a_C = 1: apply exactly the frozen candidate geometry update to members of C
```

Each branch starts from byte-identical model and optimizer states and uses the same camera/view schedule, RNG stream, training inputs, topology schedule, and non-geometry operations. The branches then own separate optimizer/model state; downstream divergence caused by `a_C` is part of the cluster-level effect, not an implementation mismatch.

The full scene remains in both renders. Gaussians outside `C` receive the same rules and no direct candidate update, but they may change indirectly through rendering, gradients, optimizer coupling, or topology. Those changes are **spillover** and must be measured by preregistered protected non-target safety outcomes. They cannot be reassigned as per-Gaussian treatment effects.

Topology is handled as follows:

- the schedule and decision code are identical across branches, while action-caused differences in split/prune decisions are permitted downstream effects and must be reported;
- a surviving row follows the existing `new_to_old` mapping;
- all split/clone descendants of a pre-action member of `C` belong to that member's treatment lineage for outcome aggregation, while retaining explicit `is_new` identity;
- pruning a treated lineage member is not missing data: it contributes the preregistered disappearance penalty and may trigger an unsafe outcome;
- descendants of rows outside `C` remain outside the treated lineage, and their changes contribute only to spillover/safety summaries;
- a lineage ambiguity, unsupported topology operation, or failed branch alignment invalidates the pair.

Because rendering and optimization couple Gaussians, this charter makes no no-interference assumption at the Gaussian level. It therefore forbids any per-Gaussian treatment-effect notation or claim for branch differences.

### 2.2 Benefit and safety target

Let `Y_C(a_C; H)` be the preregistered cluster-lineage outcome after branch `a_C`, evaluated at horizon `H`. Define the paired cluster-level effect

```text
tau_C(H) = Y_C(0; H) - Y_C(1; H).
```

Positive `tau_C` means the update improves the frozen cluster-lineage outcome. Authorization requires both:

1. **benefit:** `tau_C(H)` exceeds a preregistered practical-effect margin;
2. **safety:** the branch passes preregistered finite-state, bounded-update, protected-appearance, optimizer, topology, and non-target-spillover constraints.

The target is the total paired effect of this one cluster action in this one state and horizon, including its within-lineage downstream consequences. It is not a claim that any member Gaussian is globally correct, individually causal, reliable, or permanently safe.

The exact `Y_C`, lineage aggregation, disappearance penalty, horizon `H`, benefit margin, and safety/spillover tolerances must be chosen from mathematical semantics and synthetic calibration before Tool Room target inspection. They may not be selected for favorable Tool Room or Utility metrics.

## 3. Candidate authorization signal

The future candidate is provisionally named `Q_g`. This charter does not yet define its formula. The name carries no validity until a separately approved Stage G-A exact sub-specification freezes one formula and the staged gates below pass.

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

- detached from the source production run and unable to write its parameters, gradients, optimizer, or artifacts;
- branch-private gradients/optimizer evolution is allowed only when the Stage G-A exact sub-specification explicitly requires a finite-horizon paired simulation;
- no influence on the source run's densification/pruning/topology decisions;
- no change to source/production training loss or gradient scaling;
- no checkpoint field reused from `T_g/r_g`;
- feature-off behavior remains baseline equivalent.

If a later approved implementation needs persistent history, it must receive a new, separately named state contract and schema. It may not silently reuse Evidence v4 geometry history or be called Evidence v5 merely because Task 3 once considered that number.

The temporal contract must specify when a pre-action feature is sampled, when the action occurs, how the finite horizon is measured, how topology migration maps paired rows, and when a row becomes invalid. Newly created rows cannot inherit authorization merely from a parent.

## 5. Stage G-A deliverable and later-stage firewall

### Stage G-A: analytic and synthetic contract

The only next document permitted by this charter is a **Stage G-A exact sub-specification**. Before any TDD implementation plan, synthetic execution, or target generation, that sub-specification must freeze:

- the deterministic spatial cluster unit and guard/interference rules;
- the exact candidate action and every affected parameter;
- `Y_C`, lineage aggregation, disappearance handling, and horizon `H`;
- benefit margin and every protected safety/spillover constraint;
- the unique `Q_g` formula and all constants;
- component and combined validity predicates;
- metrics, coverage definition/minimum, uncertainty, and positive/negative/inconclusive gates;
- compute, memory, branch-count, and wall-time budgets;
- synthetic cases, expected directions, and stop rules.

Only after that sub-specification is separately approved may a Stage G-A TDD implementation plan be written. The plan may cover hand-checked synthetic scenes and controlled perturbations to establish:

- exact action direction and bounded magnitude;
- benefit/safety target semantics;
- validity/support behavior;
- monotonic expectations;
- temporal ordering and topology mapping;
- degenerate, unsupported, non-finite, and contradictory cases.

The synthetic generator must include cases where geometry is wrong but the update is harmful, and cases where geometry is already good but a small update is beneficial. It must also exercise cluster spillover, protected non-target rows, split/clone descendants, pruning/disappearance, and invalid lineage. This prevents `Q_g` from collapsing into a static error score.

### Stage G-B: Tool Room development domain

Only after Stage G-A implementation and its positive gate are complete and frozen may a separate Tool Room intervention specification be drafted. This charter does not authorize that specification, its TDD plan, or its execution. A future approved Tool Room protocol may generate paired, short-horizon cluster targets under the frozen action; its spatial and seed-level development/holdout partitions must be fixed before target inspection.

Tool Room may only test or reject the unique `Q_g` formula frozen by the approved Stage G-A exact sub-specification. It may not select, delete, or recombine components, refit the formula or constants, or substitute another candidate. The final Tool Room holdout is single-use. No result may be rescued through a sign flip, threshold search, alternate horizon, alternate action, or post-hoc row filter.

### Stage G-C: pre-Utility freeze

Before any Utility GT-derived value is read, a later Stage G-C confirmation must make the following immutable and checksummed:

- exact repository commit and clean status;
- action implementation and branch-pairing procedure;
- target equation, horizon, benefit margin, and safety constraints;
- one `Q_g` formula and all constants;
- component and combined validity predicates;
- temporal/topology/state contract;
- sample domain and spatial split;
- metric, uncertainty, and stop rules;
- output schema and confirmation record.

Before this freeze, Utility GT handling is limited to path existence, byte size, and SHA256. No mesh parsing, distance query, label, overlay, summary, or GT-guided preprocessing is allowed. The alternative terminal release path is defined by the prior-risk transfer specification: an immutable checksummed geometry termination confirmation may release the prior-only probe without admitting any geometry candidate to Utility.

Passing Stage G-C authorizes only a later request for independent Utility validation. It does not authorize routing or C1.

## 6. Preregistered stop rules

At the applicable approved stage, the geometry-authorization branch stops with `NO_ACTION_SPECIFIC_SIGNAL` if any of the following is established under a valid evaluation:

- the candidate predicts static geometry error but not the paired update effect;
- synthetic benefit direction or safety monotonicity fails;
- validity or support semantics require an arbitrary imputation;
- the Tool Room holdout effect is directionally inconsistent or practically negligible;
- safety violations concentrate in the authorized tail;
- usable authorization coverage falls below the preregistered minimum;
- a useful result requires `r_g`, GT, post-action leakage, alternate signs, candidate sweeps, or threshold changes;
- compute or memory overhead violates the preregistered budget.

The result is `INCONCLUSIVE` rather than a failure when corrupted inputs, invalid paired branches, missing classes/effects, non-finite outputs, infrastructure failure, or an unfrozen contract prevents the estimand from being computed. An inconclusive result cannot trigger a fallback candidate.

When an approved stage returns `NO_ACTION_SPECIFIC_SIGNAL`, it must publish a canonical termination confirmation plus detached SHA. That record binds the exact charter/sub-specification/commit, stage, frozen tests and evidence, exhaustive outcome, and the facts that no geometry candidate enters Utility and Utility results cannot reopen or revise the terminated branch. Reopening geometry research would require a new hypothesis, new specification, and a different untouched independent validation asset; it cannot reuse Utility feedback from the prior-risk line.

The only eventual positive development outcome is `GEOMETRY_AUTHORIZATION_HYPOTHESIS_READY`. Its precise positive gate must be frozen in the Stage G-A exact sub-specification and retained through later approved stages. It means the one frozen, action-specific candidate is coherent enough for a later request for independent Utility validation. It is not a causal claim beyond the paired cluster intervention, a production authorization, a five-state authorization, or C1 approval.

## 7. Relationship to the prior-risk line

The two research lines answer different questions:

| Line | Quantity | Question | Forbidden inference |
|---|---|---|---|
| Prior-risk transfer | `1-r_p` on `V_p=True` | Does an external-prior warning add conditional predictive information across scene/seed? | permission to strengthen prior gradients or replace geometry authorization |
| Geometry-update authorization | future `Q_g` on `V_Q=True` | Is one specified geometry action beneficial and safe in the current state? | static correctness, general reliability, or reuse of `r_g` |

Neither result can substitute for the other. A positive prior-risk transfer result alone cannot reconstruct the missing geometry-authorization axis. A positive `Q_g` result would still require a later written architecture and new G0/G1/C1 approvals before it can affect training.

## 8. Current authorization boundary

This Stage G-A charter currently authorizes only formal review and, after approval, drafting of the Stage G-A exact sub-specification. It does not authorize:

- implementation or TDD planning, including a Stage G-A implementation plan;
- a Tool Room intervention specification or full-line implementation plan;
- synthetic generation or Tool Room intervention runs;
- DA3 or Utility preprocessing;
- Utility training or GT evaluation;
- Evidence-version changes;
- production routing, five-state arbitration, or C1.

The next permitted artifact after this charter is approved is the Stage G-A exact sub-specification listed in Section 5. Only after that document is separately approved may a Stage G-A TDD implementation plan be drafted. Every execution stage requires an additional explicit authorization.
