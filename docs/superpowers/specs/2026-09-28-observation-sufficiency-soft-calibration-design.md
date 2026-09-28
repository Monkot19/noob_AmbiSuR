# Observation-Sufficiency Soft Half-Saturation Design

**Status:** Approach A selected by the user on 2026-09-28. This written specification awaits user review. Implementation has not started.

**Authority:** `docs/research/ambisur-reliability-routing-design.md` remains the project method contract. This document proposes one narrowly scoped revision to Section 4.2. It does not authorize implementation until the user approves this written specification and the subsequent implementation plan.

## Intent

Preserve the scientific claim that external-prior need depends on both appearance ambiguity and insufficient observation, while restoring ranking resolution that the current hard caps discard. The revision must remain GT-free during training, deterministic, monotone, interpretable across scenes, and compatible with the frozen definition

\[
N_i=1-S_i(1-A_i).
\]

Success is not defined by improving Tool Room after tuning against its GT. Success first means implementing the selected fixed calibration exactly and preserving all isolation contracts. A new preregistered Tool Room r2/seed-0 run must still pass the unchanged formal G1 gate before work may proceed to the written C1 specification and implementation plan. That single-scene gate is an internal mechanism gate only: it is not evidence of method validity or generalization. The E3 protocol still requires Tool Room and Utility Room with the prescribed multiple seeds, and a positive Tool Room seed-0 result cannot waive or replace that evaluation.

## Evidence trigger

The corrected iteration-7000 component diagnostic evaluated all 1,424,279 finite Tool Room Gaussian centers and verified the complete frozen provenance. It found useful directional signal at the tails but severe loss of resolution:

- `S` lowest/highest decile high-error rates were `0.1813/0.0867`;
- `S_angle` lowest/highest decile high-error rates were `0.2057/0.0856`;
- `S_count` was already exactly 1 from its fifth percentile upward;
- `40.93%` of all rows had `S=1`, and `52.82%` had `S>=0.95`;
- formal G1 still failed: `AUROC(N)=0.542911`, best component `AUROC(A)=0.550844`, gain `-0.007933`.

This evidence does not show that observation sufficiency is useless. It shows that the current finite hard caps collapse a large fraction of otherwise ordered observations into ties. The selected revision addresses only that calibration defect.

## Frozen revised equations

The definitions of per-camera binary observation `o_iv`, view count `M_i`, camera direction `d_iv`, direction sum `q_i^view`, and dispersion `D_i` remain exactly as approved. The fixed constants also remain:

\[
K_c=5,
\qquad
\theta_c=30^\circ,
\qquad
D_c=\frac{1-\cos\theta_c}{2}.
\]

Only the two hard-capped calibration functions change. The revised count sufficiency is

\[
S_i^{count}=\frac{M_i}{M_i+K_c}.
\]

The revised angular sufficiency is

\[
S_i^{angle}=\frac{D_i}{D_i+D_c}.
\]

The approved composition remains

\[
S_i=\sqrt{S_i^{count}S_i^{angle}},
\qquad
N_i=1-S_i(1-A_i).
\]

`K_c` and `D_c` are now half-saturation anchors rather than terminal clipping thresholds:

- `M_i=K_c` gives `S_i^{count}=0.5`;
- `D_i=D_c` gives `S_i^{angle}=0.5`.

For non-negative finite inputs, both functions are monotone, bounded in `[0,1)`, equal zero only at zero evidence, and never create a finite hard plateau at 1. The existing clamp that makes `D_i` a valid value in `[0,1]` remains; no output quantile normalization, learned calibration, GT-derived parameter, or scene-dependent scale is introduced.

## Scientific interpretation

The revision does not redefine what `S` or `N` means. More distinct observed views and more dispersed viewing directions still increase sufficiency. Higher sufficiency still reduces external-prior need only to the extent that appearance ambiguity is low. High ambiguity can still keep `N` high even under strong observation.

The revision deliberately changes the numeric meaning of the two reference constants. Five views and the 30-degree dispersion reference now mark the midpoint of the corresponding evidence response. They no longer assert that all evidence beyond those points is indistinguishably sufficient.

## Scope and exclusions

This specification permits changes only to observation-sufficiency calibration and the minimum state/version/test changes needed to make that semantic change reproducible.

It does not change:

- appearance ambiguity `A`;
- the formula for `N`;
- EMA coefficients or refresh intervals;
- prior or geometry reliability;
- five-state arbitration, hysteresis, or thresholds;
- parameter routing, gradient projection, or lifecycle behavior;
- G1 labels, evaluation domain, thresholds, or relative-gain requirement;
- GT access rules;
- the G0 feature-off contract;
- Supporting features.

The corrected diagnostic found that current geometry multiview, depth-normal, and support components have the intended direction while historical `T_g/r_g` does not. That temporal reliability discrepancy is a separate pre-C2 investigation and must not be folded into this C1 sufficiency revision.

## State and compatibility contract

The evidence tensor inventory does not change, but the persisted `S` EMA now has different semantics. `EvidenceAccumulator.STATE_VERSION` must therefore increment from 3 to 4. New code must reject an old version-3 evidence state for training resume rather than mix hard-cap and soft-calibration EMA history.

`D0ShadowRuntime.STATE_VERSION` and the event/snapshot schemas need not change because their structures and meanings outside the nested evidence state are unchanged. The nested evidence-version check provides the fail-closed resume boundary. Old formal runs remain immutable and readable by offline evaluators; they are not resumable or promotable under the new calibration.

The same distinction must be enforced at the formal-evaluation boundary, not merely documented. Before a non-exploratory new-calibration G1 evaluation computes a decision or publishes a formal bundle, the evaluator must:

- load both iteration-3000 and iteration-7000 checkpoints through the versioned Core checkpoint parser and require `core_state.evidence.version == 4` in each;
- require the run identity, resolved configuration, seed, resolution, exact training command, dataset SHA, aligned-prior SHA, formal checkpoint/evaluation iterations, and output confirmation ID to match one immutable preregistration record;
- require the exact training commit in the run identity to match the formula commit frozen by that record;
- verify the record against its separately frozen expected SHA256 (stored in a detached sidecar or supplied as an immutable launch input) before trusting any of its fields; and
- fail before formal publication if any identity, version, or fingerprint differs.

A version-3 asset may still be opened for explicit read-only or exploratory diagnosis. It must never produce a new-calibration formal PASS: formal mode must reject it, while exploratory mode must keep `g1_pass` and every promotion decision null.

Every new experiment remains identified by exact commit, resolved configuration, run identity, seed, dataset hash, and a fresh output path. No old checkpoint, snapshot, report, or tag may be overwritten or relabelled.

## Implementation boundary

The production equation change belongs in `compute_observation_sufficiency` in `reliability/evidence.py`. It must use the existing device, dtype, chunking, detached inputs, and `torch.no_grad()` behavior. No new CLI calibration knob is added: the selected formula and constants are fixed method semantics, not experiment-time tunables.

Tests may require updates in the focused evidence, accumulator-state, D0 runtime, diagnostics, offline-input, and GPU suites. Unrelated refactors are prohibited. The implementation must be developed on an isolated `codex/` branch after the implementation plan is approved.

## Error handling and invariants

Existing malformed input checks remain. The implementation must additionally make the following behavior explicit:

- `k_c` must be positive and finite;
- `theta_c_degrees` must be finite and strictly between 0 and 180;
- empty point domains preserve the existing empty-tensor behavior;
- outputs must be finite and remain on the Gaussian input device and dtype;
- `M=0` or `D=0` yields zero combined sufficiency;
- increasing `M` with fixed positive `D`, or increasing `D` with fixed positive `M`, cannot reduce `S`;
- no finite valid `M` or `D` maps either component exactly to 1 under the supported camera-count domain.

Any violation fails closed. The implementation must not silently restore the old terminal clamp.

## TDD and verification

Implementation must begin with failing tests for hand-checked values and state compatibility. The minimum test contract is:

1. mathematically hand-checked count anchors `M=0 -> 0`, `M=5 -> 0.5`, and `M=10 -> 2/3`, compared with the existing dtype-appropriate numeric tolerance; values such as 0 and 0.5 may additionally require exact equality when the active dtype represents them exactly;
2. mathematically hand-checked angular anchors `D=0 -> 0` and `D=D_c -> 0.5`, under the same tolerance and exact-representability rule;
3. monotonicity and boundedness over representative finite counts and dispersions;
4. no hard plateau for distinct valid inputs that the former caps mapped to 1;
5. unchanged geometric-mean composition and `N=1-S(1-A)`;
6. chunked versus unchunked equality within the existing numeric tolerance;
7. dtype/device/no-grad preservation on CPU and CUDA;
8. evidence state version-4 round trip and explicit rejection of version 3 for resume;
9. synthetic CPU and CUDA topology-migration regression, including state reset/inheritance and version-4 checkpoint/snapshot alignment, without claiming that a pre-densification smoke exercised real topology change;
10. non-exploratory formal G1 rejection when either the iteration-3000 or iteration-7000 checkpoint is version 3, when the two versions differ, or when either checkpoint cannot be parsed through the versioned Core state contract;
11. canonical confirmation-record generation rejects pre-existing run/view/report/archive/output targets, writes the record before training, emits a detached SHA256, and detects any later record mutation;
12. formal G1 fail-closed rejection before publication for a wrong commit, config, seed, resolution, command, dataset/prior SHA, iteration protocol, confirmation ID, or confirmation-record SHA; version-3 exploratory diagnosis remains readable but cannot emit a formal decision;
13. feature-off dispatch and behavioral G0 tests remain unchanged.

Before any cloud experiment, run CPU tests, static compilation, focused CUDA tests, and a clean-worktree/no-active-training preflight.

## Experiment sequence and decision policy

The sequence is fixed to avoid parameter fishing:

1. Implement the single selected formula with TDD. Do not test alternative constants or transformations against GT.
2. Run a short Tool Room r2/seed-0 500-step D0 shadow smoke with five refreshes. Audit version-4 state and checkpoint round trips, finite values, no GT references, and training isolation. This run ends before the first real densification at iteration 600 and therefore supplies no real-topology-migration evidence.
3. Before creating the formal run, freeze an immutable confirmation/preflight record. It must contain the exact formula commit; the fixed equations and `K_c=5`, `theta_c=30 degrees`; seed, resolution, and exact training command; dataset and aligned-prior SHA256; the 3000/7000 checkpoint and evaluation protocol; the new confirmation ID and proof that its run, view, report, archive, and output paths are absent; and the unchanged G1 thresholds. After canonical serialization, compute its SHA256 into a detached sidecar and record that expected digest in the launch log. The record and detached digest are inputs to formal evaluation, not after-the-fact annotations.
4. If the smoke and preregistration gates pass, run one new Tool Room r2/seed-0 7000 formal D0 shadow trajectory with the same refresh/checkpoint protocol used by formal-v2. The previous formal-v2 run remains read-only. Audit the real point-count changes and require iteration-matched checkpoint/snapshot row alignment across the actual topology-changing trajectory.
5. Run the unchanged G1 metric computation at iterations 3000 and 7000, with iteration 7000 as the only decision point, but require the new formal-admission checks above before metric evaluation and atomic publication.
6. Report the result whether it passes or fails. Do not modify constants after observing it.

The unchanged hard G1 requirements remain:

\[
\operatorname{AUROC}(N)>0.60
\]

and

\[
\operatorname{AUROC}(N)-
\max\{\operatorname{AUROC}(A),\operatorname{AUROC}(1-S)\}\ge0.03.
\]

A pass authorizes only the existing written C1 specification and implementation-planning gate, not C1 execution by itself and not a scientific claim of effectiveness or generalization. Utility Room and the required multiple seeds remain mandatory under E3 and cannot be skipped because Tool Room seed 0 passes. A failure stops C1 and is recorded as evidence against this fixed operationalization; it does not authorize another same-scene calibration search.

## Completion criteria

This design is complete only when:

- the written specification and implementation plan are separately approved;
- the RED-GREEN test history is preserved in separate commits;
- local and AutoDL verification pass on exact commits;
- the 500-step smoke passes its five-refresh, version-4 round-trip, finite-value, no-GT, and training-isolation audits without being cited as real-topology evidence;
- synthetic topology migration passes on CPU/CUDA, and the new 7k run demonstrates real topology changes with exact iteration-matched checkpoint/snapshot alignment;
- a pre-run confirmation record freezes the exact commit, equations/constants, seed, resolution, command, input hashes, 3000/7000 protocol, absent output paths, and G1 gates, with a separately frozen and verified SHA256;
- formal admission rejects version-3 or mixed-version checkpoints and every mismatch against that confirmation record before publication;
- the new formal 7k run and unchanged formal G1 evaluation are published immutably;
- the result is recorded without changing the preregistered gate;
- any Tool Room seed-0 PASS is reported only as authorization to draft the C1 specification/plan, while Utility Room and multi-seed E3 obligations remain explicit;
- no C1, C2, Supporting feature, or lifecycle code is introduced under this task.
