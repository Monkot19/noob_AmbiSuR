# Geometry Stage G-A Exact Sub-Specification

**Status:** Draft for formal review. This document is the only Stage G-A exact sub-specification permitted by the approved research charter. It does not authorize a TDD implementation plan, synthetic execution, Tool Room intervention, Utility access, or production-method change.

**Parent charter:** `docs/superpowers/specs/2026-09-30-geometry-update-authorization-design.md`

**Question:** Before applying one fixed, GT-free, internal-geometry update to one preregistered spatial cluster, can one pre-action scalar `Q_g` predict that the update will be both beneficial and safe over a fixed short horizon?

## 1. Frozen exclusions and terminology

- The intervention unit and estimand are cluster-level. No branch difference may be called a per-Gaussian treatment effect.
- `T_g`, `r_g`, `1-r_g`, Evidence-v4 geometry history, Utility data, Tool Room GT, and post-action measurements are excluded from `Q_g`, validity, action construction, and target selection.
- `Q_g` is an **action-authorization candidate**, not static geometry correctness, Gaussian reliability, risk, or routing permission.
- Stage G-A is synthetic-only. It cannot change Evidence v4, production training, five-state arbitration, or C1.
- Each branch pair treats exactly one cluster. Batching is forbidden.

Unless stated otherwise, calculations use float64 and `epsilon=1e-8`; comparisons at gates are inclusive except where the word “strictly” is used.

## 2. Deterministic cluster unit

### 2.1 Coordinate frame and row identity

Clusters are constructed from finite pre-action Gaussian centers in the reconstruction world frame. Every row retains its immutable pre-action row index. Opacity, scaling, visibility, state, GT distance, and any post-action quantity cannot select a member.

For an anchor row `a`, sort all other finite rows by `(Euclidean distance to center_a, original row index)`. The proposed membership is the anchor plus the first 31 rows. It is valid only when all of the following hold:

- exactly 32 distinct rows are available;
- the farthest member is at most `R_C=0.05 m` from the anchor;
- all 32 centers and all pre-action observations used below are finite;
- the view-support and residual-validity requirements in Sections 3 and 5 pass.

Thus the frozen minimum and maximum cluster size are both 32. A cluster key is the sorted tuple of its 32 original row indices. If two anchors produce the same key, retain only the lower anchor row. No other overlap removal is performed because every branch pair starts again from the same pre-action state and processes only one cluster.

### 2.2 Guard and non-target regions

Let `d(x,C)` be the minimum Euclidean distance from non-member center `x` to a member center. The pre-action regions are:

```text
guard(C)       = non-members with d(x,C) <= 0.10 m
far(C)         = all remaining non-members
non-target(C)  = guard(C) union far(C)
```

Membership and regions are frozen before either branch starts and follow the lineage rules in Section 7. A boundary cluster is not cropped or padded: it is invalid if 32 neighbors within `R_C` or the required view support do not exist. Distance from another candidate cluster never licenses batching.

### 2.3 Anchor inventory

Stage G-A uses only the exact anchors named by the canonical synthetic fixture manifest. The manifest lists `(fixture_id, anchor_row, cluster_key)` and is hashed before execution. No cluster is added, dropped, or replaced after an outcome is observed.

## 3. Frozen observations and view partition

### 3.1 Internal-geometry observations

The action and `Q_g` use only the current model’s rendered geometry:

- cross-view relative-depth and sign-invariant normal residuals from the approved reprojection contract;
- same-view primitive-normal versus depth-normal residuals from the approved depth-normal contract;
- fixed renderer contribution weights `sg(T*alpha)` and the corresponding pre-action validity masks.

DA3 depth/confidence, image appearance loss, Ray-Color, GT, and any learned confidence are not used. Residual scales are frozen to the existing semantic constants:

```text
tau_d  = 0.05
tau_n  = 0.10
tau_dn = 0.10
```

All visibility, occlusion, finite-value, alpha, and reprojection masks are evaluated at the pre-action state and stop-gradient. Invalid pixels contribute neither a zero residual nor support.

### 3.2 Cluster support per camera

For camera `v`, define cluster support

```text
Z_Cv = sum_{i in C,r} sg(w_ivr) * m_vr.
```

Camera `v` supports `C` exactly when `Z_Cv > 1e-4` and it participates in at least one valid cross-view edge for the cluster. Supporting cameras are ordered by SHA256 of the UTF-8 string `fixture_id|cluster_key|camera_name`; the first 16 are retained. Alternating entries enter proposal set `P` and held-out set `H`, starting with `P`.

`V_views=True` requires:

- at least 8 retained cameras, with at least 4 in each of `P` and `H`;
- at least 4 valid directed reprojection edges whose source and target are both in `P`;
- at least 4 valid directed reprojection edges whose source and target are both in `H`.

No camera may occur in both partitions. A residual whose source or target crosses partitions is excluded from both.

## 4. Unique candidate action

### 4.1 Proposal residual vector

Introduce one shared translation variable `delta in R^3` for all 32 members. With all non-cluster parameters fixed, render the proposal cameras using `mu_i + delta` for `i in C`. Let `R_P(delta)` be the vector of valid, cluster-weighted residual components:

```text
e_depth / tau_d,
e_normal / tau_n,
e_depth-normal / tau_dn.
```

Each component is multiplied by the square root of its normalized fixed pre-action renderer weight. The component inventory, masks, and normalization denominator are frozen at `delta=0`; a changed mask cannot improve the proposal objective.

At `delta=0`, obtain `r=R_P(0)` and the exact autograd Jacobian `J=dR_P/d(delta)`. Define

```text
H_C = (J^T J) / max(number_of_components,1) + lambda I
b_C = (J^T r) / max(number_of_components,1)
lambda = 1e-3
delta_raw = - solve(H_C, b_C)
delta_max = 0.01 m
delta_C = delta_raw * min(1, delta_max / (norm(delta_raw)+epsilon)).
```

The sole action is:

```text
a_C=0: do not inject a translation.
a_C=1: add the same delta_C to xyz of every pre-action member of C.
```

No rotation, scaling, opacity, SH, `knn_f`, exposure, or non-member parameter is directly changed. Injection occurs after byte-identical branch restoration and before the first horizon render. It does not advance Adam, alter its moments, or count as an optimizer step.

### 4.2 Action validity

`V_action=True` requires finite `r`, `J`, `H_C`, `b_C`, and `delta_raw`; at least 256 weighted residual components with all three residual families represented; `lambda_min(H_C)>1e-8`; and a successful exact solve. A zero raw step is valid but cannot pass the practical-benefit target. A failed condition makes the cluster unknown and unauthorized; no fallback gradient, sign, loss, or step size is allowed.

## 5. Unique pre-action authorization candidate

All components below are computed before action injection. They lie in `[0,1]` when valid.

### 5.1 Direction agreement

For each proposal camera with a nonzero per-camera translation gradient `g_v`, set `u_v=-g_v/(norm(g_v)+epsilon)` and weight it by its pre-action cluster support `Z_Cv`. Then

```text
q_dir = norm(sum_v Z_Cv*u_v) / sum_v Z_Cv.
```

It is valid only with at least four finite nonzero `g_v` values.

### 5.2 Held-out virtual improvement

Using the held-out camera set and its masks fixed at `delta=0`, define the weighted RMS residual `L_H(delta)`. Then

```text
rho_H  = (L_H(0)-L_H(delta_C)) / (L_H(0)+epsilon)
q_virt = clip(rho_H / 0.10, 0, 1).
```

It is valid only when `L_H(0)>1e-4`, both RMS values are finite, and the held-out residual inventory contains at least 256 components with all three families represented.

### 5.3 Supporting-view distribution

For every retained supporting camera, let `d_v` be the unit vector from the cluster centroid to the camera center. With `M` retained cameras, use

```text
D = clip((M^2 - norm(sum_v d_v)^2) / (2*M*(M-1)), 0, 1)
D_30 = (1-cos(30 degrees))/2
q_count = clip(M/16, 0, 1)
q_angle = clip(D/D_30, 0, 1)
q_support = sqrt(q_count*q_angle).
```

The same calculation must be finite and nonzero separately in `P` and `H`; `q_support` is the minimum of the combined, proposal-only, and held-out values.

### 5.4 Conditioning and magnitude

Let `kappa_inv=lambda_min(H_C)/(lambda_max(H_C)+epsilon)`. Define

```text
q_cond = clip(kappa_inv / 0.05, 0, 1)
q_mag  = exp(-max(0, norm(delta_raw)/delta_max - 1)).
```

Both require finite eigenvalues, `lambda_max(H_C)>0`, and a finite raw-step norm.

### 5.5 Combined formula and authorization threshold

The only Stage G-A formula is

```text
V_Q = V_cluster & V_views & V_action & V_dir & V_virt & V_support & V_cond & V_mag
Q_g(C) = (q_dir*q_virt*q_support*q_cond*q_mag)^(1/5), when V_Q=True
Q_g(C) = null,                                  when V_Q=False.
```

The only authorized tail is `V_Q=True and Q_g>=0.50`. Invalid clusters are unknown, not score zero, and are excluded from predictive metrics while remaining in the coverage denominator. No component may be removed, complemented, reweighted, transformed, or refit after execution.

## 6. Paired branch timing and finite horizon

Both branches start from byte-identical model, optimizer, RNG, camera schedule, and synthetic training inputs. After the optional action injection, each branch performs exactly `H=8` completed optimizer steps. The same prespecified cameras and random draws are replayed in the same order. The full scene is rendered in both branches.

The synthetic fixture manifest also specifies whether a single topology commit occurs after optimizer step 4. When present, both branches execute the same production clone/split/prune code and thresholds; action-caused decision differences are downstream effects. No other topology event occurs inside the horizon.

The source state is immutable. Each branch owns private tensors, gradients, optimizer state, topology state, logs, and outputs. Any source-state write, branch RNG mismatch, camera-order mismatch, or direct non-cluster action change invalidates the pair.

## 7. Lineage and outcome `Y_C`

### 7.1 Lineage

The pre-action 32 rows are ancestors. Surviving rows follow `new_to_old`; every clone or split child is assigned to its mapped pre-action ancestor while preserving `is_new`. An ancestor with no surviving descendant is a disappeared lineage. A child with no unambiguous ancestor, an unsupported topology operation, or a branch that cannot be aligned invalidates the pair.

### 7.2 Cluster-lineage geometry outcome

Synthetic fixtures provide a reference triangle surface `Sigma*`, unavailable to the action and `Q_g`. Let `d(x,Sigma*)` be exact point-to-triangle distance and `d_cap=0.10 m`. For ancestor `i` in branch `a` at horizon `H`, define

```text
e_i(a,H) = min(d_cap, sqrt(mean_j d(mu_j,Sigma*)^2))  over surviving descendants j,
e_i(a,H) = d_cap                                      if the lineage disappeared.
Y_C(a,H) = mean_{i in C} e_i(a,H).
tau_C(H) = Y_C(0,H) - Y_C(1,H).
```

Ancestors, not descendants, receive equal weight, preventing split count from changing the estimand.

### 7.3 Practical benefit

The action is beneficial exactly when

```text
tau_C(H) >= max(0.001 m, 0.05*Y_C(0,H)).
```

This is the only benefit label. Immediate error, static pre-action error, and a best intermediate step cannot replace the horizon-8 outcome.

## 8. Safety and spillover gates

An action is safe only when every condition below passes against its paired control:

1. **Finite/bounded:** all treated branch parameters, optimizer tensors, renders, losses, and outcomes are finite; `norm(delta_C)<=0.01 m`.
2. **Lineage tail:** the treated 95th percentile of the 32 ancestor-balanced `e_i` values increases by at most `0.001 m`.
3. **Disappearance:** no ancestor that has a surviving control descendant may completely disappear only in the treated branch.
4. **Protected appearance:** held-out linear-RGB L1 over the full frame increases by no more than `max(0.002, 0.01*L1_RGB_control)`.
5. **Guard spillover:** ancestor-balanced point-to-surface error over frozen guard-region lineages increases by no more than `max(0.001 m, 0.02*Y_guard_control)`.
6. **Far spillover:** the corresponding far-region increase is no more than `max(0.0005 m, 0.01*Y_far_control)`.
7. **Direct-write isolation:** at injection, every non-member parameter and every optimizer state is byte-identical across branches.
8. **Topology/optimizer integrity:** every topology mapping is valid, tensor cardinalities agree, and any active Adam step counters differ only through legitimate branch-private topology replacement.

`beneficial_safe=True` requires both Section 7.3 benefit and all eight safety gates. A valid but unsafe outcome is negative even when mean geometry improves.

## 9. Frozen synthetic suite

### 9.1 Canonical fixtures

The suite contains three meter-scale reference surfaces: a `0.8 m x 0.8 m` plane, a radius-`0.25 m` cylinder segment of length `0.8 m`, and two perpendicular `0.8 m x 0.8 m` planes. Each uses deterministic Gaussian grids with `0.02 m` spacing, isotropic initial scale `0.015 m`, opacity `0.8`, degree-0 color, and at most 4,096 Gaussians.

Each fixture uses 12 `PINHOLE` cameras (`512 x 512`, `fx=fy=500`) on a `1.0 m` radius ring around the surface center, azimuths `0,30,...,330` degrees, alternating elevations `+15/-15` degrees, all looking at the center. Reference RGB/depth/normal observations are rendered once from `Sigma*` and remain immutable.

### 9.2 Eight preregistered families

Each family contains 16 deterministic variants (`variant_id=0..15`) formed only by rotating the fixture and camera ring together by `22.5*variant_id` degrees. The 128 branch pairs are fixed before execution.

| Family | Frozen perturbation/control | Required semantic direction |
|---|---|---|
| coherent-offset | translate the cluster `+0.015 m` along the local surface normal | action beneficial and safe; high `Q_g` |
| small-correctable | translate `+0.002 m` along the local normal | permits the required “already good but small update beneficial” case |
| self-consistent-wrong | translate cluster and its internal rendered observations together by `+0.015 m` | static error high, proposal benefit absent; not authorized |
| contradictory-views | add equal and opposite `0.015 m` depth residuals to alternating proposal views | low direction agreement; not authorized |
| ill-conditioned | retain cameras within one `20 degree` azimuth sector | invalid or low conditioning/support; never authorized by imputation |
| occlusion-harm | inject a pre-action foreground occluder into proposal views only | proposed action worsens `Y_C` or safety; not authorized |
| guard-spillover | place a protected parallel surface `0.03 m` behind the cluster | any guard degradation beyond Section 8 fails safety |
| topology-lineage | schedule deterministic split/clone/prune candidates at step 4 | lineage accounting, disappearance penalty, and invalid mapping behavior are exact |

The topology family includes four variants each for survivor-only, clone, split, and treatment-only prune pressure. A separate malformed mapping fixture must return `INCONCLUSIVE`; it is a contract test, not one of the 128 predictive pairs.

No family parameter, expected direction, fixture count, or variant may be changed after any `Y_C` is computed.

## 10. Metrics, coverage, uncertainty, and decision

### 10.1 Coverage and predictive metrics

Coverage is

```text
count(V_Q=True clusters) / 128.
```

Report it overall and per family. On valid clusters, report prevalence, tie-safe AUROC and AUPRC of `Q_g` for `beneficial_safe`, the complete score/outcome table, and at threshold `0.50`: authorized count, precision, recall, unsafe-authorization rate, and family inventory.

Unsafe-authorization rate is `authorized_and_unsafe / authorized`; if no cluster is authorized, the result is valid-negative rather than a vacuous pass.

### 10.2 Family-stratified uncertainty

Using the already computed 128 score/outcome rows, run exactly 2,000 family-stratified bootstrap replicates with NumPy `PCG64` seed `20260930`. Each replicate samples 16 variants with replacement independently inside each of the eight families and concatenates them. `Q_g` and branch outcomes are not recomputed. Report percentile `[2.5%,97.5%]` intervals for AUROC, precision, recall, and unsafe-authorization rate. A classless or non-finite replicate makes the stage `INCONCLUSIVE`; it is not redrawn.

### 10.3 Positive gate

`GEOMETRY_AUTHORIZATION_HYPOTHESIS_READY` requires all of the following:

- overall validity coverage at least `0.80` and every family coverage at least `0.60`;
- every exact action, isolation, validity, monotonic-direction, timing, and lineage contract test passes;
- AUROC at least `0.80` and its bootstrap lower bound strictly greater than `0.70`;
- AUPRC at least `prevalence + 0.20`;
- authorized-tail precision at least `0.90` and bootstrap lower bound at least `0.80`;
- authorized-tail recall at least `0.50`;
- unsafe-authorization rate at most `0.05` and bootstrap upper bound at most `0.10`;
- every coherent-offset family variant is authorized, and no contradictory-view, ill-conditioned, occlusion-harm, or guard-spillover negative is authorized;
- the full suite stays within the compute budget in Section 11.

A valid evaluation that misses any gate returns `NO_ACTION_SPECIFIC_SIGNAL`. Corrupt input, invalid branch alignment, classless bootstrap, non-finite output, infrastructure failure, or inability to compute the frozen estimand returns `INCONCLUSIVE`. There is no fallback formula, threshold, action, horizon, filter, or synthetic suite.

## 11. Compute and publication budget

- Exactly 128 branch pairs, two branches per pair, eight optimizer steps per branch: at most 2,048 synthetic optimizer steps.
- One pair at a time; no batch intervention and no concurrent branch execution.
- At most 4,096 Gaussians, 16 retained support cameras, and `512 x 512` renders per fixture.
- Peak allocated GPU memory at most `12 GiB` on one RTX 4090.
- Wall time at most `30 seconds` per pair and `60 minutes` for the complete suite, excluding one-time environment import/compile.
- Diagnostic output at most `5 GiB`; publish canonical inputs, exact fixture manifest, per-pair outcomes, metrics/bootstrap, logs, and SHA256 manifest only. No field-render gallery or training checkpoint is required.

Exceeding any limit under otherwise valid execution returns `NO_ACTION_SPECIFIC_SIGNAL`, because the candidate violates its preregistered practical budget. Infrastructure failure before a valid measurement is `INCONCLUSIVE`.

## 12. Stop rule and authorization boundary

The first complete valid Stage G-A execution is single-use. `NO_ACTION_SPECIFIC_SIGNAL` requires a canonical, detached-SHA termination confirmation and closes this geometry branch against later Utility feedback. `INCONCLUSIVE` permits only repair of the same measurement contract; it does not permit a new candidate. A positive result permits only a request to draft the separate Stage G-B Tool Room intervention specification.

Approval of this sub-specification would authorize only a later request to write a Stage G-A TDD implementation plan. It would not authorize implementation, synthetic execution, Tool Room work, DA3, Utility preprocessing/training/GT access, Evidence changes, production routing, five-state arbitration, or C1.
