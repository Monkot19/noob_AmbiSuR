from dataclasses import dataclass

import torch

from reliability.arbitration import (
    ArbitrationState,
    ArbitrationStateMachine,
    candidate_state,
)
from reliability.topology import migrate_tensor
from reliability.transition_diagnostics import TemporalTransitionDiagnostics


@dataclass(frozen=True)
class ObservationSufficiency:
    M_obs: torch.Tensor
    S_count: torch.Tensor
    S_angle: torch.Tensor
    S: torch.Tensor


@dataclass(frozen=True)
class PGConsistency:
    Z_pg: torch.Tensor
    V_pg: torch.Tensor
    K_raw: torch.Tensor


@dataclass(frozen=True)
class ReprojectionEvidence:
    valid: torch.Tensor
    depth_error: torch.Tensor
    normal_error: torch.Tensor


@dataclass(frozen=True)
class Reliability:
    T: torch.Tensor
    V: torch.Tensor
    r: torch.Tensor


@dataclass(frozen=True)
class GeometryStability:
    score: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class EvidenceRefreshInputs:
    sh_coefficients: torch.Tensor
    sh_degree: int
    pixel_hits: torch.Tensor
    camera_centers: torch.Tensor
    centers: torch.Tensor
    normals: torch.Tensor
    scale_reference: torch.Tensor
    prior_confidence: torch.Tensor
    prior_multiview: torch.Tensor
    prior_support_views: torch.Tensor
    geometry_multiview: torch.Tensor
    geometry_depth_normal: torch.Tensor
    geometry_support_views: torch.Tensor
    pg_weighted_support: torch.Tensor
    pg_depth_error_sum: torch.Tensor
    pg_normal_error_sum: torch.Tensor


@dataclass(frozen=True)
class EvidenceSnapshot:
    A: torch.Tensor
    S: torch.Tensor
    N: torch.Tensor
    T_p: torch.Tensor
    V_p: torch.Tensor
    r_p: torch.Tensor
    T_g: torch.Tensor
    V_g: torch.Tensor
    r_g: torch.Tensor
    Z_pg: torch.Tensor
    V_pg: torch.Tensor
    K: torch.Tensor
    delta: torch.Tensor
    candidate: torch.Tensor
    stable: torch.Tensor

    def tensors(self):
        return (
            self.A,
            self.S,
            self.N,
            self.T_p,
            self.V_p,
            self.r_p,
            self.T_g,
            self.V_g,
            self.r_g,
            self.Z_pg,
            self.V_pg,
            self.K,
            self.delta,
            self.candidate,
            self.stable,
        )


def active_non_dc(coefficients, degree):
    """Return only the non-DC coefficients active at the current SH degree."""
    if coefficients.ndim != 3 or coefficients.shape[-1] != 3:
        raise ValueError("SH coefficients must have shape [P, C, 3]")
    count = (int(degree) + 1) ** 2 - 1
    if degree < 0 or count > coefficients.shape[1]:
        raise ValueError("SH degree is incompatible with stored coefficients")
    return coefficients[:, :count, :]


@torch.no_grad()
def compute_appearance_ambiguity(coefficients, degree, eps=1e-8):
    active = active_non_dc(coefficients.detach(), degree)
    energy = active.square().sum(dim=(1, 2)).sqrt()
    low = torch.quantile(energy, 0.10)
    high = torch.quantile(energy, 0.95)
    return ((energy - low) / (high - low + eps)).clamp(0.0, 1.0)


@torch.no_grad()
def view_count(pixel_hits):
    if pixel_hits.ndim != 2:
        raise ValueError("pixel hits must have shape [V, P]")
    return (pixel_hits.detach() > 0).sum(dim=0)


@torch.no_grad()
def compute_observation_sufficiency(
    pixel_hits,
    camera_centers,
    gaussian_centers,
    *,
    k_c=5,
    theta_c_degrees=30.0,
    chunk_size=8192,
    eps=1e-8,
):
    hits = pixel_hits.detach()
    cameras = camera_centers.detach().to(device=gaussian_centers.device)
    gaussians = gaussian_centers.detach()
    if hits.ndim != 2:
        raise ValueError("pixel hits must have shape [V, P]")
    if cameras.shape != (hits.shape[0], 3):
        raise ValueError("camera centers must have shape [V, 3]")
    if gaussians.shape != (hits.shape[1], 3):
        raise ValueError("Gaussian centers must have shape [P, 3]")
    if k_c <= 0:
        raise ValueError("k_c must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    count_chunks = []
    dispersion_chunks = []
    for start in range(0, gaussians.shape[0], chunk_size):
        end = min(start + chunk_size, gaussians.shape[0])
        observed = (hits[:, start:end] > 0).to(device=gaussians.device)
        counts = observed.sum(dim=0)
        directions = cameras[:, None, :] - gaussians[None, start:end, :]
        directions = directions / directions.norm(
            dim=-1, keepdim=True
        ).clamp_min(eps)
        direction_sum = (directions * observed[..., None]).sum(dim=0)
        count_float = counts.to(dtype=gaussians.dtype)
        numerator = count_float.square() - direction_sum.square().sum(dim=-1)
        denominator = 2.0 * count_float * (count_float - 1.0)
        dispersion = torch.where(
            counts >= 2,
            numerator / denominator.clamp_min(eps),
            torch.zeros_like(count_float),
        ).clamp(0.0, 1.0)
        count_chunks.append(counts)
        dispersion_chunks.append(dispersion)

    counts = torch.cat(count_chunks) if count_chunks else torch.empty(
        0, dtype=torch.int64, device=gaussians.device
    )
    dispersion = (
        torch.cat(dispersion_chunks)
        if dispersion_chunks
        else torch.empty(0, dtype=gaussians.dtype, device=gaussians.device)
    )
    count_float = counts.to(dtype=gaussians.dtype)

    theta = torch.as_tensor(
        theta_c_degrees * torch.pi / 180.0,
        dtype=gaussians.dtype,
        device=gaussians.device,
    )
    reference = (1.0 - torch.cos(theta)) / 2.0
    count_score = (count_float / float(k_c)).clamp(0.0, 1.0)
    angle_score = (dispersion / reference.clamp_min(eps)).clamp(0.0, 1.0)
    score = (count_score * angle_score).sqrt()
    return ObservationSufficiency(counts, count_score, angle_score, score)


@torch.no_grad()
def compute_need(ambiguity, sufficiency):
    ambiguity = ambiguity.detach()
    sufficiency = sufficiency.detach()
    if ambiguity.shape != sufficiency.shape:
        raise ValueError("ambiguity and sufficiency must have the same shape")
    return (1.0 - sufficiency * (1.0 - ambiguity)).clamp(0.0, 1.0)


@torch.no_grad()
def normalize_prior_confidence(confidence, eps=1e-8):
    confidence = confidence.detach()
    if confidence.ndim < 2:
        raise ValueError("prior confidence must have a leading view dimension")
    flattened = confidence.flatten(start_dim=1)
    low = torch.quantile(flattened, 0.05, dim=1, keepdim=True)
    high = torch.quantile(flattened, 0.95, dim=1, keepdim=True)
    normalized = ((flattened - low) / (high - low + eps)).clamp(0.0, 1.0)
    return normalized.reshape_as(confidence)


@torch.no_grad()
def reprojection_validity_and_errors(
    projected_uv,
    projected_depth,
    source_depth,
    sampled_target_depth,
    source_normal,
    sampled_target_normal,
    *,
    image_height,
    image_width,
    tau_occ=0.05,
    eps=1e-8,
):
    uv = projected_uv.detach()
    projected = projected_depth.detach()
    source = source_depth.detach()
    target = sampled_target_depth.detach()
    source_n = source_normal.detach()
    target_n = sampled_target_normal.detach()
    count = projected.shape[0]
    if uv.shape != (count, 2):
        raise ValueError("projected coordinates must have shape [R, 2]")
    if source.shape != projected.shape or target.shape != projected.shape:
        raise ValueError("depth tensors must have the same shape")
    if source_n.shape != (count, 3) or target_n.shape != (count, 3):
        raise ValueError("normal tensors must have shape [R, 3]")
    if image_height <= 0 or image_width <= 0:
        raise ValueError("image dimensions must be positive")

    source_norm = source_n.norm(dim=-1)
    target_norm = target_n.norm(dim=-1)
    finite = (
        torch.isfinite(uv).all(dim=-1)
        & torch.isfinite(projected)
        & torch.isfinite(source)
        & torch.isfinite(target)
        & torch.isfinite(source_n).all(dim=-1)
        & torch.isfinite(target_n).all(dim=-1)
    )
    in_frame = (
        (uv[:, 0] >= 0)
        & (uv[:, 0] <= image_width - 1)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] <= image_height - 1)
    )
    positive = (
        (projected > 0)
        & (source > 0)
        & (target > 0)
        & (source_norm > eps)
        & (target_norm > eps)
    )
    not_occluded = projected <= (1.0 + tau_occ) * target
    valid = finite & in_frame & positive & not_occluded

    depth_error = (projected - target).abs() / (projected + target + eps)
    source_unit = source_n / source_norm.clamp_min(eps).unsqueeze(-1)
    target_unit = target_n / target_norm.clamp_min(eps).unsqueeze(-1)
    normal_error = 1.0 - (source_unit * target_unit).sum(dim=-1).abs().clamp(0.0, 1.0)
    depth_error = torch.where(valid, depth_error, torch.zeros_like(depth_error))
    normal_error = torch.where(valid, normal_error, torch.zeros_like(normal_error))
    return ReprojectionEvidence(valid, depth_error, normal_error)


@torch.no_grad()
def combine_prior_reliability(confidence, multiview, support_views, *, min_views=2):
    confidence = confidence.detach()
    multiview = multiview.detach()
    support = support_views.detach()
    if confidence.shape != multiview.shape or confidence.shape != support.shape:
        raise ValueError("prior reliability inputs must have the same shape")
    strength = (confidence.clamp(0.0, 1.0) * multiview.clamp(0.0, 1.0)).sqrt()
    valid = support >= min_views
    return Reliability(strength, valid, torch.where(valid, strength, torch.zeros_like(strength)))


@torch.no_grad()
def combine_geometry_reliability(
    multiview,
    depth_normal,
    stability,
    support_views,
    history_valid,
    *,
    min_views=2,
):
    multiview = multiview.detach()
    depth_normal = depth_normal.detach()
    stability = stability.detach()
    support = support_views.detach()
    history = history_valid.detach().bool()
    shape = multiview.shape
    if any(value.shape != shape for value in (depth_normal, stability, support, history)):
        raise ValueError("geometry reliability inputs must have the same shape")
    product = (
        multiview.clamp(0.0, 1.0)
        * depth_normal.clamp(0.0, 1.0)
        * stability.clamp(0.0, 1.0)
    )
    strength = product.pow(1.0 / 3.0)
    valid = (support >= min_views) & history
    return Reliability(strength, valid, torch.where(valid, strength, torch.zeros_like(strength)))


@torch.no_grad()
def compute_geometry_stability(
    current_centers,
    previous_centers,
    current_normals,
    previous_normals,
    scale_reference,
    history_valid,
    *,
    tau_move=0.25,
    tau_rotation=15.0 / 180.0,
    eps=1e-8,
):
    current = current_centers.detach()
    previous = previous_centers.detach()
    current_n = current_normals.detach()
    previous_n = previous_normals.detach()
    scale = scale_reference.detach()
    history = history_valid.detach().bool()
    point_count = current.shape[0]
    if current.shape != (point_count, 3) or previous.shape != current.shape:
        raise ValueError("center tensors must have shape [P, 3]")
    if current_n.shape != current.shape or previous_n.shape != current.shape:
        raise ValueError("normal tensors must have shape [P, 3]")
    if scale.shape != (point_count,) or history.shape != (point_count,):
        raise ValueError("scale and history tensors must have shape [P]")

    current_norm = current_n.norm(dim=-1)
    previous_norm = previous_n.norm(dim=-1)
    finite = (
        torch.isfinite(current).all(dim=-1)
        & torch.isfinite(previous).all(dim=-1)
        & torch.isfinite(current_n).all(dim=-1)
        & torch.isfinite(previous_n).all(dim=-1)
        & torch.isfinite(scale)
    )
    valid = history & finite & (scale > eps) & (current_norm > eps) & (previous_norm > eps)
    movement = (current - previous).norm(dim=-1) / scale.clamp_min(eps)
    dot = (
        (current_n / current_norm.clamp_min(eps).unsqueeze(-1))
        * (previous_n / previous_norm.clamp_min(eps).unsqueeze(-1))
    ).sum(dim=-1).abs().clamp(-1.0, 1.0)
    rotation = torch.acos(dot) / torch.pi
    score = torch.exp(-movement / tau_move - rotation / tau_rotation)
    score = torch.where(valid, score, torch.zeros_like(score))
    return GeometryStability(score, valid)


@torch.no_grad()
def compute_pg_consistency(
    weighted_support,
    depth_error_sum,
    normal_error_sum,
    *,
    tau_z=1e-4,
    tau_depth=0.05,
    tau_normal=0.10,
    eps=1e-8,
):
    support = weighted_support.detach()
    depth = depth_error_sum.detach()
    normal = normal_error_sum.detach()
    if support.shape != depth.shape or support.shape != normal.shape:
        raise ValueError("joint support and error sums must have the same shape")
    valid = support > tau_z
    mean_depth = depth / (support + eps)
    mean_normal = normal / (support + eps)
    raw = torch.exp(-0.5 * (mean_depth / tau_depth + mean_normal / tau_normal))
    raw = torch.where(valid, raw, torch.zeros_like(raw))
    return PGConsistency(support, valid, raw)


class EMAState:
    def __init__(self, point_count, *, beta=0.9, device=None):
        if point_count < 0:
            raise ValueError("point_count must be nonnegative")
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must be in [0, 1)")
        self.beta = float(beta)
        self.value = torch.zeros(point_count, dtype=torch.float32, device=device)
        self.initialized = torch.zeros(point_count, dtype=torch.bool, device=device)
        self.current_valid = torch.zeros(point_count, dtype=torch.bool, device=device)

    @torch.no_grad()
    def update(self, raw, valid):
        valid = valid.detach().to(device=self.value.device, dtype=torch.bool)
        if valid.shape != self.value.shape:
            raise ValueError("valid mask must match EMA state")
        self.current_valid.copy_(valid)
        if not valid.any():
            return
        if raw is None:
            raise ValueError("raw value is required for valid EMA observations")
        raw = raw.detach().to(device=self.value.device, dtype=self.value.dtype)
        if raw.shape != self.value.shape:
            raise ValueError("raw value must match EMA state")
        first = valid & ~self.initialized
        continuing = valid & self.initialized
        self.value[first] = raw[first]
        self.value[continuing] = (
            self.beta * self.value[continuing]
            + (1.0 - self.beta) * raw[continuing]
        )
        self.initialized[valid] = True


class KEMAState(EMAState):
    """Joint-validity-gated EMA retained as a named public contract."""


class EvidenceAccumulator:
    """Persistent, detached D0 evidence and arbitration state."""

    STATE_VERSION = 3

    def __init__(
        self,
        point_count,
        *,
        cfg,
        device=None,
        ema_beta=0.9,
        k_beta=0.9,
    ):
        if point_count < 0:
            raise ValueError("point_count must be nonnegative")
        cfg.validate()
        self.cfg = cfg
        self.point_count = int(point_count)
        self.a_ema = EMAState(
            self.point_count, beta=ema_beta, device=device
        )
        self.s_ema = EMAState(
            self.point_count, beta=ema_beta, device=device
        )
        self.t_p_ema = EMAState(
            self.point_count, beta=ema_beta, device=device
        )
        self.t_g_ema = EMAState(
            self.point_count, beta=ema_beta, device=device
        )
        self.k_ema = KEMAState(
            self.point_count, beta=k_beta, device=device
        )
        self.arbitration = ArbitrationStateMachine(
            self.point_count,
            initial_state=ArbitrationState.BYPASS,
            enter_count=cfg.arbitration_enter_count,
            device=device,
        )
        self.transition_diagnostics = TemporalTransitionDiagnostics(
            self.point_count, device=device
        )
        self.previous_centers = torch.zeros(
            (self.point_count, 3), dtype=torch.float32, device=device
        )
        self.previous_normals = torch.zeros(
            (self.point_count, 3), dtype=torch.float32, device=device
        )
        self.history_valid = torch.zeros(
            self.point_count, dtype=torch.bool, device=device
        )
        self.latest = None
        self.latest_transition_diagnostics = None

    def _validate_inputs(self, inputs):
        if inputs.centers.shape != (self.point_count, 3):
            raise ValueError("centers must match accumulator point count")
        if inputs.normals.shape != (self.point_count, 3):
            raise ValueError("normals must match accumulator point count")
        if inputs.centers.device != self.previous_centers.device:
            raise ValueError("refresh inputs must share the accumulator device")

    @torch.no_grad()
    def refresh(self, inputs):
        self._validate_inputs(inputs)
        ambiguity = compute_appearance_ambiguity(
            inputs.sh_coefficients, inputs.sh_degree
        )
        sufficiency = compute_observation_sufficiency(
            inputs.pixel_hits,
            inputs.camera_centers,
            inputs.centers,
        )
        prior = combine_prior_reliability(
            inputs.prior_confidence,
            inputs.prior_multiview,
            inputs.prior_support_views,
        )
        stability = compute_geometry_stability(
            inputs.centers,
            self.previous_centers,
            inputs.normals,
            self.previous_normals,
            inputs.scale_reference,
            self.history_valid,
        )
        geometry = combine_geometry_reliability(
            inputs.geometry_multiview,
            inputs.geometry_depth_normal,
            stability.score,
            inputs.geometry_support_views,
            stability.valid,
        )
        all_valid = torch.ones(
            self.point_count,
            dtype=torch.bool,
            device=inputs.centers.device,
        )
        self.a_ema.update(ambiguity, all_valid)
        self.s_ema.update(sufficiency.S, all_valid)
        self.t_p_ema.update(prior.T, prior.V)
        self.t_g_ema.update(geometry.T, geometry.V)
        need = compute_need(self.a_ema.value, self.s_ema.value)
        prior_r = prior.V.to(self.t_p_ema.value.dtype) * self.t_p_ema.value
        geometry_r = (
            geometry.V.to(self.t_g_ema.value.dtype) * self.t_g_ema.value
        )
        pg = compute_pg_consistency(
            inputs.pg_weighted_support,
            inputs.pg_depth_error_sum,
            inputs.pg_normal_error_sum,
        )
        raw_k = pg.K_raw if bool(pg.V_pg.any()) else None
        self.k_ema.update(raw_k, pg.V_pg)
        delta = prior_r - geometry_r

        placeholder = torch.zeros(
            self.point_count,
            dtype=torch.int8,
            device=inputs.centers.device,
        )
        snapshot = EvidenceSnapshot(
            self.a_ema.value.clone(),
            self.s_ema.value.clone(),
            need,
            self.t_p_ema.value.clone(),
            prior.V,
            prior_r,
            self.t_g_ema.value.clone(),
            geometry.V,
            geometry_r,
            pg.Z_pg,
            pg.V_pg,
            self.k_ema.value.clone(),
            delta,
            placeholder,
            self.arbitration.stable_state.clone(),
        )
        candidate = candidate_state(snapshot, self.cfg)
        stable = self.arbitration.update(candidate)
        self.latest_transition_diagnostics = (
            self.transition_diagnostics.update(
                self.transition_diagnostics.previous_stable,
                stable,
            )
        )
        self.previous_centers.copy_(inputs.centers.detach())
        self.previous_normals.copy_(inputs.normals.detach())
        self.history_valid.fill_(True)
        self.latest = EvidenceSnapshot(
            snapshot.A,
            snapshot.S,
            snapshot.N,
            snapshot.T_p,
            snapshot.V_p,
            snapshot.r_p,
            snapshot.T_g,
            snapshot.V_g,
            snapshot.r_g,
            snapshot.Z_pg,
            snapshot.V_pg,
            snapshot.K,
            snapshot.delta,
            candidate,
            stable,
        )
        return self.latest

    @torch.no_grad()
    def on_topology_change(self, change):
        for state in (
            self.a_ema,
            self.s_ema,
            self.t_p_ema,
            self.t_g_ema,
            self.k_ema,
        ):
            state.value = migrate_tensor(
                state.value, change, fill_value=0.0
            )
            state.initialized = migrate_tensor(
                state.initialized, change, fill_value=False
            )
            state.current_valid = migrate_tensor(
                state.current_valid, change, fill_value=False
            )
        self.arbitration.stable_state = migrate_tensor(
            self.arbitration.stable_state,
            change,
            fill_value=int(ArbitrationState.BYPASS),
        )
        self.arbitration.candidate_state = migrate_tensor(
            self.arbitration.candidate_state,
            change,
            fill_value=int(ArbitrationState.BYPASS),
        )
        self.arbitration.consecutive_count = migrate_tensor(
            self.arbitration.consecutive_count, change, fill_value=0
        )
        self.previous_centers = migrate_tensor(
            self.previous_centers, change, fill_value=0.0
        )
        self.previous_normals = migrate_tensor(
            self.previous_normals, change, fill_value=0.0
        )
        self.history_valid = migrate_tensor(
            self.history_valid, change, fill_value=False
        )
        self.transition_diagnostics.on_topology_change(change)
        self.point_count = int(change.new_to_old.shape[0])
        self.latest = None
        self.latest_transition_diagnostics = None

    def state_dict(self):
        def clone(value):
            return value.detach().clone()

        return {
            "version": self.STATE_VERSION,
            "point_count": self.point_count,
            "ema_beta": self.a_ema.beta,
            "k_beta": self.k_ema.beta,
            "a_value": clone(self.a_ema.value),
            "a_initialized": clone(self.a_ema.initialized),
            "a_current_valid": clone(self.a_ema.current_valid),
            "s_value": clone(self.s_ema.value),
            "s_initialized": clone(self.s_ema.initialized),
            "s_current_valid": clone(self.s_ema.current_valid),
            "t_p_value": clone(self.t_p_ema.value),
            "t_p_initialized": clone(self.t_p_ema.initialized),
            "t_p_current_valid": clone(self.t_p_ema.current_valid),
            "t_g_value": clone(self.t_g_ema.value),
            "t_g_initialized": clone(self.t_g_ema.initialized),
            "t_g_current_valid": clone(self.t_g_ema.current_valid),
            "k_value": clone(self.k_ema.value),
            "k_initialized": clone(self.k_ema.initialized),
            "k_current_valid": clone(self.k_ema.current_valid),
            "stable_state": clone(self.arbitration.stable_state),
            "candidate_state": clone(self.arbitration.candidate_state),
            "consecutive_count": clone(
                self.arbitration.consecutive_count
            ),
            "previous_centers": clone(self.previous_centers),
            "previous_normals": clone(self.previous_normals),
            "history_valid": clone(self.history_valid),
            "temporal_transition_diagnostics": (
                self.transition_diagnostics.state_dict()
            ),
        }

    @torch.no_grad()
    def load_state_dict(self, state):
        if state.get("version") != self.STATE_VERSION:
            raise ValueError("unsupported evidence state version")
        if state.get("point_count") != self.point_count:
            raise ValueError("evidence state point count mismatch")
        if float(state.get("ema_beta")) != self.a_ema.beta:
            raise ValueError("evidence state EMA beta mismatch")
        if float(state.get("k_beta")) != self.k_ema.beta:
            raise ValueError("evidence state K EMA beta mismatch")

        destinations = {
            "a_value": self.a_ema.value,
            "a_initialized": self.a_ema.initialized,
            "a_current_valid": self.a_ema.current_valid,
            "s_value": self.s_ema.value,
            "s_initialized": self.s_ema.initialized,
            "s_current_valid": self.s_ema.current_valid,
            "t_p_value": self.t_p_ema.value,
            "t_p_initialized": self.t_p_ema.initialized,
            "t_p_current_valid": self.t_p_ema.current_valid,
            "t_g_value": self.t_g_ema.value,
            "t_g_initialized": self.t_g_ema.initialized,
            "t_g_current_valid": self.t_g_ema.current_valid,
            "k_value": self.k_ema.value,
            "k_initialized": self.k_ema.initialized,
            "k_current_valid": self.k_ema.current_valid,
            "stable_state": self.arbitration.stable_state,
            "candidate_state": self.arbitration.candidate_state,
            "consecutive_count": self.arbitration.consecutive_count,
            "previous_centers": self.previous_centers,
            "previous_normals": self.previous_normals,
            "history_valid": self.history_valid,
        }
        for name, destination in destinations.items():
            value = state.get(name)
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"missing evidence state tensor: {name}")
            if value.shape != destination.shape:
                raise ValueError(f"evidence state shape mismatch: {name}")
            destination.copy_(
                value.detach().to(
                    device=destination.device, dtype=destination.dtype
                )
            )
        temporal_state = state.get("temporal_transition_diagnostics")
        if not isinstance(temporal_state, dict):
            raise ValueError("missing temporal transition diagnostics state")
        self.transition_diagnostics.load_state_dict(temporal_state)
        self.latest = None
        self.latest_transition_diagnostics = None
