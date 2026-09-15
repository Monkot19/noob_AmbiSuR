from dataclasses import dataclass

import torch


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
    eps=1e-8,
):
    hits = pixel_hits.detach()
    cameras = camera_centers.detach()
    gaussians = gaussian_centers.detach()
    if hits.ndim != 2:
        raise ValueError("pixel hits must have shape [V, P]")
    if cameras.shape != (hits.shape[0], 3):
        raise ValueError("camera centers must have shape [V, 3]")
    if gaussians.shape != (hits.shape[1], 3):
        raise ValueError("Gaussian centers must have shape [P, 3]")
    if k_c <= 0:
        raise ValueError("k_c must be positive")

    observed = hits > 0
    counts = observed.sum(dim=0)
    directions = cameras[:, None, :] - gaussians[None, :, :]
    directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(eps)
    direction_sum = (directions * observed[..., None]).sum(dim=0)

    count_float = counts.to(dtype=gaussians.dtype)
    numerator = count_float.square() - direction_sum.square().sum(dim=-1)
    denominator = 2.0 * count_float * (count_float - 1.0)
    dispersion = torch.where(
        counts >= 2,
        numerator / denominator.clamp_min(eps),
        torch.zeros_like(count_float),
    ).clamp(0.0, 1.0)

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


class KEMAState:
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
            raise ValueError("raw K is required for valid joint observations")
        raw = raw.detach().to(device=self.value.device, dtype=self.value.dtype)
        if raw.shape != self.value.shape:
            raise ValueError("raw K must match EMA state")
        first = valid & ~self.initialized
        continuing = valid & self.initialized
        self.value[first] = raw[first]
        self.value[continuing] = (
            self.beta * self.value[continuing]
            + (1.0 - self.beta) * raw[continuing]
        )
        self.initialized[valid] = True
