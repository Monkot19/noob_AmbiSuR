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
