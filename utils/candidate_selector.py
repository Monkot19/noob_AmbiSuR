"""Ephemeral, reference-specific additions to AmbiSuR's original SH labels."""
import hashlib

import torch
import torch.nn.functional as F

from utils.graphics_utils import patch_offsets, patch_warp
from utils.loss_utils import lncc


_EPS = 1e-6
_CORE_FLAGS = (
    "core_shadow_mode", "enable_observation_calibration", "enable_dual_reliability",
    "enable_abstention", "enable_parameter_routing", "enable_gradient_projection",
    "enable_reliability_lifecycle",
)


@torch.no_grad()
def compute_sh_ambiguity_labels(gaussians, opt):
    """The baseline's strict dual-tail rule, shared by renderer and training."""
    xyz = gaussians.get_xyz
    if opt is None or xyz.shape[0] == 0:
        return torch.zeros(xyz.shape[0], dtype=torch.bool, device=xyz.device)
    score = gaussians.compute_weighted_sh_norm("equal")
    upper = torch.quantile(score.flatten(), opt.sh_ambi_upper_ratio)
    lower = torch.quantile(score.flatten(), opt.sh_ambi_lower_ratio)
    lower = min(opt.sh_unc_lower_max, lower)
    return ((score > upper) | (score < lower)).reshape(-1).detach()


def candidate_selector_active(iteration, opt):
    if not getattr(opt, "enable_candidate_selector", False):
        return False
    budget = opt.candidate_selector_budget
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise ValueError("candidate_selector_budget must be a positive integer")
    if opt.use_mono or any(getattr(opt, key, False) for key in _CORE_FLAGS):
        raise ValueError("candidate selector requires DA3 and all old Core flags off")
    return iteration > max(1000, opt.unc_from_iter)


def candidate_relocation_active(iteration, opt):
    if not getattr(opt, "enable_candidate_relocation", False):
        return False
    if not getattr(opt, "enable_candidate_selector", False):
        raise ValueError("candidate relocation requires the candidate selector")
    interval = opt.candidate_relocation_interval
    if isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0:
        raise ValueError("candidate_relocation_interval must be a positive integer")
    active = candidate_selector_active(iteration, opt)
    return active and iteration % interval == 0 and iteration < opt.iterations


def _sample_candidate_ids(pool, budget, *, seed, iteration, camera_uid):
    if pool.numel() <= budget:
        return pool
    identity = f"ambisur-selector-v1|{seed}|{iteration}|{camera_uid}".encode("utf-8")
    local_seed = int.from_bytes(hashlib.sha256(identity).digest()[:8], "little") & ((1 << 63) - 1)
    generator = torch.Generator(device=pool.device).manual_seed(local_seed)
    return pool[torch.randperm(pool.numel(), device=pool.device, generator=generator)[:budget]]


def _best_source_decision(cost_g, cost_p, common_valid):
    common_valid = common_valid & torch.isfinite(cost_g) & torch.isfinite(cost_p)
    g = cost_g.masked_fill(~common_valid, float("inf")).min(dim=-1).values
    p = cost_p.masked_fill(~common_valid, float("inf")).min(dim=-1).values
    return common_valid.any(dim=-1) & (p < g) & (p < .5)


def _relative_pose(reference, source):
    ref, src = reference.world_view_transform, source.world_view_transform
    rotation = src[:3, :3].T @ ref[:3, :3]
    translation = src[3, :3] - rotation @ ref[3, :3]
    return rotation, translation


def _plane_homography(reference, source, normals, distances):
    rotation, translation = _relative_pose(reference, source)
    plane_map = rotation[None] - translation[None, :, None] * normals[:, None, :] / distances[:, None, None]
    return source.get_k(source.ncc_scale)[None] @ plane_map @ reference.get_inv_k(reference.ncc_scale)


def _inside_patch(pixels, height, width):
    return (torch.isfinite(pixels).all(dim=(-1, -2))
            & (pixels[..., 0] >= 0).all(-1) & (pixels[..., 0] <= width - 1).all(-1)
            & (pixels[..., 1] >= 0).all(-1) & (pixels[..., 1] <= height - 1).all(-1))


def _sample_map(image, pixels):
    """image:[C,H,W], pixels:[B,P,2], with the baseline's align_corners."""
    height, width = image.shape[-2:]
    grid = pixels.clone()
    grid[..., 0] = 2 * grid[..., 0] / (width - 1) - 1
    grid[..., 1] = 2 * grid[..., 1] / (height - 1) - 1
    values = F.grid_sample(image[None], grid.reshape(1, -1, 1, 2), align_corners=True)
    return values.reshape(image.shape[0], *pixels.shape[:2]).permute(1, 2, 0)


def _oriented_plane(points, normals):
    magnitude = normals.norm(dim=-1)
    valid = torch.isfinite(normals).all(-1) & (magnitude > _EPS)
    # Only a numerical placeholder for rejected rows, never a validity clamp.
    normals = normals / torch.where(valid, magnitude, torch.ones_like(magnitude))[:, None]
    normals = torch.where(((normals * points).sum(-1) > 0)[:, None], -normals, normals)
    distances = -(normals * points).sum(-1)
    valid &= torch.isfinite(points).all(-1) & torch.isfinite(distances) & (distances > _EPS)
    return normals, distances, valid


def _source_cost(gaussians, reference, source, reference_patch, reference_values,
                 centers, points, normals, distances, source_depth, source_gray, opt):
    """One hypothesis, one source: geometry first, raw NCC without its loss mask."""
    h = _plane_homography(reference, source, normals, distances)
    ones = torch.ones_like(reference_patch[..., :1])
    homogeneous = torch.cat((reference_patch, ones), -1)
    projected_h = homogeneous @ h.transpose(1, 2)
    projected = patch_warp(h, reference_patch)
    height, width = source_gray.shape[-2:]
    valid = _inside_patch(projected, height, width)
    valid &= torch.isfinite(projected_h).all(dim=(-1, -2)) & (projected_h[..., 2].abs() > _EPS).all(-1)

    rays = homogeneous @ reference.get_inv_k(reference.ncc_scale).T
    denominator = (rays * normals[:, None]).sum(-1)
    intersection_depth = -distances[:, None] / denominator
    rotation, translation = _relative_pose(reference, source)
    source_patch_points = (rays * intersection_depth[..., None]) @ rotation.T + translation
    valid &= ((denominator.abs() > _EPS) & torch.isfinite(intersection_depth)
              & (intersection_depth > reference.znear)).all(-1)
    valid &= torch.isfinite(source_patch_points).all(dim=(-1, -2)) & (source_patch_points[..., 2] > source.znear).all(-1)

    source_points = points @ rotation.T + translation
    sampled_depth, depth_valid = gaussians.get_points_depth_in_depth_map(source, source_depth, source_points)
    sampled_depth = sampled_depth.reshape(-1)
    valid &= depth_valid.reshape(-1) & torch.isfinite(sampled_depth) & (sampled_depth > 0)
    reconstructed_source = source_points / source_points[:, 2:3] * sampled_depth[:, None]
    source_transform = source.world_view_transform
    world = (reconstructed_source - source_transform[3, :3]) @ source_transform[:3, :3].T
    ref_transform = reference.world_view_transform
    returned = world @ ref_transform[:3, :3] + ref_transform[3, :3]
    valid &= torch.isfinite(returned).all(-1) & (returned[:, 2] > reference.znear)
    returned_pixels = returned @ reference.get_k().T
    returned_pixels = returned_pixels[:, :2] / returned_pixels[:, 2:3]
    pixel_error = (returned_pixels - centers).norm(dim=-1)
    valid &= torch.isfinite(pixel_error)
    if not opt.wo_use_geo_occ_aware:
        valid &= pixel_error < opt.multi_view_pixel_noise_th

    costs = distances.new_full(distances.shape, float("inf"))
    rows = valid.nonzero(as_tuple=True)[0]
    if rows.numel():
        sampled_values = _sample_map(source_gray, projected[rows]).squeeze(-1)
        ref_values = reference_values[rows]
        finite_values = torch.isfinite(sampled_values).all(-1) & torch.isfinite(ref_values).all(-1)
        rows = rows[finite_values]
        if rows.numel():
            raw_cost, _ = lncc(ref_values[finite_values], sampled_values[finite_values])
            raw_cost = raw_cost.reshape(-1)
            if not torch.isfinite(raw_cost).all():
                raise FloatingPointError("non-finite NCC for finite selector patches")
            costs[rows] = raw_cost
        valid &= torch.isfinite(costs)
    return costs, valid


@torch.no_grad()
def _select_candidate_geometry(gaussians, reference_camera, reference_gray,
                               prior_depth, prior_normal, support_cameras, opt, *,
                               seed, iteration, support_depth_fn, include_targets=False):
    active = candidate_selector_active(iteration, opt)
    xyz = gaussians.get_xyz
    if xyz.ndim != 2 or xyz.shape[-1] != 3 or not torch.isfinite(xyz).all():
        raise ValueError("selector requires finite Gaussian xyz:[N,3]")
    if not torch.isfinite(gaussians.compute_weighted_sh_norm("equal")).all():
        raise ValueError("selector requires finite model SH scores")
    labels = compute_sh_ambiguity_labels(gaussians, opt)
    extra = torch.zeros_like(labels)
    stats = {"sampled": 0, "comparable": extra.sum(), "added": extra.sum()}
    if not active:
        return labels, extra, stats, None
    ref = reference_camera
    height, width = ref.image_height, ref.image_width
    for name, tensor, shape in (("prior_depth", prior_depth, (height, width)),
                                ("prior_normal", prior_normal, (3, height, width))):
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if tensor.device != xyz.device or tensor.dtype != xyz.dtype:
            raise ValueError(f"{name} must match Gaussian device/dtype")
    if reference_gray.ndim != 3 or reference_gray.shape[0] != 1 or reference_gray.device != xyz.device:
        raise ValueError("reference_gray must be [1,Hn,Wn] on the model device")
    if len(support_cameras) < 2 or xyz.shape[0] == 0:
        return labels, extra, stats, None

    transform = ref.world_view_transform
    camera_xyz = xyz @ transform[:3, :3] + transform[3, :3]
    projected = camera_xyz @ ref.get_k().T
    centers = projected[:, :2] / projected[:, 2:3]
    offsets = patch_offsets(opt.multi_view_patch_size, xyz.device).to(xyz.dtype)
    patch_centers = centers / ref.ncc_scale
    radius = opt.multi_view_patch_size
    gray_height, gray_width = reference_gray.shape[-2:]
    pool_mask = ~labels & (camera_xyz[:, 2] > ref.znear)
    pool_mask &= (torch.isfinite(patch_centers).all(-1)
                  & (patch_centers[:, 0] >= radius)
                  & (patch_centers[:, 0] <= gray_width - 1 - radius)
                  & (patch_centers[:, 1] >= radius)
                  & (patch_centers[:, 1] <= gray_height - 1 - radius))
    ids = _sample_candidate_ids(pool_mask.nonzero(as_tuple=True)[0], opt.candidate_selector_budget,
                               seed=seed, iteration=iteration, camera_uid=ref.uid)
    stats["sampled"] = ids.numel()
    if not ids.numel():
        return labels, extra, stats, None
    centers = centers[ids]
    reference_patch = centers[:, None] / ref.ncc_scale + offsets
    points_g = camera_xyz[ids]
    normal_world = gaussians.get_normal(ref)
    if normal_world.shape != xyz.shape or not torch.isfinite(normal_world).all():
        raise ValueError("selector requires finite model normals:[N,3]")
    normal_g, distance_g, valid_g = _oriented_plane(points_g, normal_world[ids] @ transform[:3, :3])

    depth_p = _sample_map(prior_depth[None], centers[:, None])[:, 0, 0]
    normal_p = _sample_map(prior_normal, centers[:, None])[:, 0]
    rays = torch.cat((centers, torch.ones_like(centers[:, :1])), -1) @ ref.get_inv_k().T
    points_p = rays * depth_p[:, None]
    normal_p, distance_p, valid_p = _oriented_plane(points_p, normal_p)
    valid_p &= torch.isfinite(depth_p) & (depth_p > ref.znear)
    valid_p &= ((centers[:, 0] >= 1) & (centers[:, 0] <= width - 2)
                & (centers[:, 1] >= 1) & (centers[:, 1] <= height - 2))
    # render_normal uses the four adjacent depth points, then bilinear sampling
    # blends up to four such normals; a finite normal alone does not certify
    # that its depth stencil was usable.
    depth_valid = torch.isfinite(prior_depth) & (prior_depth > ref.znear)
    normal_stencil_valid = torch.zeros_like(depth_valid)
    normal_stencil_valid[1:-1, 1:-1] = (
        depth_valid[1:-1, 1:-1] & depth_valid[:-2, 1:-1]
        & depth_valid[2:, 1:-1] & depth_valid[1:-1, :-2]
        & depth_valid[1:-1, 2:])
    stencil_support = _sample_map(normal_stencil_valid[None].to(xyz.dtype), centers[:, None])[:, 0, 0]
    valid_p &= stencil_support >= 1 - _EPS
    valid = valid_g & valid_p
    if not valid.any():
        return labels, extra, stats, None
    ids, centers, reference_patch = ids[valid], centers[valid], reference_patch[valid]
    points_g, normal_g, distance_g = points_g[valid], normal_g[valid], distance_g[valid]
    points_p, normal_p, distance_p = points_p[valid], normal_p[valid], distance_p[valid]
    reference_values = _sample_map(reference_gray, reference_patch).squeeze(-1)
    costs_g, costs_p, common = [], [], []
    for source in support_cameras[:2]:
        source_depth = support_depth_fn(source)
        if (not isinstance(source_depth, torch.Tensor)
                or tuple(source_depth.shape) != (1, source.image_height, source.image_width)
                or source_depth.device != xyz.device or source_depth.dtype != xyz.dtype):
            raise ValueError("support depth must be [1,Hs,Ws] on the model device/dtype")
        if not torch.isfinite(source_depth).all():
            raise FloatingPointError("non-finite model-rendered support depth")
        _, source_gray = source.get_image()
        if source_gray.ndim != 3 or source_gray.shape[0] != 1 or source_gray.device != xyz.device:
            raise ValueError("source gray must be [1,Hsn,Wsn] on the model device")
        g, valid_g = _source_cost(gaussians, ref, source, reference_patch, reference_values,
            centers, points_g, normal_g, distance_g, source_depth, source_gray, opt)
        p, valid_p = _source_cost(gaussians, ref, source, reference_patch, reference_values,
            centers, points_p, normal_p, distance_p, source_depth, source_gray, opt)
        costs_g.append(g)
        costs_p.append(p)
        common.append(valid_g & valid_p)
    common = torch.stack(common, -1)
    accepted = _best_source_decision(torch.stack(costs_g, -1), torch.stack(costs_p, -1), common)
    extra[ids] = accepted
    stats["comparable"] = common.any(-1).sum()
    stats["added"] = extra.sum()
    targets = None
    if include_targets and accepted.any():
        # Both tensors keep the SAME sampled/filtered order; extra.nonzero()
        # would sort the IDs and can pair targets with the wrong Gaussians.
        move_ids = ids[accepted]
        world_xyz = (points_p[accepted] - transform[3, :3]) @ transform[:3, :3].T
        targets = (move_ids.detach(), world_xyz.detach())
    return labels | extra, extra, stats, targets


def select_candidate_labels(gaussians, reference_camera, reference_gray,
                            prior_depth, prior_normal, support_cameras, opt, *,
                            seed, iteration, support_depth_fn):
    """Original V1 interface and choices, without producing relocation targets."""
    return _select_candidate_geometry(gaussians, reference_camera, reference_gray,
        prior_depth, prior_normal, support_cameras, opt, seed=seed,
        iteration=iteration, support_depth_fn=support_depth_fn)[:3]


def select_candidate_labels_with_targets(gaussians, reference_camera, reference_gray,
                                         prior_depth, prior_normal, support_cameras, opt, *,
                                         seed, iteration, support_depth_fn):
    """V2 uses exactly the V1 decisions and their evaluated DA3 world targets."""
    return _select_candidate_geometry(gaussians, reference_camera, reference_gray,
        prior_depth, prior_normal, support_cameras, opt, seed=seed,
        iteration=iteration, support_depth_fn=support_depth_fn, include_targets=True)
