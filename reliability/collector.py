import torch
import torch.nn.functional as F

from reliability.evidence import (
    EvidenceRefreshInputs,
    ReprojectionEvidence,
    normalize_prior_confidence,
    reprojection_validity_and_errors,
)


def _depth_map(value):
    value = value.detach()
    while value.ndim > 2 and value.shape[0] == 1:
        value = value.squeeze(0)
    if value.ndim != 2:
        raise ValueError("depth must have shape [H, W]")
    return value


def _normal_map(value):
    value = value.detach()
    if value.ndim != 3 or value.shape[0] != 3:
        raise ValueError("normal must have shape [3, H, W]")
    return value


def _resize_map(value, shape):
    value = _depth_map(value)
    if tuple(value.shape) == tuple(shape):
        return value
    return F.interpolate(
        value[None, None],
        size=shape,
        mode="bilinear",
        align_corners=True,
    ).squeeze(0).squeeze(0)


@torch.no_grad()
def reproject_depth_normal_maps(
    source_depth,
    target_depth,
    source_normal,
    target_normal,
    source_intrinsics,
    target_intrinsics,
    source_world_to_camera,
    target_world_to_camera,
):
    source_depth = _depth_map(source_depth)
    target_depth = _depth_map(target_depth)
    source_normal = _normal_map(source_normal)
    target_normal = _normal_map(target_normal)
    height, width = source_depth.shape
    device = source_depth.device
    dtype = source_depth.dtype
    matrices = (
        source_intrinsics,
        target_intrinsics,
        source_world_to_camera,
        target_world_to_camera,
    )
    source_k, target_k, source_w2c, target_w2c = (
        matrix.detach().to(device=device, dtype=dtype) for matrix in matrices
    )
    if source_k.shape != (3, 3) or target_k.shape != (3, 3):
        raise ValueError("intrinsics must have shape [3, 3]")
    if source_w2c.shape != (4, 4) or target_w2c.shape != (4, 4):
        raise ValueError("camera transforms must have shape [4, 4]")

    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    pixels = torch.stack((x, y, torch.ones_like(x)), dim=0).reshape(3, -1)
    source_camera = torch.linalg.inv(source_k) @ pixels
    source_camera = source_camera * source_depth.reshape(1, -1)
    source_h = torch.cat(
        (source_camera, torch.ones((1, source_camera.shape[1]), device=device, dtype=dtype)),
        dim=0,
    )
    world = torch.linalg.inv(source_w2c) @ source_h
    target_camera = target_w2c @ world
    projected_depth = target_camera[2]
    projected = target_k @ target_camera[:3]
    safe_depth = torch.where(
        projected_depth.abs() > 1e-8,
        projected_depth,
        torch.ones_like(projected_depth),
    )
    uv = (projected[:2] / safe_depth.unsqueeze(0)).transpose(0, 1)
    grid = uv.clone()
    grid[:, 0] = 2.0 * grid[:, 0] / max(width - 1, 1) - 1.0
    grid[:, 1] = 2.0 * grid[:, 1] / max(height - 1, 1) - 1.0
    grid = grid.reshape(1, height, width, 2)
    sampled_depth = F.grid_sample(
        target_depth[None, None], grid, align_corners=True
    ).reshape(-1)
    sampled_normal = F.grid_sample(
        target_normal[None], grid, align_corners=True
    ).squeeze(0).permute(1, 2, 0).reshape(-1, 3)
    evidence = reprojection_validity_and_errors(
        uv,
        projected_depth,
        source_depth.reshape(-1),
        sampled_depth,
        source_normal.permute(1, 2, 0).reshape(-1, 3),
        sampled_normal,
        image_height=height,
        image_width=width,
    )
    return ReprojectionEvidence(
        evidence.valid.reshape(height, width),
        evidence.depth_error.reshape(height, width),
        evidence.normal_error.reshape(height, width),
    )


def _normal_score(source, target, *, scale):
    source_norm = source.norm(dim=0)
    target_norm = target.norm(dim=0)
    valid = (
        torch.isfinite(source).all(dim=0)
        & torch.isfinite(target).all(dim=0)
        & (source_norm > 1e-8)
        & (target_norm > 1e-8)
    )
    dot = (
        source / source_norm.clamp_min(1e-8).unsqueeze(0)
        * target / target_norm.clamp_min(1e-8).unsqueeze(0)
    ).sum(dim=0).abs().clamp(0.0, 1.0)
    error = 1.0 - dot
    return torch.exp(-error / scale), error, valid


def camera_to_world_normal(camera, local_normal):
    local_normal = _normal_map(local_normal)
    rotation = camera.world_view_transform[:3, :3].to(
        device=local_normal.device, dtype=local_normal.dtype
    )
    return (
        local_normal.permute(1, 2, 0) @ rotation.transpose(0, 1)
    ).permute(2, 0, 1)


class D0EvidenceCollector:
    """Collect exact forward-only P/G/PG evidence over training cameras."""

    def __init__(
        self,
        cameras,
        gaussians,
        render_fn,
        pipe,
        background,
        opt,
        *,
        normal_from_depth_fn,
    ):
        self.cameras = list(cameras)
        self.gaussians = gaussians
        self.render_fn = render_fn
        self.pipe = pipe
        self.background = background
        self.opt = opt
        self.normal_from_depth_fn = normal_from_depth_fn

    def _render(self, camera, **evidence):
        return self.render_fn(
            camera,
            self.gaussians,
            self.pipe,
            self.background,
            app_model=None,
            return_plane=True,
            return_depth_normal=True,
            ray_reg=-1,
            opt=self.opt,
            **evidence,
        )

    @torch.no_grad()
    def __call__(self):
        if not self.cameras:
            raise ValueError("D0 collector requires training cameras")
        device = self.gaussians.get_xyz.device
        cached = []
        pixel_hits = []
        for camera in self.cameras:
            rendered = self._render(camera)
            geometry_depth = _depth_map(rendered["plane_depth"])
            shape = geometry_depth.shape
            prior_depth = _resize_map(
                camera.depth_dict["depth"].to(device), shape
            )
            prior_confidence = _resize_map(
                camera.depth_dict["conf"].to(device), shape
            )
            prior_normal = camera_to_world_normal(
                camera,
                self.normal_from_depth_fn(camera, prior_depth),
            )
            primitive_world = camera_to_world_normal(
                camera, rendered["rendered_normal"]
            )
            geometry_normal = camera_to_world_normal(
                camera, rendered["depth_normal"]
            )
            cached.append({
                "prior_depth": prior_depth.detach().cpu(),
                "prior_confidence": prior_confidence.detach().cpu(),
                "prior_normal": _normal_map(prior_normal).detach().cpu(),
                "geometry_depth": geometry_depth.detach().cpu(),
                "geometry_normal": geometry_normal.detach().cpu(),
                "primitive_normal": primitive_world.detach().cpu(),
                "alpha": _depth_map(rendered["rendered_alpha"]).detach().cpu(),
            })
            pixel_hits.append(
                rendered["out_observe"].detach().gt(0).cpu()
            )

        point_count = self.gaussians.get_xyz.shape[0]
        # P/G multi-view sums need separate numerator/denominator transport
        # channels because the CUDA validity contract is strictly boolean.
        numerator = torch.zeros((point_count, 8), device=device)
        denominator = torch.zeros_like(numerator)
        prior_support = torch.zeros(point_count, dtype=torch.int64, device=device)
        geometry_support = torch.zeros_like(prior_support)

        for source_index, camera in enumerate(self.cameras):
            source = {
                name: value.to(device) for name, value in cached[source_index].items()
            }
            height, width = source["geometry_depth"].shape
            prior_score_sum = torch.zeros((height, width), device=device)
            prior_valid_count = torch.zeros_like(prior_score_sum)
            geometry_score_sum = torch.zeros_like(prior_score_sum)
            geometry_valid_count = torch.zeros_like(prior_score_sum)
            source_k = camera.get_k().to(device)
            source_w2c = camera.world_view_transform.transpose(0, 1).to(device)
            for target_index in camera.nearest_id:
                target = {
                    name: value.to(device)
                    for name, value in cached[target_index].items()
                }
                target_camera = self.cameras[target_index]
                target_k = target_camera.get_k().to(device)
                target_w2c = target_camera.world_view_transform.transpose(0, 1).to(device)
                prior_pair = reproject_depth_normal_maps(
                    source["prior_depth"],
                    target["prior_depth"],
                    source["prior_normal"],
                    target["prior_normal"],
                    source_k,
                    target_k,
                    source_w2c,
                    target_w2c,
                )
                geometry_pair = reproject_depth_normal_maps(
                    source["geometry_depth"],
                    target["geometry_depth"],
                    source["geometry_normal"],
                    target["geometry_normal"],
                    source_k,
                    target_k,
                    source_w2c,
                    target_w2c,
                )
                prior_valid = prior_pair.valid.to(prior_score_sum.dtype)
                geometry_valid = geometry_pair.valid.to(prior_score_sum.dtype)
                prior_score_sum += prior_valid * torch.exp(
                    -0.5 * (prior_pair.depth_error / 0.05 + prior_pair.normal_error / 0.10)
                )
                prior_valid_count += prior_valid
                geometry_score_sum += geometry_valid * torch.exp(
                    -0.5 * (geometry_pair.depth_error / 0.05 + geometry_pair.normal_error / 0.10)
                )
                geometry_valid_count += geometry_valid

            prior_confidence = normalize_prior_confidence(
                source["prior_confidence"].unsqueeze(0)
            ).squeeze(0)
            prior_conf_valid = (
                torch.isfinite(source["prior_depth"])
                & (source["prior_depth"] > 0)
                & torch.isfinite(prior_confidence)
            )
            dn_score, _, dn_valid = _normal_score(
                source["primitive_normal"], source["geometry_normal"], scale=0.10
            )
            dn_valid &= source["alpha"] >= 0.5
            pg_normal_score, pg_normal_error, pg_normal_valid = _normal_score(
                source["prior_normal"], source["geometry_normal"], scale=1.0
            )
            del pg_normal_score
            pg_valid = (
                pg_normal_valid
                & torch.isfinite(source["prior_depth"])
                & torch.isfinite(source["geometry_depth"])
                & (source["prior_depth"] > 0)
                & (source["geometry_depth"] > 0)
                & (source["alpha"] >= 0.5)
            )
            pg_depth_error = (
                source["prior_depth"] - source["geometry_depth"]
            ).abs() / (
                source["prior_depth"] + source["geometry_depth"] + 1e-8
            )
            values = torch.stack((
                prior_confidence,
                prior_score_sum,
                prior_valid_count,
                geometry_score_sum,
                geometry_valid_count,
                dn_score,
                pg_depth_error,
                pg_normal_error,
            )).to(dtype=torch.float32)
            validity = torch.stack((
                prior_conf_valid,
                prior_valid_count > 0,
                prior_valid_count > 0,
                geometry_valid_count > 0,
                geometry_valid_count > 0,
                dn_valid,
                pg_valid,
                pg_valid,
            ))
            rendered = self._render(
                camera,
                evidence_values=values,
                evidence_validity=validity,
            )
            view_numerator = rendered["evidence_numerator"]
            view_denominator = rendered["evidence_denominator"]
            numerator += view_numerator
            denominator += view_denominator
            prior_support += (view_numerator[:, 2] > 1e-4).to(torch.int64)
            geometry_support += (view_numerator[:, 4] > 1e-4).to(torch.int64)

        prior_confidence = numerator[:, 0] / denominator[:, 0].clamp_min(1e-8)
        prior_multiview = numerator[:, 1] / numerator[:, 2].clamp_min(1e-8)
        geometry_multiview = numerator[:, 3] / numerator[:, 4].clamp_min(1e-8)
        geometry_depth_normal = numerator[:, 5] / denominator[:, 5].clamp_min(1e-8)
        sh_coefficients = getattr(
            self.gaussians, "_features_rest", self.gaussians.get_features[:, 1:]
        )
        return EvidenceRefreshInputs(
            sh_coefficients=sh_coefficients.detach(),
            sh_degree=self.gaussians.active_sh_degree,
            pixel_hits=torch.stack(pixel_hits),
            camera_centers=torch.stack([
                camera.camera_center.detach().to(device) for camera in self.cameras
            ]),
            centers=self.gaussians.get_xyz.detach(),
            normals=self.gaussians.get_smallest_axis().detach(),
            scale_reference=self.gaussians.get_scaling.detach().max(dim=1).values,
            prior_confidence=prior_confidence,
            prior_multiview=prior_multiview,
            prior_support_views=prior_support,
            geometry_multiview=geometry_multiview,
            geometry_depth_normal=geometry_depth_normal,
            geometry_support_views=geometry_support,
            pg_weighted_support=denominator[:, 6],
            pg_depth_error_sum=numerator[:, 6],
            pg_normal_error_sum=numerator[:, 7],
        )
