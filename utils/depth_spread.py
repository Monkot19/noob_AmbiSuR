"""Normal-guided candidate depths, shared by training and static capture.

Adapted from DP-GS dc1ce835, train.py:301-335 and graphics_utils.py:212-254
(the latter credits IronDepth). Host adaptations: integer K^-1 z-depth rays,
invalid-transfer rejection, and proposal output rather than a new depth loss.

DP-GS LICENSE.md notice:
Permission to use, copy, modify and distribute this software and its
documentation for educational, research and non-profit purposes only.
Any modification based on this work must be open-source and prohibited
for commercial use.
You must retain, in the source form of any derivative works that you
distribute, all copyright, patent, trademark, and attribution notices
from the source form of this work.
"""
import torch
import torch.nn.functional as F

from utils.candidate_selector import (
    _EPS, _depth_stencil_valid, _roundtrip_geometry, _sample_candidate_ids,
    _sample_map, _source_cost,
)
from utils.graphics_utils import patch_offsets


SPREAD_PATCH_SIZE = 5
SPREAD_STEPS = 10
ANCHOR_CONFIDENCE = .6
NORMAL_COS = .99
MASK_THRESHOLD = .01


def _camera_grid(camera, depth):
    h, w = depth.shape
    y, x = torch.meshgrid(torch.arange(h, device=depth.device),
                          torch.arange(w, device=depth.device), indexing='ij')
    centers = torch.stack((x, y), -1).to(depth.dtype)
    homogeneous = torch.cat((centers, torch.ones_like(centers[..., :1])), -1)
    return centers, homogeneous @ camera.get_inv_k().T


@torch.no_grad()
def propagate_depth(model_depth, prior_normal, rays, anchor_mask, normal_valid, *,
                    patch_size=SPREAD_PATCH_SIZE, steps=SPREAD_STEPS,
                    normal_cos=NORMAL_COS, mask_threshold=MASK_THRESHOLD):
    """DP-GS iteration with camera-space neighbor normals; no hard anchor reset."""
    h, w = model_depth.shape
    if (prior_normal.shape != (3,h,w) or rays.shape != (h,w,3)
            or anchor_mask.shape != (h,w) or normal_valid.shape != (h,w)
            or anchor_mask.dtype != torch.bool or normal_valid.dtype != torch.bool
            or any(x.device != model_depth.device for x in (prior_normal,rays,anchor_mask,normal_valid))
            or prior_normal.dtype != model_depth.dtype or rays.dtype != model_depth.dtype
            or not model_depth.is_floating_point()):
        raise ValueError('spread requires compatible depth, camera normal/rays and boolean masks')
    if (isinstance(patch_size,bool) or not isinstance(patch_size,int) or patch_size < 1
            or patch_size % 2 != 1 or isinstance(steps,bool) or not isinstance(steps,int) or steps < 1):
        raise ValueError('positive odd patch_size and positive steps required')
    normal_valid = (normal_valid & torch.isfinite(prior_normal).all(0)
        & (prior_normal.norm(dim=0) > _EPS) & torch.isfinite(rays).all(-1))
    mask = anchor_mask & normal_valid & torch.isfinite(model_depth) & (model_depth > 0)
    empty = torch.zeros_like(model_depth)
    if not mask.any():
        return empty, torch.zeros_like(mask)
    normals = F.normalize(torch.where(normal_valid[None],prior_normal,0.),dim=0)
    safe_rays = torch.where(torch.isfinite(rays),rays,0.).permute(2,0,1)
    pad = patch_size // 2
    count_neighbors = patch_size ** 2
    def neighbors(value):
        padded = F.pad(value[None],(pad,pad,pad,pad),mode='replicate')
        return F.unfold(padded,(patch_size,patch_size)).view(value.shape[0],count_neighbors,h,w)
    normal_neighbors = neighbors(normals)
    normal_edge = ((normal_neighbors * normals[:,None]).sum(0) > normal_cos).all(0)
    denominator = (normal_neighbors * safe_rays[:,None]).sum(0)
    numerator = (normal_neighbors * neighbors(safe_rays)).sum(0)
    coefficient_valid = torch.isfinite(denominator) & (denominator.abs() > _EPS)
    weights = numerator / torch.where(coefficient_valid,denominator,torch.ones_like(denominator))
    coefficient_valid &= torch.isfinite(weights) & (weights > 0)
    weights = torch.where(coefficient_valid,weights,0.)
    del normal_neighbors, numerator, denominator
    depth = torch.where(mask,model_depth,0.)
    for _ in range(steps):
        depth_neighbors = neighbors((depth * mask)[None])[0]
        mask_neighbors = neighbors(mask[None].to(depth.dtype))[0]
        available = (mask_neighbors > 0) & coefficient_valid
        num = (weights * depth_neighbors).sum(0)
        divisor = available.sum(0)
        updated = num / (divisor + 1e-6)
        # The donor applies a center 5x5 normal gate to the mean weighted mask.
        mask = ((weights * mask_neighbors).mean(0) * normal_edge > mask_threshold)
        mask &= normal_valid & (divisor > 0) & torch.isfinite(updated) & (updated > 0)
        depth = torch.where(mask,updated,0.)
    return depth, mask


def _rgb_at_ncc_size(camera, prototype):
    rgb, gray = camera.get_image()
    if rgb.ndim != 3 or rgb.shape[0] != 3:
        raise ValueError('anchor NCC requires RGB images')
    rgb = rgb.to(prototype)
    if rgb.shape[-2:] != gray.shape[-2:]:
        rgb = F.interpolate(rgb[None],size=gray.shape[-2:],mode='bilinear',align_corners=True)[0]
    return rgb


@torch.no_grad()
def build_spread_proposal(gaussians, reference, supports, prior_depth, prior_normal,
                          reference_geometry, opt, *, seed, iteration, support_depth_fn):
    """Rendered-plane RGB/roundtrip anchors -> propagated candidate depth/mask."""
    h, w = reference.image_height, reference.image_width
    prototype = gaussians.get_xyz
    output = prototype.new_zeros((h,w))
    mask = torch.zeros((h,w),dtype=torch.bool,device=prototype.device)
    stats = dict(anchors=mask.sum(),spread_pixels=mask.sum())
    if len(supports) < 2:
        return output, mask, stats
    for name, shape in (('plane_depth',(1,h,w)),('rendered_normal',(3,h,w)),
                         ('rendered_distance',(1,h,w))):
        value = reference_geometry[name]
        if value.shape != shape or value.dtype != prototype.dtype or value.device != prototype.device:
            raise ValueError('invalid reference geometry for spread: '+name)
    if prior_depth.shape != (h,w) or prior_normal.shape != (3,h,w):
        raise ValueError('spread prior dimensions differ from reference')
    depth = reference_geometry['plane_depth'][0]
    normal = reference_geometry['rendered_normal'].permute(1,2,0).reshape(-1,3)
    distance = reference_geometry['rendered_distance'].reshape(-1)
    normal_valid = _depth_stencil_valid(prior_depth,reference.znear)
    normal_valid &= torch.isfinite(prior_normal).all(0) & (prior_normal.norm(dim=0) > _EPS)
    centers, rays = _camera_grid(reference,depth)
    points = (rays * depth[...,None]).reshape(-1,3)
    centers = centers.reshape(-1,2)
    rgb = _rgb_at_ncc_size(reference,depth)
    patch_center = centers / reference.ncc_scale
    radius = opt.multi_view_patch_size
    pool = (torch.isfinite(points).all(-1) & (depth.reshape(-1) > reference.znear)
        & torch.isfinite(normal).all(-1) & (normal.norm(dim=-1) > _EPS)
        & torch.isfinite(distance) & (distance > _EPS) & normal_valid.reshape(-1)
        & (patch_center[:,0] >= radius) & (patch_center[:,0] <= rgb.shape[2]-1-radius)
        & (patch_center[:,1] >= radius) & (patch_center[:,1] <= rgb.shape[1]-1-radius))
    ids = pool.nonzero(as_tuple=True)[0]
    if not ids.numel():
        return output, mask, stats
    source = supports[0]
    source_depth = support_depth_fn(source)
    if (source_depth.shape != (1,source.image_height,source.image_width)
            or source_depth.device != prototype.device or source_depth.dtype != prototype.dtype
            or not torch.isfinite(source_depth).all()):
        raise ValueError('invalid source depth for spread')
    geometry_valid, errors = _roundtrip_geometry(gaussians,reference,source,
        centers[ids],points[ids],source_depth)
    good = geometry_valid & (errors < opt.multi_view_pixel_noise_th)
    # Keep the actual roundtrip error associated with the sampled image indices.
    error_map = depth.new_zeros(h*w)
    error_map[ids] = errors
    budget = opt.multi_view_sample_num
    if isinstance(budget,bool) or not isinstance(budget,int) or budget <= 0:
        raise ValueError('multi_view_sample_num must be a positive integer')
    ids = _sample_candidate_ids(ids[good],budget,seed=seed,iteration=iteration,
        camera_uid=reference.uid,stream='selector-v3-spread-anchors')
    if not ids.numel():
        return output, mask, stats
    patch = centers[ids,None]/reference.ncc_scale + patch_offsets(radius,depth.device).to(depth.dtype)
    cost, valid = _source_cost(gaussians,reference,source,patch,_sample_map(rgb,patch),
        centers[ids],points[ids],normal[ids],distance[ids],source_depth,
        _rgb_at_ncc_size(source,depth),opt)
    confidence = (2-cost)/2 * torch.exp(-error_map[ids])
    anchors = ids[valid & (confidence > ANCHOR_CONFIDENCE)]
    mask.reshape(-1)[anchors] = True
    stats['anchors'] = mask.sum()
    output, mask = propagate_depth(depth,prior_normal,rays,mask,normal_valid)
    stats['spread_pixels'] = mask.sum()
    return output, mask, stats
