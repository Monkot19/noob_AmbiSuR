#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

# 精读地图：本文件组织训练初始化与主循环，汇总各项 Loss、执行反向传播，并在指定迭代阶段触发 Gaussian 稠密化与剪枝。

import os
from datetime import datetime
import torch
import random
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim, lncc, get_img_grad_weight, DepthAnythingv2Loss
from utils.graphics_utils import patch_offsets, patch_warp
from gaussian_renderer import render, network_gui, render_normal
import sys, time
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import cv2
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, erode
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.app_model import AppModel
from scene.cameras import Camera
from utils.mono_utils import prepare_depthanythingv2
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import time
import torch.nn.functional as F

from scene.cameras import get_camera_optimizer
from reliability.config import CoreConfig
from reliability.runtime import build_checkpoint_payload, select_training_path


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(22)

def gen_virtul_cam(cam, trans_noise=1.0, deg_noise=15.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = cam.R.transpose()
    Rt[:3, 3] = cam.T
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)

    translation_perturbation = np.random.uniform(-trans_noise, trans_noise, 3)
    rotation_perturbation = np.random.uniform(-deg_noise, deg_noise, 3)
    rx, ry, rz = np.deg2rad(rotation_perturbation)
    Rx = np.array([[1, 0, 0],
                    [0, np.cos(rx), -np.sin(rx)],
                    [0, np.sin(rx), np.cos(rx)]])
    
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]])
    
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]])
    R_perturbation = Rz @ Ry @ Rx

    C2W[:3, :3] = C2W[:3, :3] @ R_perturbation
    C2W[:3, 3] = C2W[:3, 3] + translation_perturbation
    Rt = np.linalg.inv(C2W)
    virtul_cam = Camera(100000, Rt[:3, :3].transpose(), Rt[:3, 3], cam.FoVx, cam.FoVy,
                        cam.image_width, cam.image_height,
                        cam.image_path, cam.image_name, 100000,
                        trans=np.array([0.0, 0.0, 0.0]), scale=1.0, 
                        preload_img=False, data_device = "cuda")
    return virtul_cam

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, core_config=None):
    if core_config is None:
        core_config = CoreConfig()
    training_path = select_training_path(core_config)
    if training_path != "legacy":
        raise NotImplementedError("Core training path is not implemented in E0")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, opt)
    # # backup main code
    # cmd = f'cp ./train.py {dataset.model_path}/'
    # os.system(cmd)
    # cmd = f'cp -rf ./arguments {dataset.model_path}/'
    # os.system(cmd)
    # cmd = f'cp -rf ./gaussian_renderer {dataset.model_path}/'
    # os.system(cmd)
    # cmd = f'cp -rf ./scene {dataset.model_path}/'
    # os.system(cmd)
    # cmd = f'cp -rf ./utils {dataset.model_path}/'
    # os.system(cmd)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    app_model = AppModel()
    app_model.train()
    app_model.cuda()

    first_iter = 0
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        app_model.load_weights(scene.model_path)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_single_view_for_log = 0.0
    ema_multi_view_geo_for_log = 0.0
    ema_multi_view_pho_for_log = 0.0
    image_loss, normal_loss, geo_loss, ncc_loss = None, None, None, None
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress", ascii=True, dynamic_ncols=True)
    first_iter += 1
    debug_path = os.path.join(scene.model_path, "debug")
    os.makedirs(debug_path, exist_ok=True)

    if opt.use_mono:
        prepare_depthanythingv2(
            cameras=scene.getTrainCameras(),
            source_path=dataset.source_path,
            force_rerun=False)
        
        # mono 分支虽在 iteration>1000 时进入，但该 Loss 内部从 3000 才激活，因此默认首个有效 Loss 在第 3000 轮。
        depthanythingv2_loss = DepthAnythingv2Loss(
            iter_from=3000,
            iter_end=opt.iterations,
            end_mult=0.1,
            overall=opt.use_mono_overall)



    # 每轮训练可分为三段：前向渲染并汇总 Loss -> backward 计算梯度 -> 无梯度区内统计、增删点和 optimizer 更新参数。
    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        loss = 0

        gaussians.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        gt_image, gt_image_gray = viewpoint_cam.get_image()
        if iteration > 1000 and opt.exposure_compensation:
            gaussians.use_app = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        # ray_reg 就是 ray_color_lambda 传入 renderer/CUDA 后的名字；默认在第 5001 轮到 densification 结束（含）之间开启。
        # 它不形成 Python 可见的 Loss，而在每次 rasterizer backward 中惩罚同一 ray 上“单个 Gaussian 颜色与最终像素颜色的差异”。
        # 额外项先直接加到 dL_dcolors；默认 SH 路径会继续更新 SH，并可能经视角方向更新 means3D，但不直接写 densification 的屏幕梯度。
        # 耦合风险：7001..densify_until_iter 还会执行 ALR 的独立 backward，因而同一 Ray-Color 项可能先后累计两次。
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, app_model=app_model,
                            return_plane=iteration>0, return_depth_normal=True, 
                            ray_reg=(-1 if iteration > opt.densify_until_iter else opt.ray_color_lambda) if iteration > 5000 else -1,
                            opt=opt)
        image, viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        # Loss
        ssim_loss = (1.0 - ssim(image, gt_image))
        if 'app_image' in render_pkg and ssim_loss < 0.5:
            app_image = render_pkg['app_image']
            Ll1 = l1_loss(app_image, gt_image)
            ssim_loss = (1.0 - ssim(render_pkg['app_image_detach'], gt_image))
        else:
            Ll1 = l1_loss(image, gt_image)
        image_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss
        loss = image_loss.clone()
        
        # scale loss
        if visibility_filter.sum() > 0:
            scale = gaussians.get_scaling[visibility_filter]
            sorted_scale, _ = torch.sort(scale, dim=-1)
            min_scale_loss = sorted_scale[...,0]
            loss += opt.scale_loss_weight * min_scale_loss.mean()
        # single-view loss
        # single-view、multi-view 与 ALR 都对默认 from_iter=7000 使用严格的 >，所以实际首轮是 7001。
        if iteration > opt.single_view_weight_from_iter:
            weight = opt.single_view_weight
            normal = render_pkg["rendered_normal"]
            depth_normal = render_pkg["depth_normal"]

            image_weight = (1.0 - get_img_grad_weight(gt_image))
            image_weight = (image_weight).clamp(0,1).detach() ** 2
            if not opt.wo_image_weight:
                # image_weight = erode(image_weight[None,None]).squeeze()
                normal_loss = weight * (image_weight * (((depth_normal - normal)).abs().sum(0))).mean()
            else:
                normal_loss = weight * (((depth_normal - normal)).abs().sum(0)).mean()
            loss += (normal_loss)

        # multi-view loss
        if iteration > opt.multi_view_weight_from_iter:
            nearest_cam = None if len(viewpoint_cam.nearest_id) == 0 else scene.getTrainCameras()[random.sample(viewpoint_cam.nearest_id,1)[0]]
            use_virtul_cam = False
            if opt.use_virtul_cam and (np.random.random() < opt.virtul_cam_prob or nearest_cam is None):
                nearest_cam = gen_virtul_cam(viewpoint_cam, trans_noise=dataset.multi_view_max_dis, deg_noise=dataset.multi_view_max_angle)
                use_virtul_cam = True
            if nearest_cam is not None:
                patch_size = opt.multi_view_patch_size
                sample_num = opt.multi_view_sample_num
                pixel_noise_th = opt.multi_view_pixel_noise_th
                total_patch_size = (patch_size * 2 + 1) ** 2
                ncc_weight = opt.multi_view_ncc_weight
                geo_weight = opt.multi_view_geo_weight
                ## compute geometry consistency mask and loss
                H, W = render_pkg['plane_depth'].squeeze().shape
                ix, iy = torch.meshgrid(
                    torch.arange(W), torch.arange(H), indexing='xy')
                pixels = torch.stack([ix, iy], dim=-1).float().to(render_pkg['plane_depth'].device)

                nearest_render_pkg = render(nearest_cam, gaussians, pipe, bg, app_model=app_model,
                                            return_plane=True, return_depth_normal=False)

                pts = gaussians.get_points_from_depth(viewpoint_cam, render_pkg['plane_depth'])
                pts_in_nearest_cam = pts @ nearest_cam.world_view_transform[:3,:3] + nearest_cam.world_view_transform[3,:3]
                pts_in_nearest_cam[:, 2] = torch.where(pts_in_nearest_cam[:, 2] <= 1e-3, 1e-3, pts_in_nearest_cam[:, 2])
                map_z, d_mask = gaussians.get_points_depth_in_depth_map(nearest_cam, nearest_render_pkg['plane_depth'], pts_in_nearest_cam)
                
                pts_in_nearest_cam = pts_in_nearest_cam / (pts_in_nearest_cam[:,2:3])
                pts_in_nearest_cam = pts_in_nearest_cam * map_z.squeeze()[...,None]
                R = torch.tensor(nearest_cam.R).float().cuda()
                T = torch.tensor(nearest_cam.T).float().cuda()
                pts_ = (pts_in_nearest_cam-T)@R.transpose(-1,-2)
                pts_in_view_cam = pts_ @ viewpoint_cam.world_view_transform[:3,:3] + viewpoint_cam.world_view_transform[3,:3]
                pts_in_view_cam[:, 2] = torch.where(pts_in_view_cam[:, 2] <= 1e-3, 1e-3, pts_in_view_cam[:, 2])
                pts_projections = torch.stack(
                            [pts_in_view_cam[:,0] * viewpoint_cam.Fx / pts_in_view_cam[:,2] + viewpoint_cam.Cx,
                            pts_in_view_cam[:,1] * viewpoint_cam.Fy / pts_in_view_cam[:,2] + viewpoint_cam.Cy], -1).float()
                pixel_noise = torch.norm(pts_projections - pixels.reshape(*pts_projections.shape), dim=-1)
                if not opt.wo_use_geo_occ_aware:
                    d_mask = d_mask & (pixel_noise < pixel_noise_th)
                    weights = (1.0 / torch.exp(pixel_noise)).detach()
                    weights[~d_mask] = 0
                else:
                    d_mask = d_mask
                    weights = torch.ones_like(pixel_noise)
                    weights[~d_mask] = 0
                # d_mask_inv = (~d_mask).reshape(render_pkg['plane_depth'].shape)
                if iteration % 200 == 0:
                    gt_img_show = ((gt_image).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                    if 'app_image' in render_pkg:
                        img_show = ((render_pkg['app_image']).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                    else:
                        img_show = ((image).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                    normal_show = (((normal+1.0)*0.5).permute(1,2,0)[:,:,[2,1,0]].clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
                    depth_normal_show = (((depth_normal+1.0)*0.5).permute(1,2,0)[:,:,[2,1,0]].clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
                    d_mask_show = (weights.float()*255).detach().cpu().numpy().astype(np.uint8).reshape(H,W)
                    d_mask_show_color = cv2.applyColorMap(d_mask_show, cv2.COLORMAP_JET)
                    depth = render_pkg['plane_depth'].squeeze().detach().cpu().numpy()
                    depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
                    depth_i = depth_i + (1 - render_pkg["rendered_alpha"].squeeze().detach().cpu().numpy()) * depth_i.max()
                    depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
                    depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)
                    distance = render_pkg['rendered_distance'].squeeze().detach().cpu().numpy()
                    distance_i = (distance - distance.min()) / (distance.max() - distance.min() + 1e-20)
                    distance_i = (distance_i * 255).clip(0, 255).astype(np.uint8)
                    distance_color = cv2.applyColorMap(distance_i, cv2.COLORMAP_JET)
                    # image_weight = image_weight.detach().cpu().numpy()
                    # image_weight = (image_weight * 255).clip(0, 255).astype(np.uint8)
                    # image_weight_color = cv2.applyColorMap(image_weight, cv2.COLORMAP_JET)
                    rendered_unc = (render_pkg["rendered_unc"] / render_pkg["rendered_unc"].max()).squeeze().detach().cpu().numpy()
                    rendered_unc = (rendered_unc * 255).clip(0, 255).astype(np.uint8)
                    rendered_unc_color = cv2.applyColorMap(rendered_unc, cv2.COLORMAP_JET)
                    row0 = np.concatenate([gt_img_show, img_show, normal_show, distance_color], axis=1)
                    row1 = np.concatenate([d_mask_show_color, depth_color, depth_normal_show, rendered_unc_color], axis=1)
                    image_to_show = np.concatenate([row0, row1], axis=0)
                    cv2.imwrite(os.path.join(debug_path, "%05d"%iteration + "_" + viewpoint_cam.image_name + ".jpg"), image_to_show)

                if d_mask.sum() > 0:
                    geo_loss = geo_weight * ((weights * pixel_noise)[d_mask]).mean()
                    loss += geo_loss
                    if use_virtul_cam is False:
                        with torch.no_grad():
                            '''debug'''
                            if iteration % 200 == 0:
                                sample_num = 1e9
                                d_mask_real = d_mask.clone()
                                d_mask = torch.ones_like(d_mask)

                            ## sample mask
                            d_mask = d_mask.reshape(-1)
                            valid_indices = torch.arange(d_mask.shape[0], device=d_mask.device)[d_mask]
                            if d_mask.sum() > sample_num:
                                index = np.random.choice(d_mask.sum().cpu().numpy(), sample_num, replace = False)
                                valid_indices = valid_indices[index]

                            weights = weights.reshape(-1)[valid_indices]
                            ## sample ref frame patch
                            pixels = pixels.reshape(-1,2)[valid_indices]
                            offsets = patch_offsets(patch_size, pixels.device)
                            ori_pixels_patch = pixels.reshape(-1, 1, 2) / viewpoint_cam.ncc_scale + offsets.float()
                            
                            H, W = gt_image_gray.squeeze().shape
                            pixels_patch = ori_pixels_patch.clone()
                            pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (W - 1) - 1.0
                            pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (H - 1) - 1.0
                            ref_gray_val = F.grid_sample(gt_image_gray.unsqueeze(1), pixels_patch.view(1, -1, 1, 2), align_corners=True)
                            ref_gray_val = ref_gray_val.reshape(-1, total_patch_size)

                            ref_to_neareast_r = nearest_cam.world_view_transform[:3,:3].transpose(-1,-2) @ viewpoint_cam.world_view_transform[:3,:3]
                            ref_to_neareast_t = -ref_to_neareast_r @ viewpoint_cam.world_view_transform[3,:3] + nearest_cam.world_view_transform[3,:3]

                        ## compute Homography
                        ref_local_n = render_pkg["rendered_normal"].permute(1,2,0)
                        ref_local_n = ref_local_n.reshape(-1,3)[valid_indices]

                        ref_local_d = render_pkg['rendered_distance'].squeeze()
                        # rays_d = viewpoint_cam.get_rays()
                        # rendered_normal2 = render_pkg["rendered_normal"].permute(1,2,0).reshape(-1,3)
                        # ref_local_d = render_pkg['plane_depth'].view(-1) * ((rendered_normal2 * rays_d.reshape(-1,3)).sum(-1).abs())
                        # ref_local_d = ref_local_d.reshape(*render_pkg['plane_depth'].shape)

                        ref_local_d = ref_local_d.reshape(-1)[valid_indices].clamp(min=1e-6)
                        H_ref_to_neareast = ref_to_neareast_r[None] - \
                            torch.matmul(ref_to_neareast_t[None,:,None].expand(ref_local_d.shape[0],3,1), 
                                        ref_local_n[:,:,None].expand(ref_local_d.shape[0],3,1).permute(0, 2, 1))/ref_local_d[...,None,None]
                        H_ref_to_neareast = torch.matmul(nearest_cam.get_k(nearest_cam.ncc_scale)[None].expand(ref_local_d.shape[0], 3, 3), H_ref_to_neareast)
                        H_ref_to_neareast = H_ref_to_neareast @ viewpoint_cam.get_inv_k(viewpoint_cam.ncc_scale)
                        
                        ## compute neareast frame patch
                        grid = patch_warp(H_ref_to_neareast.reshape(-1,3,3), ori_pixels_patch)
                        grid[:, :, 0] = 2 * grid[:, :, 0] / (W - 1) - 1.0
                        grid[:, :, 1] = 2 * grid[:, :, 1] / (H - 1) - 1.0
                        _, nearest_image_gray = nearest_cam.get_image()
                        sampled_gray_val = F.grid_sample(nearest_image_gray[None], grid.reshape(1, -1, 1, 2), align_corners=True)
                        sampled_gray_val = sampled_gray_val.reshape(-1, total_patch_size)


                        '''debug'''
                        if iteration % 200 == 0:
                            import torchvision
                            outdir = os.path.join(scene.model_path, "pg_view")
                            os.makedirs(outdir, exist_ok=True)
                            
                            ref_gray_val[~d_mask_real[..., None].repeat(1, total_patch_size)] = 0
                            sampled_gray_val[~d_mask_real[..., None].repeat(1, total_patch_size)] = 0
                            
                            torchvision.utils.save_image(ref_gray_val.reshape(viewpoint_cam.image_height, viewpoint_cam.image_width, total_patch_size)[..., -1], fp = os.path.join(scene.model_path, "pg_view", f"iter{iteration:06d}_ref.jpg"))
                            torchvision.utils.save_image(sampled_gray_val.reshape(viewpoint_cam.image_height, viewpoint_cam.image_width, total_patch_size)[..., -1], fp = os.path.join(scene.model_path, "pg_view", f"iter{iteration:06d}_sample.jpg"))
                            # torchvision.utils.save_image(nearest_image_gray, fp = os.path.join(voxel_model.model_path, "pg_view", f"iter{iteration:06d}_image_nearest.jpg"))
                            # torchvision.utils.save_image(gt_image_gray, fp = os.path.join(voxel_model.model_path, "pg_view", f"iter{iteration:06d}_image_gt.jpg"))


                        ## compute loss
                        ncc, ncc_mask = lncc(ref_gray_val, sampled_gray_val)
                        mask = ncc_mask.reshape(-1)
                        ncc = ncc.reshape(-1) * weights
                        ncc = ncc[mask].squeeze()

                        if mask.sum() > 0:
                            ncc_loss = ncc_weight * ncc.mean()
                            loss += ncc_loss

        # Depth loss
        if iteration > 1000:
            if not opt.use_mono:
                metric_depth = viewpoint_cam.depth_dict["depth"].detach().squeeze()
                metric_depth = torch.nn.functional.interpolate(metric_depth[None, None], size=render_pkg["plane_depth"].shape[-2:], mode='bilinear', align_corners=True).squeeze()
                if "conf" in viewpoint_cam.depth_dict:
                    metric_depth_conf = torch.nn.functional.interpolate(viewpoint_cam.depth_dict["conf"][None, None].cuda(), size=render_pkg["plane_depth"].shape[-2:], mode='bilinear', align_corners=True).squeeze()
                    conf_thresh = torch.quantile(metric_depth_conf.flatten(), opt.unc_conf_thresh_ratio)
                else:
                    metric_depth_conf = 1
                    conf_thresh = -1
                d_l1 = (metric_depth - render_pkg["plane_depth"].squeeze()).abs() * (metric_depth_conf >= conf_thresh)
                loss += opt.depth_weight * d_l1.mean() / scene.cameras_extent
            else:
                loss += opt.depth_weight * depthanythingv2_loss(viewpoint_cam, render_pkg, iteration)
                mono_depth = viewpoint_cam.depthanythingv2.cuda().squeeze()
                pseudo_metric_depth = torch.nn.functional.interpolate(mono_depth[None, None], size=render_pkg["plane_depth"].shape[-2:], mode='bilinear', align_corners=True).squeeze()
                metric_depth = 1.0 / pseudo_metric_depth.clamp(min=1e-3)
                metric_depth = metric_depth / metric_depth.max() * render_pkg["plane_depth"].max().detach()
                metric_depth_conf = 1
                conf_thresh = -1
            
            # loss_unc 对应论文的 Amorphous Local Regularizer (ALR)，不是预测“不确定度数值”的监督 Loss。
            # 它用 SH 双端歧义区域的软 Mask，加权“深度先验法线”和“渲染深度法线”的方向差异。
            # rendered_unc.detach() 表示该 Mask 在此处只决定惩罚位置，不通过本 Loss 反向更新 Mask 的生成路径。
            if iteration > opt.unc_from_iter:
                # renderer 已计算像素级 Mask；这里重新计算 Gaussian 级 Mask，供后面的参数级选择使用。
                sh_uncertainty = gaussians.compute_weighted_sh_norm("equal")
                unc_thresh = torch.quantile(sh_uncertainty.flatten(), opt.sh_ambi_upper_ratio)
                unc_thresh_min = torch.quantile(sh_uncertainty.flatten(), opt.sh_ambi_lower_ratio)
                unc_thresh_min = min(opt.sh_unc_lower_max, unc_thresh_min)
                unc_mask = (sh_uncertainty > unc_thresh) | (sh_uncertainty < unc_thresh_min)

                # 配置耦合风险：ALR 衰减进度以 multi_view_weight_from_iter 为起点，而不是 unc_from_iter；默认二者恰好同为 7000。
                ratio = (iteration - opt.multi_view_weight_from_iter) / (opt.iterations - opt.multi_view_weight_from_iter)
                # 当 0<unc_decay<1 时，mult 让 ALR 在后期逐渐减弱、避免过度约束；默认 unc_decay=1.0 时 mult 始终为 1。
                mult = opt.unc_decay ** ratio

                normal_from_metric_depth = render_normal(viewpoint_cam, metric_depth)
                # 深度先验法线、置信度条件和 detach 后的像素 Mask 都只充当固定权重；梯度来自渲染 depth_normal 一侧。
                # depth_normal 由 plane_depth 的邻域点叉积得到；CUDA 再把深度梯度分到 normal/distance 特征及 alpha 混合权重。
                # 因而 ALR 可到 xyz/rotation，若参数分离失效还可经 alpha 路径到 opacity/scale；其屏幕代理梯度会在下方单独清零。
                unc_diff = 1 - (normal_from_metric_depth * render_pkg["depth_normal"]).sum(0)
                loss_unc = opt.unc_weight * mult * (unc_diff * render_pkg["rendered_unc"].detach() * (metric_depth_conf >= conf_thresh)).mean()

                # 代码风险（服务器待验证）：requires_grad_ 是方法，下面却给方法名赋值，没有调用 PyTorch 的冻结接口，可能直接报只读属性错误。
                # 即使改成 requires_grad_(False)，布尔索引得到的临时张量也不能实现“按 Gaussian 元素冻结”，且前向计算图已经建立。
                # 论文/代码不符：论文明确冻结非风险点并排除 opacity/scale；结合 normal-loss 路径，风险点的位置/旋转应是几何优化对象。
                # 下方代码的表面意图却是保留风险点 opacity、冻结全部 rotation，与该 Parameter Separation 描述相反。
                # 因而当前代码不能据此证明 ALR 只更新 unc_mask 命中的 Gaussian；真正实现需对 ALR 单独产生的参数梯度做 Mask 等处理。
                gaussians._xyz[~unc_mask].requires_grad_ = False
                gaussians._opacity[~unc_mask].requires_grad_ = False
                gaussians._scaling.requires_grad_ = False
                gaussians._rotation.requires_grad_ = False

                # 先单独反传 ALR，把它的贡献累积进参数 .grad；这并不会自动避免与主 Loss 梯度混合。
                # retain_graph=True 保留同一计算图，供后面的主 loss.backward() 再次反传并继续累加梯度。
                loss_unc.backward(retain_graph=True)
                # 只清除 ALR 的屏幕空间代理梯度，使它不进入 densification 统计；Gaussian Parameter 上的 ALR 梯度仍被保留。
                # viewspace_point_tensor 就是 render_pkg["viewspace_points"]，所以第一、第三行实际上重复清零同一个张量。
                render_pkg["viewspace_points"].grad *= 0
                render_pkg["viewspace_points_abs"].grad *= 0
                viewspace_point_tensor.grad *= 0

                gaussians._xyz[~unc_mask].requires_grad_ = True
                gaussians._opacity[~unc_mask].requires_grad_ = True
                gaussians._scaling.requires_grad_ = True
                gaussians._rotation.requires_grad_ = True


        # 此处只计算主 Loss 的梯度贡献，但会累加到同一份 .grad；若 ALR 已执行，最终 .grad 同时包含 ALR 与主 Loss。
        # backward 仍不改参数数值，真正应用这两部分累计梯度要等后面的 optimizer.step()。
        loss.backward()
        iter_end.record()

        # no_grad 禁止下面的管理操作建立新计算图，但仍可读取刚才 backward 已写好的 .grad。
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * image_loss.item() if image_loss is not None else 0.0 + 0.6 * ema_loss_for_log
            ema_single_view_for_log = 0.4 * normal_loss.item() if normal_loss is not None else 0.0 + 0.6 * ema_single_view_for_log
            ema_multi_view_geo_for_log = 0.4 * geo_loss.item() if geo_loss is not None else 0.0 + 0.6 * ema_multi_view_geo_for_log
            ema_multi_view_pho_for_log = 0.4 * ncc_loss.item() if ncc_loss is not None else 0.0 + 0.6 * ema_multi_view_pho_for_log
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "Single": f"{ema_single_view_for_log:.{5}f}",
                    "Geo": f"{ema_multi_view_geo_for_log:.{5}f}",
                    "Pho": f"{ema_multi_view_pho_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            # 日志风险：这里传入的 loss 没有加上单独计算的 loss_unc，因此记录的“总 Loss”不含 ALR，尽管优化器会使用其累计梯度。
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), app_model)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                    
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                mask = (render_pkg["out_observe"] > 0) & visibility_filter
                gaussians.max_radii2D[mask] = torch.max(gaussians.max_radii2D[mask], radii[mask])
                viewspace_point_tensor_abs = render_pkg["viewspace_points_abs"]

                # 将本轮可见 Gaussian 的屏幕空间 xy 梯度累计起来，供周期性的 clone/split 判断使用。
                # Primitive Truncation 不直接提供 densify Mask，但会改变同一 rasterizer backward 的屏幕梯度，因而间接影响增点。
                gaussians.add_densification_stats(viewspace_point_tensor, viewspace_point_tensor_abs, visibility_filter)

                # 默认条件是 iteration>500 且每 100 轮一次，因此首次 densification 为 600，末次为 14900。
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    # 精读风险（服务器待验证）：densify/prune 在 optimizer.step() 前重建 Gaussian Parameter，
                    # 只迁移 Adam 动量而未显式迁移旧 .grad；发生替换的本轮可能跳过部分或全部 Gaussian 参数更新。
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.densify_abs_grad_threshold, 
                                                opt.opacity_cull_threshold, scene.cameras_extent, size_threshold)
            
            # multi-view observe trim
            if opt.use_multi_view_trim and iteration % 1000 == 0 and iteration < opt.densify_until_iter:
                observe_the = 2
                observe_cnt = torch.zeros_like(gaussians.get_opacity)
                for view in scene.getTrainCameras():
                    render_pkg_tmp = render(view, gaussians, pipe, bg, app_model=app_model, return_plane=False, return_depth_normal=False)
                    out_observe = render_pkg_tmp["out_observe"]
                    observe_cnt[out_observe > 0] += 1
                prune_mask = (observe_cnt < observe_the).squeeze()
                if prune_mask.sum() > 0:
                    gaussians.prune_points(prune_mask)

            # reset_opacity
            if iteration < opt.densify_until_iter:
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            # 默认第 30000 轮仍会渲染、反传和保存，但只有 1..29999 满足此条件并真正更新参数。
            if iteration < opt.iterations:
                # step 按当前 .grad 改参数；随后设为 None，防止旧梯度被下一轮继续累加。
                gaussians.optimizer.step()
                app_model.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                app_model.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save(
                    build_checkpoint_payload(
                        gaussians.capture(), iteration, core_config
                    ),
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                )
                app_model.save_weights(scene.model_path, iteration)

    app_model.save_weights(scene.model_path, opt.iterations)
    torch.cuda.empty_cache()

def prepare_output_and_logger(args, opt):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    with open(os.path.join(args.model_path, "cfg_opts"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(opt))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, app_model):
    if tb_writer:
        # total_loss 只记录 Python 显式 loss；Ray-Color 在自定义 backward 注入梯度，此处没有它的独立数值。
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    out = renderFunc(viewpoint, scene.gaussians, *renderArgs, app_model=app_model, return_plane=True, return_depth_normal=True)
                    image = out["render"]
                    if 'app_image' in out:
                        image = out['app_image']
                    depth_normal_show = (out['depth_normal']+1.0)*0.5
                    depth = out['plane_depth'].squeeze().detach().cpu().numpy()
                    depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
                    depth_i = depth_i + (1 - out["rendered_alpha"].squeeze().detach().cpu().numpy()) * 1
                    depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
                    depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)
                    depth_color = (torch.tensor(depth_color) / 255.0).permute(2,0,1)

                    image = torch.clamp(image, 0.0, 1.0)
                    gt_image, _ = viewpoint.get_image()
                    gt_image = torch.clamp(gt_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/depth_normal".format(viewpoint.image_name), depth_normal_show[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth_color[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    torch.set_num_threads(8)
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6007)
    parser.add_argument('--debug_from', type=int, default=-100)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3000, 7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
