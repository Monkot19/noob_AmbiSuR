"""One shared selector entry: CPU geometry/decisions and native GPU integration."""
import ast
import importlib.util
import math
import io
import json
import shutil
import subprocess
import tempfile
from argparse import ArgumentParser
from pathlib import Path
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from arguments import OptimizationParams
from utils.run_metadata import build_resolved_config, collect_run_identity, write_run_metadata
from utils.graphics_utils import normal_from_depth_image, patch_warp
from utils.loss_utils import lncc
from utils import candidate_selector as selector


def options(**overrides):
    parser = ArgumentParser()
    group = OptimizationParams(parser)
    result = group.extract(parser.parse_args([]))
    result.enable_candidate_selector = True
    result.multi_view_pixel_noise_th = 100.0
    for key, value in overrides.items():
        setattr(result, key, value)
    return result


class AnalyticCamera:
    """Known intrinsics and poses, independent of the renderer's camera loader."""
    def __init__(self, uid=0, translation=0.0, width=32, height=32,
                 ncc_scale=1, device="cpu"):
        self.uid, self.ncc_scale = uid, ncc_scale
        self.image_width, self.image_height = width, height
        self.Fx = self.Fy = 40.0
        self.Cx = self.Cy = 15.5
        self.znear = .01
        az, ay = math.pi / 6, math.pi / 9
        rz = torch.tensor([[math.cos(az), -math.sin(az), 0],
                           [math.sin(az), math.cos(az), 0], [0, 0, 1.]])
        ry = torch.tensor([[math.cos(ay), 0, math.sin(ay)], [0, 1., 0],
                           [-math.sin(ay), 0, math.cos(ay)]])
        self.world_view_transform = torch.eye(4, device=device)
        self.world_view_transform[:3, :3] = (rz @ ry).to(device)
        self.world_view_transform[3, :3] = torch.tensor(
            [.2 + translation, -.1, .3], device=device)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.nearest_id = [1, 2]
        self.depth_dict = {"depth": torch.full((height, width), 2., device=device)}
        h, w = int(height / ncc_scale), int(width / ncc_scale)
        y, x = torch.meshgrid(torch.arange(h, device=device),
                              torch.arange(w, device=device), indexing="ij")
        # Plane at camera z=2: a +0.1 source translation gives +2 full pixels.
        x = x * ncc_scale - 20 * translation
        y = y * ncc_scale
        self.gray = (.45 + .2 * torch.sin(x * 1.1 + y * .3)
                     + .15 * torch.cos(x * .4 - y * .9))[None]

    def get_k(self, scale=1):
        return self.world_view_transform.new_tensor(
            [[40 / scale, 0, 15.5 / scale], [0, 40 / scale, 15.5 / scale], [0, 0, 1]])

    def get_inv_k(self, scale=1):
        return self.get_k(scale).inverse()

    def get_image(self):
        return self.gray.expand(3, -1, -1), self.gray


def production_depth_sampler():
    # Execute the actual existing method body, without importing unavailable
    # GaussianModel CUDA dependencies; native class integration is tested below.
    path = Path(__file__).resolve().parents[1] / "scene/gaussian_model.py"
    model = next(n for n in ast.parse(path.read_text(encoding="utf-8")).body
                 if isinstance(n, ast.ClassDef) and n.name == "GaussianModel")
    method = next(n for n in model.body if isinstance(n, ast.FunctionDef)
                  and n.name == "get_points_depth_in_depth_map")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[method.name]


class AnalyticGaussians:
    get_points_depth_in_depth_map = production_depth_sampler()

    def __init__(self, reference, count=4, z=3.0):
        transform = reference.world_view_transform
        camera_points = transform.new_tensor([[0., 0., z]]).repeat(count, 1)
        self._xyz = (camera_points - transform[3, :3]) @ transform[:3, :3].T
        self.scores = transform.new_full((count,), 2.)
        self.normals = transform.new_tensor([[0., 0., -1.]]) @ transform[:3, :3].T

    @property
    def get_xyz(self):
        return self._xyz

    def compute_weighted_sh_norm(self, weighting):
        assert weighting == "equal"
        return self.scores

    def get_normal(self, reference):
        return self.normals.expand(self._xyz.shape[0], -1).clone()


def analytic_fixture(**source_kwargs):
    ref = AnalyticCamera()
    sources = [AnalyticCamera(1, .1, **source_kwargs),
               AnalyticCamera(2, -.1, **source_kwargs)]
    model = AnalyticGaussians(ref)
    depth = ref.depth_dict["depth"]
    normal = normal_from_depth_image(depth, ref.get_k(),
                                     ref.world_view_transform.T).permute(2, 0, 1)
    return ref, sources, model, depth, normal


class CandidateMathTests(unittest.TestCase):
    def test_cli_defaults_and_real_metadata_without_old_core(self):
        parser = ArgumentParser()
        group = OptimizationParams(parser)
        default = group.extract(parser.parse_args([]))
        self.assertFalse(default.enable_candidate_selector)
        self.assertEqual(default.candidate_selector_budget, 4096)
        self.assertEqual(default.seed, 0)
        enabled = group.extract(parser.parse_args(["--enable_candidate_selector", "--seed", "5"]))
        self.assertEqual(enabled.seed, 5)
        record = build_resolved_config(SimpleNamespace(), enabled, SimpleNamespace())
        self.assertTrue(record["optimization"]["enable_candidate_selector"])
        self.assertNotIn("core", record)
        self.assertEqual(record["training_path"], "candidate_selector")
        self.assertFalse(hasattr(default, "core_shadow_mode"))

    def test_activation_and_unsupported_inputs(self):
        for opts, iteration, expected in [(options(), 7000, False), (options(), 7001, True),
                (options(unc_from_iter=0), 1000, False), (options(unc_from_iter=0), 1001, True),
                (options(enable_candidate_selector=False), 7001, False)]:
            with self.subTest(iteration=iteration, expected=expected):
                self.assertEqual(selector.candidate_selector_active(iteration, opts), expected)
        for changes in ({"candidate_selector_budget": 0}, {"candidate_selector_budget": 1.5},
                        {"use_mono": True}, {"enable_observation_calibration": True},
                        {"core_shadow_mode": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                selector.candidate_selector_active(7001, options(**changes))

    def test_original_sh_strict_quantiles_and_low_cap(self):
        ref = AnalyticCamera()
        model = AnalyticGaussians(ref, 5)
        model.scores = torch.tensor([0., 1., 2., 3., 4.])
        labels = selector.compute_sh_ambiguity_labels(model, options(
            sh_ambi_upper_ratio=.75, sh_ambi_lower_ratio=.5, sh_unc_lower_max=1.))
        self.assertEqual(labels.tolist(), [True, False, False, False, True])

    def test_best_source_raw_cost_common_support_and_strict_thresholds(self):
        cases = [([.95, .10], [.20, .80], [1, 1], False),
                 ([.95, .90], [.20, .80], [1, 1], True),
                 ([.95, .10], [.20, .80], [1, 0], True),
                 ([.95, .90], [.20, .80], [0, 0], False),
                 ([.95, .90], [.50, .80], [1, 1], False),
                 ([.20, .90], [.20, .80], [1, 1], False)]
        for g, p, valid, expected in cases:
            with self.subTest(g=g, p=p, valid=valid):
                decision = selector._best_source_decision(torch.tensor([g]), torch.tensor([p]),
                                                          torch.tensor([valid], dtype=torch.bool))
                self.assertEqual(decision.item(), expected)

    def test_sampling_is_bounded_repeatable_and_does_not_consume_global_rng(self):
        pool = torch.arange(5000)
        before = torch.random.get_rng_state().clone()
        py_before, np_before = random.getstate(), np.random.get_state()
        ids = selector._sample_candidate_ids(pool, 4096, seed=0, iteration=7001, camera_uid=17)
        again = selector._sample_candidate_ids(pool, 4096, seed=0, iteration=7001, camera_uid=17)
        self.assertEqual(ids.numel(), 4096)
        self.assertEqual(ids.unique().numel(), 4096)
        self.assertTrue(torch.equal(ids, again))
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual(py_before, random.getstate())
        np.testing.assert_equal(np_before, np.random.get_state())
        self.assertEqual(selector._sample_candidate_ids(pool[:3], 4096,
            seed=0, iteration=7001, camera_uid=17).tolist(), [0, 1, 2])

    def test_known_plane_homography_nonidentity_pose_and_sign_pair(self):
        ref, sources, _, _, _ = analytic_fixture()
        n = torch.tensor([[0., 0., -1.]])
        d = torch.tensor([2.])
        uv = torch.tensor([[[15.5, 15.5], [10., 20.]]])
        h = selector._plane_homography(ref, sources[0], n, d)
        projected = patch_warp(h, uv)
        torch.testing.assert_close(projected, torch.tensor([[[17.5, 15.5], [12., 20.]]]))
        torch.testing.assert_close(h, selector._plane_homography(ref, sources[0], -n, -d))

    def test_real_ncc_returns_bad_g_raw_cost_and_constant_patch_rejection(self):
        pattern = torch.arange(49).float()[None] / 49
        good, valid_good = lncc(pattern, pattern)
        bad, valid_bad = lncc(pattern, torch.ones_like(pattern))
        self.assertEqual(good.shape, (1, 1))
        self.assertLess(good.item(), .001)
        self.assertTrue(valid_good.item())
        self.assertGreater(bad.item(), .9)
        self.assertFalse(valid_bad.item())
        self.assertTrue(selector._best_source_decision(bad, good, torch.ones_like(bad, dtype=torch.bool)).item())

    def run_selector(self, ref, sources, model, depth, normal, **changes):
        calls = []
        def source_depth(camera):
            self.assertFalse(torch.is_grad_enabled())
            calls.append(camera.uid)
            return depth.new_full((1, camera.image_height, camera.image_width), 2.)
        result = selector.select_candidate_labels(model, ref, ref.gray, depth, normal,
            sources, options(**changes), seed=0, iteration=7001, support_depth_fn=source_depth)
        return result, calls

    def test_full_selector_real_warp_ncc_and_depth_sampler_adds_labels(self):
        ref, sources, model, depth, normal = analytic_fixture()
        (labels, extra, stats), calls = self.run_selector(ref, sources, model, depth, normal)
        self.assertEqual(calls, [1, 2])
        self.assertTrue(extra.all())
        self.assertTrue(torch.equal(labels, extra))
        self.assertEqual(int(stats["sampled"]), 4)
        self.assertEqual(int(stats["comparable"]), 4)
        self.assertEqual(int(stats["added"]), 4)
        self.assertEqual(labels.dtype, torch.bool)
        self.assertFalse(labels.requires_grad)

    def test_each_source_uses_its_own_dimensions_and_ncc_scale(self):
        ref, sources, model, depth, normal = analytic_fixture(width=40, height=36, ncc_scale=2)
        (labels, extra, stats), calls = self.run_selector(ref, sources, model, depth, normal)
        self.assertEqual(calls, [1, 2])
        self.assertEqual(int(stats["comparable"]), 4)
        self.assertTrue(extra.all())

    def test_no_sources_empty_pool_and_invalid_prior_fall_back(self):
        for case in ("one_source", "empty_pool", "bad_prior", "bad_normal", "border"):
            ref, sources, model, depth, normal = analytic_fixture()
            if case == "one_source": sources = sources[:1]
            if case == "empty_pool": model._xyz[:, 2] = -10
            if case == "bad_prior": depth[:] = float("nan")
            if case == "bad_normal": normal[:] = 0
            if case == "border": model._xyz[:] = (torch.tensor([[-.775, 0., 2.]]) - ref.world_view_transform[3, :3]) @ ref.world_view_transform[:3, :3].T
            with self.subTest(case=case):
                (labels, extra, _), calls = self.run_selector(ref, sources, model, depth, normal)
                self.assertFalse(extra.any())
                self.assertTrue(torch.equal(labels, selector.compute_sh_ambiguity_labels(model, options())))
                self.assertEqual(calls, [])

    def test_bad_sampled_candidates_are_not_refilled_and_a_hits_survive(self):
        ref, sources, model, depth, normal = analytic_fixture()
        model.scores = torch.tensor([0., 2., 2., 20.])
        depth[:] = -1
        (labels, extra, stats), calls = self.run_selector(ref, sources, model, depth, normal,
                                                       candidate_selector_budget=1)
        self.assertEqual(int(stats["sampled"]), 1)
        self.assertEqual(labels.tolist(), [True, False, False, True])
        self.assertFalse(extra.any())
        self.assertEqual(calls, [])

    def test_invalid_depth_in_normal_stencil_is_rejected_before_source_render(self):
        ref, sources, model, depth, normal = analytic_fixture()
        # Bilinear normal at (15.5,15.5) depends on this left-hand depth even
        # when the sampled center depth and resulting normal are finite.
        depth[15, 14] = -1
        normal = normal_from_depth_image(depth, ref.get_k(),
            ref.world_view_transform.T).permute(2, 0, 1)
        (labels, extra, stats), calls = self.run_selector(ref, sources, model, depth, normal)
        self.assertEqual(calls, [])
        self.assertFalse(extra.any())

    def test_common_geometry_rejects_bad_source_patches_and_respects_occ_flag(self):
        for case, expected_count in (("one_source", 4), ("outside", 0),
                                     ("behind", 0), ("roundtrip", 0), ("wo_occ", 4)):
            ref, sources, model, depth, normal = analytic_fixture()
            changes = {}
            if case == "one_source": sources[1] = AnalyticCamera(2, 10.)
            if case == "outside": sources = [AnalyticCamera(1, 10.), AnalyticCamera(2, 20.)]
            if case == "behind":
                for source in sources: source.world_view_transform[3, 2] = -5
            if case in ("roundtrip", "wo_occ"):
                changes["multi_view_pixel_noise_th"] = .1
                changes["wo_use_geo_occ_aware"] = case == "wo_occ"
            with self.subTest(case=case):
                (labels, extra, stats), calls = self.run_selector(ref, sources, model, depth, normal, **changes)
                self.assertEqual(int(stats["comparable"]), expected_count)
                self.assertEqual(int(extra.sum()), expected_count)

    def test_model_interface_and_callback_errors_are_exposed(self):
        ref, sources, model, depth, normal = analytic_fixture()
        with self.assertRaises(FloatingPointError):
            selector.select_candidate_labels(model, ref, ref.gray, depth, normal, sources,
                options(), seed=0, iteration=7001,
                support_depth_fn=lambda c: depth.new_full((1, c.image_height, c.image_width), float("nan")))
        with self.assertRaisesRegex(RuntimeError, "source failure"):
            selector.select_candidate_labels(model, ref, ref.gray, depth, normal, sources,
                options(), seed=0, iteration=7001,
                support_depth_fn=lambda _: (_ for _ in ()).throw(RuntimeError("source failure")))
        with self.assertRaises(ValueError):
            self.run_selector(ref, sources, model, depth[:-1], normal)
        model._xyz[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            self.run_selector(ref, sources, model, depth, normal)


def native_unavailable():
    if not torch.cuda.is_available():
        return "CUDA unavailable"
    for module in ("diff_plane_rasterization_ambisur", "simple_knn", "pytorch3d",
                   "cv2", "scipy", "open3d", "plyfile"):
        if importlib.util.find_spec(module) is None:
            return "native integration dependency missing: " + module
    return ""


@unittest.skipIf(native_unavailable(), native_unavailable())
class CandidateIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from gaussian_renderer import render, render_normal
        from scene.cameras import Camera
        from scene.gaussian_model import GaussianModel
        from train import _render_training_reference, _compute_alr_loss
        cls.render = staticmethod(render)
        cls.normal = staticmethod(render_normal)
        cls.reference_render = staticmethod(_render_training_reference)
        cls.alr = staticmethod(_compute_alr_loss)
        cls.Camera, cls.GaussianModel = Camera, GaussianModel

    def test_installed_rasterizer_matches_clean_baseline(self):
        import diff_plane_rasterization_ambisur as rasterizer
        expected = Path(__file__).resolve().parents[1] / "submodules/diff-plane-rasterization-ambisur/diff_plane_rasterization_ambisur/__init__.py"
        actual = Path(rasterizer.__file__)
        self.assertEqual(ast.dump(ast.parse(actual.read_text(encoding="utf-8"))),
                         ast.dump(ast.parse(expected.read_text(encoding="utf-8"))),
                         "installed rasterizer Python wrapper is from a different checkout")
        self.assertFalse(hasattr(rasterizer._C, "rasterize_gaussians_with_evidence"),
                         "old first-run evidence extension is still installed")

    def fixture(self, invalid_prior=False):
        from pytorch3d.transforms import matrix_to_quaternion
        device = "cuda"
        cameras = []
        for uid, delta in enumerate((0., .1, -.1)):
            a = AnalyticCamera(uid, delta, device=device)
            camera = self.Camera(uid, a.world_view_transform[:3, :3].cpu().numpy(),
                a.world_view_transform[3, :3].cpu().numpy(), 2 * math.atan(32 / 80),
                2 * math.atan(32 / 80), 32, 32, "unused", str(uid), uid,
                preload_img=False, depth_dict={"depth": np.full((32, 32), 2., np.float32)})
            camera.preload_img = True
            camera.original_image = a.gray.expand(3, -1, -1).contiguous()
            camera.original_image_gray = a.gray
            camera.nearest_id = [1, 2]
            cameras.append(camera)
        ref = cameras[0]
        model = self.GaussianModel(3)
        cam_xyz = torch.tensor([[-.25, 0., 3.], [0., 0., 3.], [.25, 0., 3.]], device=device)
        model._xyz = torch.nn.Parameter((cam_xyz - ref.world_view_transform[3, :3]) @ ref.world_view_transform[:3, :3].T)
        model._features_dc = torch.nn.Parameter(torch.full((3, 1, 3), .2, device=device))
        model._features_rest = torch.nn.Parameter(torch.ones((3, 15, 3), device=device)
            * torch.tensor([0., .2, 2.], device=device)[:, None, None])
        model._scaling = torch.nn.Parameter(torch.tensor([[.35, .35, .02]], device=device).log().repeat(3, 1))
        tilt = ref.world_view_transform.new_tensor([
            [math.cos(.15), 0, math.sin(.15)], [0, 1., 0], [-math.sin(.15), 0, math.cos(.15)]])
        model._rotation = torch.nn.Parameter(matrix_to_quaternion(
            ref.world_view_transform[:3, :3] @ tilt).repeat(3, 1))
        model._opacity = torch.nn.Parameter(torch.full((3, 1), 2., device=device))
        model.disable_trunc, model.trunc_sigma = False, 2.
        if invalid_prior:
            with torch.no_grad():
                ref.depth_dict["depth"].copy_(-1 + .01 * torch.arange(32, device=device)[None].expand(32, 32))
        pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
        # The real nearest order is passed through the train helper, without a scene loader.
        ref._candidate_support_cameras = cameras[1:]
        return ref, cameras[1:], model, pipe, torch.zeros(3, device=device)

    def test_real_normal_producer_is_already_in_reference_coordinates(self):
        ref, sources, model, pipe, bg = self.fixture()
        depth = ref.depth_dict["depth"].detach()
        actual = self.normal(ref, depth)
        torch.testing.assert_close(actual[:, 10:22, 10:22],
            actual.new_tensor([0., 0., -1.])[:, None, None].expand(3, 12, 12), atol=1e-5, rtol=1e-5)
        self.assertFalse(torch.allclose(actual[:, 16, 16] @ ref.world_view_transform[:3, :3], actual[:, 16, 16]))

    def test_real_selector_to_renderer_to_alr_backward_and_step(self):
        ref, sources, model, pipe, bg = self.fixture()
        opt = options()
        depth = ref.depth_dict["depth"].detach()
        normals = self.normal(ref, depth)
        with patch("utils.candidate_selector.lncc", wraps=lncc):
            labels, extra, stats = selector.select_candidate_labels(model, ref,
                ref.original_image_gray, depth, normals, sources, opt, seed=0,
                iteration=7001, support_depth_fn=lambda c: self.render(c, model,
                    pipe, bg, return_depth_normal=False, ray_reg=False)["plane_depth"])
        a = selector.compute_sh_ambiguity_labels(model, opt)
        outputs = [self.render(ref, model, pipe, bg, opt=opt, ambiguity_labels=b)
                   for b in (labels, a, extra)]
        self.assertTrue(extra.any(), "fixture must exercise additional labels")
        torch.testing.assert_close(outputs[0]["rendered_unc"],
            outputs[1]["rendered_unc"] + outputs[2]["rendered_unc"])
        loss = self.alr(ref, outputs[0], depth, 1, -1, 7001, opt, normals)
        optimizer = torch.optim.Adam([model._xyz, model._rotation], lr=1e-3)
        loss.backward()
        self.assertGreater(float(loss.detach()), 0)
        for grad in (model._xyz.grad, model._rotation.grad):
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(float(grad.abs().sum()), 0)
        optimizer.step()
        self.assertTrue(torch.isfinite(model.get_xyz).all())

    def test_real_training_helper_off_ezero_and_active_path(self):
        ref, sources, model, pipe, bg = self.fixture(invalid_prior=True)
        from train import prepare_output_and_logger
        with tempfile.TemporaryDirectory() as output:
            dataset = SimpleNamespace(model_path=output, source_path="synthetic", sh_degree=3)
            writer = prepare_output_and_logger(dataset, options(), pipe)
            if writer is not None:
                writer.close()
            config = json.loads((Path(output) / "resolved_config.json").read_text())
            identity = json.loads((Path(output) / "run_identity.json").read_text())
            self.assertEqual(config["training_path"], "candidate_selector")
            self.assertNotIn("core", config)
            self.assertEqual(identity["seed"], 0)
            self.assertEqual(len(identity["git_commit"]), 40)
        opts = options(ray_color_lambda=-1)
        before = torch.cuda.get_rng_state().clone()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        pkg, data = self.reference_render(ref, model, pipe, bg, None, opts, iteration=7001, seed=0)
        end.record()
        torch.cuda.synchronize()
        self.assertFalse(data["extra"].any())
        self.assertTrue(torch.equal(before, torch.cuda.get_rng_state()))
        self.assertTrue(torch.isfinite(pkg["plane_depth"]).all())
        self.assertGreaterEqual(start.elapsed_time(end), 0)
        print("selector-active-ms", start.elapsed_time(end), "peak-bytes", torch.cuda.max_memory_allocated())
        off = options(enable_candidate_selector=False, ray_color_lambda=-1)
        direct = self.render(ref, model, pipe, bg, return_plane=True,
            return_depth_normal=True, ray_reg=off.ray_color_lambda, opt=off)
        self.assertTrue(selector.compute_sh_ambiguity_labels(model, off).any())
        for opts in (off, options(ray_color_lambda=-1)):
            actual, metadata = self.reference_render(ref, model, pipe, bg, None,
                opts, iteration=7001, seed=0)
            torch.testing.assert_close(actual["rendered_unc"], direct["rendered_unc"])
            for result in (actual, direct):
                model._xyz.grad = None
                model._rotation.grad = None
                loss = self.alr(ref, result, ref.depth_dict["depth"].detach(), 1, -1, 7001, opts)
                self.assertGreater(float(loss.detach()), 0)
                loss.backward(retain_graph=True)
                grad = (model._xyz.grad.clone(), model._rotation.grad.clone())
                for parameter_grad in grad:
                    self.assertTrue(torch.isfinite(parameter_grad).all())
                    self.assertGreater(float(parameter_grad.abs().sum()), 0)
                if result is actual: actual_grad = grad
                else:
                    for first, second in zip(actual_grad, grad):
                        torch.testing.assert_close(first, second)
            if opts is off: self.assertIsNone(metadata)

    def test_renderer_rejects_bad_labels_and_training_callback_preserves_grad_path(self):
        ref, sources, model, pipe, bg = self.fixture()
        for labels in (torch.zeros(3, device="cuda"), torch.zeros(2, device="cuda", dtype=torch.bool),
                       torch.zeros(3, dtype=torch.bool)):
            with self.subTest(shape=labels.shape, dtype=labels.dtype), self.assertRaises(ValueError):
                self.render(ref, model, pipe, bg, opt=options(), ambiguity_labels=labels)
        # The helper performs no-grad forwards only; a separate ordinary source
        # render still has its original differentiable depth path.
        from gaussian_renderer import render as actual_render
        calls = []
        def traced(camera, *args, **kwargs):
            calls.append((camera.uid, torch.is_grad_enabled(), kwargs))
            return actual_render(camera, *args, **kwargs)
        with patch("train.render", side_effect=traced):
            pkg, data = self.reference_render(ref, model, pipe, bg, None,
                options(), iteration=7001, seed=0)
        self.assertEqual([uid for uid, grad, kw in calls], [1, 2, 0])
        for uid, grad, kw in calls[:2]:
            self.assertFalse(grad)
            self.assertFalse(kw["ray_reg"])
            self.assertFalse(kw["return_depth_normal"])
        self.assertTrue(calls[-1][1])
        differentiable_source = self.render(sources[0], model, pipe, bg, return_depth_normal=False)
        self.assertTrue(differentiable_source["plane_depth"].requires_grad)
        ref.depth_dict = None
        for opt, iteration in ((options(enable_candidate_selector=False), 7001), (options(), 7000)):
            calls.clear()
            with patch("train.render", side_effect=traced):
                _, metadata = self.reference_render(ref, model, pipe, bg, None,
                    opt, iteration=iteration, seed=0)
            self.assertIsNone(metadata)
            self.assertEqual([uid for uid, grad, kw in calls], [0])


class RunMetadataTests(unittest.TestCase):
    def test_resolved_config_keeps_baseline_arguments_without_core(self):
        dataset = SimpleNamespace(source_path=Path("data/scene"), sh_degree=3)
        optimization = SimpleNamespace(iterations=500, seed=0)
        pipeline = SimpleNamespace(debug=False)

        resolved = build_resolved_config(dataset, optimization, pipeline)

        self.assertEqual(resolved["schema_version"], 1)
        self.assertEqual(
            resolved["model"],
            {"sh_degree": 3, "source_path": str(Path("data/scene"))},
        )
        self.assertEqual(
            resolved["optimization"], {"iterations": 500, "seed": 0}
        )
        self.assertEqual(resolved["pipeline"], {"debug": False})
        self.assertNotIn("core", resolved)
        self.assertEqual(resolved["training_path"], "baseline")

    def test_metadata_writer_creates_only_named_json_files(self):
        resolved = {"schema_version": 1, "training_path": "baseline"}
        identity = {"git_commit": "abc", "git_dirty": False}

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            write_run_metadata(output, resolved, identity)

            self.assertEqual(
                sorted(path.name for path in output.iterdir()),
                ["resolved_config.json", "run_identity.json"],
            )
            self.assertEqual(
                json.loads((output / "resolved_config.json").read_text()), resolved
            )
            self.assertEqual(
                json.loads((output / "run_identity.json").read_text()), identity
            )

    @unittest.skipUnless(shutil.which("git"), "git executable is required")
    def test_run_identity_reports_commit_and_dirty_state(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = Path(temporary_directory)
            subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
            tracked = repository / "tracked.txt"
            tracked.write_text("baseline\n", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.txt"], cwd=repository, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=E0 Test",
                    "-c",
                    "user.email=e0@example.invalid",
                    "commit",
                    "-qm",
                    "baseline",
                ],
                cwd=repository,
                check=True,
            )

            clean_identity = collect_run_identity(repository)
            tracked.write_text("changed\n", encoding="utf-8")
            dirty_identity = collect_run_identity(repository)

        self.assertEqual(len(clean_identity["git_commit"]), 40)
        self.assertFalse(clean_identity["git_dirty"])
        self.assertTrue(dirty_identity["git_dirty"])


class SeedTests(unittest.TestCase):
    def test_safe_state_default_zero_and_explicit_seed_replay(self):
        from utils.general_utils import safe_state
        python_before, numpy_before = random.getstate(), np.random.get_state()
        torch_before = torch.random.get_rng_state()
        try:
            for seed in (0, 7):
                with self.subTest(seed=seed), patch("torch.cuda.set_device"), patch("sys.stdout", io.StringIO()):
                    if seed == 0:
                        safe_state(True)
                    else:
                        safe_state(True, seed=seed)
                    self.assertEqual(random.random(), random.Random(seed).random())
                    self.assertEqual(float(np.random.rand()), float(np.random.RandomState(seed).rand()))
                    torch.testing.assert_close(torch.rand(4), torch.rand(4, generator=torch.Generator().manual_seed(seed)))
        finally:
            random.setstate(python_before)
            np.random.set_state(numpy_before)
            torch.random.set_rng_state(torch_before)



if __name__ == "__main__":
    unittest.main()
