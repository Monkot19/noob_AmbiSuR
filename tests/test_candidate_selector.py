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


def production_model_method(name):
    # Execute the actual existing method body, without importing unavailable
    # GaussianModel CUDA dependencies; native class integration is tested below.
    path = Path(__file__).resolve().parents[1] / "scene/gaussian_model.py"
    model = next(n for n in ast.parse(path.read_text(encoding="utf-8")).body
                 if isinstance(n, ast.ClassDef) and n.name == "GaussianModel")
    method = next((n for n in model.body if isinstance(n, ast.FunctionDef)
                   and n.name == name), None)
    assert method is not None, "missing production method: " + name
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[method.name]


def production_depth_sampler():
    return production_model_method("get_points_depth_in_depth_map")


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


class CandidateRelocationTests(unittest.TestCase):
    stat_names = ("xyz_gradient_accum", "xyz_gradient_accum_abs", "denom",
                  "denom_abs", "max_radii2D", "max_weight")

    def fixture(self, count=12):
        ref, sources, model, depth, normal = analytic_fixture()
        model = AnalyticGaussians(ref, count)
        transform = ref.world_view_transform
        camera_xyz = torch.stack((torch.linspace(-.25, .25, count),
            torch.linspace(.15, -.15, count), torch.full((count,), 3.)), -1)
        model._xyz = torch.nn.Parameter((camera_xyz - transform[3, :3]) @ transform[:3, :3].T)
        model._rotation = torch.nn.Parameter(torch.ones(count, 4))
        model.relocate_selected = production_model_method("relocate_selected").__get__(model)
        model.optimizer = torch.optim.Adam([
            dict(params=[model._xyz], name="xyz", lr=0.),
            dict(params=[model._rotation], name="rotation", lr=.001)], eps=1e-15)
        (model._xyz.square().sum() + model._rotation.square().sum()).backward()
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)
        model.optimizer.param_groups[0]["lr"] = .00016
        for name in self.stat_names:
            values = torch.arange(1, count + 1, dtype=model._xyz.dtype)
            setattr(model, name, values if name in ("max_radii2D", "max_weight") else values[:, None])
        return ref, sources, model, depth, normal

    def test_relocation_defaults_schedule_and_invalid_configuration(self):
        parser = ArgumentParser()
        group = OptimizationParams(parser)
        opts = group.extract(parser.parse_args([]))
        self.assertFalse(getattr(opts, "enable_candidate_relocation", True))
        self.assertEqual(opts.candidate_relocation_interval, 100)
        opts = group.extract(parser.parse_args(["--enable_candidate_selector",
            "--enable_candidate_relocation", "--candidate_relocation_interval", "100"]))
        record = build_resolved_config(SimpleNamespace(), opts, SimpleNamespace())
        self.assertTrue(record["optimization"]["enable_candidate_relocation"])
        self.assertEqual(record["optimization"]["candidate_relocation_interval"], 100)
        for iteration in (0, 7000, 7001, 7101, 30000):
            self.assertFalse(selector.candidate_relocation_active(iteration, opts))
        for iteration in (7100, 15000, 29900):
            self.assertTrue(selector.candidate_relocation_active(iteration, opts))
        for changes in ({"candidate_relocation_interval": 0},
                        {"candidate_relocation_interval": True},
                        {"candidate_relocation_interval": 1.5},
                        {"enable_candidate_selector": False}, {"use_mono": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                selector.candidate_relocation_active(0, options(enable_candidate_relocation=True, **changes))

    def test_relocation_cutoff_preserves_full_v2_and_early_schedule(self):
        parser = ArgumentParser()
        group = OptimizationParams(parser)
        default = group.extract(parser.parse_args([]))
        self.assertEqual(getattr(default, "candidate_relocation_until_iter", None), 0)
        early = group.extract(parser.parse_args(["--enable_candidate_selector",
            "--enable_candidate_relocation", "--candidate_relocation_until_iter", "15000"]))
        record = build_resolved_config(SimpleNamespace(), early, SimpleNamespace())
        self.assertEqual(record["optimization"]["candidate_relocation_until_iter"], 15000)
        actual = [i for i in range(30001) if selector.candidate_relocation_active(i, early)]
        self.assertEqual(actual, list(range(7100, 15000, 100)))
        self.assertEqual(len(actual), 79)
        full = options(enable_candidate_relocation=True)
        legacy = options(enable_candidate_relocation=True)
        del legacy.candidate_relocation_until_iter
        for opts in (full, legacy):
            actual = [i for i in range(30001) if selector.candidate_relocation_active(i, opts)]
            self.assertEqual(actual, list(range(7100, 30000, 100)))
            self.assertEqual(len(actual), 229)
        for iteration in (15000, 20000, 29900, 30000):
            self.assertFalse(selector.candidate_relocation_active(iteration, early))
            self.assertTrue(selector.candidate_selector_active(iteration, early))
        for invalid in (-1, True, 1.5, "15000"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                selector.candidate_relocation_active(0, options(enable_candidate_relocation=True,
                    candidate_relocation_until_iter=invalid))
        self.assertFalse(selector.candidate_relocation_active(7100, options(
            enable_candidate_relocation=False, candidate_relocation_until_iter=-1)))

    def test_real_selector_targets_keep_id_order_and_v1_result(self):
        ref, sources, model, depth, normal = self.fixture()
        model.scores[0], model.scores[3] = 0., 4.
        opts = options(candidate_selector_budget=7, enable_candidate_relocation=True)
        calls = []
        def source_depth(camera):
            calls.append(camera.uid)
            return depth.new_full((1, camera.image_height, camera.image_width), 2.)
        kwargs = dict(seed=0, iteration=7100, support_depth_fn=source_depth)
        rng = torch.get_rng_state().clone()
        old = selector.select_candidate_labels(model, ref, ref.gray, depth, normal, sources, opts, **kwargs)
        new = selector.select_candidate_labels_with_targets(model, ref, ref.gray, depth, normal, sources, opts, **kwargs)
        self.assertEqual(len(old), 3)
        self.assertEqual(len(new), 4)
        for i in (0, 1):
            self.assertTrue(torch.equal(old[i], new[i]))
        self.assertEqual(old[2], new[2])
        self.assertEqual(calls, [1, 2, 1, 2])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        ids, targets = new[3]
        self.assertGreater(ids.numel(), 1)
        self.assertFalse(torch.equal(ids, ids.sort().values), "exercise nonsorted sampled IDs")
        self.assertEqual(ids.dtype, torch.long)
        self.assertEqual(set(ids.tolist()), set(new[1].nonzero().flatten().tolist()))
        original_labels = selector.compute_sh_ambiguity_labels(model, opts)
        self.assertTrue(original_labels.any())
        self.assertFalse(original_labels[ids].any())
        self.assertTrue(new[0][original_labels].all())
        transform = ref.world_view_transform
        original = model._xyz.detach()[ids] @ transform[:3, :3] + transform[3, :3]
        expected_camera = original * (2. / original[:, 2:3])
        expected = (expected_camera - transform[3, :3]) @ transform[:3, :3].T
        torch.testing.assert_close(targets, expected)
        self.assertFalse(targets.requires_grad)
        parameter = model._xyz
        state = model.optimizer.state[parameter]
        before_xyz = parameter.detach().clone()
        before_moments = {k: v.clone() for k, v in state.items()}
        before_stats = {k: getattr(model, k).clone() for k in self.stat_names}
        other_state = {k: v.clone() for k, v in model.optimizer.state[model._rotation].items()}
        rotation = model._rotation.detach().clone()
        self.assertEqual(model.relocate_selected(ids, targets), len(ids))
        self.assertIs(model._xyz, parameter)
        self.assertIs(model.optimizer.param_groups[0]["params"][0], parameter)
        keep = torch.ones(len(parameter), dtype=torch.bool); keep[ids] = False
        torch.testing.assert_close(parameter[ids], expected)
        self.assertTrue(torch.equal(parameter[keep], before_xyz[keep]))
        self.assertTrue(torch.equal(state["step"], before_moments["step"]))
        self.assertEqual(model.optimizer.param_groups[0]["lr"], .00016)
        for key in ("exp_avg", "exp_avg_sq"):
            self.assertEqual(int(torch.count_nonzero(state[key][ids])), 0)
            self.assertTrue(torch.equal(state[key][keep], before_moments[key][keep]))
        for key in self.stat_names:
            self.assertEqual(int(torch.count_nonzero(getattr(model, key)[ids])), 0)
            self.assertTrue(torch.equal(getattr(model, key)[keep], before_stats[key][keep]))
        self.assertTrue(torch.equal(model._rotation, rotation))
        for key, value in other_state.items():
            self.assertTrue(torch.equal(model.optimizer.state[model._rotation][key], value))
        parameter.square().sum().backward()
        model.optimizer.step()
        self.assertTrue(torch.isfinite(parameter).all())
        empty = selector.select_candidate_labels_with_targets(model, ref, ref.gray,
            depth, normal, [], opts, **kwargs)
        self.assertIsNone(empty[3])
        self.assertFalse(empty[1].any())

    def test_relocation_invalid_inputs_do_not_partially_write(self):
        for case in ("dtype", "shape", "range", "duplicate", "nan", "binding", "state", "stats"):
            with self.subTest(case=case):
                _, _, model, _, _ = self.fixture(4)
                ids = torch.tensor([2, 0]); targets = model._xyz.detach()[ids] + .2
                if case == "dtype": ids = ids.float()
                if case == "shape": targets = targets[:, :2]
                if case == "range": ids[0] = 9
                if case == "duplicate": ids[0] = 0
                if case == "nan": targets[0, 0] = float("nan")
                if case == "binding": model.optimizer.param_groups[0]["params"] = [model._rotation]
                if case == "state": model.optimizer.state[model._xyz]["exp_avg"] = torch.zeros(1, 3)
                if case == "stats": model.denom = torch.ones(2, 1)
                xyz = model._xyz.detach().clone()
                moments = {k: v.clone() for k, v in model.optimizer.state[model._xyz].items()}
                with self.assertRaises((ValueError, TypeError)):
                    model.relocate_selected(ids, targets)
                self.assertTrue(torch.equal(model._xyz, xyz))
                for key, value in moments.items():
                    self.assertTrue(torch.equal(model.optimizer.state[model._xyz][key], value))

    def test_relocation_empty_and_uninitialized_adam(self):
        _, _, model, _, _ = self.fixture(4)
        before = model._xyz.detach().clone()
        state = model.optimizer.state[model._xyz]["exp_avg"].clone()
        self.assertEqual(model.relocate_selected(torch.empty(0, dtype=torch.long), before[:0]), 0)
        self.assertTrue(torch.equal(model._xyz, before))
        self.assertTrue(torch.equal(model.optimizer.state[model._xyz]["exp_avg"], state))
        del model.optimizer.state[model._xyz]
        model.relocate_selected(torch.tensor([1]), before[1:2] + .1)
        self.assertNotIn(model._xyz, model.optimizer.state)

    def test_real_training_helper_moves_before_main_render_only_on_event(self):
        root = Path(__file__).resolve().parents[1]
        renderer_tree = ast.parse((root / "gaussian_renderer/__init__.py").read_text(encoding="utf-8"))
        normal_node = next(n for n in renderer_tree.body if isinstance(n, ast.FunctionDef) and n.name == "render_normal")
        normal_namespace = dict(normal_from_depth_image=normal_from_depth_image)
        exec(compile(ast.Module(body=[normal_node], type_ignores=[]), "<real render_normal>", "exec"), normal_namespace)
        camera_tree = ast.parse((root / "scene/cameras.py").read_text(encoding="utf-8"))
        camera_class = next(n for n in camera_tree.body if isinstance(n, ast.ClassDef) and n.name == "Camera")
        calibration = next(n for n in camera_class.body if isinstance(n, ast.FunctionDef) and n.name == "get_calib_matrix_nerf")
        calibration_namespace = dict(torch=torch)
        exec(compile(ast.Module(body=[calibration], type_ignores=[]), "<real camera calibration>", "exec"), calibration_namespace)
        tree = ast.parse((root / "train.py").read_text(encoding="utf-8"))
        helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_render_training_reference")
        for iteration, enabled, invalid, until, should_move in (
                (7100, True, False, 0, True), (7101, True, False, 0, False),
                (7100, False, False, 0, False), (7100, True, True, 0, False),
                (14900, True, False, 15000, True), (15000, True, False, 15000, False),
                (20000, True, False, 15000, False), (30000, True, False, 15000, False)):
            with self.subTest(iteration=iteration, enabled=enabled, invalid=invalid, until=until):
                ref, sources, model, _, _ = self.fixture(4)
                ref.get_calib_matrix_nerf = calibration_namespace[calibration.name].__get__(ref)
                ref._candidate_support_cameras = sources
                if invalid: ref.depth_dict["depth"].fill_(0)
                calls = []
                def render_boundary(camera, current, *args, **kwargs):
                    calls.append(camera.uid)
                    if camera.uid != ref.uid:
                        self.assertFalse(torch.is_grad_enabled())
                        return {"plane_depth": current._xyz.new_full((1, 32, 32), 2.)}
                    self.assertTrue(torch.is_grad_enabled())
                    camera_xyz = current._xyz @ ref.world_view_transform[:3, :3] + ref.world_view_transform[3, :3]
                    torch.testing.assert_close(camera_xyz[:, 2], camera_xyz.new_full((4,), 2. if should_move else 3.))
                    return {"plane_depth": camera_xyz[:, 2].mean().expand(1, 32, 32)}
                namespace = dict(torch=torch, F=torch.nn.functional, render=render_boundary,
                    render_normal=normal_namespace["render_normal"],
                    candidate_selector_active=selector.candidate_selector_active,
                    candidate_relocation_active=getattr(selector, "candidate_relocation_active", None),
                    select_candidate_labels=selector.select_candidate_labels,
                    select_candidate_labels_with_targets=getattr(selector, "select_candidate_labels_with_targets", None))
                exec(compile(ast.Module(body=[helper], type_ignores=[]), "<real training helper>", "exec"), namespace)
                opts = options(enable_candidate_relocation=enabled, ray_color_lambda=-1,
                               candidate_relocation_until_iter=until)
                rng = torch.get_rng_state().clone()
                pkg, data = namespace[helper.name](ref, model, None, None, None, opts, iteration=iteration, seed=0)
                self.assertEqual(calls, [0] if invalid else [1, 2, 0])
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertEqual(data["stats"].get("relocated", 0), 4 if should_move else 0)
                if not invalid:
                    self.assertEqual(int(data["extra"].sum()), 4)
                pkg["plane_depth"].sum().backward()
                model.optimizer.step()
                self.assertTrue(torch.isfinite(model._xyz).all())


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

    def test_static_audit_uses_real_native_depth_without_moving_model(self):
        from scripts import evaluate_candidate_actions as audit
        ref, sources, model, pipe, bg = self.fixture()
        original = model.get_xyz.detach().clone()
        depth = ref.depth_dict['depth'].detach()
        with torch.no_grad():
            record, stats = audit.observe_selector(model,ref,ref.get_image()[1],depth,
                self.normal(ref,depth),sources,options(),seed=0,iteration=7100,
                support_depth_fn=lambda c:self.render(c,model,pipe,bg,
                    return_plane=True,return_depth_normal=False,ray_reg=False)['plane_depth'])
        audit.validate_record(record)
        self.assertGreater(stats['added'],0)
        self.assertTrue(torch.equal(model.get_xyz,original))

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

    def test_v2_real_render_relocation_alr_backward_and_step(self):
        ref, sources, model, pipe, bg = self.fixture()
        opt = options(enable_candidate_relocation=True, ray_color_lambda=-1)
        model.spatial_lr_scale = 1.
        model.training_setup(opt)
        model.max_radii2D = model._xyz.new_zeros(len(model._xyz))
        model.max_weight = model.max_radii2D.clone()
        parameter = model._xyz
        before = parameter.detach().clone()
        pkg, data = self.reference_render(ref, model, pipe, bg, None, opt, iteration=7100, seed=0)
        self.assertGreater(data["stats"]["relocated"], 0)
        self.assertIs(model._xyz, parameter)
        self.assertTrue(torch.equal(parameter[~data["extra"]], before[~data["extra"]]))
        camera_xyz = parameter[data["extra"]] @ ref.world_view_transform[:3, :3] + ref.world_view_transform[3, :3]
        torch.testing.assert_close(camera_xyz[:, 2], torch.full_like(camera_xyz[:, 2], 2.))
        loss = ((pkg["render"] - .3) ** 2).mean() + self.alr(ref, pkg,
            data["prior_depth"], 1, -1, 7100, opt, data["prior_normal"])
        loss.backward()
        self.assertTrue(torch.isfinite(parameter.grad).all())
        model.optimizer.step()
        self.assertTrue(torch.isfinite(parameter).all())

    def test_spread_real_render_to_alr_backward_and_step(self):
        ref,sources,model,pipe,bg = self.fixture()
        opt = options(candidate_depth_source='spread',enable_candidate_relocation=True,
                      candidate_relocation_until_iter=15000,ray_color_lambda=-1)
        model.spatial_lr_scale=1.;model.training_setup(opt)
        model.max_radii2D=model._xyz.new_zeros(len(model._xyz));model.max_weight=model.max_radii2D.clone()
        raw=ref.depth_dict['depth'].clone()
        pkg,data=self.reference_render(ref,model,pipe,bg,None,opt,iteration=7100,seed=0)
        torch.testing.assert_close(data['prior_depth'],raw)
        self.assertIn('anchors',data['stats']);self.assertIn('spread_pixels',data['stats'])
        self.assertTrue(torch.isfinite(pkg['plane_depth']).all())
        loss=pkg['render'].mean()+self.alr(ref,pkg,data['prior_depth'],1,-1,7100,opt,data['prior_normal'])
        loss.backward()
        self.assertTrue(torch.isfinite(model._xyz.grad).all())
        model.optimizer.step()
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


class NativeCheckpointTests(unittest.TestCase):
    def test_native_capture_to_training_restore_keeps_adam_and_exact_app_iteration(self):
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root / "scene/gaussian_model.py").read_text(encoding="utf-8"))
        model = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GaussianModel")
        methods = [n for n in model.body if isinstance(n, ast.FunctionDef)
                   and n.name in ("capture", "restore")]
        namespace = {"torch": torch}
        exec(compile(ast.Module(body=methods, type_ignores=[]), "<native Gaussian methods>", "exec"), namespace)
        parameter_names = ("_xyz", "_knn_f", "_features_dc", "_features_rest",
                           "_scaling", "_rotation", "_opacity")
        # Match Scene -> create_from_pcd -> capture: radius is a NumPy scalar,
        # which torch.load's restricted default does not accept.
        reader_tree = ast.parse((root / "scene/dataset_readers.py").read_text(encoding="utf-8"))
        normalize = next(n for n in reader_tree.body
                         if isinstance(n, ast.FunctionDef) and n.name == "getNerfppNorm")
        from utils.graphics_utils import getWorld2View2
        normalization_namespace = dict(np=np, getWorld2View2=getWorld2View2)
        exec(compile(ast.Module(body=[normalize], type_ignores=[]),
                     "<native normalization>", "exec"), normalization_namespace)
        cameras = [SimpleNamespace(R=np.eye(3), T=np.array(t))
                   for t in ([0., 0., 0.], [1., 0., 0.])]
        radius = normalization_namespace["getNerfppNorm"](cameras)["radius"]
        self.assertIsInstance(radius, np.generic)

        def setup(instance, opts):
            instance.optimizer = torch.optim.Adam(
                [getattr(instance, name) for name in parameter_names], lr=opts.position_lr_init)

        cpu_model = type("CpuNativeGaussians", (), {
            "capture": namespace["capture"], "restore": namespace["restore"], "training_setup": setup})

        def make_model(value):
            instance = cpu_model()
            for name in parameter_names:
                setattr(instance, name, torch.nn.Parameter(torch.full((2, 3), value)))
            for name in ("max_radii2D", "max_weight", "xyz_gradient_accum", "xyz_gradient_accum_abs",
                         "denom", "denom_abs"):
                setattr(instance, name, torch.full((2, 1), value))
            instance.active_sh_degree, instance.spatial_lr_scale = 3, radius
            instance.training_setup(SimpleNamespace(position_lr_init=.004))
            return instance

        def step(instance):
            sum(getattr(instance, name).square().sum() for name in parameter_names).backward()
            instance.optimizer.step()
            instance.optimizer.zero_grad(set_to_none=True)

        # Execute the real AppModel producers/consumer, bypassing only its CUDA constructor.
        app_namespace = {}
        exec(compile((root / "scene/app_model.py").read_text(encoding="utf-8"),
                     "<native AppModel>", "exec"), app_namespace)
        app_class = app_namespace["AppModel"]

        def make_app(value):
            app = app_class.__new__(app_class)
            torch.nn.Module.__init__(app)
            app.appear_ab = torch.nn.Parameter(torch.full((2, 2), value))
            return app

        original, restored = make_model(1.), make_model(8.)
        step(original)
        train_tree = ast.parse((root / "train.py").read_text(encoding="utf-8"))
        training = next(n for n in train_tree.body if isinstance(n, ast.FunctionDef) and n.name == "training")
        restore_block = next(n for n in training.body if isinstance(n, ast.If) and ast.unparse(n.test) == "checkpoint")
        with tempfile.TemporaryDirectory() as temporary:
            prefix, destination = Path(temporary) / "prefix", Path(temporary) / "new_candidate"
            prefix.mkdir()
            checkpoint = prefix / "chkpnt7000.pth"
            torch.save((original.capture(), 7000), checkpoint)
            make_app(.2).save_weights(str(prefix), 7000)
            make_app(9.).save_weights(str(prefix), 30000)
            make_app(8.).save_weights(str(destination), 30000)
            app = make_app(0.)
            context = dict(torch=torch, os=__import__("os"), checkpoint=str(checkpoint),
                           gaussians=restored, app_model=app, scene=SimpleNamespace(model_path=str(destination)),
                           opt=SimpleNamespace(position_lr_init=.123, iterations=30000), first_iter=0)
            exec(compile(ast.Module(body=[restore_block], type_ignores=[]), "<real training restore>", "exec"), context)
            self.assertEqual(context["first_iter"], 7000)
            self.assertEqual(context["opt"].iterations, 30000)
            self.assertIs(type(restored.spatial_lr_scale), type(radius))
            self.assertEqual(restored.spatial_lr_scale, radius)
            torch.testing.assert_close(app.appear_ab, torch.full((2, 2), .2))
            self.assertEqual(restored.optimizer.param_groups[0]["lr"], .004)
            torch.testing.assert_close(restored.xyz_gradient_accum, original.xyz_gradient_accum)
            torch.testing.assert_close(restored.denom_abs, original.denom_abs)
            for name in parameter_names:
                first, second = getattr(original, name), getattr(restored, name)
                torch.testing.assert_close(first, second)
                for key in ("step", "exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(original.optimizer.state[first][key], restored.optimizer.state[second][key])
            step(original)
            step(restored)
            for name in parameter_names:
                torch.testing.assert_close(getattr(original, name), getattr(restored, name), rtol=0, atol=0)


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



class StaticActionAuditTests(unittest.TestCase):
    def test_pose_reader_matches_baseline_text_fallback_order_and_binary_priority(self):
        from scripts import evaluate_candidate_actions as audit
        import struct
        spec = importlib.util.spec_from_file_location('actual_colmap_reader',
            Path(__file__).resolve().parents[1]/'scene/colmap_loader.py')
        loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loader)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, location in enumerate(('sparse/0','colmap/sparse/0','sparse')):
                source = root/str(index)
                pose = source/location; pose.mkdir(parents=True)
                (pose/'images.txt').write_text('9 1 0 0 0 .1 .2 .3 4 B.JPG\n\n2 1 0 0 0 .4 .5 .6 4 A.JPG\n\n')
                (pose/'cameras.txt').write_text('4 PINHOLE 32 24 40 41 16 12\n')
                selected, extrinsics, intrinsics, kind = audit.read_pose_files(source,loader)
                self.assertEqual(selected,pose)
                self.assertEqual(kind,'text')
                self.assertEqual(list(extrinsics),[9,2])
                self.assertEqual(extrinsics[9].name,'B.JPG')
                np.testing.assert_allclose(extrinsics[2].tvec,[.4,.5,.6])
                np.testing.assert_array_equal(intrinsics[4].params,[40,41,16,12])
                (pose/'images.bin').write_bytes(b'bad')
                self.assertEqual(audit.read_pose_files(source,loader)[3],'text')
                with (pose/'images.bin').open('wb') as f:
                    f.write(struct.pack('<Q',2))
                    for image_id,name in ((7,'D.JPG'),(3,'C.JPG')):
                        f.write(struct.pack('<idddddddi',image_id,1.,0.,0.,0.,.7,.8,.9,4))
                        f.write(name.encode()+b'\0'+struct.pack('<Q',0))
                (pose/'cameras.bin').write_bytes(struct.pack('<QiiQQdddd',1,4,1,32,24,45.,46.,16.,12.))
                _, extrinsics, intrinsics, kind = audit.read_pose_files(source,loader)
                self.assertEqual(kind,'binary')
                self.assertEqual(list(extrinsics),[7,3])
                np.testing.assert_array_equal(intrinsics[4].params,[45,46,16,12])

    def test_observer_mixed_acceptance_exports_exact_production_targets(self):
        from scripts import evaluate_candidate_actions as audit
        ref, sources, _, depth, normal = analytic_fixture()
        model = AnalyticGaussians(ref, count=16)
        rng = torch.Generator().manual_seed(0)
        rotation, _ = torch.linalg.qr(torch.randn(3,3,generator=rng))
        for camera in [ref,*sources]:
            camera.world_view_transform[:3,:3] = rotation
            camera.camera_center = camera.world_view_transform.inverse()[3,:3]
        tr = ref.world_view_transform
        model.normals = tr.new_tensor([[0.,0.,-1.]]) @ rotation.T
        points = (torch.rand(16,3,generator=rng)-.5)*.2
        points[:,2] = torch.tensor([3.,2.]*8)
        model._xyz = (points-tr[3,:3]) @ rotation.T
        fn = lambda c: depth.new_full((1,c.image_height,c.image_width),2.)
        direct = selector.select_candidate_labels_with_targets(model,ref,ref.gray,depth,
            normal,sources,options(),seed=0,iteration=7100,support_depth_fn=fn)
        self.assertGreater(int(direct[1].sum()),0)
        self.assertLess(int(direct[1].sum()),16)
        record, _ = audit.observe_selector(model,ref,ref.gray,depth,normal,sources,
            options(),seed=0,iteration=7100,support_depth_fn=fn)
        np.testing.assert_array_equal(record['ids'][record['accepted']],direct[3][0].numpy())
        np.testing.assert_array_equal(record['prior_world'][record['accepted']],direct[3][1].numpy())

    def test_observer_calls_real_selector_and_preserves_results_state_and_rng(self):
        from scripts import evaluate_candidate_actions as audit
        ref, sources, model, depth, normal = analytic_fixture()
        opts = options()
        fn = lambda c: depth.new_full((1, c.image_height, c.image_width), 2.)
        before, rng = model.get_xyz.clone(), torch.random.get_rng_state().clone()
        direct = selector.select_candidate_labels_with_targets(model, ref, ref.gray, depth,
            normal, sources, opts, seed=0, iteration=7100, support_depth_fn=fn)
        record, stats = audit.observe_selector(model, ref, ref.gray, depth, normal,
            sources, opts, seed=0, iteration=7100, support_depth_fn=fn)
        np.testing.assert_array_equal(record['ids'][record['accepted']], direct[3][0].numpy())
        np.testing.assert_allclose(record['prior_world'][record['accepted']], direct[3][1].numpy())
        self.assertEqual(stats['added'], int(direct[1].sum()))
        self.assertTrue(torch.equal(before, model.get_xyz))
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertIsNone(__import__('sys').getprofile())

    def test_observer_empty_path_and_restores_profiler_on_failure(self):
        from scripts import evaluate_candidate_actions as audit
        ref, sources, model, depth, normal = analytic_fixture()
        bad = lambda c: (_ for _ in ()).throw(RuntimeError('source failure'))
        with self.assertRaisesRegex(RuntimeError, 'source failure'):
            audit.observe_selector(model, ref, ref.gray, depth, normal, sources,
                options(), seed=0, iteration=7100, support_depth_fn=bad)
        self.assertIsNone(__import__('sys').getprofile())
        depth.fill_(float('nan'))
        record, stats = audit.observe_selector(model, ref, ref.gray, depth, normal,
            sources, options(), seed=0, iteration=7100, support_depth_fn=bad)
        self.assertEqual(len(record['ids']), 0)
        self.assertEqual(stats['sampled'], 4)

    def test_view_extension_and_literal_options(self):
        from scripts import evaluate_candidate_actions as audit
        names = ['v%02d' % i for i in range(20)]
        plan = audit.choose_views(names, ['v02', 'v17'], 5)
        self.assertEqual(plan['coverage'], ['v00','v05','v10','v14','v19'])
        self.assertEqual(plan['illustrative'], ['v02','v17'])
        self.assertEqual(len(set(plan['all'])), len(plan['all']))
        self.assertEqual(audit.literal_namespace('Namespace(seed=0, use_mono=False)'),
                         dict(seed=0, use_mono=False))
        with self.assertRaises(ValueError): audit.literal_namespace("Namespace(seed=__import__('os'))")

    @unittest.skipIf(importlib.util.find_spec('open3d') is None, 'Open3D unavailable')
    def test_real_selector_npz_to_gt_score_and_publication(self):
        from scripts import evaluate_candidate_actions as audit
        import open3d as o3d
        ref, sources, model, depth, normal = analytic_fixture()
        record, stats = audit.observe_selector(model, ref, ref.gray, depth, normal,
            sources, options(), seed=0, iteration=7100,
            support_depth_fn=lambda c: depth.new_full((1, c.image_height, c.image_width), 2.))
        camera_points = np.array([[-100.,-100.,2.], [100.,-100.,2.],
                                  [100.,100.,2.], [-100.,100.,2.]])
        tr = ref.world_view_transform.numpy()
        world = (camera_points-tr[3,:3]) @ tr[:3,:3].T
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            gt = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(world),
                o3d.utility.Vector3iVector([[0,1,2],[0,2,3]]))
            (p/'gt').mkdir()
            gt_path = p/'gt/mesh.ply'
            o3d.io.write_triangle_mesh(str(gt_path), gt)
            capture = p/'capture'; capture.mkdir()
            audit.write_capture_record(capture/'one.npz', record)
            manifest = dict(schema_version=1, records=[dict(file='one.npz',snapshot='clean7k',
                snapshot_iteration=7000,probe_iteration=7100,camera='analytic',
                groups=['illustrative','coverage'],stats=stats)])
            (capture/'capture_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            result = audit.score_capture(capture, gt_path, p/'score')
            row = next(x for x in result['rows'] if x['subset']=='accepted' and x['region']=='ALL')
            self.assertEqual(row['count'], 4)
            self.assertGreater(row['nearest_mean_gain_mm'], 999.)
            self.assertGreater(row['visible_z_mean_gain_mm'], 999.)
            self.assertEqual(row['nearest_improved_fraction'], 1.)
            self.assertLess(row['prior_normal_median_deg'], .1)
            self.assertTrue((p/'score/results.json').is_file())
            self.assertTrue((p/'score/metrics.csv').is_file())
            with self.assertRaises(FileExistsError): audit.score_capture(capture,gt_path,p/'score')


class SpreadIntegrationTests(unittest.TestCase):
    def test_real_training_helper_preview_cache_relocation_and_raw_prior(self):
        from utils.depth_spread import build_spread_proposal
        ref,sources,model,depth,normal = CandidateRelocationTests().fixture(4)
        ref.depth_dict['depth'] = depth * 2
        ref._candidate_support_cameras = sources
        ref.get_calib_matrix_nerf = lambda scale=1: (ref.get_k(scale),ref.world_view_transform.T)
        calls=[]
        def boundary(camera,current,*args,**kwargs):
            grad = torch.is_grad_enabled()
            calls.append((camera.uid,grad))
            if not grad:
                return dict(plane_depth=depth[None].clone(),rendered_normal=normal.clone(),
                            rendered_distance=depth[None].clone())
            z = (current.get_xyz@ref.world_view_transform[:3,:3]+ref.world_view_transform[3,:3])[:,2]
            return {'plane_depth':z.mean().expand(1,32,32)}
        root=Path(__file__).resolve().parents[1]
        n=next(x for x in ast.parse((root/'gaussian_renderer/__init__.py').read_text(encoding='utf-8')).body
               if isinstance(x,ast.FunctionDef) and x.name=='render_normal')
        ns=dict(torch=torch,F=torch.nn.functional,render=boundary,
                normal_from_depth_image=normal_from_depth_image,
                candidate_selector_active=selector.candidate_selector_active,
                candidate_relocation_active=selector.candidate_relocation_active,
                select_candidate_labels=selector.select_candidate_labels,
                select_candidate_labels_with_targets=selector.select_candidate_labels_with_targets,
                build_spread_proposal=build_spread_proposal)
        exec(compile(ast.Module(body=[n],type_ignores=[]),'<production normal>','exec'),ns)
        f=next(x for x in ast.parse((root/'train.py').read_text(encoding='utf-8')).body
               if isinstance(x,ast.FunctionDef) and x.name=='_render_training_reference')
        exec(compile(ast.Module(body=[f],type_ignores=[]),'<production training helper>','exec'),ns)
        opts=options(candidate_depth_source='spread',enable_candidate_relocation=True,
                     candidate_relocation_until_iter=15000,multi_view_sample_num=12)
        pkg,data=ns[f.name](ref,model,None,None,None,opts,iteration=7100,seed=0)
        self.assertEqual(calls,[(0,False),(1,False),(2,False),(0,True)])
        self.assertEqual(data['stats']['relocated'],4)
        torch.testing.assert_close(data['prior_depth'],depth*2)
        torch.testing.assert_close(data['prior_normal'],normal)
        torch.testing.assert_close(pkg['plane_depth'],depth.new_full((1,32,32),2.),atol=3e-5,rtol=2e-5)
        pkg['plane_depth'].sum().backward();model.optimizer.step()
        self.assertTrue(torch.isfinite(model.get_xyz).all())

    def test_shared_producer_to_real_observer_and_capture_config_override(self):
        from utils.depth_spread import build_spread_proposal
        from scripts import evaluate_candidate_actions as audit
        ref,sources,model,depth,normal=analytic_fixture()
        opts=options(candidate_depth_source='spread',multi_view_sample_num=12)
        geometry=dict(plane_depth=depth[None],rendered_normal=normal,rendered_distance=depth[None])
        source=lambda c: depth[None]
        proposal,valid,stats=build_spread_proposal(model,ref,sources,depth*2,normal,
            geometry,opts,seed=0,iteration=7100,support_depth_fn=source)
        kw=dict(seed=0,iteration=7100,support_depth_fn=source,
                proposal_depth=proposal,proposal_valid=valid)
        direct=selector.select_candidate_labels_with_targets(model,ref,ref.gray,depth*2,normal,
            sources,opts,**kw)
        record,counts=audit.observe_selector(model,ref,ref.gray,depth*2,normal,sources,opts,**kw)
        audit.validate_record(record)
        self.assertEqual(counts['added'],4)
        np.testing.assert_array_equal(record['prior_world'][record['accepted']],direct[3][1].numpy())
        # Exercise actual CLI parsing into capture, without invoking native CUDA.
        config=Path(__file__).resolve().parents[1]/'experiments/candidate_actions/toolroom_static_v1.json'
        with tempfile.TemporaryDirectory() as temp:
            source_path=Path(temp)
            argv=['audit','capture','--config',str(config),'--source',temp,
                  '--output',str(source_path/'new'),'--candidate-depth-source','spread']
            for name in ('native7k','c0_30k','v2_30k','early_30k'):
                argv.extend(['--snapshot',name+'='+temp])
            with patch('sys.argv',argv),patch.object(audit,'capture') as capture:
                audit.main()
            self.assertEqual(capture.call_args.args[0]['candidate_depth_source'],'spread')


class DepthSpreadTests(unittest.TestCase):
    def test_tilted_plane_from_single_anchor_and_normal_boundary(self):
        from utils.depth_spread import propagate_depth
        y,x = torch.meshgrid(torch.arange(13),torch.arange(13),indexing='ij')
        rays = torch.stack(((x-6)/40, (y-6)/40, torch.ones_like(x)), -1).float()
        n = torch.tensor([.1,.05,-1.]); n = n/n.norm()
        normals = n[:,None,None].expand(3,13,13).clone()
        truth = -2/(rays*n).sum(-1)
        start = torch.full_like(truth, 90.); start[6,6] = truth[6,6]
        anchors = torch.zeros_like(truth,dtype=torch.bool); anchors[6,6] = True
        usable = torch.ones_like(anchors)
        output, mask = propagate_depth(start,normals,rays,anchors,usable)
        self.assertTrue(mask.all())
        torch.testing.assert_close(output,truth,rtol=2e-5,atol=2e-5)
        empty, empty_mask = propagate_depth(start,normals,rays,anchors & False,usable)
        self.assertFalse(empty_mask.any()); self.assertFalse(empty.any())
        anchors.zero_(); anchors[6,2] = True; start[6,2] = truth[6,2]
        normals[:,:,6:] = torch.tensor([0.,.8,-.6])[:,None,None]
        bounded, reached = propagate_depth(start,normals,rays,anchors,usable)
        self.assertTrue(reached[:,:4].any()); self.assertFalse(reached[:,6:].any())
        normals[:,6,2] = float('nan')
        invalid, valid = propagate_depth(start,normals,rays,anchors,usable)
        self.assertTrue(torch.isfinite(invalid).all()); self.assertFalse(valid.any())
        grazing=torch.zeros_like(normals);grazing[0]=1.
        anchors.zero_();anchors[6,6]=True
        invalid,valid=propagate_depth(start,grazing,rays,anchors,usable)
        self.assertTrue(torch.isfinite(invalid).all());self.assertFalse(valid.any())

    def test_anchor_rgb_full_images_respect_ncc_scale_and_source_dimensions(self):
        from utils.depth_spread import build_spread_proposal
        ref=AnalyticCamera(ncc_scale=2)
        sources=[AnalyticCamera(1,.1,width=40,height=28,ncc_scale=2),
                 AnalyticCamera(2,-.1,ncc_scale=2)]
        for camera in [ref,*sources]:
            shift=camera.world_view_transform[3,0].item()-.2
            full=AnalyticCamera(camera.uid,shift,width=camera.image_width,height=camera.image_height)
            rgb=full.gray.expand(3,-1,-1).clone();gray=camera.gray
            camera.get_image=lambda rgb=rgb,gray=gray:(rgb,gray)
        model=AnalyticGaussians(ref)
        depth=ref.depth_dict['depth']
        normal=normal_from_depth_image(depth,ref.get_k(),ref.world_view_transform.T).permute(2,0,1)
        geometry=dict(plane_depth=depth[None],rendered_normal=normal,rendered_distance=depth[None])
        proposal,valid,stats=build_spread_proposal(model,ref,sources,depth,normal,
            geometry,options(),seed=0,iteration=7100,
            support_depth_fn=lambda c:depth.new_full((1,c.image_height,c.image_width),2.))
        self.assertGreater(int(stats['anchors']),0)
        self.assertTrue(valid[15:17,15:17].all())
        torch.testing.assert_close(proposal[valid],torch.full_like(proposal[valid],2.),atol=3e-5,rtol=2e-5)

    def test_real_rgb_anchor_producer_to_selector_targets_and_rng(self):
        from utils.depth_spread import build_spread_proposal
        ref,sources,model,depth,normal = analytic_fixture()
        opts = options(candidate_depth_source='spread',multi_view_sample_num=12)
        geometry = dict(plane_depth=depth[None],rendered_normal=normal,
                        rendered_distance=depth[None])
        rng = torch.get_rng_state().clone()
        source_calls=[]
        def source_depth(camera):
            source_calls.append(camera.uid)
            return depth[None]
        with patch.object(model,'get_normal',side_effect=AssertionError('anchor is a rendered plane')):
            proposal,valid,stats = build_spread_proposal(model,ref,sources,depth*2,normal,
                geometry,opts,seed=0,iteration=7100,support_depth_fn=source_depth)
        self.assertEqual(source_calls,[1])
        self.assertGreater(int(stats['anchors']),0)
        self.assertLessEqual(int(stats['anchors']),12)
        self.assertTrue(valid[15:17,15:17].all())
        torch.testing.assert_close(proposal[valid],torch.full_like(proposal[valid],2.),atol=3e-5,rtol=2e-5)
        self.assertTrue(torch.equal(torch.get_rng_state(),rng))
        result = selector.select_candidate_labels_with_targets(model,ref,ref.gray,
            depth*2,normal,sources,opts,seed=0,iteration=7100,support_depth_fn=source_depth,
            proposal_depth=proposal,proposal_valid=valid)
        self.assertEqual(int(result[1].sum()),4)
        camera_target = result[3][1]@ref.world_view_transform[:3,:3]+ref.world_view_transform[3,:3]
        torch.testing.assert_close(camera_target[:,2],torch.full((4,),2.),atol=3e-5,rtol=2e-5)
        missing = dict(geometry,plane_depth=torch.zeros_like(depth[None]))
        no_depth,no_mask,no_stats = build_spread_proposal(model,ref,sources,depth*2,normal,
            missing,opts,seed=0,iteration=7100,support_depth_fn=source_depth)
        self.assertFalse(no_mask.any()); self.assertFalse(no_depth.any())
        self.assertEqual(int(no_stats['anchors']),0)


class SpreadContractTests(unittest.TestCase):
    def test_gray_source_matches_frozen_pre_spread_implementation(self):
        frozen=subprocess.check_output(['git','show','43e2ea0:utils/candidate_selector.py'],text=True)
        old=next(x for x in ast.parse(frozen).body if isinstance(x,ast.FunctionDef) and x.name=='_source_cost')
        ns=dict(vars(selector))
        exec(compile(ast.Module(body=[old],type_ignores=[]),'<frozen source matcher>','exec'),ns)
        ref,sources,model,depth,_=analytic_fixture()
        centers=torch.tensor([[2.,2.],[12.,15.],[16.,16.],[30.,30.]])
        patches=centers[:,None]+selector.patch_offsets(3,'cpu')
        points=(torch.cat((centers,torch.ones(4,1)),-1)@ref.get_inv_k().T)*3
        normal=torch.tensor([[0.,0.,-1.]]).expand(4,-1)
        args=(model,ref,sources[0],patches,selector._sample_map(ref.gray,patches).squeeze(-1),
              centers,points,normal,torch.full((4,),3.),depth[None],sources[0].gray,options())
        old_cost,old_valid=ns['_source_cost'](*args)
        new_cost,new_valid=selector._source_cost(*args)
        self.assertTrue(torch.equal(old_cost,new_cost));self.assertTrue(torch.equal(old_valid,new_valid))

    def test_source_option_and_old_sampling_stream(self):
        parser = ArgumentParser()
        group = OptimizationParams(parser)
        default = group.extract(parser.parse_args([]))
        self.assertEqual(getattr(default, "candidate_depth_source", None), "da3")
        spread = group.extract(parser.parse_args([
            "--enable_candidate_selector", "--candidate_depth_source", "spread"]))
        self.assertEqual(build_resolved_config(SimpleNamespace(), spread,
            SimpleNamespace())["optimization"]["candidate_depth_source"], "spread")
        with self.assertRaises(ValueError):
            selector.candidate_selector_active(7100, options(candidate_depth_source="typo"))
        pool = torch.arange(100)
        rng = torch.get_rng_state().clone()
        old = selector._sample_candidate_ids(pool, 12, seed=0, iteration=7100, camera_uid=2)
        same = selector._sample_candidate_ids(pool, 12, seed=0, iteration=7100,
            camera_uid=2, stream="ambisur-selector-v1")
        other = selector._sample_candidate_ids(pool, 12, seed=0, iteration=7100,
            camera_uid=2, stream="selector-v3-spread-anchors")
        self.assertTrue(torch.equal(old, same))
        self.assertFalse(torch.equal(old, other))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_proposal_targets_and_uncovered_rows_keep_a_without_matching(self):
        ref, sources, model, depth, normal = analytic_fixture()
        model.scores[0] = 100.
        raw_prior = depth * 2
        source_depth = lambda c: depth[None]
        kw = dict(seed=0, iteration=7100, support_depth_fn=source_depth,
                  proposal_depth=depth, proposal_valid=torch.ones_like(depth, dtype=torch.bool))
        result = selector.select_candidate_labels_with_targets(model, ref, ref.gray,
            raw_prior, normal, sources, options(candidate_depth_source="spread"), **kw)
        self.assertEqual(int(result[1].sum()), 3)
        target_camera = result[3][1] @ ref.world_view_transform[:3,:3] + ref.world_view_transform[3,:3]
        torch.testing.assert_close(target_camera[:,2], depth.new_full((3,), 2.))
        for mask in (torch.zeros_like(depth, dtype=torch.bool), kw['proposal_valid'].clone()):
            # Gaussian projects to(15.5,15.5); one missing corner forbids bilinear mixing.
            mask[16,16] = False
            with patch.object(selector, '_source_cost', wraps=selector._source_cost) as match:
                missing = selector.select_candidate_labels_with_targets(model, ref, ref.gray,
                    raw_prior, normal, sources, options(candidate_depth_source="spread"),
                    **dict(kw, proposal_valid=mask))
            self.assertEqual(match.call_count, 0)
            self.assertTrue(torch.equal(missing[0], selector.compute_sh_ambiguity_labels(model, options())))
            self.assertFalse(missing[1].any())
            self.assertIsNone(missing[3])
        with self.assertRaises(ValueError):
            selector.select_candidate_labels_with_targets(model, ref, ref.gray, depth,
                normal, sources, options(candidate_depth_source="spread"),
                seed=0, iteration=7100, support_depth_fn=source_depth)

    def test_rgb_cost_is_mean_of_real_channel_ncc_and_geometry(self):
        ref, sources, model, depth, _ = analytic_fixture()
        src = sources[0]
        centers = torch.tensor([[12.,15.], [16.,16.]])
        patches = centers[:,None] + selector.patch_offsets(3, 'cpu')
        rays = torch.cat((centers, torch.ones(2,1)), -1) @ ref.get_inv_k().T
        points = 2 * rays
        normal = torch.tensor([[0.,0.,-1.]]).expand(2,-1)
        distance = torch.full((2,), 2.)
        rgb = torch.cat((ref.gray, ref.gray.square(), 1-ref.gray), 0)
        src_rgb = torch.cat((src.gray, src.gray.square(), 1-src.gray), 0)
        args = (model, ref, src, patches)
        tail = (centers, points, normal, distance, depth[None])
        colors, valid = selector._source_cost(*args, selector._sample_map(rgb, patches),
            *tail, src_rgb, options())
        channels = []
        for c in range(3):
            costs, mask = selector._source_cost(*args,
                selector._sample_map(rgb[c:c+1], patches).squeeze(-1), *tail,
                src_rgb[c:c+1], options())
            self.assertTrue(torch.equal(valid, mask))
            channels.append(costs)
        torch.testing.assert_close(colors, torch.stack(channels, -1).mean(-1))
        geometry_valid, error = selector._roundtrip_geometry(model, ref, src,
            centers, points, depth[None])
        self.assertTrue(geometry_valid.all())
        self.assertLess(float(error.max()), 1e-4)


if __name__ == "__main__":
    unittest.main()
