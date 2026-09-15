import inspect
from types import SimpleNamespace
import unittest

import torch

from reliability.collector import (
    D0EvidenceCollector,
    reproject_depth_normal_maps,
)


def world_normal(_camera, depth):
    normal = torch.zeros((3,) + tuple(depth.shape), device=depth.device)
    normal[2] = 1.0
    return normal


class SyntheticCamera:
    def __init__(self, uid, nearest_id):
        self.uid = uid
        self.nearest_id = nearest_id
        self.image_height = 2
        self.image_width = 2
        self.camera_center = torch.zeros(3)
        self.depth_dict = {
            "depth": torch.ones(2, 2),
            "conf": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        }
        self.world_view_transform = torch.eye(4)

    def get_k(self, scale=1.0):
        return torch.eye(3)


class SyntheticGaussians:
    active_sh_degree = 1

    def __init__(self):
        self.get_xyz = torch.zeros(1, 3)
        self.get_features = torch.zeros(1, 16, 3)
        self.get_scaling = torch.ones(1, 3)

    def get_smallest_axis(self):
        return torch.tensor([[0.0, 0.0, 1.0]])


class D0CollectorTests(unittest.TestCase):
    def test_identity_reprojection_has_zero_error_and_full_validity(self):
        depth = torch.ones(2, 2)
        normal = world_normal(None, depth)

        result = reproject_depth_normal_maps(
            depth,
            depth,
            normal,
            normal,
            torch.eye(3),
            torch.eye(3),
            torch.eye(4),
            torch.eye(4),
        )

        self.assertEqual(result.valid.tolist(), [[True, True], [True, True]])
        torch.testing.assert_close(result.depth_error, torch.zeros(2, 2))
        torch.testing.assert_close(result.normal_error, torch.zeros(2, 2))

    def test_reprojection_rejects_target_foreground_occlusion(self):
        source_depth = torch.ones(2, 2)
        target_depth = torch.full((2, 2), 0.5)
        normal = world_normal(None, source_depth)

        result = reproject_depth_normal_maps(
            source_depth,
            target_depth,
            normal,
            normal,
            torch.eye(3),
            torch.eye(3),
            torch.eye(4),
            torch.eye(4),
        )

        self.assertFalse(result.valid.any())

    def test_two_pass_collector_builds_exact_refresh_input_contract(self):
        cameras = [SyntheticCamera(0, [1]), SyntheticCamera(1, [0])]
        gaussians = SyntheticGaussians()
        grad_modes = []
        evidence_shapes = []

        def render_fn(camera, _gaussians, _pipe, _background, **kwargs):
            grad_modes.append(torch.is_grad_enabled())
            common = {
                "out_observe": torch.ones(1, dtype=torch.int32),
                "plane_depth": torch.ones(1, 2, 2),
                "depth_normal": world_normal(camera, torch.ones(2, 2)),
                "rendered_normal": world_normal(camera, torch.ones(2, 2)),
                "rendered_alpha": torch.ones(1, 2, 2),
            }
            values = kwargs.get("evidence_values")
            validity = kwargs.get("evidence_validity")
            if values is not None:
                evidence_shapes.append(tuple(values.shape))
                common["evidence_numerator"] = (
                    values * validity
                ).sum(dim=(1, 2)).unsqueeze(0)
                common["evidence_denominator"] = validity.sum(
                    dim=(1, 2)
                ).unsqueeze(0)
            return common

        collector = D0EvidenceCollector(
            cameras,
            gaussians,
            render_fn,
            SimpleNamespace(),
            torch.zeros(3),
            SimpleNamespace(),
            normal_from_depth_fn=world_normal,
        )

        inputs = collector()

        self.assertEqual(inputs.pixel_hits.tolist(), [[1], [1]])
        self.assertEqual(inputs.prior_support_views.tolist(), [2])
        self.assertEqual(inputs.geometry_support_views.tolist(), [2])
        self.assertGreater(inputs.pg_weighted_support.item(), 0.0)
        torch.testing.assert_close(inputs.pg_depth_error_sum, torch.zeros(1))
        torch.testing.assert_close(inputs.pg_normal_error_sum, torch.zeros(1))
        self.assertEqual(evidence_shapes, [(6, 2, 2), (6, 2, 2)])
        self.assertEqual(grad_modes, [False, False, False, False])

    def test_collector_boundary_has_no_gt_or_mesh_argument(self):
        names = set(inspect.signature(D0EvidenceCollector).parameters)

        self.assertNotIn("gt", names)
        self.assertNotIn("gt_mesh", names)
        self.assertNotIn("mesh", names)


if __name__ == "__main__":
    unittest.main()
