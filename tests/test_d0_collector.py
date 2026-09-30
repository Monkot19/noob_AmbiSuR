import inspect
from types import SimpleNamespace
import unittest

import torch

from reliability.collector import (
    D0EvidenceCollector,
    camera_to_world_normal,
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
    def collect_geometry(
        self,
        *,
        geometry_depths,
        geometry_normals,
        primitive_normals,
        alphas=None,
        nearest_ids=None,
    ):
        if nearest_ids is None:
            nearest_ids = ([1], [0])
        cameras = [
            SyntheticCamera(0, list(nearest_ids[0])),
            SyntheticCamera(1, list(nearest_ids[1])),
        ]
        gaussians = SyntheticGaussians()
        if alphas is None:
            alphas = (torch.ones(1, 2, 2), torch.ones(1, 2, 2))

        def render_fn(camera, _gaussians, _pipe, _background, **kwargs):
            uid = camera.uid
            common = {
                "out_observe": torch.ones(1, dtype=torch.int32),
                "plane_depth": geometry_depths[uid],
                "depth_normal": geometry_normals[uid],
                "rendered_normal": primitive_normals[uid],
                "rendered_alpha": alphas[uid],
            }
            values = kwargs.get("evidence_values")
            validity = kwargs.get("evidence_validity")
            if values is not None:
                common["evidence_numerator"] = (
                    torch.where(validity, values, torch.zeros_like(values))
                ).sum(dim=(1, 2)).unsqueeze(0)
                common["evidence_denominator"] = validity.sum(
                    dim=(1, 2)
                ).unsqueeze(0)
            return common

        return D0EvidenceCollector(
            cameras,
            gaussians,
            render_fn,
            SimpleNamespace(),
            torch.zeros(3),
            SimpleNamespace(),
            normal_from_depth_fn=world_normal,
        )()

    @staticmethod
    def constant_normal(x, z):
        normal = torch.zeros(3, 2, 2)
        normal[0] = x
        normal[2] = z
        return normal

    def test_collector_geometry_scores_decrease_with_error_and_preserve_valid_support_semantics(self):
        unit_z = self.constant_normal(0.0, 1.0)
        tilted = self.constant_normal(0.2, (1.0 - 0.2 ** 2) ** 0.5)
        unit_depths = (torch.ones(1, 2, 2), torch.ones(1, 2, 2))

        exact = self.collect_geometry(
            geometry_depths=unit_depths,
            geometry_normals=(unit_z, unit_z),
            primitive_normals=(unit_z, unit_z),
        )
        multiview_mismatch = self.collect_geometry(
            geometry_depths=(
                torch.ones(1, 2, 2),
                torch.full((1, 2, 2), 1.04),
            ),
            geometry_normals=(unit_z, tilted),
            primitive_normals=(unit_z, tilted),
        )
        depth_normal_mismatch = self.collect_geometry(
            geometry_depths=unit_depths,
            geometry_normals=(unit_z, unit_z),
            primitive_normals=(tilted, tilted),
        )
        one_supporting_source = self.collect_geometry(
            geometry_depths=unit_depths,
            geometry_normals=(unit_z, unit_z),
            primitive_normals=(unit_z, unit_z),
            nearest_ids=([1], []),
        )

        self.assertLessEqual(
            multiview_mismatch.geometry_multiview.item(),
            exact.geometry_multiview.item(),
        )
        self.assertLess(
            multiview_mismatch.geometry_multiview.item(),
            exact.geometry_multiview.item(),
        )
        self.assertLessEqual(
            depth_normal_mismatch.geometry_depth_normal.item(),
            exact.geometry_depth_normal.item(),
        )
        self.assertLess(
            depth_normal_mismatch.geometry_depth_normal.item(),
            exact.geometry_depth_normal.item(),
        )
        self.assertEqual(exact.geometry_support_views.tolist(), [2])
        self.assertEqual(
            multiview_mismatch.geometry_support_views.tolist(), [2]
        )
        self.assertEqual(
            one_supporting_source.geometry_support_views.tolist(), [1]
        )

    def test_collector_depth_normal_denominator_excludes_invalid_pixels(self):
        unit_z = self.constant_normal(0.0, 1.0)
        nonfinite = torch.full((3, 2, 2), float("nan"))
        unit_depths = (torch.ones(1, 2, 2), torch.ones(1, 2, 2))
        invalid = self.collect_geometry(
            geometry_depths=unit_depths,
            geometry_normals=(unit_z, unit_z),
            primitive_normals=(nonfinite, unit_z),
            alphas=(torch.ones(1, 2, 2), torch.full((1, 2, 2), 0.49)),
        )

        torch.testing.assert_close(
            invalid.geometry_depth_normal, torch.zeros(1)
        )
        self.assertEqual(invalid.geometry_support_views.tolist(), [2])

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

    def test_camera_frame_normals_are_rotated_to_world_frame(self):
        camera = SyntheticCamera(0, [])
        camera.world_view_transform[:3, :3] = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        local = torch.zeros(3, 2, 2)
        local[0] = 1.0

        world = camera_to_world_normal(camera, local)

        expected = torch.zeros_like(world)
        expected[1] = 1.0
        torch.testing.assert_close(world, expected)

    def test_two_pass_collector_builds_exact_refresh_input_contract(self):
        cameras = [SyntheticCamera(0, [1]), SyntheticCamera(1, [0])]
        gaussians = SyntheticGaussians()
        grad_modes = []
        evidence_shapes = []
        evidence_validity_dtypes = []

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
                evidence_validity_dtypes.append(validity.dtype)
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
        # Six conceptual evidence quantities require eight transport
        # channels because the bool CUDA validity mask cannot carry the
        # multi-view numerator and denominator counts itself.
        self.assertEqual(evidence_shapes, [(8, 2, 2), (8, 2, 2)])
        self.assertEqual(
            evidence_validity_dtypes, [torch.bool, torch.bool]
        )
        self.assertEqual(grad_modes, [False, False, False, False])

    def test_collector_boundary_has_no_gt_or_mesh_argument(self):
        names = set(inspect.signature(D0EvidenceCollector).parameters)

        self.assertNotIn("gt", names)
        self.assertNotIn("gt_mesh", names)
        self.assertNotIn("mesh", names)


if __name__ == "__main__":
    unittest.main()
