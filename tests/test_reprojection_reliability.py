import unittest

import torch

from reliability.evidence import (
    combine_geometry_reliability,
    combine_prior_reliability,
    compute_geometry_stability,
    reprojection_validity_and_errors,
)


class ReprojectionEvidenceTests(unittest.TestCase):
    def test_mask_rejects_out_of_frame_behind_camera_and_foreground_occlusion(self):
        result = reprojection_validity_and_errors(
            projected_uv=torch.tensor([[2.0, 2.0], [-1.0, 2.0], [2.0, 2.0], [2.0, 2.0]]),
            projected_depth=torch.tensor([1.0, 1.0, -1.0, 2.0]),
            source_depth=torch.ones(4),
            sampled_target_depth=torch.tensor([1.0, 1.0, 1.0, 1.0]),
            source_normal=torch.tensor([[0.0, 0.0, 1.0]] * 4),
            sampled_target_normal=torch.tensor([[0.0, 0.0, 1.0]] * 4),
            image_height=5,
            image_width=5,
        )

        self.assertEqual(result.valid.tolist(), [True, False, False, False])

    def test_occlusion_boundary_keeps_depth_up_to_five_percent(self):
        result = reprojection_validity_and_errors(
            projected_uv=torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
            projected_depth=torch.tensor([1.05, 1.0501]),
            source_depth=torch.ones(2),
            sampled_target_depth=torch.ones(2),
            source_normal=torch.tensor([[0.0, 0.0, 1.0]] * 2),
            sampled_target_normal=torch.tensor([[0.0, 0.0, 1.0]] * 2),
            image_height=3,
            image_width=3,
        )

        self.assertEqual(result.valid.tolist(), [True, False])

    def test_relative_depth_and_sign_invariant_normal_errors_follow_contract(self):
        result = reprojection_validity_and_errors(
            projected_uv=torch.tensor([[1.0, 1.0]]),
            projected_depth=torch.tensor([2.0]),
            source_depth=torch.tensor([2.0]),
            sampled_target_depth=torch.tensor([1.5]),
            source_normal=torch.tensor([[0.0, 0.0, 1.0]]),
            sampled_target_normal=torch.tensor([[0.0, 0.0, -1.0]]),
            image_height=3,
            image_width=3,
            tau_occ=1.0,
        )

        torch.testing.assert_close(result.depth_error, torch.tensor([0.5 / 3.5]))
        torch.testing.assert_close(result.normal_error, torch.zeros(1))
        self.assertFalse(result.depth_error.requires_grad)

    def test_nonfinite_depth_or_normal_is_invalid(self):
        result = reprojection_validity_and_errors(
            projected_uv=torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
            projected_depth=torch.ones(2),
            source_depth=torch.tensor([float("nan"), 1.0]),
            sampled_target_depth=torch.ones(2),
            source_normal=torch.tensor([[0.0, 0.0, 1.0], [float("nan"), 0.0, 1.0]]),
            sampled_target_normal=torch.tensor([[0.0, 0.0, 1.0]] * 2),
            image_height=3,
            image_width=3,
        )

        self.assertEqual(result.valid.tolist(), [False, False])


class ReliabilityCombinationTests(unittest.TestCase):
    def test_prior_strength_and_validity_are_separate(self):
        result = combine_prior_reliability(
            confidence=torch.tensor([0.81, 0.81]),
            multiview=torch.tensor([0.25, 0.25]),
            support_views=torch.tensor([1, 2]),
        )

        torch.testing.assert_close(result.T, torch.tensor([0.45, 0.45]))
        self.assertEqual(result.V.tolist(), [False, True])
        torch.testing.assert_close(result.r, torch.tensor([0.0, 0.45]))

    def test_geometry_requires_history_even_with_two_supporting_views(self):
        result = combine_geometry_reliability(
            multiview=torch.tensor([1.0, 1.0]),
            depth_normal=torch.tensor([1.0, 1.0]),
            stability=torch.tensor([1.0, 1.0]),
            support_views=torch.tensor([2, 2]),
            history_valid=torch.tensor([False, True]),
        )

        torch.testing.assert_close(result.T, torch.ones(2))
        self.assertEqual(result.V.tolist(), [False, True])
        torch.testing.assert_close(result.r, torch.tensor([0.0, 1.0]))

    def test_geometry_stability_is_sign_invariant_and_first_history_is_invalid(self):
        result = compute_geometry_stability(
            current_centers=torch.tensor([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]]),
            previous_centers=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            current_normals=torch.tensor([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
            previous_normals=torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
            scale_reference=torch.ones(2),
            history_valid=torch.tensor([True, False]),
        )

        torch.testing.assert_close(result.score[0], torch.tensor(1.0))
        self.assertEqual(result.valid.tolist(), [True, False])
        self.assertFalse(result.score.requires_grad)


if __name__ == "__main__":
    unittest.main()
