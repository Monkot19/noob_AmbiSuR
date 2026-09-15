import math
import unittest

import torch

from reliability.evidence import (
    KEMAState,
    active_non_dc,
    compute_appearance_ambiguity,
    compute_need,
    compute_observation_sufficiency,
    compute_pg_consistency,
    normalize_prior_confidence,
    view_count,
)


class ObservationEvidenceTests(unittest.TestCase):
    def test_active_non_dc_uses_dynamic_sh_degree(self):
        coefficients = torch.ones(2, 15, 3)

        for degree, count in ((1, 3), (2, 8), (3, 15)):
            with self.subTest(degree=degree):
                selected = active_non_dc(coefficients, degree)
                self.assertEqual(selected.shape, (2, count, 3))

    def test_appearance_ambiguity_uses_active_non_dc_energy_and_quantiles(self):
        coefficients = torch.zeros(20, 15, 3)
        coefficients[:, 0, 0] = torch.arange(20, dtype=torch.float32)
        coefficients[:, 3:, :] = 10_000.0

        ambiguity = compute_appearance_ambiguity(coefficients, degree=1)

        energy = torch.arange(20, dtype=torch.float32)
        low = torch.quantile(energy, 0.10)
        high = torch.quantile(energy, 0.95)
        expected = ((energy - low) / (high - low + 1e-8)).clamp(0.0, 1.0)
        torch.testing.assert_close(ambiguity, expected)
        self.assertFalse(ambiguity.requires_grad)

    def test_view_count_binarizes_pixel_hits_per_view(self):
        pixel_hits = torch.tensor(
            [[100, 0], [1, 0], [0, 8]], dtype=torch.int32
        )

        torch.testing.assert_close(view_count(pixel_hits), torch.tensor([2, 1]))

    def test_observation_sufficiency_combines_count_and_angular_spread(self):
        pixel_hits = torch.tensor([[9], [2]], dtype=torch.int32)
        camera_centers = torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        gaussian_centers = torch.zeros(1, 3)

        result = compute_observation_sufficiency(
            pixel_hits, camera_centers, gaussian_centers
        )

        self.assertEqual(result.M_obs.item(), 2)
        torch.testing.assert_close(result.S_count, torch.tensor([0.4]))
        torch.testing.assert_close(result.S_angle, torch.tensor([1.0]))
        torch.testing.assert_close(result.S, torch.tensor([math.sqrt(0.4)]))

    def test_repeated_view_direction_has_zero_sufficiency(self):
        pixel_hits = torch.ones(5, 1, dtype=torch.int32)
        camera_centers = torch.tensor(
            [[1.0, 0.0, 0.0]] * 5, dtype=torch.float32
        )
        gaussian_centers = torch.zeros(1, 3)

        result = compute_observation_sufficiency(
            pixel_hits, camera_centers, gaussian_centers
        )

        torch.testing.assert_close(result.S_count, torch.ones(1))
        torch.testing.assert_close(result.S_angle, torch.zeros(1), atol=1e-6, rtol=0)
        torch.testing.assert_close(result.S, torch.zeros(1), atol=1e-6, rtol=0)

    def test_need_obeys_boundary_contract(self):
        ambiguity = torch.tensor([0.0, 0.0, 1.0, 0.4])
        sufficiency = torch.tensor([0.0, 1.0, 1.0, 0.5])

        need = compute_need(ambiguity, sufficiency)

        torch.testing.assert_close(need, torch.tensor([1.0, 0.0, 1.0, 0.7]))
        self.assertFalse(need.requires_grad)

    def test_prior_confidence_uses_per_view_robust_quantiles(self):
        confidence = torch.tensor(
            [[0.0, 1.0, 2.0, 3.0, 1000.0], [7.0, 7.0, 7.0, 7.0, 7.0]]
        )

        normalized = normalize_prior_confidence(confidence)

        self.assertEqual(normalized.shape, confidence.shape)
        self.assertTrue(torch.isfinite(normalized).all())
        self.assertTrue(((0.0 <= normalized) & (normalized <= 1.0)).all())
        self.assertLess(normalized[0, 1].item(), normalized[0, 2].item())
        torch.testing.assert_close(normalized[1], torch.zeros(5))


class JointValidityTests(unittest.TestCase):
    def test_zero_joint_support_is_invalid_and_not_consensus_by_construction(self):
        result = compute_pg_consistency(
            weighted_support=torch.tensor([0.0]),
            depth_error_sum=torch.tensor([0.0]),
            normal_error_sum=torch.tensor([0.0]),
        )

        torch.testing.assert_close(result.Z_pg, torch.tensor([0.0]))
        self.assertFalse(result.V_pg.item())

    def test_joint_support_threshold_is_strict(self):
        result = compute_pg_consistency(
            weighted_support=torch.tensor([1e-4, 1.0001e-4]),
            depth_error_sum=torch.zeros(2),
            normal_error_sum=torch.zeros(2),
        )

        self.assertEqual(result.V_pg.tolist(), [False, True])

    def test_invalid_joint_refresh_keeps_history_but_clears_current_validity(self):
        state = KEMAState(point_count=1, beta=0.9, device=torch.device("cpu"))
        state.update(torch.tensor([0.8]), torch.tensor([True]))
        historical = state.value.clone()

        state.update(None, torch.tensor([False]))

        torch.testing.assert_close(state.value, historical)
        self.assertTrue(state.initialized.item())
        self.assertFalse(state.current_valid.item())

    def test_first_valid_joint_observation_initializes_ema_from_raw_value(self):
        state = KEMAState(point_count=1, beta=0.9, device=torch.device("cpu"))

        state.update(torch.tensor([0.35]), torch.tensor([True]))

        self.assertTrue(state.initialized.item())
        self.assertTrue(state.current_valid.item())
        torch.testing.assert_close(state.value, torch.tensor([0.35]))

    def test_later_valid_joint_observation_applies_ema(self):
        state = KEMAState(point_count=1, beta=0.9, device=torch.device("cpu"))
        state.update(torch.tensor([0.2]), torch.tensor([True]))

        state.update(torch.tensor([0.8]), torch.tensor([True]))

        torch.testing.assert_close(state.value, torch.tensor([0.26]))


if __name__ == "__main__":
    unittest.main()
