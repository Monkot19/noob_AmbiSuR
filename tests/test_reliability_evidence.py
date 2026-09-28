import math
import unittest

import torch

from reliability.evidence import (
    EMAState,
    KEMAState,
    active_non_dc,
    compute_appearance_ambiguity,
    compute_need,
    compute_observation_sufficiency,
    compute_pg_consistency,
    normalize_prior_confidence,
    view_count,
)


class EMAStateTests(unittest.TestCase):
    def test_first_valid_initializes_later_valid_smooths_and_invalid_holds(self):
        state = EMAState(point_count=2, beta=0.9, device="cpu")

        state.update(
            torch.tensor([0.2, 0.8]), torch.tensor([True, False])
        )
        self.assertEqual(state.initialized.tolist(), [True, False])
        torch.testing.assert_close(state.value, torch.tensor([0.2, 0.0]))

        state.update(
            torch.tensor([0.8, 0.4]), torch.tensor([True, True])
        )
        torch.testing.assert_close(state.value, torch.tensor([0.26, 0.4]))

        state.update(
            torch.tensor([1.0, 1.0]), torch.tensor([False, False])
        )
        torch.testing.assert_close(state.value, torch.tensor([0.26, 0.4]))
        self.assertEqual(state.current_valid.tolist(), [False, False])


class ObservationEvidenceTests(unittest.TestCase):
    @staticmethod
    def _sufficiency_for_pair(angle_degrees, *, dtype=torch.float64):
        angle = math.radians(angle_degrees)
        pixel_hits = torch.ones(2, 1, dtype=torch.int32)
        camera_centers = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [math.cos(angle), math.sin(angle), 0.0],
            ],
            dtype=dtype,
        )
        gaussian_centers = torch.zeros(1, 3, dtype=dtype)
        return compute_observation_sufficiency(
            pixel_hits, camera_centers, gaussian_centers
        )

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
        d_c = (1.0 - math.cos(math.radians(30.0))) / 2.0
        expected_count = torch.tensor([2.0 / 7.0])
        expected_angle = torch.tensor([1.0 / (1.0 + d_c)])
        torch.testing.assert_close(result.S_count, expected_count)
        torch.testing.assert_close(result.S_angle, expected_angle)
        torch.testing.assert_close(
            result.S, (expected_count * expected_angle).sqrt()
        )

    def test_soft_count_uses_hand_checked_half_saturation_anchors(self):
        for view_count_value, expected in ((0, 0.0), (5, 0.5), (10, 2.0 / 3.0)):
            with self.subTest(view_count=view_count_value):
                pixel_hits = torch.ones(
                    view_count_value, 1, dtype=torch.int32
                )
                camera_centers = torch.tensor(
                    [[1.0, 0.0, 0.0]] * view_count_value,
                    dtype=torch.float64,
                ).reshape(view_count_value, 3)
                result = compute_observation_sufficiency(
                    pixel_hits,
                    camera_centers,
                    torch.zeros(1, 3, dtype=torch.float64),
                )

                torch.testing.assert_close(
                    result.S_count,
                    torch.tensor([expected], dtype=torch.float64),
                    rtol=1e-12,
                    atol=1e-12,
                )

    def test_soft_angle_uses_hand_checked_half_saturation_anchors(self):
        zero = self._sufficiency_for_pair(0.0)
        half = self._sufficiency_for_pair(30.0)

        torch.testing.assert_close(
            zero.S_angle,
            torch.tensor([0.0], dtype=torch.float64),
            rtol=0.0,
            atol=1e-12,
        )
        torch.testing.assert_close(
            half.S_angle,
            torch.tensor([0.5], dtype=torch.float64),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_soft_scores_remain_strictly_below_one_without_old_plateaus(self):
        count_scores = []
        for view_count_value in (5, 10):
            result = compute_observation_sufficiency(
                torch.ones(view_count_value, 1, dtype=torch.int32),
                torch.tensor(
                    [[1.0, 0.0, 0.0]] * view_count_value,
                    dtype=torch.float64,
                ),
                torch.zeros(1, 3, dtype=torch.float64),
            )
            count_scores.append(result.S_count.item())

        angle_at_half = self._sufficiency_for_pair(30.0).S_angle.item()
        angle_at_maximum = self._sufficiency_for_pair(180.0).S_angle.item()

        self.assertLess(count_scores[0], count_scores[1])
        self.assertLess(count_scores[1], 1.0)
        self.assertLess(angle_at_half, angle_at_maximum)
        self.assertLess(angle_at_maximum, 1.0)

    def test_soft_sufficiency_and_need_follow_frozen_composition(self):
        result = self._sufficiency_for_pair(30.0)
        expected_s = torch.tensor(
            [math.sqrt((2.0 / 7.0) * 0.5)], dtype=torch.float64
        )
        ambiguity = torch.tensor([0.25], dtype=torch.float64)

        torch.testing.assert_close(
            result.S, expected_s, rtol=1e-12, atol=1e-12
        )
        torch.testing.assert_close(
            compute_need(ambiguity, result.S),
            1.0 - expected_s * (1.0 - ambiguity),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_soft_sufficiency_rejects_nonfinite_or_out_of_domain_constants(self):
        hits = torch.ones(2, 1, dtype=torch.int32)
        cameras = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        centers = torch.zeros(1, 3)

        for value in (0.0, -1.0, math.nan, math.inf):
            with self.subTest(k_c=value):
                with self.assertRaisesRegex(ValueError, "k_c"):
                    compute_observation_sufficiency(
                        hits, cameras, centers, k_c=value
                    )
        for value in (0.0, -1.0, 180.0, math.nan, math.inf):
            with self.subTest(theta_c_degrees=value):
                with self.assertRaisesRegex(ValueError, "theta_c_degrees"):
                    compute_observation_sufficiency(
                        hits,
                        cameras,
                        centers,
                        theta_c_degrees=value,
                    )

    def test_empty_point_domain_preserves_dtype_device_and_no_grad(self):
        result = compute_observation_sufficiency(
            torch.empty((2, 0), dtype=torch.int32),
            torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=torch.float64,
            ),
            torch.empty((0, 3), dtype=torch.float64, requires_grad=True),
        )

        self.assertEqual(result.M_obs.shape, (0,))
        self.assertEqual(result.M_obs.dtype, torch.int64)
        for value in (result.S_count, result.S_angle, result.S):
            self.assertEqual(value.shape, (0,))
            self.assertEqual(value.dtype, torch.float64)
            self.assertEqual(value.device.type, "cpu")
            self.assertFalse(value.requires_grad)

    def test_repeated_view_direction_has_zero_sufficiency(self):
        pixel_hits = torch.ones(5, 1, dtype=torch.int32)
        camera_centers = torch.tensor(
            [[1.0, 0.0, 0.0]] * 5, dtype=torch.float32
        )
        gaussian_centers = torch.zeros(1, 3)

        result = compute_observation_sufficiency(
            pixel_hits, camera_centers, gaussian_centers
        )

        torch.testing.assert_close(result.S_count, torch.full((1,), 0.5))
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
