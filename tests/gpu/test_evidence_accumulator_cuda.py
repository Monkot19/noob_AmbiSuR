import unittest

import torch

from diff_plane_rasterization_ambisur import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from reliability.arbitration import ArbitrationState
from reliability.config import CoreConfig
from reliability.evidence import (
    EvidenceAccumulator,
    compute_observation_sufficiency,
)
from reliability.topology import TopologyChange


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class EvidenceAccumulatorCudaTests(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda")
        self.rasterizer = GaussianRasterizer(
            GaussianRasterizationSettings(
                image_height=16,
                image_width=16,
                tanfovx=1.0,
                tanfovy=1.0,
                bg=torch.zeros(3, device=self.device),
                scale_modifier=1.0,
                viewmatrix=torch.eye(4, device=self.device),
                projmatrix=torch.eye(4, device=self.device),
                sh_degree=0,
                campos=torch.zeros(3, device=self.device),
                prefiltered=False,
                render_geo=True,
                ray_reg=0.0,
                trunc_sigma=2.0,
                disable_trunc=False,
                debug=False,
            )
        )

    def raster_inputs(self):
        return {
            "means3D": torch.tensor(
                [[0.0, 0.0, 2.0]], device=self.device
            ),
            "means2D": torch.zeros(
                (1, 3), device=self.device, requires_grad=True
            ),
            "means2D_abs": torch.zeros(
                (1, 3), device=self.device, requires_grad=True
            ),
            "opacities": torch.tensor([[0.7]], device=self.device),
            "colors_precomp": torch.tensor(
                [[0.2, 0.4, 0.6]], device=self.device
            ),
            "scales": torch.tensor(
                [[0.35, 0.35, 0.35]], device=self.device
            ),
            "rotations": torch.tensor(
                [[1.0, 0.0, 0.0, 0.0]], device=self.device
            ),
            "all_map": torch.tensor(
                [[0.0, 0.0, 1.0, 1.0, 2.0, 0.0, 2.0]],
                device=self.device,
            ),
        }

    def test_feature_off_keeps_legacy_five_output_contract(self):
        outputs = self.rasterizer(**self.raster_inputs())
        self.assertEqual(len(outputs), 5)

    def test_weighted_sums_match_rendered_alpha_oracle(self):
        inputs = self.raster_inputs()
        legacy = self.rasterizer(**inputs)
        alpha = legacy[3][3]
        self.assertGreater(float(alpha.sum()), 0.0)

        yy, xx = torch.meshgrid(
            torch.arange(16, device=self.device),
            torch.arange(16, device=self.device),
            indexing="ij",
        )
        values = torch.stack(
            (
                torch.full_like(alpha, 2.5),
                (xx + 2.0 * yy).to(torch.float32),
            )
        ).contiguous()
        validity = torch.stack(
            (
                (xx % 2 == 0),
                (yy >= 4) & (yy < 12),
            )
        ).contiguous()

        outputs = self.rasterizer(
            **inputs,
            evidence_values=values,
            evidence_validity=validity,
        )
        self.assertEqual(len(outputs), 7)
        for actual, expected in zip(outputs[:5], legacy):
            self.assertTrue(torch.equal(actual, expected))

        numerator, denominator = outputs[5:]
        self.assertEqual(tuple(numerator.shape), (1, 2))
        self.assertEqual(tuple(denominator.shape), (1, 2))
        self.assertFalse(numerator.requires_grad)
        self.assertFalse(denominator.requires_grad)

        expected_denominator = torch.stack(
            [(alpha * validity[channel]).sum() for channel in range(2)]
        )
        expected_numerator = torch.stack(
            [
                (alpha * validity[channel] * values[channel]).sum()
                for channel in range(2)
            ]
        )
        torch.testing.assert_close(
            denominator[0], expected_denominator, rtol=2e-5, atol=2e-6
        )
        torch.testing.assert_close(
            numerator[0], expected_numerator, rtol=2e-5, atol=2e-6
        )

    def test_empty_channel_input_runs_no_accumulation(self):
        empty_values = torch.empty((0, 16, 16), device=self.device)
        empty_validity = torch.empty(
            (0, 16, 16), dtype=torch.bool, device=self.device
        )
        outputs = self.rasterizer(
            **self.raster_inputs(),
            evidence_values=empty_values,
            evidence_validity=empty_validity,
        )
        self.assertEqual(tuple(outputs[5].shape), (1, 0))
        self.assertEqual(tuple(outputs[6].shape), (1, 0))

    def test_evidence_inputs_must_be_paired_and_shape_matched(self):
        values = torch.ones((1, 16, 16), device=self.device)
        validity = torch.ones(
            (1, 16, 16), dtype=torch.bool, device=self.device
        )
        with self.assertRaisesRegex(ValueError, "provided together"):
            self.rasterizer(
                **self.raster_inputs(), evidence_values=values
            )
        with self.assertRaisesRegex(ValueError, "matching shapes"):
            self.rasterizer(
                **self.raster_inputs(),
                evidence_values=values,
                evidence_validity=validity[:, :-1],
            )

    def test_evidence_values_are_explicitly_stop_gradient(self):
        values = torch.ones(
            (1, 16, 16), device=self.device, requires_grad=True
        )
        validity = torch.ones(
            (1, 16, 16), dtype=torch.bool, device=self.device
        )
        with self.assertRaisesRegex(ValueError, "must not require gradients"):
            self.rasterizer(
                **self.raster_inputs(),
                evidence_values=values,
                evidence_validity=validity,
            )

    def test_observation_sufficiency_accepts_cpu_hit_matrix_in_chunks(self):
        pixel_hits = torch.tensor(
            [[True, True, False], [True, False, True]], device="cpu"
        )
        camera_centers = torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
            device=self.device,
        )
        gaussian_centers = torch.zeros((3, 3), device=self.device)

        chunked = compute_observation_sufficiency(
            pixel_hits,
            camera_centers,
            gaussian_centers,
            chunk_size=2,
        )
        unchunked = compute_observation_sufficiency(
            pixel_hits,
            camera_centers,
            gaussian_centers,
            chunk_size=99,
        )

        self.assertEqual(chunked.M_obs.device.type, "cuda")
        self.assertEqual(chunked.M_obs.tolist(), [2, 1, 1])
        for name in ("M_obs", "S_count", "S_angle", "S"):
            actual = getattr(chunked, name)
            expected = getattr(unchunked, name)
            torch.testing.assert_close(actual, expected)
            self.assertEqual(actual.device.type, "cuda")
            self.assertFalse(actual.requires_grad)
            if actual.is_floating_point():
                self.assertEqual(actual.dtype, gaussian_centers.dtype)
                self.assertTrue(torch.isfinite(actual).all())

    def test_soft_calibration_has_no_finite_plateau_at_old_count_cap(self):
        camera_centers = torch.tensor(
            [[1.0, 0.0, 0.0]] * 10,
            dtype=torch.float64,
            device=self.device,
        )
        gaussian_centers = torch.zeros(
            (2, 3), dtype=torch.float64, device=self.device
        )
        pixel_hits = torch.zeros((10, 2), dtype=torch.bool, device="cpu")
        pixel_hits[:5, 0] = True
        pixel_hits[:, 1] = True

        result = compute_observation_sufficiency(
            pixel_hits, camera_centers, gaussian_centers, chunk_size=1
        )

        torch.testing.assert_close(
            result.S_count,
            torch.tensor(
                [0.5, 2.0 / 3.0],
                dtype=torch.float64,
                device=self.device,
            ),
        )
        self.assertLess(result.S_count[0].item(), result.S_count[1].item())
        self.assertLess(result.S_count[1].item(), 1.0)

    def test_cuda_topology_resets_evidence_but_preserves_parent_lineage(self):
        accumulator = EvidenceAccumulator(
            2,
            cfg=CoreConfig(core_shadow_mode=True),
            device=self.device,
        )
        accumulator.a_ema.value.copy_(
            torch.tensor([0.25, 0.75], device=self.device)
        )
        accumulator.a_ema.initialized.fill_(True)
        accumulator.a_ema.current_valid.fill_(True)
        accumulator.arbitration.stable_state.copy_(
            torch.tensor(
                [
                    ArbitrationState.PRIOR_LED,
                    ArbitrationState.GEOMETRY_LED,
                ],
                dtype=torch.int8,
                device=self.device,
            )
        )
        accumulator.transition_diagnostics.stable_age_refreshes.copy_(
            torch.tensor([3, 7], dtype=torch.int64, device=self.device)
        )
        accumulator.transition_diagnostics.stable_transition_count.copy_(
            torch.tensor([1, 2], dtype=torch.int64, device=self.device)
        )
        accumulator.transition_diagnostics.previous_stable.copy_(
            accumulator.arbitration.stable_state
        )
        change = TopologyChange(
            new_to_old=torch.tensor(
                [1, 1, -1, 0], dtype=torch.int64, device=self.device
            ),
            is_new=torch.tensor(
                [False, True, True, False], device=self.device
            ),
        )

        accumulator.on_topology_change(change)

        torch.testing.assert_close(
            accumulator.a_ema.value,
            torch.tensor([0.75, 0.0, 0.0, 0.25], device=self.device),
        )
        self.assertEqual(
            accumulator.a_ema.initialized.tolist(),
            [True, False, False, True],
        )
        self.assertEqual(
            accumulator.transition_diagnostics.stable_age_refreshes.tolist(),
            [7, 7, 0, 3],
        )
        self.assertEqual(
            accumulator.transition_diagnostics.stable_transition_count.tolist(),
            [2, 2, 0, 1],
        )
        self.assertEqual(
            accumulator.transition_diagnostics.previous_stable.tolist(),
            [
                ArbitrationState.GEOMETRY_LED,
                ArbitrationState.GEOMETRY_LED,
                ArbitrationState.BYPASS,
                ArbitrationState.PRIOR_LED,
            ],
        )
        self.assertEqual(accumulator.state_dict()["version"], 4)


if __name__ == "__main__":
    unittest.main()
