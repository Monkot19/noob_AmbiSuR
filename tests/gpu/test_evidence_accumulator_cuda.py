import unittest

import torch

from diff_plane_rasterization_ambisur import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from reliability.evidence import compute_observation_sufficiency


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

        result = compute_observation_sufficiency(
            pixel_hits,
            camera_centers,
            gaussian_centers,
            chunk_size=2,
        )

        self.assertEqual(result.M_obs.device.type, "cuda")
        self.assertEqual(result.M_obs.tolist(), [2, 1, 1])
        self.assertTrue(torch.isfinite(result.S).all())


if __name__ == "__main__":
    unittest.main()
