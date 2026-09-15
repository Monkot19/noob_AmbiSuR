from types import SimpleNamespace
import unittest

import torch

from gaussian_renderer import render


class TinyGaussians:
    max_sh_degree = 0
    active_sh_degree = 0
    disable_trunc = False
    trunc_sigma = 2.0
    use_app = False

    def __init__(self, device):
        self._xyz = torch.tensor(
            [[0.0, 0.0, 2.0]], device=device, requires_grad=True
        )
        self._features = torch.tensor(
            [[[0.2, 0.4, 0.6]]], device=device, requires_grad=True
        )
        self._opacity = torch.tensor(
            [[0.7]], device=device, requires_grad=True
        )
        self._scaling = torch.tensor(
            [[0.35, 0.35, 0.35]], device=device, requires_grad=True
        )
        self._rotation = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]], device=device, requires_grad=True
        )
        self.xyz_gradient_accum = torch.tensor([[3.0]], device=device)
        self.xyz_gradient_accum_abs = torch.tensor([[4.0]], device=device)
        self.denom = torch.tensor([[5.0]], device=device)
        self.denom_abs = torch.tensor([[6.0]], device=device)
        self.max_radii2D = torch.tensor([7.0], device=device)
        self.max_weight = torch.tensor([8.0], device=device)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        return self._features

    @property
    def get_opacity(self):
        return self._opacity

    @property
    def get_scaling(self):
        return self._scaling

    @property
    def get_rotation(self):
        return self._rotation

    def parameters(self):
        return (
            self._xyz,
            self._features,
            self._opacity,
            self._scaling,
            self._rotation,
        )

    def topology_state(self):
        return {
            name: getattr(self, name).clone()
            for name in (
                "xyz_gradient_accum",
                "xyz_gradient_accum_abs",
                "denom",
                "denom_abs",
                "max_radii2D",
                "max_weight",
            )
        }


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class RendererEvidenceAdapterTests(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda")
        self.gaussians = TinyGaussians(self.device)
        self.camera = SimpleNamespace(
            FoVx=1.5707963267948966,
            FoVy=1.5707963267948966,
            image_height=16,
            image_width=16,
            world_view_transform=torch.eye(4, device=self.device),
            full_proj_transform=torch.eye(4, device=self.device),
            camera_center=torch.zeros(3, device=self.device),
        )
        self.pipeline = SimpleNamespace(
            compute_cov3D_python=False,
            convert_SHs_python=False,
            debug=False,
        )
        self.background = torch.zeros(3, device=self.device)

    def render(self, **extra):
        return render(
            self.camera,
            self.gaussians,
            self.pipeline,
            self.background,
            return_plane=False,
            **extra,
        )

    def test_feature_off_keeps_the_existing_public_return_contract(self):
        result = self.render()

        self.assertEqual(
            set(result),
            {
                "render",
                "viewspace_points",
                "viewspace_points_abs",
                "visibility_filter",
                "radii",
                "out_observe",
            },
        )

    def test_evidence_path_exposes_detached_sums_without_state_writes(self):
        legacy = self.render()
        before = self.gaussians.topology_state()
        values = torch.stack(
            (
                torch.full((16, 16), 2.5, device=self.device),
                torch.arange(256, device=self.device, dtype=torch.float32)
                .reshape(16, 16),
            )
        )
        validity = torch.stack(
            (
                torch.ones((16, 16), dtype=torch.bool, device=self.device),
                torch.eye(16, dtype=torch.bool, device=self.device),
            )
        )

        result = self.render(
            evidence_values=values,
            evidence_validity=validity,
        )

        self.assertEqual(
            set(result),
            set(legacy) | {"evidence_numerator", "evidence_denominator"},
        )
        for name in ("render", "radii", "out_observe"):
            self.assertTrue(torch.equal(result[name], legacy[name]))
        self.assertEqual(tuple(result["evidence_numerator"].shape), (1, 2))
        self.assertEqual(tuple(result["evidence_denominator"].shape), (1, 2))
        self.assertFalse(result["evidence_numerator"].requires_grad)
        self.assertFalse(result["evidence_denominator"].requires_grad)
        self.assertTrue(
            all(parameter.grad is None for parameter in self.gaussians.parameters())
        )
        for name, expected in before.items():
            self.assertTrue(torch.equal(getattr(self.gaussians, name), expected))

    def test_evidence_inputs_must_be_paired_at_the_public_boundary(self):
        values = torch.ones((1, 16, 16), device=self.device)

        with self.assertRaisesRegex(ValueError, "provided together"):
            self.render(evidence_values=values)


if __name__ == "__main__":
    unittest.main()
