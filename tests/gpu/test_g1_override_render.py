from types import SimpleNamespace
import unittest

import torch

from reliability.g1_visualization import render_override_color


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

    def full_state(self):
        tensors = self.parameters() + (
            self.xyz_gradient_accum,
            self.xyz_gradient_accum_abs,
            self.denom,
            self.denom_abs,
            self.max_radii2D,
            self.max_weight,
        )
        return tuple(tensor.detach().clone() for tensor in tensors)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class G1OverrideRenderTests(unittest.TestCase):
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

    def test_override_render_is_forward_only_and_does_not_write_model_state(self):
        before = self.gaussians.full_state()
        colors = torch.tensor([[1.0, 0.0, 0.0]], device=self.device)

        image = render_override_color(
            self.camera,
            self.gaussians,
            self.pipeline,
            self.background,
            colors,
        )

        self.assertEqual(tuple(image.shape), (3, 16, 16))
        self.assertFalse(image.requires_grad)
        self.assertTrue(torch.isfinite(image).all())
        self.assertTrue(all(parameter.grad is None for parameter in self.gaussians.parameters()))
        after = self.gaussians.full_state()
        self.assertTrue(all(torch.equal(left, right) for left, right in zip(before, after)))

    def test_override_color_changes_the_real_renderer_output(self):
        red = render_override_color(
            self.camera,
            self.gaussians,
            self.pipeline,
            self.background,
            torch.tensor([[1.0, 0.0, 0.0]], device=self.device),
        )
        blue = render_override_color(
            self.camera,
            self.gaussians,
            self.pipeline,
            self.background,
            torch.tensor([[0.0, 0.0, 1.0]], device=self.device),
        )

        self.assertFalse(torch.equal(red, blue))
        self.assertGreater(float(red[0].sum()), float(red[2].sum()))
        self.assertGreater(float(blue[2].sum()), float(blue[0].sum()))

    def test_override_render_rejects_wrong_color_shape_before_cuda_call(self):
        with self.assertRaisesRegex(ValueError, "colors"):
            render_override_color(
                self.camera,
                self.gaussians,
                self.pipeline,
                self.background,
                torch.ones((2, 3), device=self.device),
            )


if __name__ == "__main__":
    unittest.main()
