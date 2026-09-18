import types
import unittest

import torch

from reliability.topology import (
    append_topology_change,
    prune_topology_change,
)
from scene.gaussian_model import GaussianModel


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class GaussianTopologyMappingTests(unittest.TestCase):
    def make_model(self, *, scale):
        device = torch.device("cuda")
        model = GaussianModel(0)
        model._xyz = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            device=device,
        )
        model._knn_f = torch.zeros((3, 1), device=device)
        model._features_dc = torch.zeros((3, 1, 3), device=device)
        model._features_rest = torch.zeros((3, 0, 3), device=device)
        model._opacity = torch.zeros((3, 1), device=device)
        model._scaling = torch.full(
            (3, 3), float(torch.log(torch.tensor(scale))), device=device
        )
        model._rotation = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * 3, device=device
        )
        model.percent_dense = 0.5
        model.max_all_points = 100
        model.max_abs_split_points = 100
        model.abs_split_radii2D_threshold = 1.0
        return model

    def record_postfix(self, model, captured):
        def postfix(
            this,
            new_xyz,
            new_knn_f,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            return_topology_change=False,
            parent_indices=None,
        ):
            captured["parent_indices"] = (
                None if parent_indices is None else parent_indices.detach().cpu()
            )
            if not return_topology_change:
                return None
            return append_topology_change(
                model.get_xyz.shape[0],
                new_xyz.shape[0],
                parent_indices=parent_indices,
                device=model.get_xyz.device,
            )

        model.densification_postfix = types.MethodType(postfix, model)

    def test_clone_children_record_selected_parent_indices(self):
        model = self.make_model(scale=0.1)
        captured = {}
        self.record_postfix(model, captured)

        change = model.densify_and_clone(
            grads=torch.tensor([[1.0], [0.0], [2.0]], device="cuda"),
            grad_threshold=0.5,
            scene_extent=1.0,
            return_topology_change=True,
        )

        self.assertEqual(captured["parent_indices"].tolist(), [0, 2])
        self.assertEqual(change.new_to_old.tolist(), [0, 1, 2, 0, 2])
        self.assertEqual(
            change.is_new.tolist(), [False, False, False, True, True]
        )

    def test_split_children_follow_repeat_order_and_keep_newness_after_prune(self):
        model = self.make_model(scale=1.0)
        captured = {}
        self.record_postfix(model, captured)

        def prune_points(this, prune_filter, return_topology_change=False):
            captured["prune_filter"] = prune_filter.detach().cpu()
            if return_topology_change:
                return prune_topology_change(prune_filter)
            return None

        model.prune_points = types.MethodType(prune_points, model)

        change = model.densify_and_split(
            grads=torch.tensor([[1.0], [0.0], [2.0]], device="cuda"),
            grad_threshold=0.5,
            grads_abs=torch.zeros((3, 1), device="cuda"),
            grad_abs_threshold=1.0,
            scene_extent=1.0,
            max_radii2D=torch.zeros(3, device="cuda"),
            N=2,
            return_topology_change=True,
        )

        self.assertEqual(captured["parent_indices"].tolist(), [0, 2, 0, 2])
        self.assertEqual(
            captured["prune_filter"].tolist(),
            [True, False, True, False, False, False, False],
        )
        self.assertEqual(change.new_to_old.tolist(), [1, 0, 2, 0, 2])
        self.assertEqual(change.is_new.tolist(), [False, True, True, True, True])


if __name__ == "__main__":
    unittest.main()
