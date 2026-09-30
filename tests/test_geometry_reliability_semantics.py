from dataclasses import replace
import math
import unittest

import torch

from reliability.config import CoreConfig
from reliability.evidence import (
    EvidenceAccumulator,
    EvidenceRefreshInputs,
    combine_geometry_reliability,
    compute_geometry_stability,
)
from reliability.topology import TopologyChange


def geometry_inputs(*, centers=None, normals=None, support=None):
    point_count = 2
    if centers is None:
        centers = torch.zeros(point_count, 3)
    if normals is None:
        normals = torch.tensor([[0.0, 0.0, 1.0]] * point_count)
    if support is None:
        support = torch.full((point_count,), 2, dtype=torch.int64)
    coefficients = torch.zeros(point_count, 15, 3)
    coefficients[1, 0, 0] = 1.0
    return EvidenceRefreshInputs(
        sh_coefficients=coefficients,
        sh_degree=1,
        pixel_hits=torch.ones(5, point_count, dtype=torch.int32),
        camera_centers=torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        centers=centers,
        normals=normals,
        scale_reference=torch.ones(point_count),
        prior_confidence=torch.ones(point_count),
        prior_multiview=torch.ones(point_count),
        prior_support_views=torch.full((point_count,), 2),
        geometry_multiview=torch.ones(point_count),
        geometry_depth_normal=torch.ones(point_count),
        geometry_support_views=support,
        pg_weighted_support=torch.ones(point_count),
        pg_depth_error_sum=torch.zeros(point_count),
        pg_normal_error_sum=torch.zeros(point_count),
    )


class GeometryReliabilitySemanticTests(unittest.TestCase):
    def config(self):
        return CoreConfig(core_shadow_mode=True)

    def test_geometry_components_are_reliability_monotone(self):
        ordered = torch.tensor([0.125, 0.5, 1.0])
        fixed = torch.full((3,), 0.5)
        support = torch.full((3,), 2)
        history = torch.ones(3, dtype=torch.bool)

        for varying_name in ("multiview", "depth_normal", "stability"):
            components = {
                "multiview": fixed,
                "depth_normal": fixed,
                "stability": fixed,
            }
            components[varying_name] = ordered
            result = combine_geometry_reliability(
                components["multiview"],
                components["depth_normal"],
                components["stability"],
                support,
                history,
            )

            self.assertTrue(bool(torch.all(result.T[1:] >= result.T[:-1])))
            expected = (
                components["multiview"]
                * components["depth_normal"]
                * components["stability"]
            ).pow(1.0 / 3.0)
            torch.testing.assert_close(result.T, expected)
            torch.testing.assert_close(result.r, result.T)

    def test_stability_decreases_only_with_movement_or_rotation(self):
        angle = math.radians(15.0)
        current_centers = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.25, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.25, 0.0, 0.0],
            ]
        )
        current_normals = torch.tensor(
            [
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 1.0],
                [math.sin(angle), 0.0, math.cos(angle)],
                [math.sin(angle), 0.0, math.cos(angle)],
            ]
        )
        result = compute_geometry_stability(
            current_centers=current_centers,
            previous_centers=torch.zeros(4, 3),
            current_normals=current_normals,
            previous_normals=torch.tensor([[0.0, 0.0, 1.0]] * 4),
            scale_reference=torch.ones(4),
            history_valid=torch.ones(4, dtype=torch.bool),
        )

        torch.testing.assert_close(
            result.score,
            torch.tensor([1.0, math.exp(-1.0), math.exp(-1.0), math.exp(-2.0)]),
            rtol=1e-5,
            atol=1e-6,
        )
        self.assertEqual(result.valid.tolist(), [True] * 4)

    def test_stability_uses_the_previous_valid_refresh_before_overwrite(self):
        accumulator = EvidenceAccumulator(
            2, cfg=self.config(), device="cpu", ema_beta=0.9
        )
        first = geometry_inputs()
        accumulator.refresh(first)
        moved_centers = torch.tensor(
            [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0]]
        )

        second = accumulator.refresh(
            replace(first, centers=moved_centers)
        )

        expected_raw = torch.full((2,), math.exp(-1.0 / 3.0))
        torch.testing.assert_close(second.T_g, expected_raw)
        torch.testing.assert_close(
            accumulator.previous_centers, moved_centers
        )

    def test_first_history_is_unknown_then_becomes_valid(self):
        accumulator = EvidenceAccumulator(2, cfg=self.config(), device="cpu")
        inputs = geometry_inputs()

        first = accumulator.refresh(inputs)
        first_initialized = accumulator.t_g_ema.initialized.clone()
        second = accumulator.refresh(inputs)

        self.assertEqual(first.V_g.tolist(), [False, False])
        torch.testing.assert_close(first.r_g, torch.zeros(2))
        self.assertEqual(first_initialized.tolist(), [False, False])
        self.assertEqual(second.V_g.tolist(), [True, True])
        torch.testing.assert_close(second.r_g, torch.ones(2))

    def test_invalid_geometry_observation_preserves_ema_but_makes_r_g_unusable(self):
        accumulator = EvidenceAccumulator(2, cfg=self.config(), device="cpu")
        inputs = geometry_inputs()
        accumulator.refresh(inputs)
        accumulator.refresh(inputs)
        historical = accumulator.t_g_ema.value.clone()

        invalid = accumulator.refresh(
            replace(
                inputs,
                geometry_support_views=torch.zeros(2, dtype=torch.int64),
            )
        )

        torch.testing.assert_close(accumulator.t_g_ema.value, historical)
        self.assertEqual(
            accumulator.t_g_ema.current_valid.tolist(), [False, False]
        )
        self.assertEqual(invalid.V_g.tolist(), [False, False])
        torch.testing.assert_close(invalid.r_g, torch.zeros(2))

    def test_mapped_new_child_resets_geometry_history_while_survivor_keeps_previous_refresh(self):
        accumulator = EvidenceAccumulator(2, cfg=self.config(), device="cpu")
        first = geometry_inputs(
            centers=torch.tensor(
                [[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]
            )
        )
        accumulator.refresh(first)
        accumulator.refresh(first)
        old_t_g = accumulator.t_g_ema.value.clone()
        old_centers = accumulator.previous_centers.clone()
        change = TopologyChange(
            new_to_old=torch.tensor([1, 1, 0], dtype=torch.int64),
            is_new=torch.tensor([False, True, False]),
        )

        accumulator.on_topology_change(change)

        torch.testing.assert_close(accumulator.t_g_ema.value[0], old_t_g[1])
        torch.testing.assert_close(
            accumulator.previous_centers[0], old_centers[1]
        )
        self.assertTrue(accumulator.history_valid[0].item())
        self.assertFalse(accumulator.t_g_ema.initialized[1].item())
        self.assertFalse(accumulator.history_valid[1].item())
        torch.testing.assert_close(
            accumulator.previous_centers[1], torch.zeros(3)
        )
        torch.testing.assert_close(accumulator.t_g_ema.value[2], old_t_g[0])
        self.assertTrue(accumulator.history_valid[2].item())

    def test_geometry_refresh_is_detached_and_does_not_change_training_state(self):
        accumulator = EvidenceAccumulator(2, cfg=self.config(), device="cpu")
        centers = torch.nn.Parameter(
            torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])
        )
        centers.grad = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        )
        optimizer = torch.optim.Adam([centers], lr=0.01)
        optimizer.state[centers] = {
            "step": torch.tensor(7.0),
            "exp_avg": torch.full_like(centers, 0.25),
            "exp_avg_sq": torch.full_like(centers, 0.5),
        }
        densification_proxy = torch.tensor([[3.0], [4.0]])
        topology_decision = torch.tensor([False, True])
        before_parameter = centers.detach().clone()
        before_grad = centers.grad.clone()
        before_optimizer = {
            name: value.clone()
            for name, value in optimizer.state[centers].items()
        }
        before_proxy = densification_proxy.clone()
        before_topology = topology_decision.clone()

        snapshot = accumulator.refresh(geometry_inputs(centers=centers))

        torch.testing.assert_close(centers.detach(), before_parameter)
        torch.testing.assert_close(centers.grad, before_grad)
        for name, expected in before_optimizer.items():
            torch.testing.assert_close(
                optimizer.state[centers][name], expected
            )
        torch.testing.assert_close(densification_proxy, before_proxy)
        self.assertTrue(torch.equal(topology_decision, before_topology))
        self.assertTrue(
            all(not value.requires_grad for value in snapshot.tensors())
        )


if __name__ == "__main__":
    unittest.main()
