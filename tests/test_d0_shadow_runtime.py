import unittest

import torch

from reliability.config import CoreConfig
from reliability.evidence import EvidenceRefreshInputs
from reliability.shadow import D0ShadowRuntime, create_shadow_runtime
from reliability.topology import TopologyChange


def refresh_inputs(centers=None):
    coefficients = torch.zeros(2, 15, 3)
    coefficients[1, 0, 0] = 1.0
    if centers is None:
        centers = torch.zeros(2, 3)
    return EvidenceRefreshInputs(
        sh_coefficients=coefficients,
        sh_degree=1,
        pixel_hits=torch.ones(5, 2, dtype=torch.int32),
        camera_centers=torch.tensor([[1.0, 0.0, 0.0]] * 5),
        centers=centers,
        normals=torch.tensor([[1.0, 0.0, 0.0]] * 2),
        scale_reference=torch.ones(2),
        prior_confidence=torch.ones(2),
        prior_multiview=torch.ones(2),
        prior_support_views=torch.full((2,), 2),
        geometry_multiview=torch.zeros(2),
        geometry_depth_normal=torch.zeros(2),
        geometry_support_views=torch.zeros(2, dtype=torch.int64),
        pg_weighted_support=torch.tensor([1.0, 0.0]),
        pg_depth_error_sum=torch.zeros(2),
        pg_normal_error_sum=torch.zeros(2),
    )


class D0ShadowRuntimeTests(unittest.TestCase):
    def config(self):
        return CoreConfig(core_shadow_mode=True)

    def test_feature_off_factory_does_not_construct_or_consume_state(self):
        runtime = create_shadow_runtime(
            CoreConfig(), point_count=2, device="cpu"
        )

        self.assertIsNone(runtime)

    def test_refresh_runs_only_once_on_each_due_iteration_under_no_grad(self):
        runtime = D0ShadowRuntime(
            2, cfg=self.config(), device="cpu", refresh_interval=1000
        )
        grad_modes = []
        call_count = 0

        def build_inputs():
            nonlocal call_count
            call_count += 1
            grad_modes.append(torch.is_grad_enabled())
            return refresh_inputs()

        self.assertIsNone(runtime.maybe_refresh(999, build_inputs))
        first = runtime.maybe_refresh(1000, build_inputs)
        self.assertIsNotNone(first)
        self.assertIsNone(runtime.maybe_refresh(1000, build_inputs))
        self.assertIsNone(runtime.maybe_refresh(1001, build_inputs))
        second = runtime.maybe_refresh(2000, build_inputs)

        self.assertIsNotNone(second)
        self.assertEqual(call_count, 2)
        self.assertEqual(grad_modes, [False, False])
        self.assertEqual(runtime.last_refresh_iteration, 2000)
        self.assertEqual(runtime.refresh_count, 2)
        self.assertEqual(
            runtime.latest_transition_diagnostics,
            runtime.accumulator.latest_transition_diagnostics,
        )
        self.assertEqual(
            sum(
                map(
                    sum,
                    runtime.latest_transition_diagnostics[
                        "transition_count_matrix"
                    ],
                )
            ),
            2,
        )

    def test_refresh_does_not_write_parameter_grad_optimizer_or_proxy(self):
        runtime = D0ShadowRuntime(
            2, cfg=self.config(), device="cpu", refresh_interval=1
        )
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
        before_parameter = centers.detach().clone()
        before_grad = centers.grad.clone()
        before_optimizer = {
            name: value.clone()
            for name, value in optimizer.state[centers].items()
        }
        before_proxy = densification_proxy.clone()

        runtime.maybe_refresh(
            1, lambda: refresh_inputs(centers=centers)
        )

        torch.testing.assert_close(centers.detach(), before_parameter)
        torch.testing.assert_close(centers.grad, before_grad)
        for name, expected in before_optimizer.items():
            torch.testing.assert_close(optimizer.state[centers][name], expected)
        torch.testing.assert_close(densification_proxy, before_proxy)

    def test_runtime_state_round_trip_preserves_schedule_and_evidence(self):
        runtime = D0ShadowRuntime(
            2, cfg=self.config(), device="cpu", refresh_interval=1000
        )
        runtime.maybe_refresh(1000, refresh_inputs)
        state = runtime.state_dict()
        restored = D0ShadowRuntime(
            2, cfg=self.config(), device="cpu", refresh_interval=1000
        )

        restored.load_state_dict(state)

        self.assertEqual(restored.last_refresh_iteration, 1000)
        self.assertEqual(restored.refresh_count, 1)
        self.assertEqual(
            restored.latest_transition_diagnostics,
            runtime.latest_transition_diagnostics,
        )
        torch.testing.assert_close(
            restored.accumulator.k_ema.value,
            runtime.accumulator.k_ema.value,
        )
        self.assertTrue(
            torch.equal(
                restored.accumulator.history_valid,
                runtime.accumulator.history_valid,
            )
        )

    def test_topology_change_migrates_state_and_keeps_schedule(self):
        runtime = D0ShadowRuntime(
            2, cfg=self.config(), device="cpu", refresh_interval=1000
        )
        runtime.maybe_refresh(1000, refresh_inputs)
        change = TopologyChange(
            new_to_old=torch.tensor([1, -1, 0], dtype=torch.int64),
            is_new=torch.tensor([False, True, False]),
        )

        runtime.on_topology_change(change)

        self.assertEqual(runtime.accumulator.point_count, 3)
        self.assertEqual(runtime.last_refresh_iteration, 1000)
        self.assertEqual(
            runtime.accumulator.history_valid.tolist(), [True, False, True]
        )

    def test_invalid_runtime_configuration_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "refresh_interval"):
            D0ShadowRuntime(
                2, cfg=self.config(), device="cpu", refresh_interval=0
            )
        with self.assertRaisesRegex(ValueError, "shadow mode"):
            D0ShadowRuntime(
                2, cfg=CoreConfig(), device="cpu", refresh_interval=1000
            )


if __name__ == "__main__":
    unittest.main()
