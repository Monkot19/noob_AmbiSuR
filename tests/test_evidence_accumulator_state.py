import inspect
import unittest
from dataclasses import replace

import torch

from reliability.arbitration import ArbitrationState
from reliability.config import CoreConfig
from reliability.evidence import EvidenceAccumulator, EvidenceRefreshInputs
from reliability.topology import TopologyChange


class EvidenceAccumulatorStateTests(unittest.TestCase):
    def setUp(self):
        self.cfg = CoreConfig(core_shadow_mode=True)

    def inputs(self):
        coefficients = torch.zeros(2, 15, 3)
        coefficients[1, 0, 0] = 1.0
        return EvidenceRefreshInputs(
            sh_coefficients=coefficients,
            sh_degree=1,
            pixel_hits=torch.ones(5, 2, dtype=torch.int32),
            camera_centers=torch.tensor([[1.0, 0.0, 0.0]] * 5),
            centers=torch.zeros(2, 3),
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

    def test_refresh_builds_detached_snapshot_and_applies_hysteresis(self):
        accumulator = EvidenceAccumulator(2, cfg=self.cfg, device="cpu")
        inputs = self.inputs()

        first = accumulator.refresh(inputs)

        self.assertEqual(first.candidate.tolist(), [2, 2])
        self.assertEqual(first.stable.tolist(), [0, 0])
        self.assertEqual(first.V_g.tolist(), [False, False])
        self.assertEqual(first.V_pg.tolist(), [True, False])
        self.assertTrue(all(not value.requires_grad for value in first.tensors()))

        accumulator.refresh(inputs)
        third = accumulator.refresh(inputs)
        self.assertEqual(
            third.stable.tolist(),
            [ArbitrationState.PRIOR_LED, ArbitrationState.PRIOR_LED],
        )

    def test_joint_invalid_retains_history_but_cannot_be_current(self):
        accumulator = EvidenceAccumulator(2, cfg=self.cfg, device="cpu")
        first = accumulator.refresh(self.inputs())
        historical = first.K.clone()
        invalid = replace(
            self.inputs(),
            pg_weighted_support=torch.zeros(2),
            pg_depth_error_sum=torch.zeros(2),
            pg_normal_error_sum=torch.zeros(2),
        )

        second = accumulator.refresh(invalid)

        torch.testing.assert_close(second.K, historical)
        self.assertEqual(second.V_pg.tolist(), [False, False])
        self.assertEqual(accumulator.k_ema.initialized.tolist(), [True, False])
        self.assertEqual(accumulator.k_ema.current_valid.tolist(), [False, False])

    def test_topology_migration_preserves_survivors_and_resets_new_rows(self):
        accumulator = EvidenceAccumulator(2, cfg=self.cfg, device="cpu")
        for _ in range(3):
            accumulator.refresh(self.inputs())
        old_k = accumulator.k_ema.value.clone()
        change = TopologyChange(
            new_to_old=torch.tensor([1, -1, 0], dtype=torch.int64),
            is_new=torch.tensor([False, True, False]),
        )

        accumulator.on_topology_change(change)

        self.assertEqual(accumulator.point_count, 3)
        torch.testing.assert_close(
            accumulator.k_ema.value,
            torch.tensor([old_k[1], 0.0, old_k[0]]),
        )
        self.assertEqual(
            accumulator.k_ema.initialized.tolist(), [False, False, True]
        )
        self.assertEqual(
            accumulator.arbitration.stable_state.tolist(),
            [ArbitrationState.PRIOR_LED, ArbitrationState.BYPASS, ArbitrationState.PRIOR_LED],
        )
        self.assertEqual(accumulator.history_valid.tolist(), [True, False, True])

    def test_state_dict_round_trip_restores_all_persistent_state(self):
        accumulator = EvidenceAccumulator(2, cfg=self.cfg, device="cpu")
        for _ in range(2):
            accumulator.refresh(self.inputs())
        state = accumulator.state_dict()
        restored = EvidenceAccumulator(2, cfg=self.cfg, device="cpu")

        restored.load_state_dict(state)

        torch.testing.assert_close(restored.k_ema.value, accumulator.k_ema.value)
        for name in ("a_ema", "s_ema", "t_p_ema", "t_g_ema"):
            actual = getattr(restored, name)
            expected = getattr(accumulator, name)
            torch.testing.assert_close(actual.value, expected.value)
            self.assertTrue(
                torch.equal(actual.initialized, expected.initialized)
            )
        self.assertTrue(
            torch.equal(restored.k_ema.initialized, accumulator.k_ema.initialized)
        )
        self.assertTrue(
            torch.equal(restored.history_valid, accumulator.history_valid)
        )
        self.assertTrue(
            torch.equal(
                restored.arbitration.stable_state,
                accumulator.arbitration.stable_state,
            )
        )
        self.assertTrue(
            torch.equal(
                restored.arbitration.consecutive_count,
                accumulator.arbitration.consecutive_count,
            )
        )

    def test_snapshot_uses_smoothed_a_s_t_and_recomputes_need(self):
        accumulator = EvidenceAccumulator(2, cfg=self.cfg, device="cpu")
        first_inputs = replace(
            self.inputs(),
            camera_centers=torch.tensor([
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]),
        )
        first = accumulator.refresh(first_inputs)
        first_a = first.A.clone()
        first_s = first.S.clone()
        torch.testing.assert_close(first.T_p, torch.ones(2))
        self.assertEqual(
            accumulator.t_g_ema.initialized.tolist(), [False, False]
        )

        second_coefficients = first_inputs.sh_coefficients.flip(0)
        second_inputs = replace(
            first_inputs,
            sh_coefficients=second_coefficients,
            pixel_hits=torch.tensor([
                [1, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
            ]),
            prior_confidence=torch.full((2,), 0.25),
            prior_multiview=torch.full((2,), 0.25),
            geometry_multiview=torch.full((2,), 0.5),
            geometry_depth_normal=torch.full((2,), 0.5),
            geometry_support_views=torch.full((2,), 2),
        )
        second = accumulator.refresh(second_inputs)

        expected_a = 0.9 * first_a + 0.1 * first_a.flip(0)
        expected_s = 0.9 * first_s
        torch.testing.assert_close(second.A, expected_a)
        torch.testing.assert_close(second.S, expected_s)
        torch.testing.assert_close(
            second.N, 1.0 - expected_s * (1.0 - expected_a)
        )
        torch.testing.assert_close(second.T_p, torch.full((2,), 0.925))
        expected_t_g = torch.full((2,), 0.25 ** (1.0 / 3.0))
        torch.testing.assert_close(second.T_g, expected_t_g)
        torch.testing.assert_close(second.r_p, second.V_p * second.T_p)
        torch.testing.assert_close(second.r_g, second.V_g * second.T_g)

    def test_invalid_reliability_refresh_retains_t_history_but_zeros_r(self):
        accumulator = EvidenceAccumulator(2, cfg=self.cfg, device="cpu")
        accumulator.refresh(self.inputs())
        historical = accumulator.t_p_ema.value.clone()
        invalid = replace(
            self.inputs(),
            prior_confidence=torch.zeros(2),
            prior_multiview=torch.zeros(2),
            prior_support_views=torch.zeros(2, dtype=torch.int64),
        )

        snapshot = accumulator.refresh(invalid)

        torch.testing.assert_close(snapshot.T_p, historical)
        self.assertEqual(snapshot.V_p.tolist(), [False, False])
        torch.testing.assert_close(snapshot.r_p, torch.zeros(2))

    def test_topology_migration_resets_all_ema_rows_for_new_gaussians(self):
        accumulator = EvidenceAccumulator(2, cfg=self.cfg, device="cpu")
        accumulator.refresh(self.inputs())
        change = TopologyChange(
            new_to_old=torch.tensor([1, -1, 0], dtype=torch.int64),
            is_new=torch.tensor([False, True, False]),
        )

        accumulator.on_topology_change(change)

        for name in ("a_ema", "s_ema", "t_p_ema", "t_g_ema", "k_ema"):
            state = getattr(accumulator, name)
            self.assertFalse(state.initialized[1].item())
            self.assertEqual(state.value[1].item(), 0.0)

    def test_refresh_contract_has_no_gt_or_mesh_input(self):
        names = set(inspect.signature(EvidenceAccumulator.refresh).parameters)

        self.assertNotIn("gt", names)
        self.assertNotIn("gt_mesh", names)
        self.assertNotIn("mesh", names)


if __name__ == "__main__":
    unittest.main()
