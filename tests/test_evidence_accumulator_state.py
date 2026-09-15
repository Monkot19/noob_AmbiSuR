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

    def test_refresh_contract_has_no_gt_or_mesh_input(self):
        names = set(inspect.signature(EvidenceAccumulator.refresh).parameters)

        self.assertNotIn("gt", names)
        self.assertNotIn("gt_mesh", names)
        self.assertNotIn("mesh", names)


if __name__ == "__main__":
    unittest.main()
