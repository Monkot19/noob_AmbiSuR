from types import SimpleNamespace
import unittest

import torch

from reliability.arbitration import (
    ArbitrationState,
    ArbitrationStateMachine,
    candidate_state,
)
from reliability.config import CoreConfig


def snapshot(*, need, t_p, v_p, t_g, v_g, v_pg, k, delta):
    return SimpleNamespace(
        N=torch.tensor([need]),
        T_p=torch.tensor([t_p]),
        V_p=torch.tensor([v_p]),
        T_g=torch.tensor([t_g]),
        V_g=torch.tensor([v_g]),
        V_pg=torch.tensor([v_pg]),
        K=torch.tensor([k]),
        delta=torch.tensor([delta]),
    )


class CandidateArbitrationTests(unittest.TestCase):
    def state(self, **values):
        return ArbitrationState(candidate_state(snapshot(**values), CoreConfig()).item())

    def test_bypass_has_highest_priority_and_ignores_joint_validity(self):
        state = self.state(
            need=0.5, t_p=1.0, v_p=True, t_g=1.0, v_g=True,
            v_pg=False, k=1.0, delta=1.0,
        )
        self.assertEqual(state, ArbitrationState.BYPASS)

    def test_consensus_requires_both_reliable_current_joint_support_and_high_k(self):
        state = self.state(
            need=0.8, t_p=0.8, v_p=True, t_g=0.9, v_g=True,
            v_pg=True, k=0.6, delta=0.0,
        )
        self.assertEqual(state, ArbitrationState.CONSENSUS)

    def test_one_sided_prior_reliability_ignores_joint_validity(self):
        state = self.state(
            need=0.8, t_p=0.8, v_p=True, t_g=0.9, v_g=False,
            v_pg=False, k=1.0, delta=-1.0,
        )
        self.assertEqual(state, ArbitrationState.PRIOR_LED)

    def test_one_sided_geometry_reliability_ignores_joint_validity(self):
        state = self.state(
            need=0.8, t_p=0.8, v_p=False, t_g=0.9, v_g=True,
            v_pg=False, k=1.0, delta=1.0,
        )
        self.assertEqual(state, ArbitrationState.GEOMETRY_LED)

    def test_joint_valid_conflict_selects_prior_only_above_positive_delta(self):
        state = self.state(
            need=0.8, t_p=0.8, v_p=True, t_g=0.8, v_g=True,
            v_pg=True, k=0.2, delta=0.11,
        )
        self.assertEqual(state, ArbitrationState.PRIOR_LED)

    def test_joint_valid_conflict_selects_geometry_only_below_negative_delta(self):
        state = self.state(
            need=0.8, t_p=0.8, v_p=True, t_g=0.8, v_g=True,
            v_pg=True, k=0.2, delta=-0.11,
        )
        self.assertEqual(state, ArbitrationState.GEOMETRY_LED)

    def test_joint_invalid_both_reliable_is_abstain_despite_historical_k(self):
        state = self.state(
            need=0.8, t_p=0.8, v_p=True, t_g=0.8, v_g=True,
            v_pg=False, k=1.0, delta=0.9,
        )
        self.assertEqual(state, ArbitrationState.ABSTAIN)

    def test_ambiguous_conflict_and_both_low_are_abstain(self):
        cases = (
            dict(need=0.8, t_p=0.8, v_p=True, t_g=0.8, v_g=True,
                 v_pg=True, k=0.2, delta=0.1),
            dict(need=0.8, t_p=0.2, v_p=True, t_g=0.2, v_g=True,
                 v_pg=True, k=1.0, delta=0.0),
        )
        for values in cases:
            with self.subTest(values=values):
                self.assertEqual(self.state(**values), ArbitrationState.ABSTAIN)


class ArbitrationHysteresisTests(unittest.TestCase):
    def test_candidate_must_repeat_three_refreshes_before_entering(self):
        machine = ArbitrationStateMachine(
            point_count=1,
            initial_state=ArbitrationState.BYPASS,
            enter_count=3,
            device=torch.device("cpu"),
        )
        candidate = torch.tensor([ArbitrationState.PRIOR_LED], dtype=torch.int8)

        self.assertEqual(machine.update(candidate).item(), ArbitrationState.BYPASS)
        self.assertEqual(machine.update(candidate).item(), ArbitrationState.BYPASS)
        self.assertEqual(machine.update(candidate).item(), ArbitrationState.PRIOR_LED)

    def test_changed_candidate_resets_consecutive_count(self):
        machine = ArbitrationStateMachine(
            point_count=1,
            initial_state=ArbitrationState.BYPASS,
            enter_count=3,
            device=torch.device("cpu"),
        )
        prior = torch.tensor([ArbitrationState.PRIOR_LED], dtype=torch.int8)
        geometry = torch.tensor([ArbitrationState.GEOMETRY_LED], dtype=torch.int8)

        machine.update(prior)
        machine.update(prior)
        machine.update(geometry)
        machine.update(geometry)

        self.assertEqual(machine.stable_state.item(), ArbitrationState.BYPASS)
        self.assertEqual(machine.consecutive_count.item(), 2)


if __name__ == "__main__":
    unittest.main()
