import json
import unittest

import torch

from reliability.topology import TopologyChange
from reliability.transition_diagnostics import TemporalTransitionDiagnostics


class TemporalTransitionDiagnosticsTests(unittest.TestCase):
    def test_first_refresh_uses_initialized_bypass_as_previous_state(self):
        tracker = TemporalTransitionDiagnostics(3, device="cpu")

        summary = tracker.update(
            torch.tensor([0, 0, 0], dtype=torch.int8),
            torch.tensor([0, 2, 4], dtype=torch.int8),
        )

        self.assertEqual(
            summary["transition_count_matrix"],
            [
                [1, 0, 1, 0, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
        )
        self.assertAlmostEqual(
            sum(map(sum, summary["transition_fraction_matrix"])), 1.0
        )
        self.assertEqual(tracker.stable_age_refreshes.tolist(), [1, 1, 1])
        self.assertEqual(tracker.stable_transition_count.tolist(), [0, 1, 1])
        self.assertEqual(summary["jitter_count"], 2)
        self.assertAlmostEqual(summary["jitter_rate"], 2.0 / 3.0)
        self.assertEqual(summary["mean_stable_age_refreshes"], 1.0)
        self.assertEqual(summary["mean_stable_transition_count"], 2.0 / 3.0)
        self.assertIsNone(
            summary["mean_stable_age_refreshes_by_state"]["Consensus"]
        )
        json.dumps(summary, allow_nan=False)

    def test_repeated_refresh_updates_age_and_transition_count(self):
        tracker = TemporalTransitionDiagnostics(3, device="cpu")
        tracker.update(
            torch.tensor([0, 0, 0], dtype=torch.int8),
            torch.tensor([0, 2, 4], dtype=torch.int8),
        )

        summary = tracker.update(
            torch.tensor([0, 2, 4], dtype=torch.int8),
            torch.tensor([0, 3, 4], dtype=torch.int8),
        )

        self.assertEqual(tracker.stable_age_refreshes.tolist(), [2, 1, 2])
        self.assertEqual(tracker.stable_transition_count.tolist(), [0, 2, 1])
        self.assertEqual(summary["transition_count_matrix"][0][0], 1)
        self.assertEqual(summary["transition_count_matrix"][2][3], 1)
        self.assertEqual(summary["transition_count_matrix"][4][4], 1)
        self.assertEqual(summary["jitter_count"], 1)
        self.assertAlmostEqual(summary["jitter_rate"], 1.0 / 3.0)
        self.assertEqual(
            summary["mean_stable_age_refreshes_by_state"]["Geometry-led"],
            1.0,
        )
        self.assertEqual(
            summary["mean_stable_transition_count_by_state"]["Geometry-led"],
            2.0,
        )

    def test_topology_migration_inherits_parent_lineage_only_when_mapped(self):
        tracker = TemporalTransitionDiagnostics(2, device="cpu")
        tracker.stable_age_refreshes.copy_(torch.tensor([3, 7]))
        tracker.stable_transition_count.copy_(torch.tensor([1, 4]))
        change = TopologyChange(
            new_to_old=torch.tensor([1, 1, -1, 0], dtype=torch.int64),
            is_new=torch.tensor([False, True, True, False]),
        )

        tracker.on_topology_change(change)

        self.assertEqual(tracker.point_count, 4)
        self.assertEqual(tracker.stable_age_refreshes.tolist(), [7, 7, 0, 3])
        self.assertEqual(
            tracker.stable_transition_count.tolist(), [4, 4, 0, 1]
        )

    def test_state_round_trip_preserves_temporal_lineage(self):
        tracker = TemporalTransitionDiagnostics(3, device="cpu")
        tracker.update(
            torch.tensor([0, 0, 0], dtype=torch.int8),
            torch.tensor([0, 2, 4], dtype=torch.int8),
        )
        state = tracker.state_dict()
        restored = TemporalTransitionDiagnostics(3, device="cpu")

        restored.load_state_dict(state)

        torch.testing.assert_close(
            restored.stable_age_refreshes, tracker.stable_age_refreshes
        )
        torch.testing.assert_close(
            restored.stable_transition_count,
            tracker.stable_transition_count,
        )
        expected = tracker.update(
            torch.tensor([0, 2, 4], dtype=torch.int8),
            torch.tensor([0, 3, 4], dtype=torch.int8),
        )
        actual = restored.update(
            torch.tensor([0, 2, 4], dtype=torch.int8),
            torch.tensor([0, 3, 4], dtype=torch.int8),
        )
        self.assertEqual(actual, expected)

    def test_invalid_state_vectors_fail_closed(self):
        tracker = TemporalTransitionDiagnostics(2, device="cpu")

        with self.assertRaisesRegex(ValueError, "int8"):
            tracker.update(torch.tensor([0, 0]), torch.tensor([0, 0]))
        with self.assertRaisesRegex(ValueError, "shape"):
            tracker.update(
                torch.tensor([0], dtype=torch.int8),
                torch.tensor([0], dtype=torch.int8),
            )
        with self.assertRaisesRegex(ValueError, r"\[0, 4\]"):
            tracker.update(
                torch.tensor([0, 0], dtype=torch.int8),
                torch.tensor([0, 5], dtype=torch.int8),
            )


if __name__ == "__main__":
    unittest.main()
