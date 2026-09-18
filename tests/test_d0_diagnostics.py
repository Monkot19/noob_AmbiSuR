import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from reliability.diagnostics import write_snapshot
from reliability.evidence import EvidenceAccumulator
from reliability.config import CoreConfig
from tests.test_d0_shadow_runtime import refresh_inputs


class D0DiagnosticsTests(unittest.TestCase):
    def snapshot_and_transition(self):
        accumulator = EvidenceAccumulator(
            2, cfg=CoreConfig(core_shadow_mode=True), device="cpu"
        )
        snapshot = accumulator.refresh(refresh_inputs())
        return snapshot, copy.deepcopy(
            accumulator.latest_transition_diagnostics
        )

    def test_snapshot_writer_emits_versioned_no_gt_artifacts(self):
        snapshot, transition = self.snapshot_and_transition()

        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = write_snapshot(
                temporary_directory,
                iteration=1000,
                snapshot=snapshot,
                transition_diagnostics=transition,
            )
            array_path = Path(paths["arrays"])
            event_path = Path(paths["events"])
            with np.load(array_path, allow_pickle=False) as arrays:
                self.assertEqual(
                    set(arrays.files),
                    {
                        "A", "S", "N", "T_p", "V_p", "r_p",
                        "T_g", "V_g", "r_g", "Z_pg", "V_pg",
                        "K", "delta", "candidate", "stable",
                    },
                )
                self.assertEqual(arrays["N"].shape, (2,))
            events = [
                json.loads(line)
                for line in event_path.read_text(encoding="utf-8").splitlines()
            ]

        expected_event_keys = {
            "schema_version",
            "iteration",
            "point_count",
            "joint_valid_count",
            "transition_count_matrix",
            "transition_fraction_matrix",
            "jitter_count",
            "jitter_rate",
            "mean_stable_age_refreshes",
            "mean_stable_age_refreshes_by_state",
            "mean_stable_transition_count",
            "mean_stable_transition_count_by_state",
        }
        self.assertEqual(set(events[0]), expected_event_keys)
        self.assertEqual(events[0]["schema_version"], 2)
        self.assertEqual(events[0]["iteration"], 1000)
        self.assertEqual(events[0]["point_count"], 2)
        self.assertEqual(
            sum(map(sum, events[0]["transition_count_matrix"])), 2
        )
        self.assertAlmostEqual(
            sum(map(sum, events[0]["transition_fraction_matrix"])), 1.0
        )
        serialized = json.dumps(events[0], sort_keys=True).lower()
        self.assertNotIn("gt", serialized)
        self.assertNotIn("mesh", serialized)

    def test_writer_rejects_duplicate_iteration_instead_of_overwriting(self):
        snapshot, transition = self.snapshot_and_transition()

        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = write_snapshot(
                temporary_directory,
                1000,
                snapshot,
                transition_diagnostics=transition,
            )
            with self.assertRaisesRegex(FileExistsError, "iteration_001000"):
                write_snapshot(
                    temporary_directory,
                    1000,
                    snapshot,
                    transition_diagnostics=transition,
                )
            event_lines = Path(paths["events"]).read_text(
                encoding="utf-8"
            ).splitlines()

        self.assertEqual(len(event_lines), 1)

    def test_invalid_transition_summary_fails_before_any_artifact_write(self):
        snapshot, transition = self.snapshot_and_transition()
        malformed = [None]

        wrong_shape = copy.deepcopy(transition)
        wrong_shape["transition_count_matrix"] = [[2]]
        malformed.append(wrong_shape)

        wrong_count = copy.deepcopy(transition)
        wrong_count["transition_count_matrix"][0][0] += 1
        malformed.append(wrong_count)

        wrong_fraction = copy.deepcopy(transition)
        wrong_fraction["transition_fraction_matrix"][0][0] += 0.25
        malformed.append(wrong_fraction)

        nonfinite = copy.deepcopy(transition)
        nonfinite["jitter_rate"] = float("nan")
        malformed.append(nonfinite)

        leaked = copy.deepcopy(transition)
        leaked["gt_mesh"] = "forbidden"
        malformed.append(leaked)

        for index, summary in enumerate(malformed):
            with self.subTest(index=index):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    with self.assertRaises((TypeError, ValueError)):
                        write_snapshot(
                            temporary_directory,
                            1000,
                            snapshot,
                            transition_diagnostics=summary,
                        )
                    evidence = Path(temporary_directory) / "d0_evidence"
                    self.assertFalse(evidence.exists())


if __name__ == "__main__":
    unittest.main()
