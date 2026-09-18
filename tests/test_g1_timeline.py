import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from reliability.diagnostics import SNAPSHOT_FIELDS
from reliability.g1_timeline import (
    EXPECTED_REFRESH_ITERATIONS,
    load_d0_timeline,
)


STATE_NAMES = (
    "Bypass",
    "Consensus",
    "Prior-led",
    "Geometry-led",
    "Abstain",
)


class G1TimelineTests(unittest.TestCase):
    def event(self, iteration, point_count):
        counts = [[0] * 5 for _ in range(5)]
        fractions = [[0.0] * 5 for _ in range(5)]
        counts[0][0] = point_count
        fractions[0][0] = 1.0
        return {
            "schema_version": 2,
            "iteration": iteration,
            "point_count": point_count,
            "joint_valid_count": point_count - 1,
            "transition_count_matrix": counts,
            "transition_fraction_matrix": fractions,
            "jitter_count": 0,
            "jitter_rate": 0.0,
            "mean_stable_age_refreshes": 1.0,
            "mean_stable_age_refreshes_by_state": {
                name: 1.0 if name == "Bypass" else None
                for name in STATE_NAMES
            },
            "mean_stable_transition_count": 0.0,
            "mean_stable_transition_count_by_state": {
                name: 0.0 if name == "Bypass" else None
                for name in STATE_NAMES
            },
        }

    def snapshot(self, point_count):
        arrays = {}
        for name in SNAPSHOT_FIELDS:
            if name in {"V_p", "V_g", "V_pg"}:
                arrays[name] = np.ones(point_count, dtype=np.bool_)
            elif name in {"candidate", "stable"}:
                arrays[name] = np.zeros(point_count, dtype=np.int8)
            else:
                arrays[name] = np.zeros(point_count, dtype=np.float32)
        arrays["V_pg"][-1] = False
        return arrays

    def write_valid_run(self, root):
        evidence = Path(root) / "d0_evidence"
        evidence.mkdir(parents=True)
        events = []
        for index, iteration in enumerate(EXPECTED_REFRESH_ITERATIONS):
            point_count = index + 2
            np.savez_compressed(
                evidence / f"iteration_{iteration:06d}.npz",
                **self.snapshot(point_count),
            )
            events.append(self.event(iteration, point_count))
        (evidence / "events.jsonl").write_text(
            "".join(json.dumps(event, allow_nan=False) + "\n" for event in events),
            encoding="utf-8",
        )
        return evidence, events

    def test_loader_preserves_event_matrices_with_changing_topology_counts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            _, events = self.write_valid_run(temporary_directory)

            timeline = load_d0_timeline(Path(temporary_directory))

        self.assertEqual(timeline.iterations, EXPECTED_REFRESH_ITERATIONS)
        self.assertEqual(tuple(timeline.events), tuple(events))
        self.assertEqual(timeline.point_counts.tolist(), list(range(2, 9)))
        self.assertEqual(timeline.joint_valid_counts.tolist(), list(range(1, 8)))
        self.assertEqual(timeline.state_counts.shape, (7, 5))
        self.assertEqual(timeline.state_counts[:, 0].tolist(), list(range(2, 9)))
        self.assertEqual(timeline.state_counts[:, 1:].sum(), 0)
        self.assertEqual(timeline.transition_count_matrices.shape, (7, 5, 5))
        self.assertEqual(
            timeline.transition_count_matrices[:, 0, 0].tolist(),
            list(range(2, 9)),
        )

    def test_loader_rejects_missing_duplicate_or_out_of_order_refreshes(self):
        mutations = ("missing", "duplicate", "out_of_order")
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    evidence, events = self.write_valid_run(temporary_directory)
                    if mutation == "missing":
                        (evidence / "iteration_004000.npz").unlink()
                    elif mutation == "duplicate":
                        events[3] = copy.deepcopy(events[2])
                    else:
                        events[2], events[3] = events[3], events[2]
                    (evidence / "events.jsonl").write_text(
                        "".join(json.dumps(event) + "\n" for event in events),
                        encoding="utf-8",
                    )

                    with self.assertRaises(ValueError):
                        load_d0_timeline(Path(temporary_directory))

    def test_loader_rejects_malformed_or_leaking_event_records(self):
        mutations = {}
        wrong_schema = self.event(1000, 2)
        wrong_schema["schema_version"] = 1
        mutations["schema"] = wrong_schema

        wrong_total = self.event(1000, 2)
        wrong_total["transition_count_matrix"][0][0] = 1
        mutations["count_total"] = wrong_total

        bad_fraction = self.event(1000, 2)
        bad_fraction["transition_fraction_matrix"][0][0] = float("nan")
        mutations["nonfinite_fraction"] = bad_fraction

        negative_age = self.event(1000, 2)
        negative_age["mean_stable_age_refreshes"] = -1.0
        mutations["negative_age"] = negative_age

        leaked = self.event(1000, 2)
        leaked["gt_mesh"] = "forbidden"
        mutations["gt_mesh_leak"] = leaked

        for name, replacement in mutations.items():
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    evidence, events = self.write_valid_run(temporary_directory)
                    events[0] = replacement
                    (evidence / "events.jsonl").write_text(
                        "".join(json.dumps(event) + "\n" for event in events),
                        encoding="utf-8",
                    )

                    with self.assertRaises(ValueError):
                        load_d0_timeline(Path(temporary_directory))


if __name__ == "__main__":
    unittest.main()
