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
    def test_snapshot_writer_emits_versioned_no_gt_artifacts(self):
        accumulator = EvidenceAccumulator(
            2, cfg=CoreConfig(core_shadow_mode=True), device="cpu"
        )
        snapshot = accumulator.refresh(refresh_inputs())

        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = write_snapshot(
                temporary_directory,
                iteration=1000,
                snapshot=snapshot,
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

        self.assertEqual(events[0]["schema_version"], 1)
        self.assertEqual(events[0]["iteration"], 1000)
        serialized = json.dumps(events[0], sort_keys=True).lower()
        self.assertNotIn("gt", serialized)
        self.assertNotIn("mesh", serialized)

    def test_writer_rejects_duplicate_iteration_instead_of_overwriting(self):
        accumulator = EvidenceAccumulator(
            2, cfg=CoreConfig(core_shadow_mode=True), device="cpu"
        )
        snapshot = accumulator.refresh(refresh_inputs())

        with tempfile.TemporaryDirectory() as temporary_directory:
            write_snapshot(temporary_directory, 1000, snapshot)
            with self.assertRaisesRegex(FileExistsError, "iteration_001000"):
                write_snapshot(temporary_directory, 1000, snapshot)


if __name__ == "__main__":
    unittest.main()
