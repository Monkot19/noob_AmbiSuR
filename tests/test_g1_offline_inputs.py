import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from reliability.diagnostics import SNAPSHOT_FIELDS
from reliability.offline_g1 import (
    build_g1_iteration_inputs,
    load_g1_iteration,
    validate_snapshot_core_state,
)

try:
    import torch
except ModuleNotFoundError:
    torch = None


def make_snapshot(rows=4):
    base = np.arange(rows, dtype=np.float32) / 10.0
    snapshot = {
        "A": base.copy(),
        "S": (1.0 - base).astype(np.float32),
        "N": (1.0 - (1.0 - base) * (1.0 - base)).astype(np.float32),
        "T_p": (0.4 + base).astype(np.float32),
        "V_p": np.array([True, False, True, True][:rows], dtype=np.bool_),
        "r_p": np.zeros(rows, dtype=np.float32),
        "T_g": (0.3 + base).astype(np.float32),
        "V_g": np.array([True, True, False, True][:rows], dtype=np.bool_),
        "r_g": np.zeros(rows, dtype=np.float32),
        "Z_pg": (1.0 + base).astype(np.float32),
        "V_pg": np.array([True, True, True, False][:rows], dtype=np.bool_),
        "K": (0.5 + base).astype(np.float32),
        "delta": np.zeros(rows, dtype=np.float32),
        "candidate": np.array([0, 2, 4, 1][:rows], dtype=np.int8),
        "stable": np.array([0, 2, 4, 1][:rows], dtype=np.int8),
    }
    snapshot["r_p"] = snapshot["T_p"] * snapshot["V_p"]
    snapshot["r_g"] = snapshot["T_g"] * snapshot["V_g"]
    snapshot["delta"] = snapshot["r_p"] - snapshot["r_g"]
    return snapshot


def make_core_state(snapshot):
    rows = snapshot["A"].shape[0]
    evidence = {
        "version": 2,
        "point_count": rows,
        "a_value": snapshot["A"].copy(),
        "s_value": snapshot["S"].copy(),
        "t_p_value": snapshot["T_p"].copy(),
        "t_p_current_valid": snapshot["V_p"].copy(),
        "t_g_value": snapshot["T_g"].copy(),
        "t_g_current_valid": snapshot["V_g"].copy(),
        "k_value": snapshot["K"].copy(),
        "candidate_state": snapshot["candidate"].copy(),
        "stable_state": snapshot["stable"].copy(),
    }
    return {
        "version": 1,
        "refresh_interval": 1000,
        "last_refresh_iteration": 3000,
        "refresh_count": 3,
        "evidence": evidence,
    }


class G1OfflineInputTests(unittest.TestCase):
    def test_build_inputs_keeps_only_nonfinite_center_rejection_map(self):
        snapshot = make_snapshot()
        centers = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [np.nan, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        joined = build_g1_iteration_inputs(
            iteration=3000,
            centers=centers,
            snapshot=snapshot,
            core_state=make_core_state(snapshot),
            checkpoint_sha256="a" * 64,
            snapshot_sha256="b" * 64,
        )

        self.assertEqual(joined.original_point_count, 4)
        self.assertEqual(joined.centers.dtype, np.float64)
        np.testing.assert_array_equal(joined.finite_row_indices, [0, 1, 3])
        np.testing.assert_array_equal(joined.rejected_center_indices, [2])
        np.testing.assert_array_equal(joined.snapshot["N"], snapshot["N"])

    def test_build_inputs_rejects_snapshot_checkpoint_row_count_mismatch(self):
        snapshot = make_snapshot(rows=3)
        centers = np.zeros((4, 3), dtype=np.float32)

        with self.assertRaisesRegex(
            ValueError, "snapshot/checkpoint row count mismatch"
        ):
            build_g1_iteration_inputs(
                iteration=3000,
                centers=centers,
                snapshot=snapshot,
                core_state=make_core_state(snapshot),
                checkpoint_sha256="a" * 64,
                snapshot_sha256="b" * 64,
            )

    def test_validate_core_state_rejects_same_count_row_permutation(self):
        snapshot = make_snapshot()
        core_state = make_core_state(snapshot)
        core_state["evidence"]["a_value"] = snapshot["A"][[1, 0, 2, 3]]

        with self.assertRaisesRegex(
            ValueError, "snapshot/core_state row contract mismatch: A"
        ):
            validate_snapshot_core_state(snapshot, core_state)

    def test_validate_core_state_checks_derived_fields(self):
        snapshot = make_snapshot()
        snapshot["N"] = snapshot["N"].copy()
        snapshot["N"][2] += 0.25

        with self.assertRaisesRegex(
            ValueError, "snapshot/core_state derived mismatch: N"
        ):
            validate_snapshot_core_state(snapshot, make_core_state(make_snapshot()))

    def test_build_inputs_rejects_wrong_refresh_iteration(self):
        snapshot = make_snapshot()
        core_state = make_core_state(snapshot)
        core_state["last_refresh_iteration"] = 2000

        with self.assertRaisesRegex(ValueError, "D0 refresh iteration mismatch"):
            build_g1_iteration_inputs(
                iteration=3000,
                centers=np.zeros((4, 3), dtype=np.float32),
                snapshot=snapshot,
                core_state=core_state,
                checkpoint_sha256="a" * 64,
                snapshot_sha256="b" * 64,
            )

    def test_build_inputs_rejects_non_boolean_validity(self):
        snapshot = make_snapshot()
        snapshot["V_p"] = snapshot["V_p"].astype(np.float32)

        with self.assertRaisesRegex(ValueError, "snapshot dtype mismatch: V_p"):
            build_g1_iteration_inputs(
                iteration=3000,
                centers=np.zeros((4, 3), dtype=np.float32),
                snapshot=snapshot,
                core_state=make_core_state(make_snapshot()),
                checkpoint_sha256="a" * 64,
                snapshot_sha256="b" * 64,
            )

    def test_build_inputs_rejects_nonfinite_evidence(self):
        snapshot = make_snapshot()
        snapshot["A"][1] = np.nan

        with self.assertRaisesRegex(ValueError, "nonfinite snapshot field: A"):
            build_g1_iteration_inputs(
                iteration=3000,
                centers=np.zeros((4, 3), dtype=np.float32),
                snapshot=snapshot,
                core_state=make_core_state(snapshot),
                checkpoint_sha256="a" * 64,
                snapshot_sha256="b" * 64,
            )

    def test_build_inputs_rejects_missing_or_extra_snapshot_fields(self):
        snapshot = make_snapshot()
        snapshot.pop("K")
        snapshot["unexpected"] = np.zeros(4, dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "snapshot fields mismatch"):
            build_g1_iteration_inputs(
                iteration=3000,
                centers=np.zeros((4, 3), dtype=np.float32),
                snapshot=snapshot,
                core_state=make_core_state(make_snapshot()),
                checkpoint_sha256="a" * 64,
                snapshot_sha256="b" * 64,
            )

    @unittest.skipIf(torch is None, "Torch is required for checkpoint I/O")
    def test_load_g1_iteration_reads_versioned_core_checkpoint(self):
        snapshot = make_snapshot()
        core_state = make_core_state(snapshot)
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            evidence = run / "d0_evidence"
            evidence.mkdir()
            snapshot_path = evidence / "iteration_003000.npz"
            np.savez_compressed(snapshot_path, **snapshot)

            centers = torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                 [float("inf"), 0.0, 0.0], [3.0, 0.0, 0.0]],
                dtype=torch.float32,
            )
            capture = (0, centers) + (None,) * 14
            torch_state = {
                key: torch.as_tensor(value) if isinstance(value, np.ndarray) else value
                for key, value in core_state["evidence"].items()
            }
            payload = {
                "schema_version": 1,
                "gaussian_state": capture,
                "iteration": 3000,
                "core_state": {**core_state, "evidence": torch_state},
            }
            checkpoint_path = run / "chkpnt3000.pth"
            torch.save(payload, checkpoint_path)

            joined = load_g1_iteration(run, 3000)

            np.testing.assert_array_equal(joined.finite_row_indices, [0, 1, 3])
            self.assertEqual(
                joined.checkpoint_sha256,
                hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                joined.snapshot_sha256,
                hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
