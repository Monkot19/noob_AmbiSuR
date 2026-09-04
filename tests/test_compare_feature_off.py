from pathlib import Path
import tempfile
import unittest

from scripts.diagnostics.compare_feature_off import (
    compare_runs,
    evaluate_numeric_field,
    evaluate_scalar_triplet,
    evaluate_triplet_report,
    load_run,
)


def _write_run(root, *, psnr=31.0, points=12, vertex_x=0.0):
    root.mkdir()
    (root / "exit_code.txt").write_text("0\n", encoding="utf-8")
    (root / "gpu_peak_mib.txt").write_text("1024\n", encoding="utf-8")
    (root / "start_utc.txt").write_text(
        "2026-09-03T00:00:00Z\n", encoding="utf-8"
    )
    (root / "end_utc.txt").write_text(
        "2026-09-03T00:01:00Z\n", encoding="utf-8"
    )
    (root / "train.log").write_text(
        "[ITER 30000] Evaluating train: L1 0.016 PSNR "
        f"{psnr}\nTraining progress: 100%|#| 30000/30000 "
        f"[00:01<00:00, Loss=0.009, Points={points}]\n",
        encoding="utf-8",
    )
    point_cloud = root / "point_cloud" / "iteration_30000"
    point_cloud.mkdir(parents=True)
    (point_cloud / "point_cloud.ply").write_text(
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {points}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
        + "".join(f"{vertex_x} 0 0\n" for _ in range(points)),
        encoding="ascii",
    )


def _pair(rmse, mae, *, exact=False, shape_equal=True):
    return {
        "exact": exact,
        "shape_equal": shape_equal,
        "rmse": rmse,
        "mean_abs": mae,
        "max_abs": max(rmse, mae),
        "mismatch_count": 0 if exact else 2,
        "element_count": 2,
    }


def _field(
    *,
    self_rmse=1.0,
    self_mae=1.0,
    b1_rmse=1.0,
    b1_mae=1.0,
    b2_rmse=3.0,
    b2_mae=3.0,
    self_exact=False,
    b1_exact=False,
    b2_exact=False,
    shape_equal=True,
):
    return {
        "name": "capture.xyz",
        "dtype": "torch.float32",
        "shape": [2],
        "b1_b2": _pair(
            self_rmse,
            self_mae,
            exact=self_exact,
            shape_equal=shape_equal,
        ),
        "b1_e0": _pair(
            b1_rmse,
            b1_mae,
            exact=b1_exact,
            shape_equal=shape_equal,
        ),
        "b2_e0": _pair(
            b2_rmse,
            b2_mae,
            exact=b2_exact,
            shape_equal=shape_equal,
        ),
    }
class FeatureOffComparatorTests(unittest.TestCase):
    def test_load_run_reads_training_and_resource_evidence(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run = Path(temporary_directory) / "run"
            _write_run(run)

            snapshot = load_run(run)

        self.assertEqual(snapshot["exit_code"], 0)
        self.assertEqual(snapshot["gpu_peak_mib"], 1024)
        self.assertEqual(snapshot["wall_time_seconds"], 60.0)
        self.assertEqual(
            snapshot["evaluations"],
            [{"iteration": 30000, "split": "train", "l1": 0.016, "psnr": 31.0}],
        )
        self.assertEqual(snapshot["final_points"], 12)
        self.assertEqual(snapshot["point_clouds"]["30000"]["vertices"], 12)

    def test_identical_runs_are_equivalent(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            baseline = root / "baseline"
            candidate = root / "candidate"
            _write_run(baseline)
            _write_run(candidate)

            comparison = compare_runs(load_run(baseline), load_run(candidate))

        self.assertTrue(comparison["equivalent"])
        self.assertEqual(comparison["differences"], [])

    def test_metric_change_is_reported(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            baseline = root / "baseline"
            candidate = root / "candidate"
            _write_run(baseline, psnr=31.0)
            _write_run(candidate, psnr=31.1)

            comparison = compare_runs(load_run(baseline), load_run(candidate))

        self.assertFalse(comparison["equivalent"])
        self.assertIn("evaluation[30000:train].psnr", comparison["differences"])

    def test_ply_content_change_is_reported_even_when_count_matches(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            baseline = root / "baseline"
            candidate = root / "candidate"
            _write_run(baseline, vertex_x=0.0)
            _write_run(candidate, vertex_x=1.0)

            comparison = compare_runs(load_run(baseline), load_run(candidate))

        self.assertFalse(comparison["equivalent"])
        self.assertIn("point_cloud[30000].sha256", comparison["differences"])


class ThreeRunEnvelopeTests(unittest.TestCase):
    def test_envelope_accepts_exact_two_times_boundary(self):
        result = evaluate_numeric_field(
            _field(b1_rmse=2.0, b1_mae=2.0, b2_rmse=3.0, b2_mae=3.0)
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["metrics"]["rmse"]["ratio"], 2.0)
        self.assertEqual(result["metrics"]["mae"]["ratio"], 2.0)

    def test_envelope_rejects_value_above_two_times_boundary(self):
        result = evaluate_numeric_field(
            _field(b1_rmse=2.000001, b1_mae=1.0, b2_rmse=3.0, b2_mae=3.0)
        )

        self.assertFalse(result["passed"])
        self.assertFalse(result["metrics"]["rmse"]["passed"])

    def test_zero_self_distance_requires_exact_candidate_pair(self):
        result = evaluate_numeric_field(
            _field(
                self_rmse=0.0,
                self_mae=0.0,
                b1_rmse=0.0,
                b1_mae=0.0,
                b2_rmse=1.0,
                b2_mae=1.0,
                self_exact=True,
                b1_exact=False,
            )
        )

        self.assertFalse(result["passed"])
        self.assertEqual(result["metrics"]["rmse"]["reason"], "zero_self_not_exact")

    def test_rmse_and_mae_must_both_pass(self):
        result = evaluate_numeric_field(
            _field(b1_rmse=1.0, b1_mae=2.1, b2_rmse=3.0, b2_mae=3.0)
        )

        self.assertFalse(result["passed"])
        self.assertTrue(result["metrics"]["rmse"]["passed"])
        self.assertFalse(result["metrics"]["mae"]["passed"])

    def test_each_metric_selects_its_nearest_baseline(self):
        result = evaluate_numeric_field(
            _field(b1_rmse=0.5, b1_mae=3.0, b2_rmse=3.0, b2_mae=0.5)
        )

        self.assertEqual(result["metrics"]["rmse"]["nearest"], "b1")
        self.assertEqual(result["metrics"]["mae"]["nearest"], "b2")
        self.assertTrue(result["passed"])

    def test_missing_nonfinite_or_shape_mismatch_is_rejected(self):
        malformed = _field()
        del malformed["b1_e0"]["rmse"]
        nonfinite = _field(b1_rmse=float("nan"), b2_rmse=float("inf"))
        wrong_shape = _field(shape_equal=False)

        for field in (malformed, nonfinite, wrong_shape):
            with self.subTest(field=field):
                self.assertFalse(evaluate_numeric_field(field)["passed"])

    def test_scalar_metric_uses_absolute_distance_and_nearest_baseline(self):
        result = evaluate_scalar_triplet("psnr", 10.0, 12.0, 11.0)

        self.assertTrue(result["passed"])
        self.assertEqual(result["self_distance"], 2.0)
        self.assertEqual(result["candidate_distance"], 1.0)
        self.assertEqual(result["nearest"], "b1")

    def test_exact_invariant_mismatch_rejects_triplet(self):
        report = {
            "exact_invariants": [
                {"name": "gaussian_count", "b1": 2, "b2": 2, "e0": 3}
            ],
            "numeric_fields": [],
            "scalar_metrics": [],
            "diagnostics": {},
        }

        result = evaluate_triplet_report(report)

        self.assertFalse(result["equivalent"])
        self.assertEqual(result["exact_failures"], ["gaussian_count"])

    def test_learned_hash_difference_is_diagnostic_only(self):
        report = {
            "exact_invariants": [
                {"name": "gaussian_count", "b1": 2, "b2": 2, "e0": 2}
            ],
            "numeric_fields": [_field()],
            "scalar_metrics": [
                {"name": "l1", "b1": 1.0, "b2": 2.0, "e0": 1.5}
            ],
            "diagnostics": {
                "learned_ply_sha": {"b1": "a", "b2": "b", "e0": "c"}
            },
        }

        result = evaluate_triplet_report(report)

        self.assertTrue(result["equivalent"])
        self.assertEqual(
            result["diagnostics"]["learned_ply_sha"],
            {"b1": "a", "b2": "b", "e0": "c"},
        )


if __name__ == "__main__":
    unittest.main()
