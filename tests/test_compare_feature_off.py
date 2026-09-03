from pathlib import Path
import tempfile
import unittest

from scripts.diagnostics.compare_feature_off import compare_runs, load_run


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


if __name__ == "__main__":
    unittest.main()
