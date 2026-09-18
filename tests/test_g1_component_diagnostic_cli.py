import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from reliability.g1_component_diagnostics import RAW_COMPONENTS
from scripts.diagnostics.diagnose_d0_g1_components import (
    build_parser,
    collect_component_arrays,
    run_diagnostic,
)
from tests.test_g1_component_diagnostics import valid_components, valid_snapshot


class G1ComponentDiagnosticCliTests(unittest.TestCase):
    @staticmethod
    def make_request(root, confirmation_id="component-diagnostic"):
        root = Path(root)
        run = root / "run"
        evidence = run / "d0_evidence"
        source = root / "source"
        output = root / "output"
        evidence.mkdir(parents=True)
        source.mkdir()
        output.mkdir()
        (run / "exit_code.txt").write_text("0\n", encoding="utf-8")
        (run / "resolved_config.json").write_text("{}\n", encoding="utf-8")
        (run / "chkpnt7000.pth").write_bytes(b"checkpoint")
        (evidence / "iteration_007000.npz").write_bytes(b"snapshot")
        (source / "source.bin").write_bytes(b"source")
        gt = root / "mesh.ply"
        gt.write_bytes(b"mesh")
        args = SimpleNamespace(
            run_dir=str(run),
            source_root=str(source),
            gt_mesh=str(gt),
            output_root=str(output),
            confirmation_id=confirmation_id,
            expected_commit="a" * 40,
            expected_dataset_sha="b" * 64,
            expected_gt_sha="c" * 64,
        )
        return args, output

    @staticmethod
    def dependencies(*, mutate=False):
        calls = {"fingerprint": 0}
        joined = SimpleNamespace(
            original_point_count=20,
            centers=np.zeros((20, 3), dtype=np.float64),
            finite_row_indices=np.arange(20, dtype=np.int64),
            rejected_center_indices=np.empty(0, dtype=np.int64),
            snapshot=valid_snapshot(),
            checkpoint_sha256="d" * 64,
            snapshot_sha256="e" * 64,
        )

        def fingerprint(_paths):
            calls["fingerprint"] += 1
            suffix = calls["fingerprint"] if mutate else 1
            return {"immutable": {"sha256": str(suffix) * 64}}

        return SimpleNamespace(
            git_head=lambda _repository: "a" * 40,
            canonical_tree_sha256=lambda _source: "b" * 64,
            sha256_file=lambda _path: "c" * 64,
            fingerprint_inputs=fingerprint,
            load_iteration=lambda _run, _iteration: joined,
            load_mesh=lambda _path: object(),
            distance_query=lambda centers, _mesh: np.linspace(
                0.0, 0.19, centers.shape[0], dtype=np.float64
            ),
            collect_components=lambda _run, _source, _iteration: valid_components(),
        )

    def test_cli_publishes_only_small_diagnostic_artifacts_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _output = self.make_request(directory)

            exit_code, publication = run_diagnostic(
                args, dependencies=self.dependencies()
            )

            names = {path.name for path in publication["output_dir"].iterdir()}
            report = json.loads(
                (publication["output_dir"] / "report.json").read_text(
                    encoding="utf-8"
                )
            )
            manifest = json.loads(
                (publication["output_dir"] / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            names,
            {"report.json", "risk_bins.csv", "inputs.json", "manifest.json"},
        )
        self.assertTrue(report["diagnostic_only"])
        self.assertIsNone(report["g1_decision"])
        self.assertFalse(report["historical_geometry_stability_reconstructable"])
        self.assertEqual(
            {item["path"] for item in manifest["files"]},
            {"report.json", "risk_bins.csv", "inputs.json"},
        )

    def test_cli_cleans_staging_when_an_immutable_input_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            args, output = self.make_request(directory, "mutated-input")
            target = output / args.confirmation_id

            with self.assertRaisesRegex(RuntimeError, "input mutated"):
                run_diagnostic(args, dependencies=self.dependencies(mutate=True))

            self.assertFalse(target.exists())
            self.assertEqual(list(output.glob(".mutated-input.tmp-*")), [])

    def test_cli_rejects_overwrite_and_never_uses_metric_exit_status(self):
        with tempfile.TemporaryDirectory() as directory:
            args, output = self.make_request(directory, "no-overwrite")
            (output / args.confirmation_id).mkdir()

            with self.assertRaisesRegex(FileExistsError, "output already exists"):
                run_diagnostic(args, dependencies=self.dependencies())

    def test_parser_exposes_only_the_frozen_diagnostic_inputs(self):
        args = build_parser().parse_args(
            [
                "--run-dir", "run",
                "--source-root", "source",
                "--gt-mesh", "mesh.ply",
                "--output-root", "output",
                "--confirmation-id", "diagnosis",
                "--expected-commit", "a" * 40,
                "--expected-dataset-sha", "b" * 64,
                "--expected-gt-sha", "c" * 64,
            ]
        )

        self.assertEqual(args.confirmation_id, "diagnosis")
        self.assertFalse(hasattr(args, "iterations"))
        self.assertFalse(hasattr(args, "g1_threshold"))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Torch is unavailable")
    def test_collector_boundary_exposes_current_components_not_fake_history(self):
        import torch

        inputs = SimpleNamespace(
            pixel_hits=torch.tensor([[1, 0], [1, 1]], dtype=torch.int64),
            camera_centers=torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]),
            centers=torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            prior_confidence=torch.tensor([0.8, 0.6]),
            prior_multiview=torch.tensor([0.7, 0.5]),
            prior_support_views=torch.tensor([2, 1]),
            geometry_multiview=torch.tensor([0.4, 0.3]),
            geometry_depth_normal=torch.tensor([0.9, 0.8]),
            geometry_support_views=torch.tensor([2, 1]),
            pg_weighted_support=torch.tensor([2.0, 1.0]),
            pg_depth_error_sum=torch.tensor([0.1, 0.1]),
            pg_normal_error_sum=torch.tensor([0.2, 0.1]),
        )

        arrays = collect_component_arrays(inputs)

        self.assertEqual(set(arrays), set(RAW_COMPONENTS))
        self.assertNotIn("geometry_stability", arrays)
        np.testing.assert_array_equal(arrays["view_count"], [2, 1])
        np.testing.assert_allclose(arrays["prior_confidence"], [0.8, 0.6])


if __name__ == "__main__":
    unittest.main()
