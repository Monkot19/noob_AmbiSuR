import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from reliability.g1_visualization import required_artifacts
from reliability.offline_g1 import G1IterationInputs, ValidatedMesh
from scripts.diagnostics import evaluate_d0_g1 as evaluator
from scripts.diagnostics.evaluate_d0_g1 import evaluate_iteration, run_evaluator


class G1OrchestrationTests(unittest.TestCase):
    @staticmethod
    def make_formal_contract(root, confirmation_id="formal-success"):
        root = Path(root)
        run = root / "run"
        evidence = run / "d0_evidence"
        source = root / "source"
        output = root / "output"
        evidence.mkdir(parents=True)
        source.mkdir()
        (source / "source.bin").write_bytes(b"frozen-source")
        gt_mesh = root / "mesh.ply"
        gt_mesh.write_bytes(b"frozen-gt")
        for iteration in range(1000, 7001, 1000):
            (evidence / f"iteration_{iteration:06d}.npz").write_bytes(
                f"snapshot-{iteration}".encode("ascii")
            )
        (evidence / "events.jsonl").write_text(
            "schema-2-events\n", encoding="utf-8"
        )
        for iteration in (3000, 7000):
            (run / f"chkpnt{iteration}.pth").write_bytes(
                f"checkpoint-{iteration}".encode("ascii")
            )
        (run / "resolved_config.json").write_text("{}\n", encoding="utf-8")
        args = SimpleNamespace(
            run_dir=str(run),
            source_root=str(source),
            gt_mesh=str(gt_mesh),
            output_root=str(output),
            confirmation_id=confirmation_id,
            iterations=[3000, 7000],
            expected_commit="a" * 40,
            expected_dataset_sha=evaluator.canonical_tree_sha256(source),
            expected_gt_sha=hashlib.sha256(gt_mesh.read_bytes()).hexdigest(),
            exploratory=False,
        )
        return args, run, source, gt_mesh, output

    @staticmethod
    def materialize_formal_artifacts(staging, report):
        for index, name in enumerate(required_artifacts()):
            path = Path(staging) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if name == "report.json":
                payload = json.dumps(report, sort_keys=True)
            elif name == "inputs.json":
                payload = json.dumps({"schema_version": 1}, sort_keys=True)
            else:
                payload = f"formal-artifact-{index}"
            path.write_text(payload + "\n", encoding="utf-8")

    def test_evaluate_iteration_uses_every_finite_center_and_matching_rows(self):
        snapshot = {
            "A": np.array([0.0, 0.2, 0.9]),
            "S": np.array([1.0, 0.8, 0.1]),
            "N": np.array([0.0, 0.36, 0.99]),
            "r_p": np.array([0.2, 0.4, 0.8]),
            "V_p": np.array([True, True, True]),
            "r_g": np.array([0.1, 0.3, 0.7]),
            "V_g": np.array([True, True, True]),
            "stable": np.array([0, 2, 4], dtype=np.int8),
        }
        inputs = G1IterationInputs(
            iteration=500,
            original_point_count=3,
            centers=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            finite_row_indices=np.array([0, 2]),
            rejected_center_indices=np.array([1]),
            snapshot=snapshot,
            checkpoint_sha256="a" * 64,
            snapshot_sha256="b" * 64,
        )
        mesh = ValidatedMesh(
            vertices=np.zeros((3, 3)),
            triangles=np.array([[0, 1, 2]]),
            source_vertex_count=3,
            source_triangle_count=1,
            nonfinite_vertex_count=0,
            rejected_nonfinite_triangle_count=0,
            rejected_degenerate_triangle_count=0,
        )
        calls = []

        def distance_fn(points, received_mesh):
            calls.append((points.copy(), received_mesh))
            return np.array([0.01, 0.20])

        result = evaluate_iteration(inputs, mesh, distance_fn=distance_fn)

        self.assertEqual(len(calls), 1)
        np.testing.assert_array_equal(calls[0][0], inputs.centers)
        self.assertIs(calls[0][1], mesh)
        np.testing.assert_array_equal(result["snapshot"]["N"], [0.0, 0.99])
        np.testing.assert_array_equal(result["distances"], [0.01, 0.20])
        self.assertEqual(result["report"]["iteration"], 500)
        self.assertIsNone(result["report"]["g1_evaluable"])
        self.assertIsNone(result["report"]["g1_pass"])
        self.assertEqual(result["rejected_center_count"], 1)

    def test_evaluate_iteration_rejects_distance_contract_violation(self):
        base = G1IterationInputs(
            iteration=500,
            original_point_count=1,
            centers=np.zeros((1, 3)),
            finite_row_indices=np.array([0]),
            rejected_center_indices=np.array([], dtype=np.int64),
            snapshot={
                "A": np.zeros(1), "S": np.ones(1), "N": np.zeros(1),
                "r_p": np.ones(1), "V_p": np.ones(1, dtype=bool),
                "r_g": np.ones(1), "V_g": np.ones(1, dtype=bool),
                "stable": np.zeros(1, dtype=np.int8),
            },
            checkpoint_sha256="a" * 64,
            snapshot_sha256="b" * 64,
        )
        mesh = ValidatedMesh(
            vertices=np.zeros((3, 3)), triangles=np.array([[0, 1, 2]]),
            source_vertex_count=3, source_triangle_count=1,
            nonfinite_vertex_count=0, rejected_nonfinite_triangle_count=0,
            rejected_degenerate_triangle_count=0,
        )
        with self.assertRaisesRegex(ValueError, "distance row count"):
            evaluate_iteration(
                base,
                mesh,
                distance_fn=lambda _points, _mesh: np.zeros(2),
            )

    def test_formal_run_publishes_108_artifacts_and_uses_7000_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _run, _source, _gt_mesh, _output = self.make_formal_contract(
                directory
            )
            captured = {}
            report = {
                "schema_version": 1,
                "exploratory": False,
                "evaluated_iterations": [3000, 7000],
                "decision_iteration": 7000,
                "g1_evaluable": True,
                "g1_pass": True,
                "iterations": {
                    "3000": {"g1_evaluable": True, "g1_pass": None},
                    "7000": {"g1_evaluable": True, "g1_pass": True},
                },
            }

            def produce(staging, **kwargs):
                captured.update(kwargs)
                self.materialize_formal_artifacts(staging, report)
                return report

            with (
                patch.object(evaluator, "_git_head", return_value="a" * 40),
                patch.object(
                    evaluator,
                    "produce_formal_3000_7000",
                    side_effect=produce,
                    create=True,
                ),
            ):
                exit_code, publication = run_evaluator(args)

            published_report = json.loads(
                (publication["output_dir"] / "report.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(publication["manifest"]["files"]), 108)
        self.assertEqual(published_report["evaluated_iterations"], [3000, 7000])
        self.assertEqual(published_report["decision_iteration"], 7000)
        self.assertFalse(published_report["exploratory"])
        self.assertEqual(
            set(captured["input_fingerprints"]),
            {
                "checkpoint_3000",
                "checkpoint_7000",
                "events",
                "gt_mesh",
                "resolved_config",
                "snapshot_1000",
                "snapshot_2000",
                "snapshot_3000",
                "snapshot_4000",
                "snapshot_5000",
                "snapshot_6000",
                "snapshot_7000",
                "source_root",
            },
        )

    def test_formal_7000_failure_leaves_no_publication_or_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _run, _source, _gt_mesh, output = self.make_formal_contract(
                directory, confirmation_id="formal-failure"
            )

            def fail_at_7000(_staging, **_kwargs):
                raise RuntimeError("7000 evaluation failed")

            with (
                patch.object(evaluator, "_git_head", return_value="a" * 40),
                patch.object(
                    evaluator,
                    "produce_formal_3000_7000",
                    side_effect=fail_at_7000,
                    create=True,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "7000 evaluation failed"):
                    run_evaluator(args)

            self.assertFalse((output / "formal-failure").exists())
            self.assertFalse((output / "formal-failure.tar.gz").exists())
            self.assertFalse((output / "formal-failure.tar.gz.sha256").exists())
            self.assertEqual(list(output.glob(".formal-failure.tmp-*")), [])


if __name__ == "__main__":
    unittest.main()
