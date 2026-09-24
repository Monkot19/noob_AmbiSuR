import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from reliability.g1_component_diagnostics import RAW_COMPONENTS
from scripts.diagnostics.diagnose_d0_g1_components import (
    _assemble_component_arrays,
    _restore_training_neighbors,
    build_parser,
    collect_component_arrays,
    run_diagnostic,
)
from tests.test_g1_component_diagnostics import valid_components, valid_snapshot


class G1ComponentDiagnosticCliTests(unittest.TestCase):
    class FakeTensor:
        def __init__(self, values):
            self.values = np.asarray(values)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.values

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
        (run / "multi_view.json").write_text(
            '{"ref_name":"camera-a","nearest_name":[]}\n',
            encoding="utf-8",
        )
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
            report["metadata"]["current_component_observation"],
            "post_training_recomputation_at_iteration_7000",
        )
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

    def test_component_assembly_uses_the_observation_count_contract_name(self):
        tensor = self.FakeTensor
        refresh = SimpleNamespace(
            prior_confidence=tensor([0.8, 0.6]),
            prior_multiview=tensor([0.7, 0.5]),
            prior_support_views=tensor([2, 1]),
            geometry_multiview=tensor([0.4, 0.3]),
            geometry_depth_normal=tensor([0.9, 0.8]),
            geometry_support_views=tensor([2, 1]),
        )
        sufficiency = SimpleNamespace(
            M_obs=tensor([2, 1]),
            S_count=tensor([0.4, 0.2]),
            S_angle=tensor([0.9, 0.8]),
            S=tensor([0.6, 0.4]),
        )
        consistency = SimpleNamespace(K_raw=tensor([0.75, 0.25]))

        arrays = _assemble_component_arrays(
            refresh, sufficiency, consistency
        )

        self.assertEqual(set(arrays), set(RAW_COMPONENTS))
        np.testing.assert_array_equal(arrays["view_count"], [2, 1])

    def test_training_neighbors_are_restored_by_name_in_offline_camera_order(self):
        cameras = [
            SimpleNamespace(image_name="camera-b", nearest_id=[], nearest_names=[]),
            SimpleNamespace(image_name="camera-a", nearest_id=[], nearest_names=[]),
            SimpleNamespace(image_name="camera-c", nearest_id=[], nearest_names=[]),
        ]
        records = [
            {"ref_name": "camera-a", "nearest_name": ["camera-c", "camera-b"]},
            {"ref_name": "camera-b", "nearest_name": ["camera-a"]},
            {"ref_name": "camera-c", "nearest_name": []},
        ]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "multi_view.json"
            path.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )
            _restore_training_neighbors(cameras, path)

        self.assertEqual(cameras[0].nearest_id, [1])
        self.assertEqual(cameras[0].nearest_names, ["camera-a"])
        self.assertEqual(cameras[1].nearest_id, [2, 0])
        self.assertEqual(cameras[1].nearest_names, ["camera-c", "camera-b"])
        self.assertEqual(cameras[2].nearest_id, [])
        self.assertEqual(cameras[2].nearest_names, [])

    def test_training_neighbor_restoration_fails_closed_on_bad_names(self):
        cases = {
            "missing camera record": [
                {"ref_name": "camera-a", "nearest_name": ["camera-b"]},
            ],
            "duplicate reference camera": [
                {"ref_name": "camera-a", "nearest_name": ["camera-b"]},
                {"ref_name": "camera-a", "nearest_name": []},
                {"ref_name": "camera-b", "nearest_name": ["camera-a"]},
            ],
            "unknown neighbor camera": [
                {"ref_name": "camera-a", "nearest_name": ["camera-z"]},
                {"ref_name": "camera-b", "nearest_name": ["camera-a"]},
            ],
        }
        for message, records in cases.items():
            cameras = [
                SimpleNamespace(image_name="camera-a", nearest_id=[], nearest_names=[]),
                SimpleNamespace(image_name="camera-b", nearest_id=[], nearest_names=[]),
            ]
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "multi_view.json"
                path.write_text(
                    "\n".join(json.dumps(record) for record in records) + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, message):
                    _restore_training_neighbors(cameras, path)

    def test_multi_view_map_is_an_immutable_diagnostic_input(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _output = self.make_request(directory, "neighbor-fingerprint")
            dependency = self.dependencies()
            seen = []
            original = dependency.fingerprint_inputs

            def fingerprint(paths):
                seen.append(dict(paths))
                return original(paths)

            dependency.fingerprint_inputs = fingerprint
            run_diagnostic(args, dependencies=dependency)

        self.assertEqual(len(seen), 2)
        for inputs in seen:
            self.assertEqual(
                inputs["multi_view"], Path(args.run_dir).resolve() / "multi_view.json"
            )

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
