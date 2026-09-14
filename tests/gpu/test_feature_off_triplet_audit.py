from contextlib import redirect_stderr, redirect_stdout
import io
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "Torch is required for checkpoint audit tests")
class FeatureOffTripletAuditTests(unittest.TestCase):
    def _write_synthetic_run(self, root, *, role, count, final_points, iteration):
        root.mkdir()
        gaussian = torch.ones(count, 1, dtype=torch.float32)
        optimizer = {
            "state": {
                0: {
                    "step": torch.tensor(float(iteration)),
                    "exp_avg": gaussian.clone(),
                    "exp_avg_sq": gaussian.clone(),
                }
            },
            "param_groups": [{"name": "xyz", "params": [0], "lr": 0.1}],
        }
        capture = (0, *(gaussian.clone() for _ in range(13)), optimizer, 1.0)
        torch.save((capture, iteration), root / f"chkpnt{iteration}.pth")

        app_directory = root / "app_model" / f"iteration_{iteration}"
        app_directory.mkdir(parents=True)
        torch.save(
            {"appear_ab": torch.tensor([1.0, 2.0])},
            app_directory / "app.pth",
        )

        point_cloud_directory = root / "point_cloud" / f"iteration_{iteration}"
        point_cloud_directory.mkdir(parents=True)
        (point_cloud_directory / "point_cloud.ply").write_text(
            "ply\n"
            "format ascii 1.0\n"
            f"element vertex {final_points}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "end_header\n",
            encoding="ascii",
        )
        (root / "train.log").write_text(
            f"Training progress: Points={final_points}\n"
            f"[ITER {iteration}] Evaluating train: L1 0.1 PSNR 20.0\n"
            "Training complete.\n",
            encoding="utf-8",
        )
        (root / "exit_code.txt").write_text("0\n", encoding="utf-8")
        (root / "cfg_args").write_text("synthetic\n", encoding="utf-8")
        (root / "cfg_opts").write_text("synthetic\n", encoding="utf-8")
        (root / "start_utc.txt").write_text(
            "2026-09-14T00:00:00Z\n", encoding="utf-8"
        )
        (root / "end_utc.txt").write_text(
            "2026-09-14T00:01:00Z\n", encoding="utf-8"
        )
        (root / "gpu_peak_mib.txt").write_text("1024\n", encoding="utf-8")
        commit = "e0-test-commit" if role == "e0" else "baseline-test-commit"
        launcher = {
            "role": role,
            "commit": commit,
            "dirty": False,
            "command": f"python train.py --model_path {root} -r 2",
            "python": "3.10.21",
            "torch": "2.7.1+cu128",
            "gpu": "NVIDIA GeForce RTX 4090",
            "dataset_manifest_sha256": "d" * 64,
            "aligned_prior_sha256": "e" * 64,
            "training_config": {
                "semantic_seed": 0,
                "resolution": 2,
                "iterations": iteration,
                "evaluation_iterations": [iteration],
            },
        }
        (root / "g0_run_contract.json").write_text(
            json.dumps(launcher, sort_keys=True) + "\n", encoding="utf-8"
        )
        (root / "launcher.log").write_text("completed\n", encoding="utf-8")
        if role == "e0":
            core = {
                "seed": 0,
                "core_shadow_mode": False,
                "enable_observation_calibration": False,
                "enable_dual_reliability": False,
                "enable_abstention": False,
                "enable_parameter_routing": False,
                "enable_gradient_projection": False,
                "enable_reliability_lifecycle": False,
            }
            resolved = {
                "training_path": "legacy",
                "model": {"resolution": 2},
                "optimization": {"iterations": iteration},
                "core": core,
            }
            identity = {"git_commit": commit, "git_dirty": False, "seed": 0}
            (root / "resolved_config.json").write_text(
                json.dumps(resolved, sort_keys=True) + "\n", encoding="utf-8"
            )
            (root / "run_identity.json").write_text(
                json.dumps(identity, sort_keys=True) + "\n", encoding="utf-8"
            )

    def _write_synthetic_triplet(self, root, *, iteration=8):
        run_directories = {}
        for role, count, final_points in (
            ("b1", 2, 5),
            ("b2", 3, 6),
            ("e0", 4, 7),
        ):
            directory = root / role
            self._write_synthetic_run(
                directory,
                role=role,
                count=count,
                final_points=final_points,
                iteration=iteration,
            )
            run_directories[role] = directory
        return run_directories

    def test_gaussian_summary_uses_frozen_statistics_and_names(self):
        from scripts.diagnostics.audit_feature_off_triplet import (
            summarize_gaussian_tensor,
        )

        tensor = torch.tensor(
            [[4.0, 0.0], [1.0, 3.0], [3.0, 1.0], [2.0, 2.0]],
            dtype=torch.float32,
        )

        summary = summarize_gaussian_tensor(tensor)

        self.assertEqual(summary["dtype"], "torch.float32")
        self.assertEqual(summary["trailing_shape"], [2])
        self.assertEqual(summary["leading_count"], 4)
        self.assertEqual(summary["values"]["channel_000.mean"], 2.5)
        self.assertAlmostEqual(
            summary["values"]["channel_000.std"], math.sqrt(1.25)
        )
        self.assertEqual(summary["values"]["channel_000.q50"], 2.5)
        self.assertEqual(summary["values"]["channel_000.q01"], 1.03)
        self.assertIn("row_l2.q99", summary["values"])
        self.assertEqual(summary["diagnostics"]["element_count"], 8)
        self.assertEqual(summary["diagnostics"]["min"], 0.0)
        self.assertEqual(summary["diagnostics"]["max"], 4.0)
        self.assertRegex(summary["diagnostics"]["sha256"], r"^[0-9a-f]{64}$")

    def test_gaussian_summary_is_exactly_row_permutation_invariant(self):
        from scripts.diagnostics.audit_feature_off_triplet import (
            summarize_gaussian_tensor,
        )

        tensor = torch.tensor(
            [[1.0, 9.0], [2.0, 8.0], [3.0, 7.0], [4.0, 6.0]],
            dtype=torch.float32,
        )
        permutation = torch.tensor([2, 0, 3, 1])
        original = summarize_gaussian_tensor(tensor)
        permuted = summarize_gaussian_tensor(tensor[permutation])

        self.assertEqual(original["values"], permuted["values"])
        self.assertNotEqual(
            original["diagnostics"]["sha256"],
            permuted["diagnostics"]["sha256"],
        )

    def test_gaussian_summary_rejects_scalar_empty_and_nonfinite_inputs(self):
        from scripts.diagnostics.audit_feature_off_triplet import (
            summarize_gaussian_tensor,
        )

        invalid_cases = (
            (torch.tensor(1.0), "Gaussian dimension"),
            (torch.empty(0, 2), "empty"),
            (torch.tensor([[float("nan")]]), "finite"),
            (torch.tensor([[float("inf")]]), "finite"),
        )

        for invalid, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    summarize_gaussian_tensor(invalid)

    def test_gaussian_summary_preserves_original_dtype_and_trailing_shape(self):
        from scripts.diagnostics.audit_feature_off_triplet import (
            summarize_gaussian_tensor,
        )

        float_summary = summarize_gaussian_tensor(torch.zeros(2, 3))
        double_summary = summarize_gaussian_tensor(
            torch.zeros(4, 2, 2, dtype=torch.float64)
        )

        self.assertEqual(float_summary["dtype"], "torch.float32")
        self.assertEqual(float_summary["trailing_shape"], [3])
        self.assertEqual(double_summary["dtype"], "torch.float64")
        self.assertEqual(double_summary["trailing_shape"], [2, 2])

    def test_gaussian_summary_metrics_allow_different_leading_counts(self):
        from scripts.diagnostics.audit_feature_off_triplet import (
            gaussian_summary_metrics,
        )

        tensors = {
            "b1": torch.tensor([[0.0, 1.0], [2.0, 3.0]]),
            "b2": torch.tensor([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]),
            "e0": torch.tensor(
                [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [2.0, 4.0]]
            ),
        }

        metrics, diagnostics = gaussian_summary_metrics("capture.xyz", tensors)
        names = [metric["name"] for metric in metrics]

        self.assertEqual(len(names), 27)
        self.assertIn("capture.xyz.channel_000.mean", names)
        self.assertIn("capture.xyz.channel_001.q99", names)
        self.assertIn("capture.xyz.row_l2.std", names)
        self.assertFalse(any("leading_count" in name for name in names))
        self.assertTrue(diagnostics["summary_keys_equal"])
        self.assertEqual(diagnostics["roles"]["b1"]["leading_count"], 2)
        self.assertEqual(diagnostics["roles"]["b2"]["leading_count"], 3)
        self.assertEqual(diagnostics["roles"]["e0"]["leading_count"], 4)
        channel_one = next(
            metric
            for metric in metrics
            if metric["name"] == "capture.xyz.channel_001.mean"
        )
        self.assertEqual(channel_one["b1"], 2.0)
        self.assertEqual(channel_one["b2"], 2.0)
        self.assertEqual(channel_one["e0"], 2.5)

    def test_gaussian_summary_metrics_do_not_align_incompatible_channels(self):
        from scripts.diagnostics.audit_feature_off_triplet import (
            gaussian_summary_metrics,
        )

        metrics, diagnostics = gaussian_summary_metrics(
            "capture.xyz",
            {
                "b1": torch.zeros(2, 3),
                "b2": torch.zeros(3, 4),
                "e0": torch.zeros(4, 3),
            },
        )

        self.assertEqual(metrics, [])
        self.assertFalse(diagnostics["summary_keys_equal"])

    def test_topology_aware_evidence_reclassifies_dynamic_counts_and_shapes(self):
        from scripts.diagnostics.audit_feature_off_triplet import (
            build_topology_aware_tensor_evidence,
        )

        counts = {"b1": 2, "b2": 3, "e0": 4}
        captures = {
            role: {"xyz": torch.ones(count, 2)}
            for role, count in counts.items()
        }
        optimizer_tensors = {role: {} for role in counts}
        app_tensors = {
            role: {"appear_ab": torch.tensor([1.0, 2.0])}
            for role in counts
        }
        final_points = {"b1": 5, "b2": 6, "e0": 7}
        snapshots = {
            role: {"final_points": value}
            for role, value in final_points.items()
        }
        learned_ply = {
            role: {"vertices": value}
            for role, value in final_points.items()
        }

        evidence = build_topology_aware_tensor_evidence(
            captures,
            optimizer_tensors,
            app_tensors,
            snapshots,
            learned_ply,
        )
        exact_by_name = {
            item["name"]: item for item in evidence["exact_invariants"]
        }
        scalar_names = {item["name"] for item in evidence["scalar_metrics"]}
        numeric_names = [item["name"] for item in evidence["numeric_fields"]]

        self.assertNotIn("gaussian_count", exact_by_name)
        self.assertNotIn("run.final_points", exact_by_name)
        self.assertIn("capture.xyz.trailing_shape", exact_by_name)
        self.assertIn("checkpoint.gaussian_count", scalar_names)
        self.assertIn("run.final_points", scalar_names)
        self.assertIn("capture.xyz.channel_000.mean", scalar_names)
        self.assertEqual(numeric_names, ["appear_ab"])
        self.assertEqual(
            {
                role: exact_by_name["run.final_points_matches_ply"][role]
                for role in ("b1", "b2", "e0")
            },
            {"b1": True, "b2": True, "e0": True},
        )

        mismatched_ply = {
            **learned_ply,
            "e0": {"vertices": final_points["e0"] + 1},
        }
        mismatched = build_topology_aware_tensor_evidence(
            captures,
            optimizer_tensors,
            app_tensors,
            snapshots,
            mismatched_ply,
        )
        match_invariant = next(
            item
            for item in mismatched["exact_invariants"]
            if item["name"] == "run.final_points_matches_ply"
        )
        self.assertTrue(match_invariant["b1"])
        self.assertTrue(match_invariant["b2"])
        self.assertFalse(match_invariant["e0"])

    def test_topology_aware_evidence_classifies_optimizer_moments(self):
        from scripts.diagnostics.audit_feature_off_triplet import (
            build_topology_aware_tensor_evidence,
        )

        counts = {"b1": 2, "b2": 3, "e0": 4}
        captures = {
            role: {"xyz": torch.ones(count, 2)}
            for role, count in counts.items()
        }
        optimizer_tensors = {
            role: {
                "optimizer.xyz.step": torch.tensor(8.0),
                "optimizer.xyz.exp_avg": torch.ones(count, 2),
                "optimizer.xyz.exp_avg_sq": torch.ones(count, 2),
            }
            for role, count in counts.items()
        }
        snapshots = {role: {"final_points": count} for role, count in counts.items()}
        learned_ply = {role: {"vertices": count} for role, count in counts.items()}

        evidence = build_topology_aware_tensor_evidence(
            captures,
            optimizer_tensors,
            {role: {} for role in counts},
            snapshots,
            learned_ply,
        )
        scalar_names = {item["name"] for item in evidence["scalar_metrics"]}

        self.assertIn("optimizer.xyz.exp_avg.channel_000.mean", scalar_names)
        self.assertIn("optimizer.xyz.exp_avg_sq.row_l2.q99", scalar_names)
        self.assertFalse(any("optimizer.xyz.step." in name for name in scalar_names))

        unsupported = {
            role: {
                **values,
                "optimizer.xyz.momentum": torch.ones(counts[role], 2),
            }
            for role, values in optimizer_tensors.items()
        }
        with self.assertRaisesRegex(ValueError, "unsupported optimizer tensor state"):
            build_topology_aware_tensor_evidence(
                captures,
                unsupported,
                {role: {} for role in counts},
                snapshots,
                learned_ply,
            )

    def test_build_report_topology_aware_is_explicit_schema_two(self):
        from scripts.diagnostics.audit_feature_off_triplet import build_report

        with TemporaryDirectory() as temporary_directory:
            run_directories = self._write_synthetic_triplet(
                Path(temporary_directory)
            )
            legacy = build_report(run_directories, 8, exploratory=True)
            topology_aware = build_report(
                run_directories,
                8,
                exploratory=True,
                topology_aware=True,
            )

        legacy_exact = {item["name"] for item in legacy["exact_invariants"]}
        topology_exact = {
            item["name"] for item in topology_aware["exact_invariants"]
        }
        topology_scalars = {
            item["name"] for item in topology_aware["scalar_metrics"]
        }

        self.assertEqual(legacy["schema_version"], 1)
        self.assertEqual(topology_aware["schema_version"], 2)
        self.assertIn("gaussian_count", legacy_exact)
        self.assertIn("capture.xyz.shape", legacy_exact)
        self.assertNotIn("gaussian_count", topology_exact)
        self.assertIn("capture.xyz.trailing_shape", topology_exact)
        self.assertIn("checkpoint.gaussian_count", topology_scalars)

    def test_behavioral_schema_three_keeps_old_modes_and_partitions_metrics(self):
        # Catches a global gate change or silent deletion of internal summaries.
        from scripts.diagnostics.audit_feature_off_triplet import build_report

        with TemporaryDirectory() as temporary_directory:
            runs = self._write_synthetic_triplet(Path(temporary_directory))
            schema_one = build_report(runs, 8, exploratory=True)
            schema_two = build_report(runs, 8, exploratory=True, topology_aware=True)
            schema_three = build_report(
                runs, 8, exploratory=True, topology_aware=True, behavioral_g0=True
            )

        self.assertEqual(schema_one["schema_version"], 1)
        self.assertEqual(schema_two["schema_version"], 2)
        self.assertEqual(schema_three["schema_version"], 3)
        old_by_name = {item["name"]: item for item in schema_two["scalar_metrics"]}
        hard = {item["name"] for item in schema_three["scalar_metrics"]}
        diagnostic = {
            item["name"]: item for item in schema_three["diagnostic_scalar_metrics"]
        }
        self.assertEqual(
            hard,
            {"checkpoint.gaussian_count", "run.final_points",
             "evaluation[8:train].l1", "evaluation[8:train].psnr"},
        )
        self.assertIn("capture.xyz.channel_000.mean", diagnostic)
        self.assertIn("optimizer.xyz.exp_avg.row_l2.q99", diagnostic)
        self.assertEqual(set(old_by_name), hard | set(diagnostic))
        self.assertFalse(hard & set(diagnostic))
        for name, item in diagnostic.items():
            self.assertEqual(item, old_by_name[name])
        self.assertEqual(schema_three["numeric_fields"], schema_two["numeric_fields"])

    def test_behavioral_without_topology_rejects_before_checkpoint_loading(self):
        # Catches accepting an ambiguous schema-3 invocation after artifact reads.
        from scripts.diagnostics.audit_feature_off_triplet import build_report

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            runs = {role: root / role for role in ("b1", "b2", "e0")}
            with patch(
                "scripts.diagnostics.audit_feature_off_triplet.load_checkpoint",
                side_effect=AssertionError("checkpoint was read"),
            ) as load:
                with self.assertRaisesRegex(ValueError, "topology-aware"):
                    build_report(runs, 8, exploratory=True, behavioral_g0=True)
                load.assert_not_called()

    def test_behavioral_missing_completion_is_a_hard_failure(self):
        # Catches treating an incomplete training log as an internal diagnostic.
        from scripts.diagnostics.audit_feature_off_triplet import build_report
        from scripts.diagnostics.compare_feature_off import evaluate_triplet_report

        with TemporaryDirectory() as temporary_directory:
            runs = self._write_synthetic_triplet(Path(temporary_directory))
            log = runs["e0"] / "train.log"
            log.write_text(
                log.read_text(encoding="utf-8").replace("Training complete.\n", ""),
                encoding="utf-8",
            )
            report = build_report(
                runs, 8, exploratory=True, topology_aware=True, behavioral_g0=True
            )

        failures = evaluate_triplet_report(report)["exact_failures"]
        self.assertIn("log.training_complete_count", failures)

    def test_behavioral_duplicate_train_evaluation_is_a_hard_failure(self):
        # Catches a duplicated boundary record that the old presence-only gate misses.
        from scripts.diagnostics.audit_feature_off_triplet import build_report
        from scripts.diagnostics.compare_feature_off import evaluate_triplet_report

        with TemporaryDirectory() as temporary_directory:
            runs = self._write_synthetic_triplet(Path(temporary_directory))
            log = runs["e0"] / "train.log"
            log.write_text(
                log.read_text(encoding="utf-8")
                + "[ITER 8] Evaluating train: L1 0.1 PSNR 20.0\n",
                encoding="utf-8",
            )
            report = build_report(
                runs, 8, exploratory=True, topology_aware=True, behavioral_g0=True
            )

        failures = evaluate_triplet_report(report)["exact_failures"]
        self.assertIn("evaluation.iteration_8.train_count", failures)

    def test_behavioral_empty_required_artifact_is_a_hard_failure(self):
        # Catches treating mere path existence as complete evidence.
        from scripts.diagnostics.audit_feature_off_triplet import build_report
        from scripts.diagnostics.compare_feature_off import evaluate_triplet_report

        with TemporaryDirectory() as temporary_directory:
            runs = self._write_synthetic_triplet(Path(temporary_directory))
            (runs["e0"] / "cfg_opts").write_bytes(b"")
            report = build_report(
                runs, 8, exploratory=True, topology_aware=True, behavioral_g0=True
            )

        failures = evaluate_triplet_report(report)["exact_failures"]
        self.assertIn("artifact.cfg_opts.nonempty", failures)

    def test_behavioral_gpu_peak_and_wall_time_are_hard_safety_gates(self):
        # Catches allowing a feature-off resource regression as diagnostic only.
        from scripts.diagnostics.audit_feature_off_triplet import build_report
        from scripts.diagnostics.compare_feature_off import evaluate_triplet_report

        with TemporaryDirectory() as temporary_directory:
            runs = self._write_synthetic_triplet(Path(temporary_directory))
            (runs["e0"] / "gpu_peak_mib.txt").write_text("23000\n", encoding="utf-8")
            (runs["e0"] / "end_utc.txt").write_text(
                "2026-09-14T00:03:01Z\n", encoding="utf-8"
            )
            report = build_report(
                runs, 8, exploratory=True, topology_aware=True, behavioral_g0=True
            )

        failures = evaluate_triplet_report(report)["exact_failures"]
        self.assertIn("resource.gpu_peak_below_22gib", failures)
        self.assertIn("resource.e0_wall_within_baseline_2x", failures)

    def test_cli_topology_aware_writes_exploratory_schema_two(self):
        from scripts.diagnostics.audit_feature_off_triplet import main

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run_directories = self._write_synthetic_triplet(root)
            output = root / "report.json"
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                return_code = main(
                    [
                        str(run_directories["b1"]),
                        str(run_directories["b2"]),
                        str(run_directories["e0"]),
                        "--iteration",
                        "8",
                        "--evaluation-iterations",
                        "8",
                        "--topology-aware",
                        "--exploratory",
                        "--output",
                        str(output),
                    ]
                )
            report = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(return_code, 0, stderr.getvalue())
        self.assertEqual(report["schema_version"], 2)
        self.assertTrue(report["gate"]["exploratory"])
        self.assertFalse(report["gate"]["g0_equivalent"])

    def test_tensor_pair_stats_uses_float64_rmse_and_mae(self):
        from scripts.diagnostics.audit_feature_off_triplet import tensor_pair_stats

        result = tensor_pair_stats(
            torch.tensor([0.0, 2.0], dtype=torch.float32),
            torch.tensor([0.0, 4.0], dtype=torch.float32),
        )

        self.assertFalse(result["exact"])
        self.assertTrue(result["shape_equal"])
        self.assertEqual(result["element_count"], 2)
        self.assertEqual(result["mismatch_count"], 1)
        self.assertEqual(result["max_abs"], 2.0)
        self.assertEqual(result["mean_abs"], 1.0)
        self.assertAlmostEqual(result["rmse"], math.sqrt(2.0))

    def test_tensor_pair_stats_reports_shape_mismatch_without_broadcasting(self):
        from scripts.diagnostics.audit_feature_off_triplet import tensor_pair_stats

        result = tensor_pair_stats(torch.zeros(2), torch.zeros(2, 1))

        self.assertFalse(result["shape_equal"])
        self.assertFalse(result["exact"])
        self.assertIsNone(result["rmse"])
        self.assertIsNone(result["mean_abs"])

    def test_unpack_legacy_checkpoint_rejects_outer_and_capture_schema(self):
        from scripts.diagnostics.audit_feature_off_triplet import (
            unpack_legacy_checkpoint,
        )

        valid_capture = tuple(range(16))
        self.assertEqual(
            unpack_legacy_checkpoint((valid_capture, 8000), 8000), valid_capture
        )
        with self.assertRaisesRegex(ValueError, "outer checkpoint"):
            unpack_legacy_checkpoint((valid_capture,), 8000)
        with self.assertRaisesRegex(ValueError, "capture.*16"):
            unpack_legacy_checkpoint((tuple(range(15)), 8000), 8000)
        with self.assertRaisesRegex(ValueError, "iteration"):
            unpack_legacy_checkpoint((valid_capture, 7999), 8000)

    def test_name_capture_exposes_frozen_sixteen_field_contract(self):
        from scripts.diagnostics.audit_feature_off_triplet import name_capture

        capture = name_capture(tuple(range(16)))

        self.assertEqual(
            list(capture),
            [
                "active_sh_degree", "xyz", "knn_f", "features_dc",
                "features_rest", "scaling", "rotation", "opacity",
                "max_radii2D", "max_weight", "xyz_gradient_accum",
                "xyz_gradient_accum_abs", "denom", "denom_abs",
                "optimizer", "spatial_lr_scale",
            ],
        )
        self.assertEqual(capture["optimizer"], 14)

    def test_normalize_optimizer_separates_structure_and_tensor_moments(self):
        from scripts.diagnostics.audit_feature_off_triplet import normalize_optimizer

        optimizer = {
            "state": {
                0: {
                    "step": torch.tensor(7.0),
                    "exp_avg": torch.tensor([1.0, 2.0]),
                    "exp_avg_sq": torch.tensor([3.0, 4.0]),
                }
            },
            "param_groups": [{
                "name": "xyz", "params": [0], "lr": 0.1,
                "betas": (0.9, 0.999),
            }],
        }

        structure, tensors = normalize_optimizer(optimizer)

        self.assertEqual(structure["group_names"], ["xyz"])
        self.assertEqual(structure["groups"]["xyz"]["lr"], 0.1)
        self.assertEqual(
            structure["groups"]["xyz"]["state_keys"],
            ["exp_avg", "exp_avg_sq", "step"],
        )
        self.assertEqual(structure["groups"]["xyz"]["step"], 7.0)
        self.assertTrue(torch.equal(
            tensors["optimizer.xyz.exp_avg"], torch.tensor([1.0, 2.0])
        ))
        self.assertTrue(torch.equal(
            tensors["optimizer.xyz.exp_avg_sq"], torch.tensor([3.0, 4.0])
        ))

    def test_finalize_gate_never_promotes_exploratory_replay_to_g0(self):
        from scripts.diagnostics.audit_feature_off_triplet import finalize_gate

        exploratory = finalize_gate({"equivalent": True}, exploratory=True)
        confirmation = finalize_gate({"equivalent": True}, exploratory=False)
        failure = finalize_gate({"equivalent": False}, exploratory=False)

        self.assertTrue(exploratory["audit_completed"])
        self.assertFalse(exploratory["g0_equivalent"])
        self.assertEqual(exploratory["exit_code"], 0)
        self.assertTrue(confirmation["g0_equivalent"])
        self.assertEqual(confirmation["exit_code"], 0)
        self.assertFalse(failure["g0_equivalent"])
        self.assertEqual(failure["exit_code"], 1)


if __name__ == "__main__":
    unittest.main()
