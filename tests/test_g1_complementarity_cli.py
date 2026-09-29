import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from reliability.g1_complementarity import ProbeInconclusiveError
from scripts.diagnostics.probe_g1_prior_complementarity import (
    build_parser,
    run_probe,
)


REQUIRED_ARTIFACTS = (
    "bootstrap.csv",
    "folds.csv",
    "inputs.json",
    "report.json",
    "risk_bins.csv",
)


def iteration_summary(iteration, *, passing=False):
    role = "primary" if iteration == 7000 else "direction_stability"
    validation_counts = [21, 21, 21, 20, 20]
    validation_positive_counts = [5, 5, 5, 4, 4]
    folds = []
    for fold in range(5):
        validation_count = validation_counts[fold]
        validation_positive_count = validation_positive_counts[fold]
        training_count = 103 - validation_count
        training_positive_count = 23 - validation_positive_count
        folds.append({
            "fold": fold,
            "training_count": training_count,
            "validation_count": validation_count,
            "training_positive_count": training_positive_count,
            "training_negative_count": 64,
            "validation_positive_count": validation_positive_count,
            "validation_negative_count": 16,
            "baseline_auroc": 0.55,
            "augmented_auroc": 0.59,
            "auroc_gain": 0.04,
            "candidate_relative_residual": 0.2,
            "positive_weight": training_count / (2 * training_positive_count),
            "negative_weight": training_count / (2 * 64),
            "baseline_a_mean": 0.5,
            "baseline_one_minus_s_mean": 0.4,
            "baseline_a_scale": 0.2,
            "baseline_one_minus_s_scale": 0.1,
            "candidate_mean": 0.3,
            "candidate_scale": 0.15,
            "baseline_iterations": 6,
            "augmented_iterations": 7,
            "baseline_converged": True,
            "augmented_converged": True,
        })
    return {
        "iteration": iteration,
        "role": role,
        "domain": {
            "original_point_count": 110,
            "finite_center_count": 103,
            "eligible_count": 103,
            "positive_count": 23,
            "negative_count": 80,
            "coverage": 1.0,
            "label": "distance_gt_0.05_m",
        },
        "direction": {
            "lowest_quintile_count": 21,
            "highest_quintile_count": 20,
            "lowest_quintile_high_error_rate": 0.0,
            "highest_quintile_high_error_rate": 1.0,
            "high_minus_low_error_rate": 1.0,
            "spearman_risk_distance": 0.5,
            "marginal": {"auroc": 0.65, "auprc": 0.50},
            "risk_bins": [
                {
                    "bin": index,
                    "count": 6 if index < 3 else 5,
                    "risk_min": index / 20,
                    "risk_max": (index + 1) / 20,
                    "mean_distance_m": 0.01 + index / 1000,
                    "high_error_rate": index / 20,
                }
                for index in range(20)
            ],
        },
        "numerical_independence": {
            "threshold_strictly_greater_than": 1e-8,
            "fold_relative_residuals": [0.2] * 5,
        },
        "crossfit": {
            "folds": folds,
            "baseline": {"auroc": 0.55, "auprc": 0.35},
            "augmented": {"auroc": 0.59 if passing else 0.56, "auprc": 0.39},
            "pooled_auroc_gain": 0.04 if passing else 0.01,
        },
        "bootstrap": {
            "seed": 20260928,
            "replicate_count": 2000,
            "interval_percentiles": [2.5, 97.5],
            "voxel_size_m": 0.5,
            "voxel_origin": [0.0, 0.0, 0.0],
            "voxel_count": 4,
            "lower": 0.04 if passing else 0.0,
            "upper": 0.04 if passing else 0.0,
            "auroc_gain_replicates": [0.04 if passing else 0.0] * 2000,
        },
    }


class FakeDependencies:
    def __init__(
        self,
        *,
        passing=False,
        computation_error=None,
        confirmation_error=None,
        mutate=False,
    ):
        self.passing = passing
        self.computation_error = computation_error
        self.confirmation_error = confirmation_error
        self.mutate = mutate
        self.load_calls = []
        self.distance_counts = []
        self.evaluation_rows = []
        self.fingerprint_calls = 0

    def git_identity(self, _repository):
        return {"commit": "a" * 40, "clean": True}

    def load_confirmation(self, _path, _sha):
        if self.confirmation_error:
            raise ValueError(self.confirmation_error)
        return {"confirmation_id": "formal-v4", "formula_commit": "b" * 40}

    def validate_admission(self, **_kwargs):
        return None

    def load_iteration(self, _run_dir, iteration, *, expected_evidence_version):
        self.load_calls.append((iteration, expected_evidence_version))
        rows = 10
        return SimpleNamespace(
            iteration=iteration,
            original_point_count=12,
            centers=np.column_stack((np.arange(rows), np.zeros(rows), np.zeros(rows))),
            finite_row_indices=np.arange(rows, dtype=np.int64),
            snapshot={
                "A": np.linspace(0.1, 0.9, 12),
                "S": np.linspace(0.9, 0.1, 12),
                "r_p": np.linspace(0.95, 0.05, 12),
                "V_p": np.ones(12, dtype=np.bool_),
            },
        )

    def load_mesh(self, _path):
        return object()

    def distances(self, centers, _mesh):
        self.distance_counts.append(len(centers))
        return np.linspace(0.01, 0.10, len(centers))

    def evaluate_iteration(self, joined, distances, *, iteration, config):
        if self.computation_error:
            raise ProbeInconclusiveError(self.computation_error)
        self.evaluation_rows.append(
            (iteration, len(joined.centers), len(distances), config.fold_count)
        )
        return iteration_summary(iteration, passing=self.passing)

    def fingerprint(self, paths):
        self.fingerprint_calls += 1
        result = {
            name: {"kind": "file", "bytes": 1, "sha256": hashlib.sha256(name.encode()).hexdigest()}
            for name in sorted(paths)
        }
        if self.mutate and self.fingerprint_calls > 1:
            result[sorted(result)[0]]["sha256"] = "0" * 64
        return result

    @staticmethod
    def assert_unchanged(before, after):
        if before != after:
            raise RuntimeError("input mutated during evaluation")


class G1ComplementarityCliTests(unittest.TestCase):
    def make_args(self, root, diagnostic_id="probe-a"):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        run = root / "run"
        source = root / "source"
        output = root / "output"
        run.mkdir()
        source.mkdir()
        gt = root / "mesh.ply"
        confirmation = root / "confirmation.json"
        gt.write_bytes(b"ply\n")
        confirmation.write_text("{}\n", encoding="utf-8")
        return build_parser().parse_args(
            [
                "--run-dir", str(run),
                "--source-root", str(source),
                "--gt-mesh", str(gt),
                "--confirmation-contract", str(confirmation),
                "--output-root", str(output),
                "--diagnostic-id", diagnostic_id,
                "--expected-diagnostic-commit", "a" * 40,
                "--expected-confirmation-sha", "c" * 64,
                "--expected-dataset-sha", "d" * 64,
                "--expected-prior-sha", "e" * 64,
                "--expected-gt-sha", "f" * 64,
            ]
        )

    def test_parser_exposes_only_frozen_inputs_not_analysis_overrides(self):
        parser = build_parser()
        destinations = {action.dest for action in parser._actions}
        for required in (
            "run_dir", "source_root", "gt_mesh", "confirmation_contract",
            "output_root", "diagnostic_id", "expected_diagnostic_commit",
            "expected_confirmation_sha", "expected_dataset_sha",
            "expected_prior_sha", "expected_gt_sha",
        ):
            self.assertIn(required, destinations)
        for forbidden in (
            "candidate", "model", "fold_count", "solver", "threshold",
            "bootstrap_replicates", "crop", "iterations",
        ):
            self.assertNotIn(forbidden, destinations)

    def test_boundary_requires_v4_and_queries_every_finite_center_before_vp(self):
        with tempfile.TemporaryDirectory() as directory:
            dependencies = FakeDependencies()
            code, publication = run_probe(
                self.make_args(directory), dependencies=dependencies
            )

        self.assertEqual(code, 1)
        self.assertEqual(dependencies.load_calls, [(3000, 4), (7000, 4)])
        self.assertEqual(dependencies.distance_counts, [10, 10])
        self.assertEqual(
            dependencies.evaluation_rows,
            [(3000, 10, 10, 5), (7000, 10, 10, 5)],
        )
        self.assertEqual(publication["report"]["outcome"], "NO_CLEAR_COMPLEMENT")
        self.assertFalse(publication["report"]["training_started"])

    def test_publication_is_atomic_compact_and_hash_manifested(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_args(directory)
            code, publication = run_probe(args, dependencies=FakeDependencies(passing=True))
            target = Path(publication["output_dir"])

            self.assertEqual(code, 0)
            self.assertEqual(
                sorted(path.name for path in target.iterdir()),
                sorted((*REQUIRED_ARTIFACTS, "manifest.json")),
            )
            manifest = json.loads((target / "manifest.json").read_text())
            self.assertEqual([row["path"] for row in manifest["files"]], list(REQUIRED_ARTIFACTS))
            for row in manifest["files"]:
                payload = (target / row["path"]).read_bytes()
                self.assertEqual(row["bytes"], len(payload))
                self.assertEqual(row["sha256"], hashlib.sha256(payload).hexdigest())
            with (target / "folds.csv").open(newline="", encoding="utf-8") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 10)
            with (target / "risk_bins.csv").open(newline="", encoding="utf-8") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 40)
            with (target / "bootstrap.csv").open(newline="", encoding="utf-8") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 4000)
            self.assertEqual(list(Path(args.output_root).glob("*.tar.gz")), [])

    def test_valid_negative_and_safe_inconclusive_are_both_published(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            negative_args = self.make_args(root / "negative", "negative")
            negative_code, negative = run_probe(negative_args, dependencies=FakeDependencies())
            inconclusive_args = self.make_args(root / "inconclusive", "inconclusive")
            inconclusive_code, inconclusive = run_probe(
                inconclusive_args,
                dependencies=FakeDependencies(computation_error="classless fold"),
            )
            damaged_args = self.make_args(root / "damaged", "damaged")
            damaged_code, damaged = run_probe(
                damaged_args,
                dependencies=FakeDependencies(
                    confirmation_error="confirmation SHA256 mismatch"
                ),
            )

            self.assertEqual(negative_code, 1)
            self.assertTrue(Path(negative["output_dir"]).is_dir())
            self.assertEqual(inconclusive_code, 2)
            self.assertTrue(Path(inconclusive["output_dir"]).is_dir())
            self.assertEqual(inconclusive["report"]["outcome"], "INCONCLUSIVE")
            self.assertEqual(
                [row["iteration"] for row in inconclusive["report"]["iterations"]],
                [3000, 7000],
            )
            self.assertTrue(all(row["status"] == "INCONCLUSIVE" for row in inconclusive["report"]["iterations"]))
            self.assertEqual(damaged_code, 2)
            self.assertTrue(Path(damaged["output_dir"]).is_dir())
            self.assertIn(
                "confirmation SHA256 mismatch",
                damaged["report"]["inconclusive_reasons"][0],
            )

    def test_overwrite_unsafe_output_and_mutation_leave_no_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.make_args(root / "overwrite")
            target = Path(args.output_root) / args.diagnostic_id
            target.mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                run_probe(args, dependencies=FakeDependencies())

            unsafe = self.make_args(root / "unsafe")
            unsafe.output_root = str(Path(unsafe.run_dir) / "diagnostics")
            with self.assertRaisesRegex(ValueError, "outside immutable"):
                run_probe(unsafe, dependencies=FakeDependencies())

            mutated = self.make_args(root / "mutated")
            with self.assertRaisesRegex(RuntimeError, "input mutated"):
                run_probe(mutated, dependencies=FakeDependencies(mutate=True))
            output = Path(mutated.output_root)
            self.assertFalse((output / mutated.diagnostic_id).exists())
            self.assertEqual(list(output.glob("*.tmp-*")), [])


if __name__ == "__main__":
    unittest.main()
