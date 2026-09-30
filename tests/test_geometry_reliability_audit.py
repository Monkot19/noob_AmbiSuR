import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from reliability.geometry_reliability_audit import (
    audit_exit_code,
    build_geometry_semantic_report,
    validate_geometry_semantic_report,
)
from scripts.diagnostics.audit_geometry_reliability_semantics import (
    build_parser,
    run_audit,
)

import numpy as np


COMMIT = "a" * 40
DATASET_SHA = "b" * 64
PRIOR_SHA = "c" * 64


def semantic_checks(*, state_7000="PASS", immutable="PASS"):
    return [
        {
            "name": "task1_formula_conformance",
            "status": "PASS",
            "expected": "23 focused semantic tests pass",
            "actual": "23 focused semantic tests pass",
        },
        {
            "name": "version4_state_3000",
            "status": "PASS",
            "expected": "r_g == V_g * T_g",
            "actual": "r_g == V_g * T_g",
        },
        {
            "name": "version4_state_7000",
            "status": state_7000,
            "expected": "r_g == V_g * T_g",
            "actual": "r_g mismatch" if state_7000 == "FAIL" else "r_g == V_g * T_g",
        },
        {
            "name": "collector_component_contract",
            "status": "PASS",
            "expected": "finite reliability components and support",
            "actual": "finite reliability components and support",
        },
        {
            "name": "immutable_inputs",
            "status": immutable,
            "expected": "before and after fingerprints match",
            "actual": "input mutation" if immutable != "PASS" else "before and after fingerprints match",
        },
    ]


def provenance():
    return {
        "diagnostic_commit": COMMIT,
        "dataset_sha256": DATASET_SHA,
        "aligned_prior_sha256": PRIOR_SHA,
        "evidence_version": 4,
        "iterations": [3000, 7000],
        "task1_outcome": "NO_SEMANTIC_REPAIR_JUSTIFIED",
        "task1_test_count": 23,
    }


class GeometryReliabilityAuditTests(unittest.TestCase):
    def test_validation_returns_none_after_checking_the_report(self):
        report = build_geometry_semantic_report(semantic_checks(), provenance())

        self.assertIsNone(validate_geometry_semantic_report(report))

    def test_green_semantics_force_no_repair_and_stop_later_tasks(self):
        report = build_geometry_semantic_report(semantic_checks(), provenance())

        self.assertEqual(report["outcome"], "NO_SEMANTIC_REPAIR_JUSTIFIED")
        self.assertEqual(audit_exit_code(report), 0)
        self.assertFalse(report["repair_authorized"])
        self.assertIsNone(report["authorized_evidence_version"])
        self.assertFalse(report["utility_authorized"])
        self.assertFalse(report["c1_authorized"])
        self.assertTrue(report["tasks_4_to_13_stopped"])

    def test_named_formula_or_state_failure_is_the_only_repair_branch(self):
        report = build_geometry_semantic_report(
            semantic_checks(state_7000="FAIL"), provenance()
        )

        self.assertEqual(report["outcome"], "SEMANTIC_DEFECT_CONFIRMED")
        self.assertEqual(audit_exit_code(report), 1)
        self.assertTrue(report["repair_authorized"])
        self.assertEqual(report["authorized_evidence_version"], 5)
        self.assertFalse(report["utility_authorized"])
        self.assertFalse(report["c1_authorized"])

    def test_incomplete_or_mutated_inputs_are_inconclusive_not_a_repair(self):
        report = build_geometry_semantic_report(
            semantic_checks(immutable="INCONCLUSIVE"), provenance()
        )

        self.assertEqual(report["outcome"], "INCONCLUSIVE")
        self.assertEqual(audit_exit_code(report), 2)
        self.assertFalse(report["repair_authorized"])
        self.assertFalse(report["utility_authorized"])

    def test_validation_rejects_gt_direction_checks_and_manual_override(self):
        report = build_geometry_semantic_report(semantic_checks(), provenance())
        report["checks"].append(
            {"name": "gt_direction", "status": "PASS", "expected": "x", "actual": "x"}
        )
        with self.assertRaisesRegex(ValueError, "check inventory"):
            validate_geometry_semantic_report(report)

        report = build_geometry_semantic_report(semantic_checks(), provenance())
        report["manual_override"] = "SEMANTIC_DEFECT_CONFIRMED"
        with self.assertRaisesRegex(ValueError, "field inventory"):
            validate_geometry_semantic_report(report)


class GeometryReliabilityAuditCliTests(unittest.TestCase):
    @staticmethod
    def make_request(root, diagnostic_id="geometry-semantics"):
        root = Path(root)
        repository = root / "repository"
        run = root / "run"
        evidence = run / "d0_evidence"
        source = root / "source"
        prior = source / "sparse_da3_aligned" / "0"
        output = root / "output"
        repository.mkdir()
        evidence.mkdir(parents=True)
        prior.mkdir(parents=True)
        output.mkdir()
        (run / "exit_code.txt").write_text("0\n", encoding="utf-8")
        (run / "resolved_config.json").write_text("{}\n", encoding="utf-8")
        (run / "multi_view.json").write_text("{}\n", encoding="utf-8")
        for iteration in (3000, 7000):
            (run / f"chkpnt{iteration}.pth").write_bytes(b"checkpoint")
            (evidence / f"iteration_{iteration:06d}.npz").write_bytes(b"snapshot")
        (source / "source.bin").write_bytes(b"source")
        (prior / "prior.bin").write_bytes(b"prior")
        args = SimpleNamespace(
            repository=str(repository),
            run_dir=str(run),
            source_root=str(source),
            output_root=str(output),
            diagnostic_id=diagnostic_id,
            expected_commit=COMMIT,
            expected_dataset_sha=DATASET_SHA,
            expected_prior_sha=PRIOR_SHA,
        )
        return args, run, source, output

    @staticmethod
    def dependencies(*, bad_iteration=None, mutate=False):
        calls = {"loads": [], "fingerprints": 0}

        def joined(iteration):
            t_g = np.array([0.2, 0.5, 0.8], dtype=np.float32)
            v_g = np.array([True, False, True], dtype=np.bool_)
            r_g = v_g.astype(np.float32) * t_g
            if iteration == bad_iteration:
                r_g = r_g.copy()
                r_g[0] = 0.9
            return SimpleNamespace(
                original_point_count=3,
                centers=np.zeros((3, 3), dtype=np.float64),
                finite_row_indices=np.arange(3, dtype=np.int64),
                rejected_center_indices=np.empty(0, dtype=np.int64),
                snapshot={"T_g": t_g, "V_g": v_g, "r_g": r_g},
                checkpoint_sha256=str(iteration // 1000) * 64,
                snapshot_sha256=str(iteration // 1000 + 3) * 64,
            )

        def load_iteration(_run, iteration, *, expected_evidence_version=None):
            calls["loads"].append((iteration, expected_evidence_version))
            return joined(iteration)

        def fingerprint(_paths):
            calls["fingerprints"] += 1
            suffix = calls["fingerprints"] if mutate else 1
            return {"immutable": {"sha256": str(suffix) * 64}}

        components = {
            "geometry_multiview": np.array([0.9, 0.7, 0.4]),
            "geometry_depth_normal": np.array([0.8, 0.6, 0.3]),
            "geometry_support_views": np.array([2, 3, 1]),
        }
        deps = SimpleNamespace(
            git_head=lambda _repository: COMMIT,
            canonical_tree_sha256=lambda path: (
                PRIOR_SHA if Path(path).name == "0" else DATASET_SHA
            ),
            fingerprint_inputs=fingerprint,
            load_iteration=load_iteration,
            collect_components=lambda _run, _source, _iteration: components,
        )
        return deps, calls

    def test_parser_exposes_only_frozen_no_gt_inputs(self):
        parser = build_parser()
        destinations = {
            action.dest for action in parser._actions if action.dest != "help"
        }

        self.assertEqual(
            destinations,
            {
                "repository",
                "run_dir",
                "source_root",
                "output_root",
                "diagnostic_id",
                "expected_commit",
                "expected_dataset_sha",
                "expected_prior_sha",
            },
        )
        self.assertTrue(all(action.required for action in parser._actions[1:]))

    def test_audit_requires_version_four_and_publishes_only_three_files(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _run, _source, _output = self.make_request(directory)
            dependencies, calls = self.dependencies()

            exit_code, publication = run_audit(args, dependencies=dependencies)
            names = {path.name for path in publication["output_dir"].iterdir()}
            report = json.loads(
                (publication["output_dir"] / "report.json").read_text(encoding="utf-8")
            )
            manifest = json.loads(
                (publication["output_dir"] / "manifest.json").read_text(encoding="utf-8")
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(calls["loads"], [(3000, 4), (7000, 4)])
        self.assertEqual(names, {"inputs.json", "report.json", "manifest.json"})
        self.assertEqual(report["outcome"], "NO_SEMANTIC_REPAIR_JUSTIFIED")
        self.assertEqual(len(manifest["files"]), 2)
        self.assertNotIn("gt", json.dumps(report).lower())
        self.assertNotIn("mesh", json.dumps(report).lower())

    def test_version_four_state_mismatch_names_the_only_repairable_defect(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _run, _source, _output = self.make_request(directory)
            dependencies, _calls = self.dependencies(bad_iteration=7000)

            exit_code, publication = run_audit(args, dependencies=dependencies)
            report = json.loads(
                (publication["output_dir"] / "report.json").read_text(encoding="utf-8")
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(report["outcome"], "SEMANTIC_DEFECT_CONFIRMED")
        self.assertEqual(report["failed_checks"], ["version4_state_7000"])
        self.assertTrue(report["repair_authorized"])
        self.assertEqual(report["authorized_evidence_version"], 5)

    def test_input_mutation_cleans_staging_and_publishes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _run, _source, output = self.make_request(directory)
            dependencies, _calls = self.dependencies(mutate=True)

            with self.assertRaisesRegex(ValueError, "immutable input"):
                run_audit(args, dependencies=dependencies)

            self.assertFalse((output / args.diagnostic_id).exists())
            self.assertEqual(list(output.glob(f".{args.diagnostic_id}.staging-*")), [])

    def test_output_must_be_outside_run_and_source_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            args, run, _source, _output = self.make_request(directory)
            args.output_root = str(run / "diagnostics")
            dependencies, _calls = self.dependencies()

            with self.assertRaisesRegex(ValueError, "outside run and source"):
                run_audit(args, dependencies=dependencies)
