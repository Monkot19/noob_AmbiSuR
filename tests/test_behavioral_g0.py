"""Dependency-free contracts for the explicit schema-3 G0 gate."""

import copy
import hashlib
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.diagnostics.behavioral_g0 import evaluate_behavioral_report


HARD_PSNR = {
    "name": "evaluation[500:train].psnr",
    "b1": 20.0,
    "b2": 21.0,
    "e0": 20.5,
}
DIAGNOSTIC_MEAN = {
    "name": "capture.xyz.channel_000.mean",
    "b1": 0.0,
    "b2": 1.0,
    "e0": 4.0,
}


def _report():
    return {
        "schema_version": 3,
        "exact_invariants": [],
        "numeric_fields": [],
        "scalar_metrics": [copy.deepcopy(HARD_PSNR)],
        "diagnostic_scalar_metrics": [copy.deepcopy(DIAGNOSTIC_MEAN)],
        "diagnostics": {},
    }


def _numeric_pair(distance):
    return {
        "exact": False,
        "shape_equal": True,
        "rmse": distance,
        "mean_abs": distance,
    }


class BehavioralG0GateTests(unittest.TestCase):
    def test_internal_outlier_is_reported_but_does_not_fail_hard_gate(self):
        # Catches accidentally folding Gaussian summaries back into hard scalars.
        report = _report()

        gate = evaluate_behavioral_report(report, {DIAGNOSTIC_MEAN["name"]})

        self.assertTrue(gate["hard_equivalent"])
        self.assertTrue(gate["equivalent"])
        self.assertEqual(gate["factor"], 2.0)
        self.assertEqual(gate["numeric_failures"], [])
        self.assertEqual(
            [item["name"] for item in gate["diagnostic_results"]],
            [DIAGNOSTIC_MEAN["name"]],
        )
        self.assertEqual(
            [item["name"] for item in gate["diagnostic_outliers"]],
            [DIAGNOSTIC_MEAN["name"]],
        )
        self.assertEqual(
            gate["diagnostic_outlier_counts"],
            {"capture": {"capture.xyz": {"mean": 1}}},
        )
        diagnostic = gate["diagnostic_outliers"][0]
        self.assertEqual(diagnostic["self_distance"], 1.0)
        self.assertEqual(diagnostic["candidate_distance"], 3.0)
        self.assertEqual(diagnostic["nearest"], "b2")
        self.assertEqual(diagnostic["ratio"], 3.0)

    def test_observable_scalar_outlier_remains_hard(self):
        # Catches accidentally demoting PSNR alongside Gaussian summaries.
        report = _report()
        report["scalar_metrics"][0]["e0"] = 24.0

        gate = evaluate_behavioral_report(report, {DIAGNOSTIC_MEAN["name"]})

        self.assertFalse(gate["hard_equivalent"])
        self.assertFalse(gate["equivalent"])
        self.assertEqual(gate["numeric_failures"], [HARD_PSNR["name"]])

    def test_fixed_tensor_outlier_remains_hard(self):
        # Catches an accidental hard/diagnostic split based on tensor shape.
        report = _report()
        report["numeric_fields"] = [
            {
                "name": "application.appear_ab",
                "b1_b2": _numeric_pair(1.0),
                "b1_e0": _numeric_pair(4.0),
                "b2_e0": _numeric_pair(3.0),
            }
        ]

        gate = evaluate_behavioral_report(report, {DIAGNOSTIC_MEAN["name"]})

        self.assertFalse(gate["hard_equivalent"])
        self.assertEqual(gate["numeric_failures"], ["application.appear_ab"])

    def test_exact_invariant_mismatch_remains_hard(self):
        # Catches silently treating provenance failures as diagnostics.
        report = _report()
        report["exact_invariants"] = [
            {
                "name": "launcher_contract.clean",
                "b1": False,
                "b2": False,
                "e0": True,
                "expected": {"b1": False, "b2": False, "e0": False},
            }
        ]

        gate = evaluate_behavioral_report(report, {DIAGNOSTIC_MEAN["name"]})

        self.assertFalse(gate["hard_equivalent"])
        self.assertEqual(gate["exact_failures"], ["launcher_contract.clean"])

    def test_incomplete_duplicate_nonfinite_or_unexpected_diagnostic_is_invalid(self):
        # Catches making diagnostic-only mean optional or accepting corrupt data.
        expected = {DIAGNOSTIC_MEAN["name"]}
        cases = {}
        cases["missing"] = _report()
        cases["missing"]["diagnostic_scalar_metrics"] = []
        cases["duplicate"] = _report()
        cases["duplicate"]["diagnostic_scalar_metrics"].append(
            copy.deepcopy(DIAGNOSTIC_MEAN)
        )
        cases["nan"] = _report()
        cases["nan"]["diagnostic_scalar_metrics"][0]["e0"] = math.nan
        cases["unexpected"] = _report()
        cases["unexpected"]["diagnostic_scalar_metrics"][0]["name"] = (
            "capture.opacity.channel_000.mean"
        )

        for name, report in cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                evaluate_behavioral_report(report, expected)

    def test_schema_and_factor_cannot_be_silently_changed(self):
        # Catches schema-2 promotion and a post-result factor override.
        report = _report()
        report["schema_version"] = 2
        with self.assertRaises(ValueError):
            evaluate_behavioral_report(report, {DIAGNOSTIC_MEAN["name"]})
        with self.assertRaises(TypeError):
            evaluate_behavioral_report(
                _report(), {DIAGNOSTIC_MEAN["name"]}, factor=3.0
            )


def _contract_fixture(root):
    source = root / "canonical_source"
    prior = root / "canonical_prior"
    source.mkdir()
    prior.mkdir()
    (source / "image.bin").write_bytes(b"source-bytes")
    (prior / "points3D.bin").write_bytes(b"prior-bytes")
    runs = {role: root / role for role in ("b1", "b2", "e0")}
    commands = {}
    launchers = {}
    for role, directory in runs.items():
        directory.mkdir()
        for relative in (
            "train.log", "chkpnt8000.pth", "point_cloud/iteration_8000/point_cloud.ply",
            "app_model/iteration_8000/app.pth", "cfg_args", "cfg_opts",
            "g0_run_contract.json", "exit_code.txt", "start_utc.txt",
            "end_utc.txt", "gpu_peak_mib.txt", "launcher.log",
        ):
            path = directory / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{role}:{relative}".encode("ascii"))
        (directory / "start_utc.txt").write_text(
            "2026-09-14T01:00:00Z\n", encoding="utf-8"
        )
        if role == "e0":
            (directory / "resolved_config.json").write_text("{}\n", encoding="utf-8")
            (directory / "run_identity.json").write_text("{}\n", encoding="utf-8")
        commands[role] = f"python train.py --model_path {directory} -r 2"
        launchers[role] = {
            "role": role,
            "command": commands[role],
            "commit": "e0-commit" if role == "e0" else "baseline-commit",
            "dirty": False,
            "dataset_manifest_sha256": "d" * 64,
            "aligned_prior_sha256": "e" * 64,
            "training_config": {
                "semantic_seed": 0,
                "resolution": 2,
                "iterations": 8000,
                "evaluation_iterations": [500, 1000, 5001, 7001, 8000],
            },
        }
    contract = {
        "contract_version": 1,
        "confirmation_id": "g0-schema3-unseen-e0-test",
        "preflight_utc": "2026-09-14T00:00:00Z",
        "e0_output_absent_at_preflight": True,
        "paths": {role: str(directory.resolve()) for role, directory in runs.items()},
        "canonical_source_root": str(source.resolve()),
        "canonical_aligned_prior_root": str(prior.resolve()),
        "canonical_source_tree_sha256": "0" * 64,
        "canonical_prior_tree_sha256": "0" * 64,
        "baseline_artifact_fingerprints": {"b1": "0" * 64, "b2": "0" * 64},
        "commits": {"b1": "baseline-commit", "b2": "baseline-commit", "e0": "e0-commit"},
        "dataset_sha256": "d" * 64,
        "prior_sha256": "e" * 64,
        "seed": 0,
        "resolution": 2,
        "iteration": 8000,
        "evaluation_iterations": [500, 1000, 5001, 7001, 8000],
        "normalized_commands": commands,
    }
    resolved = {"b1": None, "b2": None, "e0": {
        "training_path": "legacy",
        "model": {"resolution": 2},
        "optimization": {"iterations": 8000},
        "core": {"seed": 0},
    }}
    return contract, runs, launchers, resolved


class ConfirmationContractTests(unittest.TestCase):
    def test_loader_requires_a_recorded_matching_content_hash(self):
        # Catches substituting a post-run contract or computing SHA from live input.
        from scripts.diagnostics.behavioral_g0 import load_confirmation_contract

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract, _, _, _ = _contract_fixture(root)
            path = root / "confirmation.json"
            path.write_text(json.dumps(contract, sort_keys=True), encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(load_confirmation_contract(path, digest), contract)
            with self.assertRaises(ValueError):
                load_confirmation_contract(path, "f" * 64)
            with self.assertRaises(ValueError):
                load_confirmation_contract(path, "")
            path.write_text("[]", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_confirmation_contract(
                    path, hashlib.sha256(path.read_bytes()).hexdigest()
                )

    def test_fingerprint_detects_input_mutation_and_supports_preflight_subset(self):
        # Catches missing run files and a preflight that requires unseen E0 output.
        from scripts.diagnostics.behavioral_g0 import fingerprint_immutable_inputs

        with TemporaryDirectory() as temporary:
            contract, runs, _, _ = _contract_fixture(Path(temporary))
            first = fingerprint_immutable_inputs(
                contract, runs, 8000, roles=("b1", "b2")
            )
            self.assertEqual(
                set(first),
                {"canonical_source_tree_sha256", "canonical_prior_tree_sha256",
                 "runs.b1", "runs.b2"},
            )
            self.assertEqual(
                first,
                fingerprint_immutable_inputs(contract, runs, 8000, roles=("b1", "b2")),
            )
            source = Path(contract["canonical_source_root"]) / "image.bin"
            source.write_bytes(b"source-bytEs")
            changed = fingerprint_immutable_inputs(
                contract, runs, 8000, roles=("b1", "b2")
            )
            self.assertNotEqual(
                first["canonical_source_tree_sha256"],
                changed["canonical_source_tree_sha256"],
            )
            run_file = runs["b1"] / "chkpnt8000.pth"
            run_file.write_bytes(b"changed-checkpoint")
            changed_again = fingerprint_immutable_inputs(
                contract, runs, 8000, roles=("b1", "b2")
            )
            self.assertNotEqual(changed["runs.b1"], changed_again["runs.b1"])
            run_file.write_bytes(b"")
            with self.assertRaises((ValueError, FileNotFoundError)):
                fingerprint_immutable_inputs(contract, runs, 8000, roles=("b1",))

    def test_canonical_tree_rejects_symlink_outside_root(self):
        # Catches a hash that follows an unapproved external data target.
        from scripts.diagnostics.behavioral_g0 import fingerprint_immutable_inputs

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract, runs, _, _ = _contract_fixture(root)
            external = root / "outside.bin"
            external.write_bytes(b"outside")
            link = Path(contract["canonical_source_root"]) / "escape.bin"
            try:
                os.symlink(external, link)
            except OSError as error:
                self.skipTest(f"OS disallows symlink creation: {error}")
            with self.assertRaises(ValueError):
                fingerprint_immutable_inputs(contract, runs, 8000, roles=("b1",))

    def test_confirmation_invariants_reject_each_contract_mismatch(self):
        # Catches accepting stale paths, commands, inputs or preflight chronology.
        from scripts.diagnostics.behavioral_g0 import (
            confirmation_invariants, fingerprint_immutable_inputs,
        )
        from scripts.diagnostics.compare_feature_off import evaluate_triplet_report

        with TemporaryDirectory() as temporary:
            contract, runs, launchers, resolved = _contract_fixture(Path(temporary))
            before = fingerprint_immutable_inputs(contract, runs, 8000)
            contract["canonical_source_tree_sha256"] = before["canonical_source_tree_sha256"]
            contract["canonical_prior_tree_sha256"] = before["canonical_prior_tree_sha256"]
            contract["baseline_artifact_fingerprints"] = {
                "b1": before["runs.b1"], "b2": before["runs.b2"]
            }

            def failures(candidate):
                invariants = confirmation_invariants(
                    candidate, runs, launchers, resolved, before,
                    iteration=8000,
                    evaluation_iterations=[500, 1000, 5001, 7001, 8000],
                    expected_baseline_commit="baseline-commit",
                    expected_e0_commit="e0-commit",
                    expected_dataset_sha="d" * 64,
                    expected_prior_sha="e" * 64,
                )
                return evaluate_triplet_report({"exact_invariants": invariants})["exact_failures"]

            self.assertEqual(failures(contract), [])
            changes = {
                "path": ("paths", "e0", str(Path(temporary) / "old_e0")),
                "commit": ("commits", "b1", "wrong"),
                "command": ("normalized_commands", "e0", "python wrong.py"),
                "data": ("dataset_sha256", None, "wrong"),
                "prior": ("prior_sha256", None, "wrong"),
                "source_tree": ("canonical_source_tree_sha256", None, "0" * 64),
                "prior_tree": ("canonical_prior_tree_sha256", None, "0" * 64),
                "baseline_file": ("baseline_artifact_fingerprints", "b1", "0" * 64),
                "seed": ("seed", None, 1),
                "resolution": ("resolution", None, 4),
                "iteration": ("iteration", None, 7000),
                "evaluations": ("evaluation_iterations", None, [500]),
                "absence": ("e0_output_absent_at_preflight", None, False),
                "time_missing": ("preflight_utc", None, None),
            }
            for name, (key, subkey, value) in changes.items():
                candidate = copy.deepcopy(contract)
                if subkey is None:
                    candidate[key] = value
                else:
                    candidate[key][subkey] = value
                with self.subTest(name=name):
                    self.assertTrue(failures(candidate), name)
            (runs["e0"] / "start_utc.txt").write_text(
                "2026-09-13T23:59:00Z\n", encoding="utf-8"
            )
            self.assertTrue(failures(contract))


if __name__ == "__main__":
    unittest.main()
