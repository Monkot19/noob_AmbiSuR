"""Dependency-free checks of the audit's metadata and inactive-state contract."""

import subprocess
import sys
import unittest

from scripts.diagnostics.audit_feature_off_triplet import (
    _add_contract_invariants,
    _feature_off_metadata_valid,
    normalize_optimizer,
)
from scripts.diagnostics.compare_feature_off import evaluate_triplet_report


class AuditImportTests(unittest.TestCase):
    def test_metadata_audit_can_be_imported_without_torch(self):
        result = subprocess.run(
            [sys.executable, "-B", "-c", (
                "import sys; sys.modules['torch'] = None; "
                "from scripts.diagnostics.audit_feature_off_triplet import finalize_gate; "
                "assert finalize_gate({'equivalent': True}, exploratory=True)"
                "['g0_equivalent'] is False"
            )],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


class OptimizerContractTests(unittest.TestCase):
    def test_inactive_parameter_group_without_adam_state_is_preserved(self):
        structure, tensors = normalize_optimizer({
            "state": {},
            "param_groups": [{
                "name": "knn_f",
                "params": [1],
                "lr": 0.01,
                "betas": (0.9, 0.999),
            }],
        })

        self.assertEqual(structure["group_names"], ["knn_f"])
        self.assertEqual(structure["groups"]["knn_f"]["state_keys"], [])
        self.assertIsNone(structure["groups"]["knn_f"]["step"])
        self.assertEqual(tensors, {})


class ProvenanceContractTests(unittest.TestCase):
    def _resolved_config(self):
        return {
            "core": {
                "seed": 0,
                "core_shadow_mode": False,
                "enable_observation_calibration": False,
                "enable_dual_reliability": False,
                "enable_abstention": False,
                "enable_parameter_routing": False,
                "enable_gradient_projection": False,
                "enable_reliability_lifecycle": False,
            }
        }

    def test_e0_metadata_uses_the_runtime_writer_field_names(self):
        self.assertTrue(_feature_off_metadata_valid(
            self._resolved_config(),
            {"git_commit": "e0-sha", "git_dirty": False, "seed": 0},
            {"commit": "e0-sha"},
        ))

    def test_confirmation_rejects_required_contract_fields_missing_in_all_runs(self):
        contracts = {
            "b1": {"role": "b1", "commit": "base", "dirty": False, "command": "train"},
            "b2": {"role": "b2", "commit": "base", "dirty": False, "command": "train"},
            "e0": {"role": "e0", "commit": "e0", "dirty": False, "command": "train"},
        }
        invariants = []
        _add_contract_invariants(
            invariants,
            contracts,
            {"b1": None, "b2": None, "e0": self._resolved_config()},
            {
                "b1": None,
                "b2": None,
                "e0": {"git_commit": "e0", "git_dirty": False, "seed": 0},
            },
            exploratory=False,
            expected_baseline_commit="base",
            expected_e0_commit="e0",
            expected_dataset_sha=None,
            expected_prior_sha=None,
        )

        result = evaluate_triplet_report({"exact_invariants": invariants})

        self.assertFalse(result["equivalent"])
        self.assertIn("launcher_contract.python", result["exact_failures"])
        self.assertIn("launcher_contract.training_config", result["exact_failures"])


if __name__ == "__main__":
    unittest.main()
