import math
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "Torch is required for checkpoint audit tests")
class FeatureOffTripletAuditTests(unittest.TestCase):
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
