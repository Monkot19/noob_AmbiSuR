import math
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "Torch is required for checkpoint audit tests")
class FeatureOffTripletAuditTests(unittest.TestCase):
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
