import unittest

import numpy as np

from reliability.g1_component_diagnostics import (
    RAW_COMPONENTS,
    SNAPSHOT_COMPONENTS,
    build_component_report,
    risk_bin_rows,
)


def valid_snapshot(rows=20):
    base = np.linspace(0.0, 1.0, rows, dtype=np.float64)
    return {
        "A": base.copy(),
        "S": np.concatenate(
            (np.linspace(0.0, 0.9, rows // 2), np.ones(rows - rows // 2))
        ),
        "N": base.copy(),
        "T_p": 1.0 - base,
        "T_g": base.copy(),
        "K": 1.0 - base,
        "r_p": 1.0 - base,
        "r_g": base.copy(),
    }


def valid_components(rows=20):
    base = np.linspace(0.0, 1.0, rows, dtype=np.float64)
    return {
        "view_count": np.arange(1, rows + 1, dtype=np.float64),
        "s_count": base.copy(),
        "s_angle": base[::-1].copy(),
        "s_raw": np.sqrt(base * base[::-1]),
        "prior_confidence": 1.0 - base,
        "prior_multiview": 1.0 - base / 2.0,
        "prior_support_views": np.arange(rows, dtype=np.float64),
        "geometry_multiview": base.copy(),
        "geometry_depth_normal": base / 2.0,
        "geometry_support_views": np.arange(rows, dtype=np.float64),
        "pg_raw_k": 1.0 - base,
    }


class G1ComponentDiagnosticTests(unittest.TestCase):
    def test_report_is_diagnostic_only_and_preserves_frozen_scope(self):
        report = build_component_report(
            valid_snapshot(),
            valid_components(),
            np.linspace(0.0, 0.10, 20, dtype=np.float64),
            iteration=7000,
            metadata={"checkpoint_sha256": "a" * 64},
        )

        self.assertEqual(report["schema_version"], 1)
        self.assertTrue(report["diagnostic_only"])
        self.assertIsNone(report["g1_decision"])
        self.assertFalse(
            report["historical_geometry_stability_reconstructable"]
        )
        self.assertEqual(report["iteration"], 7000)
        self.assertEqual(report["point_count"], 20)
        self.assertEqual(report["metadata"]["checkpoint_sha256"], "a" * 64)
        self.assertEqual(
            set(report["fields"]),
            set(SNAPSHOT_COMPONENTS) | set(RAW_COMPONENTS),
        )
        self.assertNotIn("g1_pass", report)

    def test_report_uses_hand_checked_fixed_quantiles_and_risk_bins(self):
        report = build_component_report(
            valid_snapshot(),
            valid_components(),
            np.linspace(0.0, 0.19, 20, dtype=np.float64),
            iteration=7000,
            metadata={},
        )

        need = report["fields"]["N"]
        self.assertAlmostEqual(need["quantiles"]["0.50"], 0.5)
        self.assertAlmostEqual(need["pearson_distance"], 1.0)
        self.assertAlmostEqual(need["spearman_distance"], 1.0)
        self.assertEqual(len(need["risk_bins"]), 10)
        self.assertEqual(need["risk_bins"][0]["count"], 2)
        self.assertEqual(need["risk_bins"][0]["high_error_count"], 0)
        self.assertEqual(need["risk_bins"][-1]["high_error_count"], 2)
        self.assertAlmostEqual(need["risk_bins"][0]["mean_distance_m"], 0.005)
        self.assertAlmostEqual(need["risk_bins"][-1]["mean_distance_m"], 0.185)

    def test_report_names_saturation_instead_of_changing_the_gate(self):
        report = build_component_report(
            valid_snapshot(),
            valid_components(),
            np.linspace(0.0, 0.10, 20, dtype=np.float64),
            iteration=7000,
            metadata={},
        )

        saturation = report["s_saturation"]
        self.assertAlmostEqual(saturation[">=0.50"], 0.75)
        self.assertAlmostEqual(saturation[">=0.95"], 0.5)
        self.assertAlmostEqual(saturation["at_maximum"], 0.5)
        self.assertIsNone(report["g1_decision"])

    def test_report_rejects_wrong_iteration_shape_and_nonfinite_values(self):
        snapshot = valid_snapshot()
        components = valid_components()
        distances = np.linspace(0.0, 0.10, 20, dtype=np.float64)

        with self.assertRaisesRegex(ValueError, "iteration 7000"):
            build_component_report(
                snapshot, components, distances, iteration=3000, metadata={}
            )

        wrong_rows = dict(components)
        wrong_rows["view_count"] = np.ones(19)
        with self.assertRaisesRegex(ValueError, "row count"):
            build_component_report(
                snapshot, wrong_rows, distances, iteration=7000, metadata={}
            )

        nonfinite = dict(snapshot)
        nonfinite["A"] = snapshot["A"].copy()
        nonfinite["A"][3] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            build_component_report(
                nonfinite, components, distances, iteration=7000, metadata={}
            )

    def test_risk_rows_are_flat_and_carry_no_pass_decision(self):
        report = build_component_report(
            valid_snapshot(),
            valid_components(),
            np.linspace(0.0, 0.19, 20, dtype=np.float64),
            iteration=7000,
            metadata={},
        )

        rows = risk_bin_rows(report)

        self.assertEqual(len(rows), 190)
        self.assertEqual(
            set(rows[0]),
            {
                "field",
                "bin",
                "count",
                "score_min",
                "score_max",
                "mean_distance_m",
                "high_error_count",
                "high_error_rate",
            },
        )
        self.assertNotIn("g1_pass", rows[0])


if __name__ == "__main__":
    unittest.main()
