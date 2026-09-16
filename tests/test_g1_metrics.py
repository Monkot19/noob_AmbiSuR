import unittest

import numpy as np

from reliability.g1_metrics import (
    COVERAGES,
    binary_curves,
    evaluate_g1_gate,
    fixed_risk_coverage,
)


def valid_snapshot(scores):
    scores = np.asarray(scores, dtype=np.float64)
    count = scores.size
    stable = np.resize(np.array([1, 2, 3], dtype=np.int8), count)
    validity = np.ones(count, dtype=np.bool_)
    reliability = 1.0 - scores
    return {
        "A": np.full(count, 0.5, dtype=np.float64),
        "S": np.full(count, 0.5, dtype=np.float64),
        "N": scores,
        "T_p": reliability,
        "V_p": validity.copy(),
        "r_p": reliability.copy(),
        "T_g": reliability,
        "V_g": validity.copy(),
        "r_g": reliability.copy(),
        "Z_pg": np.ones(count, dtype=np.float64),
        "V_pg": validity.copy(),
        "K": np.full(count, 0.8, dtype=np.float64),
        "delta": np.zeros(count, dtype=np.float64),
        "candidate": stable.copy(),
        "stable": stable,
    }


def separated_case(count, positive_count):
    labels = np.zeros(count, dtype=np.bool_)
    labels[-positive_count:] = True
    distances = np.where(labels, 0.08, 0.01).astype(np.float64)
    scores = np.linspace(0.0, 1.0, count, dtype=np.float64)
    return valid_snapshot(scores), distances


class G1MetricTests(unittest.TestCase):
    def test_binary_curves_are_tie_safe_with_hand_checked_values(self):
        result = binary_curves(
            np.array([0.1, 0.4, 0.35, 0.8]),
            np.array([False, False, True, True]),
        )

        self.assertAlmostEqual(result["auroc"], 0.75)
        self.assertAlmostEqual(result["auprc"], 5.0 / 6.0)

        tied = binary_curves(
            np.ones(4), np.array([False, False, True, True])
        )
        self.assertAlmostEqual(tied["auroc"], 0.5)
        self.assertAlmostEqual(tied["auprc"], 0.5)

    def test_primary_label_uses_strict_five_centimeters(self):
        snapshot = valid_snapshot([0.1, 0.2, 0.8, 0.9])
        report = evaluate_g1_gate(
            snapshot,
            np.array([0.01, 0.05, 0.0500001, 0.12]),
            iteration=7000,
        )

        self.assertEqual(report["primary"]["positive_count"], 2)
        self.assertEqual(report["primary"]["threshold_m"], 0.05)

    def test_prevalence_boundaries_are_inclusive(self):
        for positives, expected in ((1, True), (19, True)):
            snapshot, distances = separated_case(20, positives)
            report = evaluate_g1_gate(snapshot, distances, iteration=7000)
            self.assertEqual(report["g1_evaluable"], expected)

        for positives in (49, 951):
            snapshot, distances = separated_case(1000, positives)
            report = evaluate_g1_gate(snapshot, distances, iteration=7000)
            self.assertFalse(report["g1_evaluable"])
            self.assertFalse(report["g1_pass"])

    def test_iteration_3000_never_returns_a_pass_decision(self):
        snapshot, distances = separated_case(20, 10)

        report = evaluate_g1_gate(snapshot, distances, iteration=3000)

        self.assertEqual(report["role"], "early_diagnostic")
        self.assertIsNone(report["g1_evaluable"])
        self.assertIsNone(report["g1_pass"])

    def test_primary_gate_compares_n_to_the_better_component(self):
        snapshot, distances = separated_case(20, 10)

        report = evaluate_g1_gate(snapshot, distances, iteration=7000)

        self.assertGreater(report["primary"]["auroc_n"], 0.60)
        self.assertGreaterEqual(report["primary"]["auroc_gain"], 0.03)
        self.assertTrue(report["g1_pass"])

    def test_primary_report_names_and_uses_the_better_component(self):
        snapshot = valid_snapshot([0.1, 0.2, 0.8, 0.9])
        snapshot["A"] = np.array([0.1, 0.4, 0.35, 0.8])
        snapshot["S"] = np.full(4, 0.5)

        report = evaluate_g1_gate(
            snapshot,
            np.array([0.01, 0.02, 0.08, 0.12]),
            iteration=7000,
        )

        self.assertEqual(report["primary"]["component_best_name"], "A")
        self.assertAlmostEqual(report["primary"]["auroc_component_best"], 0.75)
        self.assertAlmostEqual(report["primary"]["auroc_gain"], 0.25)

    def test_sensitivity_labels_cannot_change_primary_pass(self):
        snapshot, _ = separated_case(20, 10)
        distances = np.array([0.03] * 10 + [0.08] * 10, dtype=np.float64)
        report = evaluate_g1_gate(snapshot, distances, iteration=7000)

        self.assertTrue(report["g1_pass"])
        self.assertFalse(report["sensitivity"]["0.02m"]["evaluable"])
        self.assertFalse(report["sensitivity"]["0.10m"]["evaluable"])
        self.assertEqual(
            sorted(report["sensitivity"]), ["0.02m", "0.10m", "top20"]
        )

    def test_risk_coverage_uses_frozen_need_and_reliability_directions(self):
        distances = np.array([0.01, 0.02, 0.08, 0.12])
        labels = distances > 0.05

        need = fixed_risk_coverage(
            np.array([0.1, 0.2, 0.8, 0.9]),
            distances,
            labels,
            direction="retain_low",
            coverages=np.array([0.5, 1.0]),
        )
        reliability = fixed_risk_coverage(
            np.array([0.9, 0.8, 0.2, 0.1]),
            distances,
            labels,
            direction="retain_high",
            coverages=np.array([0.5, 1.0]),
        )

        self.assertAlmostEqual(need["points"][0]["mean_distance"], 0.015)
        self.assertEqual(need["points"][0]["high_error_rate"], 0.0)
        self.assertAlmostEqual(
            reliability["points"][0]["mean_distance"], 0.015
        )
        self.assertEqual(reliability["points"][0]["high_error_rate"], 0.0)

    def test_risk_coverage_applies_channel_validity_before_ranking(self):
        result = fixed_risk_coverage(
            scores=np.array([0.9, 0.8, 0.7, 0.6]),
            distances=np.array([0.9, 0.1, 0.2, 0.8]),
            labels=np.array([True, False, False, True]),
            valid=np.array([False, True, True, False]),
            direction="retain_high",
            coverages=np.array([0.5, 1.0]),
        )

        self.assertEqual(result["valid_count"], 2)
        self.assertAlmostEqual(result["points"][0]["mean_distance"], 0.1)

    def test_single_bypass_or_abstain_state_fails_primary_gate(self):
        for state in (0, 4):
            snapshot, distances = separated_case(20, 10)
            snapshot["stable"][:] = state
            report = evaluate_g1_gate(snapshot, distances, iteration=7000)
            self.assertTrue(report["g1_evaluable"])
            self.assertFalse(report["g1_pass"])
            self.assertTrue(report["state"]["blocking_collapse"])

    def test_fixed_coverage_grid_is_five_through_one_hundred_percent(self):
        np.testing.assert_allclose(COVERAGES, np.arange(0.05, 1.01, 0.05))


if __name__ == "__main__":
    unittest.main()
