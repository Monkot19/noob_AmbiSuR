from types import SimpleNamespace
import unittest

import numpy as np

from reliability.g1_complementarity import (
    ProbeConfig,
    ProbeDomain,
    ProbeInconclusiveError,
    bootstrap_rows,
    build_probe_domain,
    build_probe_report,
    column_independence,
    crossfit_comparison,
    fit_logistic,
    fold_rows,
    make_spatial_folds,
    paired_voxel_bootstrap,
    probe_exit_code,
    raw_risk_direction,
    risk_bin_rows,
    summarize_oof,
    validate_probe_report,
)


def joined_fixture():
    original_count = 9
    finite_rows = np.array([0, 2, 3, 5, 6, 8], dtype=np.int64)
    centers = np.array(
        [
            [-0.8, 0.0, 0.0],
            [-0.2, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.8, 0.0, 0.0],
            [1.2, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    snapshot = {
        "A": np.linspace(0.1, 0.9, original_count, dtype=np.float32),
        "S": np.linspace(0.9, 0.1, original_count, dtype=np.float32),
        "r_p": np.linspace(0.95, 0.15, original_count, dtype=np.float32),
        "V_p": np.array(
            [True, True, False, True, True, True, True, False, True],
            dtype=np.bool_,
        ),
        # These fields deliberately contain extreme values. They are not
        # permitted to influence the comparison domain.
        "opacity": np.array([0.0, 1.0] * 4 + [0.0], dtype=np.float32),
        "scaling": np.arange(original_count, dtype=np.float32),
    }
    return SimpleNamespace(
        iteration=7000,
        original_point_count=original_count,
        centers=centers,
        finite_row_indices=finite_rows,
        rejected_center_indices=np.array([1, 4, 7], dtype=np.int64),
        snapshot=snapshot,
    )


def synthetic_domain(rows=40):
    index = np.arange(rows, dtype=np.int64)
    x = np.linspace(-2.0, 2.0, rows)
    centers = np.column_stack((x, np.sin(x), np.cos(x)))
    labels = (index % 2 == 0)
    a = 0.2 + 0.6 * (index / max(1, rows - 1))
    one_minus_s = 0.15 + 0.25 * np.sin(index * 0.7) ** 2
    prior_risk = np.where(labels, 0.75, 0.25) + 0.02 * np.cos(index)
    distances = np.where(labels, 0.08, 0.02).astype(np.float64)
    return ProbeDomain(
        iteration=7000,
        original_point_count=rows,
        finite_center_count=rows,
        row_indices=index,
        centers=centers.astype(np.float64),
        distances=distances,
        labels=labels,
        a=a.astype(np.float64),
        one_minus_s=one_minus_s.astype(np.float64),
        prior_risk=prior_risk.astype(np.float64),
        coverage=1.0,
    )


def iteration_summary(iteration, *, passing=True):
    role = "primary" if iteration == 7000 else "direction_stability"
    direction = {
        "lowest_quintile_count": 20,
        "highest_quintile_count": 20,
        "lowest_quintile_high_error_rate": 0.10,
        "highest_quintile_high_error_rate": 0.20 if passing else 0.08,
        "high_minus_low_error_rate": 0.10 if passing else -0.02,
        "spearman_risk_distance": 0.15 if passing else -0.01,
        "risk_bins": [
            {
                "bin": index,
                "count": 5,
                "risk_min": index / 20.0,
                "risk_max": (index + 1) / 20.0,
                "mean_distance_m": 0.02 + index * 0.002,
                "high_error_rate": 0.05 + index * 0.01,
            }
            for index in range(20)
        ],
    }
    folds = [
        {
            "fold": fold,
            "training_count": 80,
            "validation_count": 20,
            "baseline_auroc": 0.55,
            "augmented_auroc": 0.59,
            "auroc_gain": 0.04,
            "candidate_relative_residual": 0.25,
            "positive_weight": 1.0,
            "negative_weight": 1.0,
        }
        for fold in range(5)
    ]
    return {
        "iteration": iteration,
        "role": role,
        "domain": {
            "original_point_count": 110,
            "finite_center_count": 100,
            "eligible_count": 90 if passing else 79,
            "positive_count": 30,
            "negative_count": 60 if passing else 49,
            "coverage": 0.90 if passing else 0.79,
            "label": "distance_gt_0.05_m",
        },
        "direction": direction,
        "numerical_independence": {
            "threshold_strictly_greater_than": 1e-8,
            "fold_relative_residuals": [0.25] * 5,
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
            "voxel_count": 25,
            "lower": 0.04 if passing else 0.0,
            "upper": 0.04 if passing else 0.0,
            "auroc_gain_replicates": [0.04 if passing else 0.0] * 2000,
        },
    }


def probe_provenance():
    return {
        "diagnostic_commit": "a" * 40,
        "formula_commit": "b" * 40,
        "confirmation_id": "soft_v4_formal_toolroom_seed0",
        "confirmation_sha256": "c" * 64,
        "dataset_sha256": "d" * 64,
        "aligned_prior_sha256": "e" * 64,
        "gt_mesh_sha256": "f" * 64,
        "run_identity_sha256": "1" * 64,
    }


class G1ComplementarityDomainTests(unittest.TestCase):
    def test_domain_uses_only_finite_joined_prior_valid_rows(self):
        joined = joined_fixture()
        distances = np.array([0.01, 0.02, 0.08, 0.03, 0.09, 0.10])

        domain = build_probe_domain(joined, distances, iteration=7000)

        np.testing.assert_array_equal(domain.row_indices, [0, 3, 5, 6, 8])
        np.testing.assert_allclose(domain.distances, [0.01, 0.08, 0.03, 0.09, 0.10])
        np.testing.assert_array_equal(domain.labels, [False, True, False, True, True])
        np.testing.assert_allclose(
            domain.prior_risk,
            1.0 - joined.snapshot["r_p"][domain.row_indices],
        )
        self.assertEqual(domain.finite_center_count, 6)
        self.assertEqual(domain.coverage, 5 / 6)

        changed = joined_fixture()
        changed.snapshot["opacity"][:] = 999.0
        changed.snapshot["scaling"][:] = -999.0
        second = build_probe_domain(changed, distances, iteration=7000)
        np.testing.assert_array_equal(second.row_indices, domain.row_indices)

    def test_domain_rejects_misalignment_nonboolean_validity_and_nonfinite_selected(self):
        joined = joined_fixture()
        distances = np.arange(6, dtype=np.float64)
        cases = []

        bad_rows = joined_fixture()
        bad_rows.finite_row_indices = np.array([0, 2, 3, 5, 6], dtype=np.int64)
        cases.append((bad_rows, distances, "finite row"))

        bad_valid = joined_fixture()
        bad_valid.snapshot["V_p"] = bad_valid.snapshot["V_p"].astype(np.int8)
        cases.append((bad_valid, distances, "V_p"))

        bad_value = joined_fixture()
        bad_value.snapshot["A"][0] = np.nan
        cases.append((bad_value, distances, "nonfinite"))

        for candidate, values, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    build_probe_domain(candidate, values, iteration=7000)

    def test_spatial_folds_are_complete_deterministic_and_tie_safe(self):
        row_indices = np.array([9, 3, 8, 1, 7, 2, 6, 0, 5, 4])
        centers = np.column_stack(
            (
                np.array([-1, -1, -0.5, -0.5, 0, 0, 0.5, 0.5, 1, 1]),
                np.zeros(10),
                np.linspace(0.0, 0.1, 10),
            )
        )

        first = make_spatial_folds(centers, row_indices, fold_count=5)
        second = make_spatial_folds(centers, row_indices, fold_count=5)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(sorted(np.bincount(first).tolist()), [2, 2, 2, 2, 2])
        self.assertEqual(set(first.tolist()), set(range(5)))
        ordered = np.lexsort((row_indices, centers[:, 0]))
        for fold, expected in enumerate(np.array_split(ordered, 5)):
            np.testing.assert_array_equal(np.sort(np.flatnonzero(first == fold)), np.sort(expected))

    def test_raw_direction_uses_candidate_order_and_tie_aware_spearman(self):
        risk = np.repeat(np.arange(5, dtype=np.float64), 4)
        distances = np.repeat([0.01, 0.03, 0.06, 0.08, 0.12], 4)
        rows = np.arange(20, dtype=np.int64)[::-1]

        summary = raw_risk_direction(risk, distances, rows)

        self.assertEqual(summary["lowest_quintile_count"], 4)
        self.assertEqual(summary["highest_quintile_count"], 4)
        self.assertEqual(summary["lowest_quintile_high_error_rate"], 0.0)
        self.assertEqual(summary["highest_quintile_high_error_rate"], 1.0)
        self.assertEqual(summary["high_minus_low_error_rate"], 1.0)
        self.assertGreater(summary["spearman_risk_distance"], 0.9)

    def test_column_independence_rejects_dependent_and_keeps_independent_column(self):
        baseline = np.column_stack(
            (np.linspace(-1.0, 1.0, 8), np.array([0, 1] * 4, dtype=np.float64))
        )
        dependent = 2.0 * baseline[:, 0] - 3.0 * baseline[:, 1] + 4.0
        independent = np.array([0, 0, 1, 1, 4, 4, 9, 9], dtype=np.float64)

        dependent_ratio = column_independence(baseline, dependent)
        independent_ratio = column_independence(baseline, independent)

        self.assertLessEqual(dependent_ratio, 1e-8)
        self.assertGreater(independent_ratio, 1e-8)


class G1ComplementarityModelTests(unittest.TestCase):
    def test_logistic_fit_is_deterministic_finite_and_uses_frozen_contract(self):
        config = ProbeConfig()
        features = np.array(
            [[-2.0], [-1.0], [-0.5], [0.5], [1.0], [2.0]], dtype=np.float64
        )
        labels = np.array([False, False, False, True, True, True])

        first = fit_logistic(features, labels, config=config)
        second = fit_logistic(features, labels, config=config)

        np.testing.assert_array_equal(first.coefficients, second.coefficients)
        probabilities = first.predict_proba(features)
        self.assertEqual(probabilities.dtype, np.float64)
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities > 0.0) & (probabilities < 1.0)).all())
        self.assertLess(probabilities[0], probabilities[-1])
        self.assertEqual(config.l2_penalty, 1e-4)
        self.assertEqual(config.max_iterations, 100)
        self.assertEqual(config.gradient_tolerance, 1e-8)
        self.assertEqual(config.armijo_constant, 1e-4)
        self.assertEqual(config.backtracking_factor, 0.5)
        self.assertEqual(config.minimum_step, 2.0 ** -20)

    def test_invalid_model_inputs_and_budget_fail_inconclusive(self):
        config = ProbeConfig()
        valid_x = np.array([[-1.0], [0.0], [1.0], [2.0]], dtype=np.float64)
        valid_y = np.array([False, True, False, True])
        cases = (
            (valid_x, np.zeros(4, dtype=np.bool_), config, "both classes"),
            (
                np.array([[-1.0], [np.nan], [1.0], [2.0]]),
                valid_y,
                config,
                "finite",
            ),
            (valid_x, valid_y, ProbeConfig(max_iterations=0), "converge"),
        )
        for features, labels, candidate_config, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ProbeInconclusiveError, message):
                    fit_logistic(features, labels, config=candidate_config)

    def test_crossfit_uses_identical_rows_and_fold_local_preprocessing(self):
        domain = synthetic_domain()
        folds = make_spatial_folds(domain.centers, domain.row_indices, fold_count=5)

        result = crossfit_comparison(domain, folds, config=ProbeConfig())

        np.testing.assert_array_equal(result.fold_assignments, folds)
        self.assertTrue(np.isfinite(result.baseline_oof).all())
        self.assertTrue(np.isfinite(result.augmented_oof).all())
        self.assertEqual(result.baseline_oof.shape, domain.labels.shape)
        self.assertEqual(result.augmented_oof.shape, domain.labels.shape)
        self.assertEqual(len(result.folds), 5)
        for fold in result.folds:
            self.assertEqual(fold["baseline_validation_rows"], fold["augmented_validation_rows"])
            train = folds != fold["fold"]
            np.testing.assert_allclose(
                fold["baseline_feature_mean"],
                np.column_stack((domain.a, domain.one_minus_s))[train].mean(axis=0),
            )
            self.assertEqual(
                fold["positive_weight"],
                int(train.sum()) / (2 * int(domain.labels[train].sum())),
            )


class G1ComplementarityReportTests(unittest.TestCase):
    def report(self, *, passing=True, inconclusive_reasons=()):
        return build_probe_report(
            [
                iteration_summary(3000, passing=passing),
                iteration_summary(7000, passing=passing),
            ],
            provenance=probe_provenance(),
            config=ProbeConfig(),
            inconclusive_reasons=list(inconclusive_reasons),
        )

    def test_exact_report_schema_preserves_diagnostic_only_scope(self):
        report = self.report()

        self.assertEqual(
            set(report),
            {
                "schema_version",
                "diagnostic_only",
                "training_started",
                "candidate",
                "baseline",
                "configuration",
                "gates",
                "provenance",
                "iterations",
                "outcome",
                "failed_gates",
                "inconclusive_reasons",
                "c1_authorized",
                "causal_claim",
                "cross_scene_claim",
            },
        )
        self.assertTrue(report["diagnostic_only"])
        self.assertFalse(report["training_started"])
        self.assertEqual(report["candidate"], {"name": "one_minus_r_p", "expression": "1-r_p"})
        self.assertEqual(report["baseline"], ["A", "1-S"])
        self.assertEqual([row["iteration"] for row in report["iterations"]], [3000, 7000])
        self.assertEqual([row["role"] for row in report["iterations"]], ["direction_stability", "primary"])
        self.assertFalse(report["c1_authorized"])
        self.assertIsNone(report["causal_claim"])
        self.assertIsNone(report["cross_scene_claim"])
        self.assertEqual(validate_probe_report(report), report)

    def test_three_outcomes_are_exhaustive_and_have_frozen_exit_codes(self):
        feasible = self.report()
        negative = self.report(passing=False)
        inconclusive = self.report(inconclusive_reasons=["solver did not converge"])

        self.assertEqual(feasible["outcome"], "INDEPENDENT_EVIDENCE_FEASIBLE")
        self.assertEqual(probe_exit_code(feasible), 0)
        self.assertEqual(negative["outcome"], "NO_CLEAR_COMPLEMENT")
        self.assertEqual(probe_exit_code(negative), 1)
        self.assertEqual(inconclusive["outcome"], "INCONCLUSIVE")
        self.assertEqual(probe_exit_code(inconclusive), 2)
        self.assertIn("solver did not converge", inconclusive["inconclusive_reasons"])

    def test_all_gates_are_conjunctive_and_3000_cannot_replace_7000(self):
        early_good = iteration_summary(3000)
        primary_bad = iteration_summary(7000)
        primary_bad["crossfit"]["pooled_auroc_gain"] = 0.019
        primary_bad["crossfit"]["augmented"]["auroc"] = 0.569
        primary_bad["crossfit"]["folds"][0]["auroc_gain"] = 0.50

        report = build_probe_report(
            [early_good, primary_bad],
            provenance=probe_provenance(),
            config=ProbeConfig(),
        )

        self.assertEqual(report["outcome"], "NO_CLEAR_COMPLEMENT")
        self.assertIn("7000_POOLED_AUROC_GAIN_BELOW_0.02", report["failed_gates"])
        self.assertEqual(probe_exit_code(report), 1)

    def test_flat_csv_rows_have_frozen_complete_inventories(self):
        report = self.report()

        folds = fold_rows(report)
        bins = risk_bin_rows(report)
        bootstrap = bootstrap_rows(report)

        self.assertEqual(len(folds), 10)
        self.assertEqual([(row["iteration"], row["fold"]) for row in folds], [(iteration, fold) for iteration in (3000, 7000) for fold in range(5)])
        self.assertEqual(len(bins), 40)
        self.assertEqual([(row["iteration"], row["bin"]) for row in bins], [(iteration, index) for iteration in (3000, 7000) for index in range(20)])
        self.assertEqual(len(bootstrap), 4000)
        self.assertEqual(bootstrap[0], {"iteration": 3000, "replicate": 0, "auroc_gain": 0.04})
        self.assertEqual(bootstrap[-1], {"iteration": 7000, "replicate": 1999, "auroc_gain": 0.04})

    def test_validation_rejects_schema_constant_and_scope_mutations(self):
        mutations = []
        missing = self.report()
        del missing["candidate"]
        mutations.append((missing, "fields"))
        extra = self.report()
        extra["unexpected"] = True
        mutations.append((extra, "fields"))
        candidate = self.report()
        candidate["candidate"] = {"name": "one_minus_r_g", "expression": "1-r_g"}
        mutations.append((candidate, "candidate"))
        role = self.report()
        role["iterations"][0]["role"] = "primary"
        mutations.append((role, "role"))
        nonfinite = self.report()
        nonfinite["iterations"][1]["crossfit"]["pooled_auroc_gain"] = np.nan
        mutations.append((nonfinite, "finite"))
        wrong_bootstrap = self.report()
        wrong_bootstrap["iterations"][1]["bootstrap"]["seed"] = 7
        mutations.append((wrong_bootstrap, "bootstrap"))
        authorized = self.report()
        authorized["c1_authorized"] = True
        mutations.append((authorized, "C1"))
        causal = self.report()
        causal["causal_claim"] = "causal"
        mutations.append((causal, "causal"))

        for report, message in mutations:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validate_probe_report(report)

    def test_crossfit_rejects_zero_scale_or_classless_training_fold(self):
        domain = synthetic_domain()
        folds = make_spatial_folds(domain.centers, domain.row_indices, fold_count=5)
        constant = ProbeDomain(**{**domain.__dict__, "a": np.ones_like(domain.a)})
        with self.assertRaisesRegex(ProbeInconclusiveError, "standard deviation"):
            crossfit_comparison(constant, folds, config=ProbeConfig())

        labels = folds == 0
        classless = ProbeDomain(
            **{
                **domain.__dict__,
                "labels": labels,
                "distances": np.where(labels, 0.08, 0.02),
            }
        )
        with self.assertRaisesRegex(ProbeInconclusiveError, "both classes"):
            crossfit_comparison(classless, folds, config=ProbeConfig())

    def test_pooled_oof_gain_is_not_mean_fold_gain(self):
        labels = np.array([0, 1, 0, 1, 0, 0, 0, 1], dtype=np.bool_)
        folds = np.array([0, 0, 1, 1, 1, 1, 1, 1], dtype=np.int64)
        baseline = np.array([0.1, 0.9, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
        augmented = np.array([0.2, 0.8, 0.1, 0.9, 0.2, 0.3, 0.4, 0.8])

        summary = summarize_oof(labels, baseline, augmented, folds)

        mean_fold_gain = np.mean([row["auroc_gain"] for row in summary["folds"]])
        self.assertAlmostEqual(
            summary["pooled_auroc_gain"],
            summary["augmented"]["auroc"] - summary["baseline"]["auroc"],
        )
        self.assertNotAlmostEqual(summary["pooled_auroc_gain"], mean_fold_gain)

    def test_voxel_bootstrap_is_paired_fixed_origin_and_deterministic(self):
        centers = np.array(
            [
                [-0.75, 0, 0], [-0.70, 0, 0],
                [-0.25, 0, 0], [-0.20, 0, 0],
                [0.25, 0, 0], [0.30, 0, 0],
                [0.75, 0, 0], [0.80, 0, 0],
            ],
            dtype=np.float64,
        )
        labels = np.array([False, True] * 4)
        baseline = np.array([0.4, 0.6, 0.45, 0.55, 0.48, 0.52, 0.49, 0.51])
        augmented = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])

        first = paired_voxel_bootstrap(
            centers, labels, baseline, augmented, config=ProbeConfig()
        )
        second = paired_voxel_bootstrap(
            centers, labels, baseline, augmented, config=ProbeConfig()
        )

        self.assertEqual(first["seed"], 20260928)
        self.assertEqual(first["replicate_count"], 2000)
        self.assertEqual(first["interval_percentiles"], [2.5, 97.5])
        self.assertEqual(len(first["auroc_gain_replicates"]), 2000)
        np.testing.assert_array_equal(
            first["auroc_gain_replicates"], second["auroc_gain_replicates"]
        )
        self.assertEqual(first["voxel_count"], 4)
        self.assertTrue(np.isfinite(first["lower"]))
        self.assertTrue(np.isfinite(first["upper"]))

    def test_voxel_bootstrap_rejects_classless_or_nonfinite_inputs(self):
        centers = np.array([[0.0, 0, 0], [0.1, 0, 0]], dtype=np.float64)
        labels = np.array([False, False])
        scores = np.array([0.1, 0.2])
        with self.assertRaisesRegex(ProbeInconclusiveError, "both classes"):
            paired_voxel_bootstrap(
                centers, labels, scores, scores, config=ProbeConfig()
            )
        with self.assertRaisesRegex(ProbeInconclusiveError, "finite"):
            paired_voxel_bootstrap(
                centers,
                np.array([False, True]),
                np.array([0.1, np.nan]),
                scores,
                config=ProbeConfig(),
            )


if __name__ == "__main__":
    unittest.main()
