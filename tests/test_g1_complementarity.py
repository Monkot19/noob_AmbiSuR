from types import SimpleNamespace
import unittest

import numpy as np

from reliability.g1_complementarity import (
    ProbeConfig,
    ProbeDomain,
    ProbeInconclusiveError,
    build_probe_domain,
    column_independence,
    crossfit_comparison,
    fit_logistic,
    make_spatial_folds,
    paired_voxel_bootstrap,
    raw_risk_direction,
    summarize_oof,
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
