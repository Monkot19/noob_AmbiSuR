import copy
import unittest
from unittest import mock

import numpy as np

from reliability.g1_complementarity import ProbeConfig
from reliability.g1_prior_transfer import (
    TRANSFER_OUTCOMES,
    build_transfer_report,
    evaluate_transfer_iteration,
    transfer_exit_code,
    utility_probe_config,
    validate_transfer_report,
)


def _iteration(iteration, *, gain, replicates, coverage=0.95, separation=0.10):
    return {
        "iteration": iteration,
        "role": "primary" if iteration == 7000 else "direction_stability",
        "domain": {"coverage": coverage},
        "direction": {
            "lowest_quintile_high_error_rate": 0.10,
            "highest_quintile_high_error_rate": 0.10 + separation,
            "high_minus_low_error_rate": separation,
            "spearman_risk_distance": 0.12,
        },
        "numerical_independence": {
            "fold_relative_residuals": [0.8, 0.81, 0.82, 0.83, 0.84],
        },
        "crossfit": {"pooled_auroc_gain": gain},
        "bootstrap": {
            "replicate_count": len(replicates),
            "auroc_gain_replicates": list(replicates),
        },
    }


def _seed_results(*, early_gain=-0.40, primary_gains=(0.03, 0.04, 0.05)):
    rows = {}
    for seed, gain in enumerate(primary_gains):
        rows[seed] = {
            3000: _iteration(
                3000,
                gain=early_gain,
                replicates=np.full(2000, early_gain, dtype=np.float64),
            ),
            7000: _iteration(
                7000,
                gain=gain,
                replicates=np.full(2000, gain, dtype=np.float64),
            ),
        }
    return rows


class PriorTransferReuseTests(unittest.TestCase):
    def test_single_seed_delegates_to_frozen_toolroom_evaluator(self):
        sentinel = {"iteration": 7000, "frozen": True}
        with mock.patch(
            "reliability.g1_prior_transfer.frozen_evaluate_iteration",
            return_value=sentinel,
        ) as frozen:
            result = evaluate_transfer_iteration(
                object(), np.array([0.1]), training_seed=2, iteration=7000
            )

        self.assertIs(result, sentinel)
        frozen.assert_called_once()
        args, kwargs = frozen.call_args
        self.assertEqual(args[2:], ())
        self.assertEqual(kwargs["iteration"], 7000)
        config = kwargs["config"]
        self.assertIsInstance(config, ProbeConfig)
        self.assertEqual(config.fold_count, ProbeConfig().fold_count)
        self.assertEqual(config.l2_penalty, ProbeConfig().l2_penalty)
        self.assertEqual(config.bootstrap_replicates, 2000)
        self.assertEqual(config.bootstrap_seed.entropy, [20260930, 7000, 2])

    def test_utility_seed_sequence_is_exact_and_deterministic(self):
        first = utility_probe_config(iteration=3000, training_seed=1)
        second = utility_probe_config(iteration=3000, training_seed=1)
        other = utility_probe_config(iteration=3000, training_seed=2)
        first_rng = np.random.Generator(np.random.PCG64(first.bootstrap_seed))
        second_rng = np.random.Generator(np.random.PCG64(second.bootstrap_seed))
        other_rng = np.random.Generator(np.random.PCG64(other.bootstrap_seed))
        np.testing.assert_array_equal(
            first_rng.integers(0, 2**31, 16),
            second_rng.integers(0, 2**31, 16),
        )
        self.assertFalse(
            np.array_equal(
                np.random.Generator(np.random.PCG64(first.bootstrap_seed)).integers(
                    0, 2**31, 16
                ),
                other_rng.integers(0, 2**31, 16),
            )
        )


class PriorTransferAggregationTests(unittest.TestCase):
    def test_3000_gain_is_descriptive_and_primary_macro_is_paired(self):
        report = build_transfer_report(
            _seed_results(early_gain=-0.40),
            provenance={"confirmation_sha256": "a" * 64},
        )
        self.assertEqual(report["outcome"], "PRIOR_RISK_TRANSFER_SUPPORTED")
        self.assertEqual(report["failed_gates"], [])
        self.assertAlmostEqual(report["macro_7000"]["mean_auroc_gain"], 0.04)
        self.assertAlmostEqual(report["macro_7000"]["bootstrap"]["lower"], 0.04)
        self.assertEqual(
            report["macro_7000"]["bootstrap"]["replicate_count"], 2000
        )
        self.assertFalse(report["c1_authorized"])
        self.assertFalse(report["routing_authorized"])

    def test_primary_gates_are_conjunctive_and_outcomes_are_exhaustive(self):
        cases = []
        low_coverage = _seed_results()
        low_coverage[1][7000]["domain"]["coverage"] = 0.79
        cases.append((low_coverage, "SEED_1_7000_COVERAGE_BELOW_0.80"))

        negative_seed = _seed_results(primary_gains=(0.03, -0.001, 0.05))
        cases.append((negative_seed, "SEED_1_7000_AUROC_GAIN_NEGATIVE"))

        weak_mean = _seed_results(primary_gains=(0.01, 0.01, 0.01))
        cases.append((weak_mean, "7000_MEAN_AUROC_GAIN_BELOW_0.02"))

        weak_direction = _seed_results()
        weak_direction[0][3000]["direction"]["spearman_risk_distance"] = -0.01
        cases.append((weak_direction, "SEED_0_3000_RAW_RISK_SPEARMAN_NEGATIVE"))

        for rows, expected_gate in cases:
            with self.subTest(expected_gate=expected_gate):
                report = build_transfer_report(rows, provenance={"x": "y"})
                self.assertEqual(report["outcome"], "NO_CROSS_SCENE_REPLICATION")
                self.assertIn(expected_gate, report["failed_gates"])
                self.assertEqual(transfer_exit_code(report), 1)

        inconclusive = build_transfer_report(
            {}, provenance={"x": "y"}, inconclusive_reasons=["solver failed"]
        )
        self.assertEqual(inconclusive["outcome"], "INCONCLUSIVE")
        self.assertEqual(transfer_exit_code(inconclusive), 2)
        self.assertEqual(set(TRANSFER_OUTCOMES), {
            "PRIOR_RISK_TRANSFER_SUPPORTED",
            "NO_CROSS_SCENE_REPLICATION",
            "INCONCLUSIVE",
        })

    def test_rejects_wrong_inventory_nonfinite_or_misaligned_bootstrap(self):
        invalid = _seed_results()
        invalid.pop(2)
        with self.assertRaisesRegex(ValueError, "seed inventory"):
            build_transfer_report(invalid, provenance={"x": "y"})

        invalid = _seed_results()
        invalid[0].pop(3000)
        with self.assertRaisesRegex(ValueError, "iteration inventory"):
            build_transfer_report(invalid, provenance={"x": "y"})

        invalid = _seed_results()
        invalid[1][7000]["bootstrap"]["auroc_gain_replicates"][5] = float("nan")
        with self.assertRaisesRegex(ValueError, "bootstrap"):
            build_transfer_report(invalid, provenance={"x": "y"})

        invalid = _seed_results()
        invalid[2][7000]["bootstrap"]["auroc_gain_replicates"].pop()
        invalid[2][7000]["bootstrap"]["replicate_count"] = 1999
        with self.assertRaisesRegex(ValueError, "2000"):
            build_transfer_report(invalid, provenance={"x": "y"})

    def test_validation_rejects_scope_gate_and_aggregation_mutations(self):
        report = build_transfer_report(
            _seed_results(), provenance={"confirmation_sha256": "a" * 64}
        )
        validate_transfer_report(report)
        for path, value in (
            (("candidate", "expression"), "r_p"),
            (("baseline",), ["N"]),
            (("c1_authorized",), True),
            (("routing_authorized",), True),
            (("gates", "mean_auroc_gain_at_least"), 0.01),
            (("macro_7000", "mean_auroc_gain"), 0.5),
        ):
            changed = copy.deepcopy(report)
            target = changed
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    validate_transfer_report(changed)


if __name__ == "__main__":
    unittest.main()
