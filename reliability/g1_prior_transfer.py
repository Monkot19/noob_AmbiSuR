"""Utility transfer adapter around the frozen Tool Room complementarity probe.

This module deliberately owns no model fitting, spatial-fold construction, risk
partitioning, or voxel-bootstrap implementation.  A single Utility seed is
evaluated by the already frozen Tool Room evaluator; this module only supplies
the preregistered seed sequence and aggregates the three seed-local outputs.
"""

from __future__ import annotations

import copy
import math
from dataclasses import replace
from typing import Mapping

import numpy as np

from reliability.g1_complementarity import ProbeConfig
from scripts.diagnostics.probe_g1_prior_complementarity import (
    evaluate_iteration as frozen_evaluate_iteration,
)


TRANSFER_OUTCOMES = (
    "PRIOR_RISK_TRANSFER_SUPPORTED",
    "NO_CROSS_SCENE_REPLICATION",
    "INCONCLUSIVE",
)
TRAINING_SEEDS = (0, 1, 2)
TRANSFER_ITERATIONS = (3000, 7000)
BOOTSTRAP_ROOT_SEED = 20260930

_GATES = {
    "coverage_at_7000_at_least": 0.80,
    "direction_spearman_at_both_at_least": 0.0,
    "direction_high_minus_low_at_both_at_least": 0.0,
    "direction_high_minus_low_at_7000_at_least": 0.05,
    "fold_relative_residual_strictly_greater_than": 1e-8,
    "each_seed_7000_auroc_gain_at_least": 0.0,
    "mean_auroc_gain_at_least": 0.02,
    "macro_bootstrap_95pct_lower_strictly_greater_than": 0.005,
}

_REPORT_FIELDS = {
    "schema_version",
    "diagnostic_only",
    "training_started",
    "candidate",
    "baseline",
    "frozen_probe_reuse",
    "configuration",
    "gates",
    "provenance",
    "seeds",
    "macro_7000",
    "outcome",
    "failed_gates",
    "inconclusive_reasons",
    "c1_authorized",
    "routing_authorized",
    "causal_claim",
    "statistical_independence_claim",
    "broad_generalization_claim",
}


class _IntSeedSequence(np.random.SeedSequence):
    """SeedSequence accepted by PCG64 and serializable by the frozen probe."""

    def __int__(self):
        return int(self.generate_state(1, dtype=np.uint32)[0])


def utility_probe_config(*, iteration: int, training_seed: int) -> ProbeConfig:
    """Return the frozen probe config with only its preregistered RNG changed."""

    iteration = int(iteration)
    training_seed = int(training_seed)
    if iteration not in TRANSFER_ITERATIONS:
        raise ValueError("Utility probe iteration must be 3000 or 7000")
    if training_seed not in TRAINING_SEEDS:
        raise ValueError("Utility training seed must be 0, 1, or 2")
    sequence = _IntSeedSequence(
        [BOOTSTRAP_ROOT_SEED, iteration, training_seed]
    )
    return replace(ProbeConfig(), bootstrap_seed=sequence)


def evaluate_transfer_iteration(
    joined,
    distances,
    *,
    training_seed: int,
    iteration: int,
):
    """Delegate one seed/iteration unchanged to the frozen Tool Room evaluator."""

    config = utility_probe_config(
        iteration=iteration, training_seed=training_seed
    )
    return frozen_evaluate_iteration(
        joined,
        distances,
        iteration=int(iteration),
        config=config,
    )


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _normalize_seed_results(seed_results):
    if not isinstance(seed_results, Mapping) or set(seed_results) != set(
        TRAINING_SEEDS
    ):
        raise ValueError("Utility seed inventory must be exactly 0, 1, 2")
    normalized = []
    for seed in TRAINING_SEEDS:
        iterations = seed_results[seed]
        if not isinstance(iterations, Mapping) or set(iterations) != set(
            TRANSFER_ITERATIONS
        ):
            raise ValueError(
                f"seed {seed} iteration inventory must be exactly 3000 and 7000"
            )
        records = []
        for iteration in TRANSFER_ITERATIONS:
            row = copy.deepcopy(iterations[iteration])
            if not isinstance(row, dict):
                raise ValueError("frozen iteration result must be a mapping")
            expected_role = (
                "primary" if iteration == 7000 else "direction_stability"
            )
            if (
                row.get("iteration") != iteration
                or row.get("role") != expected_role
            ):
                raise ValueError("frozen iteration identity mismatch")
            domain = row.get("domain")
            direction = row.get("direction")
            independence = row.get("numerical_independence")
            crossfit = row.get("crossfit")
            bootstrap = row.get("bootstrap")
            if not all(
                isinstance(value, dict)
                for value in (
                    domain,
                    direction,
                    independence,
                    crossfit,
                    bootstrap,
                )
            ):
                raise ValueError("frozen iteration result is incomplete")
            coverage = _finite_number(domain.get("coverage"), "coverage")
            if not 0.0 <= coverage <= 1.0:
                raise ValueError("coverage is outside the unit interval")
            low = _finite_number(
                direction.get("lowest_quintile_high_error_rate"),
                "lowest-risk error rate",
            )
            high = _finite_number(
                direction.get("highest_quintile_high_error_rate"),
                "highest-risk error rate",
            )
            separation = _finite_number(
                direction.get("high_minus_low_error_rate"),
                "direction separation",
            )
            if not math.isclose(
                separation, high - low, rel_tol=0.0, abs_tol=1e-15
            ):
                raise ValueError("direction separation mismatch")
            _finite_number(
                direction.get("spearman_risk_distance"),
                "direction Spearman",
            )
            residuals = independence.get("fold_relative_residuals")
            if not isinstance(residuals, list) or len(residuals) != 5:
                raise ValueError("frozen fold independence inventory mismatch")
            for residual in residuals:
                _finite_number(residual, "fold independence residual")
            _finite_number(
                crossfit.get("pooled_auroc_gain"), "pooled AUROC gain"
            )
            replicates = bootstrap.get("auroc_gain_replicates")
            if (
                bootstrap.get("replicate_count") != 2000
                or not isinstance(replicates, list)
                or len(replicates) != 2000
            ):
                raise ValueError("bootstrap must contain exactly 2000 replicates")
            for replicate in replicates:
                _finite_number(replicate, "bootstrap replicate")
            records.append(row)
        normalized.append({"training_seed": seed, "iterations": records})
    return normalized


def _by_seed_and_iteration(seeds):
    return {
        seed_row["training_seed"]: {
            row["iteration"]: row for row in seed_row["iterations"]
        }
        for seed_row in seeds
    }


def _macro_primary(seeds):
    indexed = _by_seed_and_iteration(seeds)
    gains = [
        float(indexed[seed][7000]["crossfit"]["pooled_auroc_gain"])
        for seed in TRAINING_SEEDS
    ]
    replicate_matrix = np.asarray(
        [
            indexed[seed][7000]["bootstrap"]["auroc_gain_replicates"]
            for seed in TRAINING_SEEDS
        ],
        dtype=np.float64,
    )
    if replicate_matrix.shape != (3, 2000) or not np.isfinite(
        replicate_matrix
    ).all():
        raise ValueError("bootstrap replicate matrix mismatch")
    macro = replicate_matrix.mean(axis=0)
    lower, upper = np.percentile(macro, (2.5, 97.5))
    return {
        "seed_auroc_gains": gains,
        "mean_auroc_gain": float(np.mean(gains)),
        "bootstrap": {
            "replicate_count": 2000,
            "interval_percentiles": [2.5, 97.5],
            "lower": float(lower),
            "upper": float(upper),
            "auroc_gain_replicates": macro.tolist(),
        },
    }


def _failed_gates(seeds, macro):
    indexed = _by_seed_and_iteration(seeds)
    failures = []
    for seed in TRAINING_SEEDS:
        for iteration in TRANSFER_ITERATIONS:
            row = indexed[seed][iteration]
            direction = row["direction"]
            if (
                direction["highest_quintile_high_error_rate"]
                < direction["lowest_quintile_high_error_rate"]
            ):
                failures.append(
                    f"SEED_{seed}_{iteration}_RAW_RISK_QUINTILE_DIRECTION_NEGATIVE"
                )
            if direction["spearman_risk_distance"] < 0.0:
                failures.append(
                    f"SEED_{seed}_{iteration}_RAW_RISK_SPEARMAN_NEGATIVE"
                )
            if any(
                residual <= _GATES[
                    "fold_relative_residual_strictly_greater_than"
                ]
                for residual in row["numerical_independence"][
                    "fold_relative_residuals"
                ]
            ):
                failures.append(
                    f"SEED_{seed}_{iteration}_CANDIDATE_COLUMN_NOT_INDEPENDENT"
                )
        primary = indexed[seed][7000]
        if primary["domain"]["coverage"] < _GATES[
            "coverage_at_7000_at_least"
        ]:
            failures.append(f"SEED_{seed}_7000_COVERAGE_BELOW_0.80")
        if primary["direction"]["high_minus_low_error_rate"] < _GATES[
            "direction_high_minus_low_at_7000_at_least"
        ]:
            failures.append(
                f"SEED_{seed}_7000_RAW_RISK_SEPARATION_BELOW_0.05"
            )
        if primary["crossfit"]["pooled_auroc_gain"] < _GATES[
            "each_seed_7000_auroc_gain_at_least"
        ]:
            failures.append(f"SEED_{seed}_7000_AUROC_GAIN_NEGATIVE")
    if macro["mean_auroc_gain"] < _GATES["mean_auroc_gain_at_least"]:
        failures.append("7000_MEAN_AUROC_GAIN_BELOW_0.02")
    if macro["bootstrap"]["lower"] <= _GATES[
        "macro_bootstrap_95pct_lower_strictly_greater_than"
    ]:
        failures.append("7000_MACRO_BOOTSTRAP_LOWER_NOT_GREATER_THAN_0.005")
    return failures


def build_transfer_report(
    seed_results,
    *,
    provenance,
    inconclusive_reasons=None,
):
    """Aggregate three frozen seed-local results without refitting anything."""

    reasons = [str(reason) for reason in (inconclusive_reasons or [])]
    if any(not reason for reason in reasons):
        raise ValueError("inconclusive reasons must be nonempty strings")
    seeds = [] if reasons else _normalize_seed_results(seed_results)
    macro = None if reasons else _macro_primary(seeds)
    failures = [] if reasons else _failed_gates(seeds, macro)
    report = {
        "schema_version": 1,
        "diagnostic_only": True,
        "training_started": False,
        "candidate": {"name": "one_minus_r_p", "expression": "1-r_p"},
        "baseline": ["A", "1-S"],
        "frozen_probe_reuse": {
            "module": "scripts.diagnostics.probe_g1_prior_complementarity",
            "function": "evaluate_iteration",
            "model_fold_solver_bootstrap_reimplemented": False,
        },
        "configuration": {
            "training_seeds": list(TRAINING_SEEDS),
            "iterations": list(TRANSFER_ITERATIONS),
            "bootstrap_seed_sequence": [
                BOOTSTRAP_ROOT_SEED,
                "iteration",
                "training_seed",
            ],
            "bootstrap_replicates": 2000,
            "macro_seed_resampling": False,
        },
        "gates": copy.deepcopy(_GATES),
        "provenance": copy.deepcopy(dict(provenance)),
        "seeds": seeds,
        "macro_7000": macro,
        "outcome": (
            "INCONCLUSIVE"
            if reasons
            else (
                "NO_CROSS_SCENE_REPLICATION"
                if failures
                else "PRIOR_RISK_TRANSFER_SUPPORTED"
            )
        ),
        "failed_gates": failures,
        "inconclusive_reasons": reasons,
        "c1_authorized": False,
        "routing_authorized": False,
        "causal_claim": None,
        "statistical_independence_claim": None,
        "broad_generalization_claim": None,
    }
    return validate_transfer_report(report)


def validate_transfer_report(report):
    if not isinstance(report, dict) or set(report) != _REPORT_FIELDS:
        raise ValueError("transfer report fields mismatch")
    if report["schema_version"] != 1:
        raise ValueError("transfer report schema mismatch")
    if report["diagnostic_only"] is not True or report["training_started"] is not False:
        raise ValueError("transfer report scope mismatch")
    if report["candidate"] != {"name": "one_minus_r_p", "expression": "1-r_p"}:
        raise ValueError("transfer candidate mismatch")
    if report["baseline"] != ["A", "1-S"]:
        raise ValueError("transfer baseline mismatch")
    expected_reuse = {
        "module": "scripts.diagnostics.probe_g1_prior_complementarity",
        "function": "evaluate_iteration",
        "model_fold_solver_bootstrap_reimplemented": False,
    }
    if report["frozen_probe_reuse"] != expected_reuse:
        raise ValueError("frozen probe reuse contract mismatch")
    expected_configuration = {
        "training_seeds": [0, 1, 2],
        "iterations": [3000, 7000],
        "bootstrap_seed_sequence": [20260930, "iteration", "training_seed"],
        "bootstrap_replicates": 2000,
        "macro_seed_resampling": False,
    }
    if report["configuration"] != expected_configuration:
        raise ValueError("transfer configuration mismatch")
    if report["gates"] != _GATES:
        raise ValueError("transfer gate constants mismatch")
    if not isinstance(report["provenance"], dict):
        raise ValueError("transfer provenance must be a mapping")
    if report["outcome"] not in TRANSFER_OUTCOMES:
        raise ValueError("transfer outcome mismatch")
    reasons = report["inconclusive_reasons"]
    if not isinstance(reasons, list) or any(
        not isinstance(reason, str) or not reason for reason in reasons
    ):
        raise ValueError("inconclusive reasons mismatch")
    if report["c1_authorized"] is not False or report["routing_authorized"] is not False:
        raise ValueError("C1 and routing authorization are forbidden")
    for name in (
        "causal_claim",
        "statistical_independence_claim",
        "broad_generalization_claim",
    ):
        if report[name] is not None:
            raise ValueError(f"{name} is forbidden")
    if reasons:
        if (
            report["outcome"] != "INCONCLUSIVE"
            or report["seeds"] != []
            or report["macro_7000"] is not None
            or report["failed_gates"] != []
        ):
            raise ValueError("inconclusive report contract mismatch")
        return report
    seed_mapping = {
        row["training_seed"]: {
            iteration["iteration"]: iteration
            for iteration in row["iterations"]
        }
        for row in report["seeds"]
        if isinstance(row, dict)
        and set(row) == {"training_seed", "iterations"}
    }
    normalized = _normalize_seed_results(seed_mapping)
    if normalized != report["seeds"]:
        raise ValueError("seed-local result normalization mismatch")
    expected_macro = _macro_primary(normalized)
    if expected_macro != report["macro_7000"]:
        raise ValueError("macro 7000 aggregation mismatch")
    expected_failures = _failed_gates(normalized, expected_macro)
    if report["failed_gates"] != expected_failures:
        raise ValueError("transfer failed-gate inventory mismatch")
    expected_outcome = (
        "NO_CROSS_SCENE_REPLICATION"
        if expected_failures
        else "PRIOR_RISK_TRANSFER_SUPPORTED"
    )
    if report["outcome"] != expected_outcome:
        raise ValueError("transfer decision mismatch")
    return report


def transfer_exit_code(report):
    validate_transfer_report(report)
    return {
        "PRIOR_RISK_TRANSFER_SUPPORTED": 0,
        "NO_CROSS_SCENE_REPLICATION": 1,
        "INCONCLUSIVE": 2,
    }[report["outcome"]]
