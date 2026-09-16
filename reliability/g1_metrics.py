import math

import numpy as np


PRIMARY_THRESHOLD_M = 0.05
COVERAGES = np.arange(0.05, 1.0001, 0.05, dtype=np.float64)
STATE_NAMES = (
    "Bypass",
    "Consensus",
    "Prior-led",
    "Geometry-led",
    "Abstain",
)


def _finite_vector(name, values, *, dtype=np.float64):
    values = np.asarray(values, dtype=dtype)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must be a nonempty vector")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite")
    return values


def _average_ranks(values):
    values = _finite_vector("rank values", values)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    return ranks


def _spearman(x, y):
    if len(x) < 2:
        return None
    x_rank = _average_ranks(x)
    y_rank = _average_ranks(y)
    x_centered = x_rank - x_rank.mean()
    y_centered = y_rank - y_rank.mean()
    denominator = math.sqrt(
        float(np.dot(x_centered, x_centered))
        * float(np.dot(y_centered, y_centered))
    )
    if denominator == 0.0:
        return 0.0
    return float(np.dot(x_centered, y_centered) / denominator)


def binary_curves(scores, labels):
    scores = _finite_vector("scores", scores)
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("labels must match scores")
    if not np.issubdtype(labels.dtype, np.bool_):
        raise ValueError("labels must be boolean")
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("binary curves require both classes")

    ranks = _average_ranks(scores)
    auroc = (
        float(ranks[labels].sum()) - positives * (positives + 1) / 2.0
    ) / (positives * negatives)

    original_indices = np.arange(scores.size, dtype=np.int64)
    order = np.lexsort((original_indices, -scores))
    sorted_scores = scores[order]
    sorted_labels = labels[order].astype(np.int64)
    cumulative_true = np.cumsum(sorted_labels)
    cumulative_false = np.cumsum(1 - sorted_labels)
    group_ends = np.flatnonzero(
        np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    )
    true_positive = cumulative_true[group_ends].astype(np.float64)
    false_positive = cumulative_false[group_ends].astype(np.float64)
    recall = true_positive / positives
    precision = true_positive / (true_positive + false_positive)
    auprc = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))

    return {
        "auroc": float(auroc),
        "auprc": auprc,
        "thresholds": np.r_[np.inf, sorted_scores[group_ends]],
        "fpr": np.r_[0.0, false_positive / negatives],
        "tpr": np.r_[0.0, true_positive / positives],
        "recall": np.r_[0.0, recall],
        "precision": np.r_[1.0, precision],
        "positive_count": positives,
        "negative_count": negatives,
    }


def _trapezoid(y, x):
    if len(y) < 2:
        return 0.0
    return float(
        np.sum((y[1:] + y[:-1]) * 0.5 * (x[1:] - x[:-1]))
    )


def fixed_risk_coverage(
    scores,
    distances,
    labels,
    *,
    valid=None,
    direction,
    coverages=COVERAGES,
):
    scores = _finite_vector("risk scores", scores)
    distances = _finite_vector("GT distances", distances)
    labels = np.asarray(labels)
    if scores.shape != distances.shape or labels.shape != scores.shape:
        raise ValueError("risk inputs must have matching shapes")
    if not np.issubdtype(labels.dtype, np.bool_):
        raise ValueError("risk labels must be boolean")
    if valid is None:
        valid = np.ones(scores.size, dtype=np.bool_)
    valid = np.asarray(valid)
    if valid.shape != scores.shape or not np.issubdtype(valid.dtype, np.bool_):
        raise ValueError("risk validity must be a matching boolean vector")
    if not bool(valid.any()):
        raise ValueError("risk coverage has no valid samples")
    scores = scores[valid]
    distances = distances[valid]
    labels = labels[valid]
    coverages = _finite_vector("coverages", coverages)
    if (
        np.any(coverages <= 0.0)
        or np.any(coverages > 1.0)
        or np.any(np.diff(coverages) <= 0.0)
    ):
        raise ValueError("coverages must increase within (0,1]")
    indices = np.arange(scores.size, dtype=np.int64)
    if direction == "retain_low":
        order = np.lexsort((indices, scores))
        priority = -scores
    elif direction == "retain_high":
        order = np.lexsort((indices, -scores))
        priority = scores
    else:
        raise ValueError("direction must be retain_low or retain_high")

    points = []
    mean_risk = []
    binary_risk = []
    for coverage in coverages:
        count = max(1, int(math.ceil(float(coverage) * scores.size)))
        selected = order[:count]
        mean_distance = float(distances[selected].mean())
        high_error_rate = float(labels[selected].mean())
        points.append(
            {
                "coverage": float(coverage),
                "count": count,
                "mean_distance": mean_distance,
                "high_error_rate": high_error_rate,
            }
        )
        mean_risk.append(mean_distance)
        binary_risk.append(high_error_rate)
    mean_risk = np.asarray(mean_risk, dtype=np.float64)
    binary_risk = np.asarray(binary_risk, dtype=np.float64)
    return {
        "direction": direction,
        "valid_count": int(scores.size),
        "points": points,
        "aurc_mean_distance": _trapezoid(mean_risk, coverages),
        "aurc_high_error_rate": _trapezoid(binary_risk, coverages),
        "spearman_priority_distance": _spearman(priority, distances),
        "mean_distance_monotonic_violations": int(
            (np.diff(mean_risk) < 0.0).sum()
        ),
        "high_error_monotonic_violations": int(
            (np.diff(binary_risk) < 0.0).sum()
        ),
    }


def _optional_curves(scores, labels):
    positives = int(labels.sum())
    if positives == 0 or positives == labels.size:
        return {
            "evaluable": False,
            "prevalence": float(labels.mean()),
            "positive_count": positives,
            "curves": None,
        }
    return {
        "evaluable": True,
        "prevalence": float(labels.mean()),
        "positive_count": positives,
        "curves": binary_curves(scores, labels),
    }


def _state_summary(stable, distances, labels):
    stable = np.asarray(stable)
    if stable.ndim != 1 or stable.shape != distances.shape:
        raise ValueError("stable state must match GT distances")
    if not np.issubdtype(stable.dtype, np.integer):
        raise ValueError("stable state must be integer")
    if np.any(stable < 0) or np.any(stable >= len(STATE_NAMES)):
        raise ValueError("stable state id is out of range")
    per_state = {}
    for state_id, name in enumerate(STATE_NAMES):
        mask = stable == state_id
        per_state[name] = {
            "count": int(mask.sum()),
            "fraction": float(mask.mean()),
            "mean_distance": float(distances[mask].mean()) if mask.any() else None,
            "high_error_rate": float(labels[mask].mean()) if mask.any() else None,
        }
    blocking = bool(np.all(stable == 0) or np.all(stable == 4))
    return {"blocking_collapse": blocking, "per_state": per_state}


def _top_fraction_labels(values, fraction):
    values = _finite_vector("top-fraction values", values)
    count = max(1, int(math.ceil(float(fraction) * values.size)))
    indices = np.arange(values.size, dtype=np.int64)
    order = np.lexsort((indices, -values))
    labels = np.zeros(values.size, dtype=np.bool_)
    labels[order[:count]] = True
    return labels


def evaluate_g1_gate(snapshot, distances, iteration):
    distances = _finite_vector("GT distances", distances)
    if np.any(distances < 0.0):
        raise ValueError("GT distances must be nonnegative")
    required = {"A", "S", "N", "r_p", "V_p", "r_g", "V_g", "stable"}
    missing = sorted(required - set(snapshot))
    if missing:
        raise ValueError(f"missing G1 snapshot fields: {missing}")
    arrays = {name: np.asarray(snapshot[name]) for name in required}
    for name, values in arrays.items():
        if values.ndim != 1 or values.shape != distances.shape:
            raise ValueError(f"G1 snapshot shape mismatch: {name}")
        if name not in {"V_p", "V_g", "stable"} and not np.isfinite(values).all():
            raise ValueError(f"nonfinite G1 snapshot field: {name}")
    if not np.issubdtype(arrays["V_p"].dtype, np.bool_) or not np.issubdtype(
        arrays["V_g"].dtype, np.bool_
    ):
        raise ValueError("G1 reliability validity must be boolean")

    labels = distances > PRIMARY_THRESHOLD_M
    primary_optional = _optional_curves(arrays["N"], labels)
    prevalence = primary_optional["prevalence"]
    prevalence_ok = 0.05 <= prevalence <= 0.95
    curves_available = primary_optional["curves"] is not None
    if curves_available:
        curves_n = primary_optional["curves"]
        curves_a = binary_curves(arrays["A"], labels)
        curves_one_minus_s = binary_curves(1.0 - arrays["S"], labels)
        if curves_a["auroc"] >= curves_one_minus_s["auroc"]:
            component_best_name = "A"
            component_best = curves_a["auroc"]
        else:
            component_best_name = "one_minus_S"
            component_best = curves_one_minus_s["auroc"]
        gain = curves_n["auroc"] - component_best
    else:
        curves_n = curves_a = curves_one_minus_s = None
        component_best_name = component_best = gain = None

    risk = {}
    risk_errors = []
    for name, scores, valid, direction in (
        ("N", arrays["N"], None, "retain_low"),
        ("r_p", arrays["r_p"], arrays["V_p"], "retain_high"),
        ("r_g", arrays["r_g"], arrays["V_g"], "retain_high"),
    ):
        try:
            risk[name] = fixed_risk_coverage(
                scores,
                distances,
                labels,
                valid=valid,
                direction=direction,
            )
        except ValueError as exc:
            risk[name] = None
            risk_errors.append(f"{name}: {exc}")

    state = _state_summary(arrays["stable"], distances, labels)
    sensitivity_labels = {
        "0.02m": distances > 0.02,
        "0.10m": distances > 0.10,
        "top20": _top_fraction_labels(distances, 0.20),
    }
    sensitivity = {
        name: _optional_curves(arrays["N"], alternative_labels)
        for name, alternative_labels in sensitivity_labels.items()
    }

    iteration = int(iteration)
    role = (
        "primary_gate"
        if iteration == 7000
        else "early_diagnostic"
        if iteration == 3000
        else "exploratory"
    )
    prediction_pass = bool(
        curves_available
        and curves_n["auroc"] > 0.60
        and gain >= 0.03
    )
    evaluable = bool(prevalence_ok and curves_available and not risk_errors)
    gate_pass = bool(
        evaluable and prediction_pass and not state["blocking_collapse"]
    )
    primary = {
        "threshold_m": PRIMARY_THRESHOLD_M,
        "positive_count": int(labels.sum()),
        "negative_count": int(labels.size - labels.sum()),
        "prevalence": prevalence,
        "prevalence_ok": bool(prevalence_ok),
        "auroc_n": curves_n["auroc"] if curves_available else None,
        "auprc_n": curves_n["auprc"] if curves_available else None,
        "auroc_a": curves_a["auroc"] if curves_available else None,
        "auroc_one_minus_s": (
            curves_one_minus_s["auroc"] if curves_available else None
        ),
        "auroc_component_best": component_best,
        "component_best_name": component_best_name,
        "auroc_gain": gain,
        "prediction_pass": prediction_pass,
        "curves": {
            "N": curves_n,
            "A": curves_a,
            "one_minus_S": curves_one_minus_s,
        },
    }
    return {
        "schema_version": 1,
        "iteration": iteration,
        "role": role,
        "g1_evaluable": evaluable if role == "primary_gate" else None,
        "g1_pass": gate_pass if role == "primary_gate" else None,
        "primary": primary,
        "sensitivity": sensitivity,
        "risk_coverage": risk,
        "risk_errors": risk_errors,
        "state": state,
    }
