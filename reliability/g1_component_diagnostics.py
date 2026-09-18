"""Diagnostic-only summaries for explaining a failed formal G1 result."""

from collections.abc import Mapping

import numpy as np


QUANTILES = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)
RISK_BIN_COUNT = 10
SNAPSHOT_COMPONENTS = ("A", "S", "N", "T_p", "T_g", "K", "r_p", "r_g")
RAW_COMPONENTS = (
    "view_count",
    "s_count",
    "s_angle",
    "s_raw",
    "prior_confidence",
    "prior_multiview",
    "prior_support_views",
    "geometry_multiview",
    "geometry_depth_normal",
    "geometry_support_views",
    "pg_raw_k",
)


def _vector(name, value, rows):
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a vector")
    if array.shape[0] != rows:
        raise ValueError(f"{name} row count mismatch")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be numeric")
    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _average_ranks(values):
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and ordered[end] == ordered[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _correlation(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = np.sqrt(
        np.dot(left_centered, left_centered)
        * np.dot(right_centered, right_centered)
    )
    if denominator == 0.0:
        return None
    return float(np.dot(left_centered, right_centered) / denominator)


def _risk_bins(scores, distances):
    labels = distances > 0.05
    order = np.argsort(scores, kind="mergesort")
    bins = np.array_split(order, RISK_BIN_COUNT)
    if any(indices.size == 0 for indices in bins):
        raise ValueError("at least ten rows are required for fixed risk bins")
    rows = []
    for index, indices in enumerate(bins):
        selected_scores = scores[indices]
        selected_distances = distances[indices]
        selected_labels = labels[indices]
        rows.append(
            {
                "bin": index,
                "count": int(indices.size),
                "score_min": float(selected_scores.min()),
                "score_max": float(selected_scores.max()),
                "mean_distance_m": float(selected_distances.mean()),
                "high_error_count": int(selected_labels.sum()),
                "high_error_rate": float(selected_labels.mean()),
            }
        )
    return rows


def _field_summary(values, distances):
    quantiles = np.quantile(values, QUANTILES)
    return {
        "count": int(values.size),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "quantiles": {
            f"{quantile:.2f}": float(value)
            for quantile, value in zip(QUANTILES, quantiles)
        },
        "pearson_distance": _correlation(values, distances),
        "spearman_distance": _correlation(
            _average_ranks(values), _average_ranks(distances)
        ),
        "risk_bins": _risk_bins(values, distances),
    }


def build_component_report(
    snapshot,
    components,
    distances,
    *,
    iteration,
    metadata,
):
    """Build a fixed diagnostic report without producing a G1 decision."""
    if int(iteration) != 7000:
        raise ValueError("component diagnosis requires iteration 7000")
    if not isinstance(snapshot, Mapping) or set(snapshot) != set(SNAPSHOT_COMPONENTS):
        raise ValueError("snapshot component fields do not match the fixed contract")
    if not isinstance(components, Mapping) or set(components) != set(RAW_COMPONENTS):
        raise ValueError("raw component fields do not match the fixed contract")
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping")

    distances = np.asarray(distances)
    if distances.ndim != 1:
        raise ValueError("distances must be a vector")
    rows = int(distances.shape[0])
    if rows < RISK_BIN_COUNT:
        raise ValueError("at least ten rows are required")
    distances = _vector("distances", distances, rows)
    if np.any(distances < 0.0):
        raise ValueError("distances must be nonnegative")

    arrays = {}
    for name in SNAPSHOT_COMPONENTS:
        arrays[name] = _vector(name, snapshot[name], rows)
    for name in RAW_COMPONENTS:
        arrays[name] = _vector(name, components[name], rows)

    sufficiency = arrays["S"]
    maximum = sufficiency.max()
    report = {
        "schema_version": 1,
        "diagnostic_only": True,
        "g1_decision": None,
        "historical_geometry_stability_reconstructable": False,
        "iteration": 7000,
        "point_count": rows,
        "primary_error_threshold_m": 0.05,
        "positive_count": int((distances > 0.05).sum()),
        "prevalence": float((distances > 0.05).mean()),
        "metadata": dict(metadata),
        "fields": {
            name: _field_summary(values, distances)
            for name, values in arrays.items()
        },
        "s_saturation": {
            f">={threshold:.2f}": float((sufficiency >= threshold).mean())
            for threshold in (0.50, 0.75, 0.90, 0.95, 0.98, 0.99)
        },
    }
    report["s_saturation"]["at_maximum"] = float(
        (sufficiency == maximum).mean()
    )
    return report


def risk_bin_rows(report):
    """Flatten fixed risk bins for a compact CSV artifact."""
    rows = []
    for field, summary in report["fields"].items():
        for risk_bin in summary["risk_bins"]:
            rows.append({"field": field, **risk_bin})
    return tuple(rows)
