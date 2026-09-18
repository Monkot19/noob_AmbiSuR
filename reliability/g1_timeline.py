"""Strict loading for formal no-GT D0 temporal diagnostics."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from reliability.diagnostics import SNAPSHOT_FIELDS
from reliability.transition_diagnostics import (
    STATE_NAMES,
    TRANSITION_SUMMARY_FIELDS,
    validate_transition_summary,
)


EXPECTED_REFRESH_ITERATIONS = (1000, 2000, 3000, 4000, 5000, 6000, 7000)
_EVENT_FIELDS = {
    "schema_version",
    "iteration",
    "point_count",
    "joint_valid_count",
    *TRANSITION_SUMMARY_FIELDS,
}
_VALIDITY_FIELDS = {"V_p", "V_g", "V_pg"}
_STATE_FIELDS = {"candidate", "stable"}


@dataclass(frozen=True)
class D0Timeline:
    iterations: tuple
    events: tuple
    point_counts: np.ndarray
    joint_valid_counts: np.ndarray
    state_counts: np.ndarray
    transition_count_matrices: np.ndarray
    transition_fraction_matrices: np.ndarray


def _contains_forbidden_key(value):
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if "gt" in lowered or "mesh" in lowered:
                return True
            if _contains_forbidden_key(child):
                return True
    elif isinstance(value, list):
        return any(_contains_forbidden_key(child) for child in value)
    return False


def _require_int(value, name, *, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} is below its valid range")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} is above its valid range")
    return value


def _load_snapshot(path, point_count):
    if not path.is_file():
        raise ValueError(f"missing D0 timeline snapshot: {path.stem[-4:]}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(SNAPSHOT_FIELDS):
                raise ValueError("D0 timeline snapshot field mismatch")
            arrays = {name: np.asarray(archive[name]) for name in SNAPSHOT_FIELDS}
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid D0 timeline snapshot: {path.name}") from error

    for name, values in arrays.items():
        if values.shape != (point_count,):
            raise ValueError(f"D0 timeline snapshot {name} shape mismatch")
        if name in _VALIDITY_FIELDS:
            if values.dtype != np.bool_:
                raise ValueError(f"D0 timeline snapshot {name} must be boolean")
        elif name in _STATE_FIELDS:
            if not np.issubdtype(values.dtype, np.integer):
                raise ValueError(f"D0 timeline snapshot {name} must be integer")
            if np.any(values < 0) or np.any(values >= len(STATE_NAMES)):
                raise ValueError(f"D0 timeline snapshot {name} is out of range")
        elif not np.issubdtype(values.dtype, np.number):
            raise ValueError(f"D0 timeline snapshot {name} must be numeric")
        if not np.isfinite(values).all():
            raise ValueError(f"D0 timeline snapshot {name} must be finite")
    return arrays


def load_d0_timeline(run_directory):
    """Load seven current-domain snapshots without cross-refresh row matching."""
    evidence = Path(run_directory) / "d0_evidence"
    event_path = evidence / "events.jsonl"
    if not event_path.is_file():
        raise ValueError("missing D0 timeline events.jsonl")
    try:
        lines = event_path.read_text(encoding="utf-8").splitlines()
        if any(not line.strip() for line in lines):
            raise ValueError("empty D0 timeline event line")
        events = tuple(json.loads(line) for line in lines)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("invalid D0 timeline events.jsonl") from error

    if len(events) != len(EXPECTED_REFRESH_ITERATIONS):
        raise ValueError("D0 timeline requires exactly seven event records")
    observed_iterations = tuple(
        event.get("iteration") if isinstance(event, dict) else None
        for event in events
    )
    if observed_iterations != EXPECTED_REFRESH_ITERATIONS:
        raise ValueError("D0 timeline iterations are missing, duplicate, or out of order")

    point_counts = []
    joint_valid_counts = []
    state_counts = []
    count_matrices = []
    fraction_matrices = []
    for iteration, event in zip(EXPECTED_REFRESH_ITERATIONS, events):
        if not isinstance(event, dict) or set(event) != _EVENT_FIELDS:
            raise ValueError("invalid D0 timeline event fields")
        if _contains_forbidden_key(event):
            raise ValueError("D0 timeline event contains GT or mesh data")
        if event["schema_version"] != 2:
            raise ValueError("D0 timeline event schema must be 2")
        point_count = _require_int(
            event["point_count"], "point_count", minimum=1
        )
        joint_valid_count = _require_int(
            event["joint_valid_count"],
            "joint_valid_count",
            minimum=0,
            maximum=point_count,
        )
        summary = {
            name: event[name] for name in TRANSITION_SUMMARY_FIELDS
        }
        validate_transition_summary(summary, point_count)

        snapshot = _load_snapshot(
            evidence / f"iteration_{iteration:06d}.npz", point_count
        )
        if int(snapshot["V_pg"].sum()) != joint_valid_count:
            raise ValueError("joint_valid_count does not match current snapshot")
        current_state_counts = np.bincount(
            snapshot["stable"].astype(np.int64, copy=False), minlength=5
        )
        current_matrix = np.asarray(
            event["transition_count_matrix"], dtype=np.int64
        )
        if not np.array_equal(current_matrix.sum(axis=0), current_state_counts):
            raise ValueError("transition columns do not match current stable states")

        point_counts.append(point_count)
        joint_valid_counts.append(joint_valid_count)
        state_counts.append(current_state_counts)
        count_matrices.append(current_matrix)
        fraction_matrices.append(
            np.asarray(event["transition_fraction_matrix"], dtype=np.float64)
        )

    return D0Timeline(
        iterations=EXPECTED_REFRESH_ITERATIONS,
        events=events,
        point_counts=np.asarray(point_counts, dtype=np.int64),
        joint_valid_counts=np.asarray(joint_valid_counts, dtype=np.int64),
        state_counts=np.asarray(state_counts, dtype=np.int64),
        transition_count_matrices=np.asarray(count_matrices, dtype=np.int64),
        transition_fraction_matrices=np.asarray(
            fraction_matrices, dtype=np.float64
        ),
    )


def _state_proportion_source(timeline):
    rows = []
    records = []
    for iteration, point_count, counts in zip(
        timeline.iterations, timeline.point_counts, timeline.state_counts
    ):
        fractions = counts.astype(np.float64) / float(point_count)
        records.append(
            {
                "iteration": int(iteration),
                "point_count": int(point_count),
                "counts": {
                    name: int(value) for name, value in zip(STATE_NAMES, counts)
                },
                "fractions": {
                    name: float(value)
                    for name, value in zip(STATE_NAMES, fractions)
                },
            }
        )
        rows.extend(
            (
                int(iteration),
                name,
                int(count),
                float(fraction),
            )
            for name, count, fraction in zip(STATE_NAMES, counts, fractions)
        )
    metadata = {
        "schema_version": 1,
        "iterations": list(timeline.iterations),
        "state_order": list(STATE_NAMES),
        "records": records,
    }
    return rows, metadata


def _transition_source(timeline):
    rows = []
    for iteration, event in zip(timeline.iterations, timeline.events):
        for previous_index, previous_name in enumerate(STATE_NAMES):
            for current_index, current_name in enumerate(STATE_NAMES):
                rows.append(
                    (
                        int(iteration),
                        "transition",
                        previous_name,
                        current_name,
                        "",
                        int(
                            event["transition_count_matrix"][previous_index][
                                current_index
                            ]
                        ),
                        float(
                            event["transition_fraction_matrix"][previous_index][
                                current_index
                            ]
                        ),
                        "",
                    )
                )
        for name in (
            "jitter_count",
            "jitter_rate",
            "mean_stable_age_refreshes",
            "mean_stable_transition_count",
        ):
            rows.append((int(iteration), name, "", "", "", "", "", event[name]))
        for name in (
            "mean_stable_age_refreshes_by_state",
            "mean_stable_transition_count_by_state",
        ):
            rows.extend(
                (
                    int(iteration),
                    name,
                    "",
                    "",
                    state,
                    "",
                    "",
                    event[name][state],
                )
                for state in STATE_NAMES
            )
    metadata = {
        "schema_version": 1,
        "iterations": list(timeline.iterations),
        "state_order": list(STATE_NAMES),
        "matrix_orientation": "previous_rows_current_columns",
        "records": [dict(event) for event in timeline.events],
    }
    return rows, metadata


def _joint_coverage_source(timeline):
    rows = []
    records = []
    for iteration, point_count, valid_count in zip(
        timeline.iterations,
        timeline.point_counts,
        timeline.joint_valid_counts,
    ):
        fraction = float(valid_count) / float(point_count)
        row = (
            int(iteration),
            int(point_count),
            int(valid_count),
            fraction,
        )
        rows.append(row)
        records.append(
            {
                "iteration": row[0],
                "point_count": row[1],
                "joint_valid_count": row[2],
                "joint_valid_fraction": row[3],
            }
        )
    metadata = {
        "schema_version": 1,
        "iterations": list(timeline.iterations),
        "records": records,
    }
    return rows, metadata


def write_timeline_artifacts(timeline, output_directory):
    """Write the three frozen formal timeline bundles and source data."""
    from reliability.g1_visualization import (
        STATE_PALETTE,
        _matplotlib_pyplot,
        _save_figure_bundle,
    )

    if not isinstance(timeline, D0Timeline):
        raise ValueError("timeline must be a validated D0Timeline")
    plt = _matplotlib_pyplot()
    timeline_directory = Path(output_directory) / "timeline"
    written = []

    state_rows, state_metadata = _state_proportion_source(timeline)
    state_fractions = timeline.state_counts.astype(np.float64)
    state_fractions /= timeline.point_counts[:, None]
    figure, axis = plt.subplots(figsize=(6.8, 3.0), constrained_layout=True)
    for state_index, state_name in enumerate(STATE_NAMES):
        axis.plot(
            timeline.iterations,
            state_fractions[:, state_index],
            marker="o",
            markersize=3,
            color=STATE_PALETTE[state_index] / 255.0,
            label=state_name,
        )
    axis.set(
        xlabel="Iteration",
        ylabel="Stable-state fraction",
        ylim=(0.0, 1.0),
    )
    axis.legend(ncol=3)
    written.extend(
        _save_figure_bundle(
            figure,
            timeline_directory,
            "state_proportion",
            ("iteration", "state", "count", "fraction"),
            state_rows,
            state_metadata,
        )
    )
    plt.close(figure)

    transition_rows, transition_metadata = _transition_source(timeline)
    figure, axes = plt.subplots(
        3, 3, figsize=(8.4, 7.8), constrained_layout=True
    )
    for index, (iteration, matrix) in enumerate(
        zip(timeline.iterations, timeline.transition_fraction_matrices)
    ):
        axis = axes.flat[index]
        axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
        axis.set_title(str(iteration))
        axis.set_xticks(range(5), STATE_NAMES, rotation=45, ha="right")
        axis.set_yticks(range(5), STATE_NAMES)
        axis.set_xlabel("Current")
        axis.set_ylabel("Previous")
    summary_axis = axes.flat[7]
    summary_axis.plot(
        timeline.iterations,
        [event["mean_stable_age_refreshes"] for event in timeline.events],
        marker="o",
        label="Mean stable age",
    )
    summary_axis.plot(
        timeline.iterations,
        [event["mean_stable_transition_count"] for event in timeline.events],
        marker="o",
        label="Mean transitions",
    )
    summary_axis.plot(
        timeline.iterations,
        [event["jitter_rate"] for event in timeline.events],
        marker="o",
        label="Jitter rate",
    )
    summary_axis.set(xlabel="Iteration", ylabel="Event statistic")
    summary_axis.legend()
    axes.flat[8].axis("off")
    written.extend(
        _save_figure_bundle(
            figure,
            timeline_directory,
            "state_transition",
            (
                "iteration",
                "record_type",
                "previous_state",
                "current_state",
                "state",
                "count",
                "fraction",
                "value",
            ),
            transition_rows,
            transition_metadata,
        )
    )
    plt.close(figure)

    coverage_rows, coverage_metadata = _joint_coverage_source(timeline)
    figure, axis = plt.subplots(figsize=(6.8, 3.0), constrained_layout=True)
    axis.plot(
        timeline.iterations,
        [row[3] for row in coverage_rows],
        marker="o",
        color="#1F77B4",
    )
    axis.set(
        xlabel="Iteration",
        ylabel="Joint-valid fraction",
        ylim=(0.0, 1.0),
    )
    written.extend(
        _save_figure_bundle(
            figure,
            timeline_directory,
            "joint_coverage",
            (
                "iteration",
                "point_count",
                "joint_valid_count",
                "joint_valid_fraction",
            ),
            coverage_rows,
            coverage_metadata,
        )
    )
    plt.close(figure)
    return tuple(written)
