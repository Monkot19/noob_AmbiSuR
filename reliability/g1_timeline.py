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
