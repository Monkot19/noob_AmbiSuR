"""Detached temporal diagnostics for topology-aligned D0 stable states."""

import math

import torch

from reliability.topology import migrate_lineage_tensor


STATE_NAMES = (
    "Bypass",
    "Consensus",
    "Prior-led",
    "Geometry-led",
    "Abstain",
)

TRANSITION_SUMMARY_FIELDS = {
    "transition_count_matrix",
    "transition_fraction_matrix",
    "jitter_count",
    "jitter_rate",
    "mean_stable_age_refreshes",
    "mean_stable_age_refreshes_by_state",
    "mean_stable_transition_count",
    "mean_stable_transition_count_by_state",
}


def _finite_nonnegative(value, name, *, allow_none=False):
    if value is None and allow_none:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative number")


def validate_transition_summary(summary, point_count):
    """Validate one JSON-safe temporal summary before persistence."""
    if not isinstance(summary, dict) or set(summary) != TRANSITION_SUMMARY_FIELDS:
        raise ValueError("invalid temporal transition summary fields")
    if not isinstance(point_count, int) or point_count <= 0:
        raise ValueError("point_count must be positive")

    counts = summary["transition_count_matrix"]
    fractions = summary["transition_fraction_matrix"]
    if (
        not isinstance(counts, list)
        or len(counts) != len(STATE_NAMES)
        or any(not isinstance(row, list) or len(row) != len(STATE_NAMES) for row in counts)
    ):
        raise ValueError("transition_count_matrix must have shape [5, 5]")
    if (
        not isinstance(fractions, list)
        or len(fractions) != len(STATE_NAMES)
        or any(not isinstance(row, list) or len(row) != len(STATE_NAMES) for row in fractions)
    ):
        raise ValueError("transition_fraction_matrix must have shape [5, 5]")

    count_total = 0
    fraction_total = 0.0
    for row_index, (count_row, fraction_row) in enumerate(
        zip(counts, fractions)
    ):
        for column_index, (count, fraction) in enumerate(
            zip(count_row, fraction_row)
        ):
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("transition counts must be nonnegative integers")
            _finite_nonnegative(fraction, "transition fraction")
            expected = count / float(point_count)
            if not math.isclose(float(fraction), expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    "transition fraction does not match transition count"
                )
            count_total += count
            fraction_total += float(fraction)
    if count_total != point_count:
        raise ValueError("transition count total must equal point_count")
    if not math.isclose(fraction_total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("transition fractions must sum to one")

    jitter_count = summary["jitter_count"]
    if (
        isinstance(jitter_count, bool)
        or not isinstance(jitter_count, int)
        or not 0 <= jitter_count <= point_count
    ):
        raise ValueError("jitter_count must be within the point population")
    _finite_nonnegative(summary["jitter_rate"], "jitter_rate")
    if not math.isclose(
        float(summary["jitter_rate"]),
        jitter_count / float(point_count),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("jitter_rate does not match jitter_count")

    for name in (
        "mean_stable_age_refreshes",
        "mean_stable_transition_count",
    ):
        _finite_nonnegative(summary[name], name)
    for name in (
        "mean_stable_age_refreshes_by_state",
        "mean_stable_transition_count_by_state",
    ):
        values = summary[name]
        if not isinstance(values, dict) or set(values) != set(STATE_NAMES):
            raise ValueError(f"{name} must use the five frozen state names")
        for state, value in values.items():
            _finite_nonnegative(value, f"{name}.{state}", allow_none=True)


class TemporalTransitionDiagnostics:
    """Track stable-state age and changes along the mapped lineage."""

    STATE_VERSION = 1

    def __init__(self, point_count, *, device=None):
        if point_count <= 0:
            raise ValueError("point_count must be positive")
        self.point_count = int(point_count)
        self.stable_age_refreshes = torch.zeros(
            self.point_count, dtype=torch.int64, device=device
        )
        self.stable_transition_count = torch.zeros(
            self.point_count, dtype=torch.int64, device=device
        )
        self.previous_stable = torch.zeros(
            self.point_count, dtype=torch.int8, device=device
        )

    def _validate_states(self, states, name):
        if not isinstance(states, torch.Tensor):
            raise ValueError(f"{name} must be a tensor")
        if states.dtype != torch.int8:
            raise ValueError(f"{name} must use int8 state values")
        if states.shape != (self.point_count,):
            raise ValueError(
                f"{name} shape must be ({self.point_count},)"
            )
        if states.device != self.stable_age_refreshes.device:
            raise ValueError(f"{name} must share the tracker device")
        detached = states.detach()
        if bool(((detached < 0) | (detached >= len(STATE_NAMES))).any()):
            raise ValueError(f"{name} values must be in [0, 4]")
        return detached

    @staticmethod
    def _mean_or_none(values, mask):
        if not bool(mask.any()):
            return None
        return float(values[mask].to(torch.float64).mean().item())

    @torch.no_grad()
    def update(self, previous_stable, current_stable):
        previous = self._validate_states(previous_stable, "previous_stable")
        current = self._validate_states(current_stable, "current_stable")
        if not torch.equal(previous, self.previous_stable):
            raise ValueError(
                "previous_stable must match topology-aligned temporal history"
            )
        changed = current != previous
        self.stable_age_refreshes.copy_(
            torch.where(
                changed,
                torch.ones_like(self.stable_age_refreshes),
                self.stable_age_refreshes + 1,
            )
        )
        self.stable_transition_count.add_(changed.to(torch.int64))

        flattened = previous.to(torch.int64) * len(STATE_NAMES)
        flattened = flattened + current.to(torch.int64)
        counts = torch.bincount(
            flattened, minlength=len(STATE_NAMES) ** 2
        ).reshape(len(STATE_NAMES), len(STATE_NAMES))
        fractions = counts.to(torch.float64) / float(self.point_count)

        mean_age_by_state = {}
        mean_transition_by_state = {}
        for state, name in enumerate(STATE_NAMES):
            mask = current == state
            mean_age_by_state[name] = self._mean_or_none(
                self.stable_age_refreshes, mask
            )
            mean_transition_by_state[name] = self._mean_or_none(
                self.stable_transition_count, mask
            )

        self.previous_stable.copy_(current)

        return {
            "transition_count_matrix": counts.cpu().tolist(),
            "transition_fraction_matrix": fractions.cpu().tolist(),
            "jitter_count": int(changed.sum().item()),
            "jitter_rate": float(
                changed.to(torch.float64).mean().item()
            ),
            "mean_stable_age_refreshes": float(
                self.stable_age_refreshes.to(torch.float64).mean().item()
            ),
            "mean_stable_age_refreshes_by_state": mean_age_by_state,
            "mean_stable_transition_count": float(
                self.stable_transition_count.to(torch.float64).mean().item()
            ),
            "mean_stable_transition_count_by_state": (
                mean_transition_by_state
            ),
        }

    @torch.no_grad()
    def on_topology_change(self, change):
        self.stable_age_refreshes = migrate_lineage_tensor(
            self.stable_age_refreshes, change, fill_value=0
        )
        self.stable_transition_count = migrate_lineage_tensor(
            self.stable_transition_count, change, fill_value=0
        )
        self.previous_stable = migrate_lineage_tensor(
            self.previous_stable, change, fill_value=0
        )
        self.point_count = int(change.new_to_old.shape[0])

    def state_dict(self):
        return {
            "version": self.STATE_VERSION,
            "point_count": self.point_count,
            "stable_age_refreshes": self.stable_age_refreshes.clone(),
            "stable_transition_count": self.stable_transition_count.clone(),
            "previous_stable": self.previous_stable.clone(),
        }

    @torch.no_grad()
    def load_state_dict(self, state):
        if state.get("version") != self.STATE_VERSION:
            raise ValueError("unsupported temporal transition state version")
        if state.get("point_count") != self.point_count:
            raise ValueError("temporal transition point count mismatch")
        expected_keys = {
            "version",
            "point_count",
            "stable_age_refreshes",
            "stable_transition_count",
            "previous_stable",
        }
        if set(state) != expected_keys:
            raise ValueError("invalid temporal transition state fields")
        for name in (
            "stable_age_refreshes",
            "stable_transition_count",
        ):
            value = state[name]
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"{name} must be a tensor")
            if value.dtype != torch.int64 or value.shape != (self.point_count,):
                raise ValueError(
                    f"{name} must be an int64 point-count vector"
                )
            if bool((value < 0).any()):
                raise ValueError(f"{name} must be nonnegative")
            getattr(self, name).copy_(
                value.detach().to(device=self.stable_age_refreshes.device)
            )
        previous = state["previous_stable"]
        if not isinstance(previous, torch.Tensor):
            raise ValueError("previous_stable must be a tensor")
        if previous.dtype != torch.int8 or previous.shape != (self.point_count,):
            raise ValueError("previous_stable must be an int8 point-count vector")
        previous = previous.detach().to(device=self.previous_stable.device)
        if bool(((previous < 0) | (previous >= len(STATE_NAMES))).any()):
            raise ValueError("previous_stable values must be in [0, 4]")
        self.previous_stable.copy_(previous)
