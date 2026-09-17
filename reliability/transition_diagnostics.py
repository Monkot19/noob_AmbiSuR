"""Detached temporal diagnostics for topology-aligned D0 stable states."""

import torch


STATE_NAMES = (
    "Bypass",
    "Consensus",
    "Prior-led",
    "Geometry-led",
    "Abstain",
)


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
        mapping = change.new_to_old.to(
            device=self.stable_age_refreshes.device
        )
        valid = mapping >= 0
        if bool(valid.any()) and int(mapping[valid].max().item()) >= self.point_count:
            raise ValueError("new_to_old source index is out of range")

        def migrate(values):
            result = torch.zeros(
                mapping.shape[0], dtype=values.dtype, device=values.device
            )
            result[valid] = values[mapping[valid]]
            return result

        self.stable_age_refreshes = migrate(self.stable_age_refreshes)
        self.stable_transition_count = migrate(
            self.stable_transition_count
        )
        self.point_count = int(mapping.shape[0])

    def state_dict(self):
        return {
            "version": self.STATE_VERSION,
            "point_count": self.point_count,
            "stable_age_refreshes": self.stable_age_refreshes.clone(),
            "stable_transition_count": self.stable_transition_count.clone(),
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
