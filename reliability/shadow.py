import copy

import torch

from reliability.evidence import EvidenceAccumulator
from reliability.transition_diagnostics import validate_transition_summary


class D0ShadowRuntime:
    """Schedule detached D0 evidence refreshes without training writes."""

    STATE_VERSION = 2

    def __init__(
        self,
        point_count,
        *,
        cfg,
        device=None,
        refresh_interval=1000,
    ):
        cfg.validate()
        if not cfg.core_shadow_mode:
            raise ValueError("D0 runtime requires shadow mode")
        if refresh_interval <= 0:
            raise ValueError("refresh_interval must be positive")
        self.cfg = cfg
        self.refresh_interval = int(refresh_interval)
        self.last_refresh_iteration = None
        self.refresh_count = 0
        self.latest_transition_diagnostics = None
        self.accumulator = EvidenceAccumulator(
            point_count, cfg=cfg, device=device
        )

    @torch.no_grad()
    def maybe_refresh(self, iteration, build_inputs):
        iteration = int(iteration)
        if iteration <= 0 or iteration % self.refresh_interval != 0:
            return None
        if iteration == self.last_refresh_iteration:
            return None
        snapshot = self.accumulator.refresh(build_inputs())
        self.latest_transition_diagnostics = copy.deepcopy(
            self.accumulator.latest_transition_diagnostics
        )
        validate_transition_summary(
            self.latest_transition_diagnostics,
            self.accumulator.point_count,
        )
        self.last_refresh_iteration = iteration
        self.refresh_count += 1
        return snapshot

    @torch.no_grad()
    def on_topology_change(self, change):
        self.accumulator.on_topology_change(change)

    def state_dict(self):
        return {
            "version": self.STATE_VERSION,
            "refresh_interval": self.refresh_interval,
            "last_refresh_iteration": self.last_refresh_iteration,
            "refresh_count": self.refresh_count,
            "latest_transition_diagnostics": copy.deepcopy(
                self.latest_transition_diagnostics
            ),
            "evidence": self.accumulator.state_dict(),
        }

    def load_state_dict(self, state):
        if state.get("version") != self.STATE_VERSION:
            raise ValueError("unsupported D0 runtime state version")
        if state.get("refresh_interval") != self.refresh_interval:
            raise ValueError("D0 runtime refresh interval mismatch")
        last = state.get("last_refresh_iteration")
        count = state.get("refresh_count")
        if last is not None and (not isinstance(last, int) or last <= 0):
            raise ValueError("invalid D0 last refresh iteration")
        if not isinstance(count, int) or count < 0:
            raise ValueError("invalid D0 refresh count")
        evidence = state.get("evidence")
        if not isinstance(evidence, dict):
            raise ValueError("missing D0 evidence state")
        latest = state.get("latest_transition_diagnostics")
        if count == 0:
            if latest is not None:
                raise ValueError("unrefreshed D0 runtime cannot have a summary")
        else:
            validate_transition_summary(latest, self.accumulator.point_count)
        self.accumulator.load_state_dict(evidence)
        self.last_refresh_iteration = last
        self.refresh_count = count
        self.latest_transition_diagnostics = copy.deepcopy(latest)


def create_shadow_runtime(
    cfg, *, point_count, device=None, refresh_interval=1000
):
    cfg.validate()
    if not cfg.core_shadow_mode:
        return None
    return D0ShadowRuntime(
        point_count,
        cfg=cfg,
        device=device,
        refresh_interval=refresh_interval,
    )
