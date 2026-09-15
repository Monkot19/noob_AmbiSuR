from enum import IntEnum

import torch


class ArbitrationState(IntEnum):
    BYPASS = 0
    CONSENSUS = 1
    PRIOR_LED = 2
    GEOMETRY_LED = 3
    ABSTAIN = 4


@torch.no_grad()
def candidate_state(snapshot, cfg):
    need = snapshot.N.detach()
    h_p = snapshot.V_p.detach().bool() & (snapshot.T_p.detach() >= cfg.tau_p)
    h_g = snapshot.V_g.detach().bool() & (snapshot.T_g.detach() >= cfg.tau_g)
    joint = snapshot.V_pg.detach().bool()
    agreement = snapshot.K.detach()
    delta = snapshot.delta.detach()

    result = torch.full_like(need, ArbitrationState.ABSTAIN, dtype=torch.int8)
    unresolved = need > cfg.tau_n

    mask = unresolved & h_p & h_g & joint & (agreement >= cfg.tau_k)
    result[mask] = ArbitrationState.CONSENSUS

    mask = unresolved & h_p & ~h_g
    result[mask] = ArbitrationState.PRIOR_LED
    mask = unresolved & ~h_p & h_g
    result[mask] = ArbitrationState.GEOMETRY_LED

    conflict = unresolved & h_p & h_g & joint & (agreement < cfg.tau_k)
    result[conflict & (delta > cfg.reliability_delta)] = ArbitrationState.PRIOR_LED
    result[conflict & (delta < -cfg.reliability_delta)] = ArbitrationState.GEOMETRY_LED

    result[~unresolved] = ArbitrationState.BYPASS
    return result


class ArbitrationStateMachine:
    def __init__(
        self,
        point_count,
        *,
        initial_state=ArbitrationState.BYPASS,
        enter_count=3,
        device=None,
    ):
        if point_count < 0:
            raise ValueError("point_count must be nonnegative")
        if enter_count <= 0:
            raise ValueError("enter_count must be positive")
        self.enter_count = int(enter_count)
        self.stable_state = torch.full(
            (point_count,), int(initial_state), dtype=torch.int8, device=device
        )
        self.candidate_state = self.stable_state.clone()
        self.consecutive_count = torch.zeros(
            point_count, dtype=torch.int32, device=device
        )

    @torch.no_grad()
    def update(self, candidate):
        candidate = candidate.detach().to(
            device=self.stable_state.device, dtype=torch.int8
        )
        if candidate.shape != self.stable_state.shape:
            raise ValueError("candidate state must match stable state")
        repeated = candidate == self.candidate_state
        self.consecutive_count = torch.where(
            repeated,
            self.consecutive_count + 1,
            torch.ones_like(self.consecutive_count),
        )
        self.candidate_state.copy_(candidate)
        ready = self.consecutive_count >= self.enter_count
        self.stable_state[ready] = candidate[ready]
        return self.stable_state.clone()
