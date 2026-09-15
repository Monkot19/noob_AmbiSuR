from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TopologyChange:
    new_to_old: torch.Tensor
    is_new: torch.Tensor

    def __post_init__(self):
        if self.new_to_old.ndim != 1 or self.new_to_old.dtype != torch.int64:
            raise ValueError("new_to_old must be an int64 vector")
        if self.is_new.ndim != 1 or self.is_new.dtype != torch.bool:
            raise ValueError("is_new must be a bool vector")
        if self.new_to_old.shape != self.is_new.shape:
            raise ValueError("new_to_old and is_new must have the same shape")
        if (self.new_to_old < -1).any():
            raise ValueError("new_to_old may use only -1 for new rows")
        if not torch.equal(self.is_new, self.new_to_old == -1):
            raise ValueError("is_new must exactly identify new_to_old == -1")


@torch.no_grad()
def migrate_tensor(old, change, *, fill_value=0):
    if old.ndim == 0:
        raise ValueError("state tensor must have a leading point dimension")
    mapping = change.new_to_old.to(device=old.device)
    survivors = ~change.is_new.to(device=old.device)
    if survivors.any() and mapping[survivors].max() >= old.shape[0]:
        raise ValueError("new_to_old source index is out of range")
    shape = (mapping.shape[0],) + tuple(old.shape[1:])
    migrated = torch.full(shape, fill_value, dtype=old.dtype, device=old.device)
    migrated[survivors] = old.detach()[mapping[survivors]]
    return migrated
