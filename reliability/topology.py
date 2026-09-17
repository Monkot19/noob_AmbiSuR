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
        if ((self.new_to_old == -1) & ~self.is_new).any():
            raise ValueError("new_to_old=-1 requires is_new=True")


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


@torch.no_grad()
def migrate_lineage_tensor(old, change, *, fill_value=0):
    """Migrate by parent identity, including mapped newly created rows."""
    if old.ndim == 0:
        raise ValueError("state tensor must have a leading point dimension")
    mapping = change.new_to_old.to(device=old.device)
    mapped = mapping >= 0
    if mapped.any() and mapping[mapped].max() >= old.shape[0]:
        raise ValueError("new_to_old source index is out of range")
    shape = (mapping.shape[0],) + tuple(old.shape[1:])
    migrated = torch.full(shape, fill_value, dtype=old.dtype, device=old.device)
    migrated[mapped] = old.detach()[mapping[mapped]]
    return migrated


def identity_topology_change(point_count, *, device=None):
    mapping = torch.arange(point_count, dtype=torch.int64, device=device)
    return TopologyChange(mapping, torch.zeros_like(mapping, dtype=torch.bool))


def append_topology_change(
    old_count,
    appended_count,
    *,
    parent_indices=None,
    device=None,
):
    if old_count < 0 or appended_count < 0:
        raise ValueError("topology counts must be nonnegative")
    old = torch.arange(old_count, dtype=torch.int64, device=device)
    if parent_indices is None:
        new = torch.full(
            (appended_count,), -1, dtype=torch.int64, device=device
        )
    else:
        new = parent_indices.detach().to(device=device)
        if new.dtype != torch.int64 or new.shape != (appended_count,):
            raise ValueError(
                "parent_indices must be an int64 appended-count vector"
            )
        if ((new < 0) | (new >= old_count)).any():
            raise ValueError("parent index is out of range")
    mapping = torch.cat((old, new))
    is_new = torch.cat(
        (
            torch.zeros(old_count, dtype=torch.bool, device=device),
            torch.ones(appended_count, dtype=torch.bool, device=device),
        )
    )
    return TopologyChange(mapping, is_new)


def prune_topology_change(prune_mask):
    if prune_mask.ndim != 1 or prune_mask.dtype != torch.bool:
        raise ValueError("prune_mask must be a bool vector")
    mapping = torch.arange(
        prune_mask.shape[0], dtype=torch.int64, device=prune_mask.device
    )[~prune_mask]
    return TopologyChange(mapping, torch.zeros_like(mapping, dtype=torch.bool))


def compose_topology_changes(first, second):
    """Compose old->intermediate and intermediate->new row mappings."""
    mapping = torch.full_like(second.new_to_old, -1)
    mapped = second.new_to_old >= 0
    if mapped.any():
        indices = second.new_to_old[mapped].to(first.new_to_old.device)
        if indices.max() >= first.new_to_old.shape[0]:
            raise ValueError("topology composition index is out of range")
        mapping[mapped] = first.new_to_old[indices].to(mapping.device)
    is_new = second.is_new.clone()
    if mapped.any():
        inherited_new = first.is_new[indices].to(is_new.device)
        is_new[mapped] |= inherited_new
    return TopologyChange(mapping, is_new)
