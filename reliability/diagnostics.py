import json
from pathlib import Path

import numpy as np


SNAPSHOT_FIELDS = (
    "A",
    "S",
    "N",
    "T_p",
    "V_p",
    "r_p",
    "T_g",
    "V_g",
    "r_g",
    "Z_pg",
    "V_pg",
    "K",
    "delta",
    "candidate",
    "stable",
)


def _as_numpy(tensor):
    return tensor.detach().cpu().numpy()


def write_snapshot(output_directory, iteration, snapshot):
    """Persist one no-GT D0 snapshot without overwriting prior evidence."""

    iteration = int(iteration)
    if iteration <= 0:
        raise ValueError("iteration must be positive")
    directory = Path(output_directory) / "d0_evidence"
    directory.mkdir(parents=True, exist_ok=True)
    array_path = directory / f"iteration_{iteration:06d}.npz"
    event_path = directory / "events.jsonl"
    arrays = {
        name: _as_numpy(getattr(snapshot, name))
        for name in SNAPSHOT_FIELDS
    }

    with array_path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)

    event = {
        "schema_version": 1,
        "iteration": iteration,
        "point_count": int(arrays["N"].shape[0]),
        "joint_valid_count": int(arrays["V_pg"].sum()),
    }
    with event_path.open("a", encoding="utf-8") as stream:
        json.dump(event, stream, sort_keys=True)
        stream.write("\n")
    return {"arrays": str(array_path), "events": str(event_path)}
