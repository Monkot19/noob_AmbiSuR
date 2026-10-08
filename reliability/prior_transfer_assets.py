"""GT-free qualification of one preregistered Utility shadow run.

This module never launches, repairs, resumes, or evaluates a run. Checkpoint
pickle inputs must be the user's trusted local training assets. Torch is loaded
only at the checkpoint boundary; the command has no mesh/GT input surface.
"""
from argparse import ArgumentParser
from datetime import datetime
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess

import numpy as np

from reliability.config import CoreConfig
from reliability.g1_prior_transfer_confirmation import _canonical_bytes, _file_sha, _validate_record
from reliability.utility_snapshot import _canonical_bytes as snapshot_bytes


_REFRESHES = tuple(range(1000, 7001, 1000))
_GROUPS = ("xyz", "knn_f", "f_dc", "f_rest", "opacity", "scaling", "rotation")
_GT = re.compile(r"/gt/|gt_mesh|mesh_aligned|--gt\b|\bground[_ -]?truth\b", re.I)
_BAD_LOG = re.compile(r"\b(?:nan|inf(?:inity)?|non[-_ ]?finite|traceback|error|exception)\b", re.I)


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _no_gt(value):
    if _GT.search(json.dumps(value, allow_nan=False).replace("\\\\", "/")):
        raise ValueError("GT reference in training assets")


def expected_training_config(row):
    """Resolve the frozen argv through the actual production argument defaults."""
    from arguments import ModelParams, OptimizationParams, PipelineParams
    from reliability.runtime import build_resolved_config
    parser = ArgumentParser()
    model, opt, pipe = ModelParams(parser), OptimizationParams(parser), PipelineParams(parser)
    for name in ("test_iterations", "save_iterations", "checkpoint_iterations"):
        parser.add_argument("--" + name, nargs="+", type=int)
    args = parser.parse_args(row["training_argv"][2:])
    return build_resolved_config(model.extract(args), opt.extract(args), pipe.extract(args),
                                 CoreConfig(seed=row["seed"], core_shadow_mode=True))


def _processes():
    """Fail closed if the host cannot provide a complete process inventory."""
    output = subprocess.check_output(["ps", "-eo", "pid=,stat=,args="], text=True)
    rows = []
    for line in output.splitlines():
        pid, state, command = line.strip().split(None, 2)
        rows.append((int(pid), state, command))
    return rows


def validate_completion(run_dir, row, confirmation_sha, *, processes):
    run_dir, launch = Path(run_dir), Path(row["launcher_dir"])
    record = _json(launch / "launch_record.json")
    expected = {"schema_version": 1, "attempt": 1, "resumed": False,
                "replaces_completed_run": False, "confirmation_sha256": confirmation_sha,
                **{key: row[key] for key in ("seed", "run_dir", "view_dir", "state_file",
                                            "launcher_dir", "training_argv")}}
    if record != expected:
        raise ValueError("launch record binding/attempt mismatch")
    # Parse literal shell assignments only. Never source/eval a state file.
    state = {}
    for line in Path(row["state_file"]).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if any(token in line for token in ("$", "`", ";")):
            raise ValueError("state file must contain literal assignments")
        tokens = shlex.split(line, posix=True)
        if len(tokens) != 1 or "=" not in tokens[0]:
            raise ValueError("invalid state assignment")
        key, value = tokens[0].split("=", 1)
        if key in state:
            raise ValueError("duplicate state assignment")
        state[key] = value
    expected_state = {"RUN_DIR": row["run_dir"], "VIEW_DIR": row["view_dir"],
                      "LAUNCH_DIR": row["launcher_dir"], "SEED": str(row["seed"]),
                      "CONFIRMATION_SHA256": confirmation_sha}
    if state != expected_state:
        raise ValueError("state file binding mismatch")
    _no_gt(record)
    for root in (run_dir, launch):
        if (root / ".training_active").exists():
            raise ValueError("training sentinel remains")
    pids = [int((launch / name).read_text().strip()) for name in ("launcher.pid", "training.pid")]
    if any(pid <= 0 for pid in pids) or len(set(pids)) != 2:
        raise ValueError("invalid process identity")
    for pid, status, command in processes:
        if pid in pids or re.search(r"(?:^|[/\s])python(?:3(?:\.\d+)?)?(?:\s|$).*?(?:^|[/\s])train\.py(?:\s|$)", command):
            raise ValueError("launcher or training process remains active")
    if (launch / "exit_code.txt").read_text().strip() != "0":
        raise ValueError("training exit code is not zero")
    if (launch / "confirmation_sha256.txt").read_text().strip() != confirmation_sha:
        raise ValueError("launcher confirmation SHA mismatch")
    start, end = [datetime.fromisoformat((launch / name).read_text().strip().replace("Z", "+00:00"))
                  for name in ("start_utc.txt", "end_utc.txt")]
    wall = int((launch / "wall_seconds.txt").read_text().strip())
    gpu = int((launch / "gpu_peak_mib.txt").read_text().strip())
    if start.tzinfo is None or end.tzinfo is None or end <= start or wall <= 0 or gpu <= 0:
        raise ValueError("invalid completion timing/resources")
    if abs((end - start).total_seconds() - wall) > 2:
        raise ValueError("wall time mismatch")
    train_log = (run_dir / "train.log").read_text(encoding="utf-8")
    if train_log.count("Training complete.") != 1:
        raise ValueError("completion marker count mismatch")
    if tuple(map(int, re.findall(r"\[ITER (\d+)\] Evaluating train:", train_log))) != _REFRESHES:
        raise ValueError("evaluation schedule mismatch")
    for log in (train_log, (launch / "launcher.log").read_text(encoding="utf-8")):
        _no_gt(log)
        if _BAD_LOG.search(log):
            raise ValueError("error/nonfinite log token")
    return {"wall_seconds": wall, "gpu_peak_mib": gpu,
            "start_utc": start.isoformat(), "end_utc": end.isoformat()}


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def validate_optimizer_state(optimizer, parameters, schedule, iteration):
    """Adam is lazy: only knn_f may be completely dormant; active state is exact."""
    until = min(iteration, int(schedule["iterations"]) - 1)
    replacements = sum(index > schedule["densify_from_iter"]
                       and index < schedule["densify_until_iter"]
                       and index % schedule["densification_interval"] == 0
                       for index in range(1, until + 1))
    expected = until - replacements
    groups, state = optimizer["param_groups"], optimizer["state"]
    if len(parameters) != 7 or len(groups) != 7 or tuple(group["name"] for group in groups) != _GROUPS:
        raise ValueError("optimizer group inventory mismatch")
    identifiers, dormant = [], []
    for group, parameter in zip(groups, parameters):
        if len(group["params"]) != 1:
            raise ValueError("optimizer parameter inventory mismatch")
        identifier = group["params"][0]
        identifiers.append(identifier)
        values = state.get(identifier, {})
        if not values and group["name"] == "knn_f":
            dormant.append("knn_f")
            continue
        if set(values) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("active optimizer state is incomplete")
        step = _numpy(values["step"])
        if step.size != 1 or not np.isfinite(step).all() or float(step.reshape(-1)[0]) != expected:
            raise ValueError("optimizer step does not match replacement schedule")
        for name in ("exp_avg", "exp_avg_sq"):
            value = _numpy(values[name])
            if value.shape != _numpy(parameter).shape or not np.isfinite(value).all():
                raise ValueError("optimizer moment shape/finite mismatch")
    if len(set(identifiers)) != 7 or not set(state) <= set(identifiers):
        raise ValueError("duplicate/extra optimizer parameter state")
    return {"expected_step": expected, "replacement_count": replacements, "dormant_groups": dormant}


def _safe_relative(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name or not path.parts:
        raise ValueError("unsafe manifest path")
    _no_gt(name)
    return path


def _manifest(root, entries):
    expected = {}
    for entry in entries:
        relative = _safe_relative(entry["path"])
        if relative.as_posix() in expected:
            raise ValueError("duplicate manifest entry")
        path = Path(root) / relative
        if not path.is_file() or path.stat().st_size != entry["bytes"] or _file_sha(path) != entry["sha256"]:
            raise ValueError(f"immutable input changed: {path}")
        expected[relative.as_posix()] = entry
    return expected


def _snapshot_binding(confirmation, row):
    source = _json(confirmation["source_record"]["path"])
    snapshot = _json(confirmation["snapshot_record"]["path"])
    if source.get("audit_kind") != "utility_source" or source.get("gt_access") != "NONE":
        raise ValueError("source audit identity mismatch")
    if (source["expected_count"], source["image_count"], source["registered_image_count"]) != (147, 147, 147):
        raise ValueError("Utility source count mismatch")
    identity_keys = ("source_sha256", "repository_commit", "da3_checkpoint", "environment",
                     "preprocessing", "command", "array_shapes", "array_scales", "alignment", "derived_manifest")
    digest = hashlib.sha256(snapshot_bytes({key: snapshot[key] for key in identity_keys})).hexdigest()
    if snapshot.get("kind") != "utility_da3_snapshot" or snapshot.get("gt_access") != "NONE" or digest != snapshot["snapshot_sha256"] or digest != confirmation["snapshot_record"]["snapshot_sha256"]:
        raise ValueError("snapshot identity mismatch")
    if snapshot["source_sha256"] != source["source_sha256"]:
        raise ValueError("snapshot/source mismatch")
    if hashlib.sha256(snapshot_bytes(source["files"])).hexdigest() != source["source_sha256"]:
        raise ValueError("source manifest digest mismatch")
    derived = snapshot["derived_manifest"]
    roots = ("estimated_depths", "estimated_confs", "sparse_da3", "sparse_da3_aligned")
    if any(_safe_relative(entry["path"]).parts[0] not in roots for entry in derived):
        raise ValueError("derived manifest root mismatch")
    names = source["image_names"]
    if len(names) != 147 or len(set(names)) != 147:
        raise ValueError("source image inventory mismatch")
    for prefix in roots[:2]:
        actual = {entry["path"] for entry in derived if entry["path"].startswith(prefix + "/") and entry["path"].endswith(".npy")}
        if actual != {f"{prefix}/{name}.npy" for name in names}:
            raise ValueError("loader array inventory mismatch")
    source_files = _manifest(source["source_root"], source["files"])
    if _file_inventory(Path(source["source_root"])) != set(source_files):
        raise ValueError("source file inventory changed")
    for root in (Path(snapshot["snapshot_root"]), Path(row["view_dir"])):
        _manifest(root, source["files"])
        expected = _manifest(root, derived)
        actual = _file_inventory(root)
        allowed = set(expected) | set(source_files)
        # readColmapSceneInfo always writes this cache, only in the private view.
        if root == Path(row["view_dir"]):
            actual.discard("sparse_da3_aligned/0/points3D.ply")
        if actual != allowed:
            raise ValueError("source/derived file inventory changed")
    return source, snapshot


def _file_inventory(root):
    # Explicit prefixes also traverse intentional top-level symlinks in views.
    paths = {path for path in root.rglob("*") if path.is_file()}
    for name in ("images", "sparse", "estimated_depths", "estimated_confs", "sparse_da3", "sparse_da3_aligned"):
        paths.update(path for path in (root / name).rglob("*") if path.is_file())
    return {path.relative_to(root).as_posix() for path in paths}


def input_fingerprints(confirmation, row):
    """Capture all read-only trees and records, explicitly excluding the GT asset."""
    source = _json(confirmation["source_record"]["path"])
    snapshot = _json(confirmation["snapshot_record"]["path"])
    roots = [Path(source["source_root"]), Path(snapshot["snapshot_root"]),
             Path(row["view_dir"]), Path(row["run_dir"]), Path(row["launcher_dir"])]
    paths = {Path(row["state_file"]), Path(confirmation["source_record"]["path"]),
             Path(confirmation["snapshot_record"]["path"])}
    for root in roots:
        if not root.is_dir():
            raise ValueError(f"input tree missing: {root}")
        paths.update(root / name for name in _file_inventory(root))
    result = {}
    for path in sorted(paths):
        _no_gt(str(path))
        resolved = path.resolve()
        if resolved == Path(confirmation["gt_mesh"]["path"]).resolve():
            raise ValueError("GT alias in training inputs")
        _no_gt(str(resolved))
        result[str(path)] = {"resolved_path": str(resolved), "bytes": path.stat().st_size, "sha256": _file_sha(path)}
    return result


def _same_state(before, after):
    if isinstance(before, dict):
        return set(before) == set(after) and all(_same_state(value, after[key]) for key, value in before.items())
    if hasattr(before, "detach"):
        return before.dtype == after.dtype and np.array_equal(_numpy(before), _numpy(after))
    return before == after


def validate_checkpoint_metadata(payload):
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "iteration", "gaussian_state", "core_state"}:
        raise ValueError("checkpoint root fields mismatch")

    def visit(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if isinstance(key, str) and re.search(r"(?:^|_)(?:gt|mesh|ground_truth)(?:_|$)", key, re.I):
                    raise ValueError("GT metadata in checkpoint")
                visit(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child)
        elif isinstance(value, str):
            _no_gt(value)
    visit(payload)


def validate_checkpoint_lineage(centers, evidence, event, refresh_count):
    """Refresh and capture are adjacent: centers and current lineage must agree."""
    centers = _numpy(centers)
    history = _numpy(evidence["history_valid"])
    if not np.array_equal(centers, _numpy(evidence["previous_centers"])):
        raise ValueError("checkpoint/evidence center row mismatch")
    if history.shape != (len(centers),) or history.dtype != np.bool_ or not history.all():
        raise ValueError("checkpoint current history mismatch")
    stable = _numpy(evidence["stable_state"])
    temporal = evidence["temporal_transition_diagnostics"]
    if not np.array_equal(_numpy(temporal["previous_stable"]), stable):
        raise ValueError("temporal previous stable mismatch")
    names = ("Bypass", "Consensus", "Prior-led", "Geometry-led", "Abstain")
    for field, mean_field, minimum, maximum in (
        ("stable_age_refreshes", "mean_stable_age_refreshes", 1, refresh_count),
        ("stable_transition_count", "mean_stable_transition_count", 0, refresh_count - 1),
    ):
        value = _numpy(temporal[field])
        if value.dtype != np.int64 or value.shape != stable.shape or np.any(value < minimum) or np.any(value > maximum):
            raise ValueError("temporal counter refresh bounds mismatch")
        if not np.isclose(value.astype(np.float64).mean(), event[mean_field], rtol=0, atol=1e-12):
            raise ValueError("temporal counter overall mean mismatch")
        for index, name in enumerate(names):
            selected = value[stable == index]
            actual = event[mean_field + "_by_state"][name]
            if not selected.size:
                if actual is not None:
                    raise ValueError("empty temporal state mean mismatch")
            elif actual is None or not np.isclose(selected.astype(np.float64).mean(), actual, rtol=0, atol=1e-12):
                raise ValueError("temporal counter state mean mismatch")


def _checkpoint(run, iteration, config, event, point_count):
    import torch
    from reliability.offline_g1 import load_g1_iteration
    from reliability.shadow import D0ShadowRuntime
    from reliability.transition_diagnostics import TRANSITION_SUMMARY_FIELDS
    joined = load_g1_iteration(run, iteration, expected_evidence_version=4)
    if joined.rejected_center_indices.size or joined.original_point_count != point_count:
        raise ValueError("nonfinite centers or checkpoint/timeline row mismatch")
    payload = torch.load(run / f"chkpnt{iteration}.pth", map_location="cpu", weights_only=False)
    validate_checkpoint_metadata(payload)
    capture, state = payload["gaussian_state"], payload["core_state"]
    if state["refresh_count"] != iteration // 1000 or state["last_refresh_iteration"] != iteration:
        raise ValueError("checkpoint refresh schedule mismatch")
    if state["latest_transition_diagnostics"] != {key: event[key] for key in TRANSITION_SUMMARY_FIELDS}:
        raise ValueError("checkpoint/event temporal summary mismatch")
    runtime = D0ShadowRuntime(point_count, cfg=CoreConfig(**{
        key: value for key, value in config["core"].items() if key != "enabled_features"
    }), device="cpu", refresh_interval=1000)
    runtime.load_state_dict(state)
    restored = runtime.state_dict()
    if not _same_state(state, restored):
        raise ValueError("runtime state round-trip mismatch")
    evidence = state["evidence"]
    validate_checkpoint_lineage(capture[1], evidence, event, state["refresh_count"])
    for value in _tensor_leaves(state):
        if not bool(torch.isfinite(value).all()):
            raise ValueError("nonfinite runtime state")
    parameters = list(capture[1:8])
    expected_shapes = [(point_count, 3), (point_count, 6), (point_count, 1, 3),
                       (point_count, 15, 3), (point_count, 3), (point_count, 4), (point_count, 1)]
    if any(not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
           or not bool(torch.isfinite(value).all()) for value, shape in zip(parameters, expected_shapes)):
        raise ValueError("Gaussian parameter shape/finite mismatch")
    # capture order differs from Adam's named-group order.
    optimizer_parameters = [parameters[index] for index in (0, 1, 2, 3, 6, 4, 5)]
    optimizer = validate_optimizer_state(capture[14], optimizer_parameters, config["optimization"], iteration)
    return {"iteration": iteration, "point_count": point_count, "rejected_centers": 0,
            "checkpoint_sha256": joined.checkpoint_sha256, "snapshot_sha256": joined.snapshot_sha256,
            "schema_versions": [1, 2, 4, 1], "temporal_round_trip": True, "optimizer": optimizer}


def _tensor_leaves(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _tensor_leaves(item)
    elif hasattr(value, "detach"):
        yield value


def qualify_prior_transfer_run(run_dir, seed, confirmation):
    record = _validate_record(copy.deepcopy(dict(confirmation)), require_targets_absent=False, verify_record_files=True)
    if type(seed) is not int or seed not in (0, 1, 2):
        raise ValueError("seed is outside frozen inventory")
    row = record["runs"][seed]
    run = Path(run_dir).resolve()
    if str(run) != row["run_dir"]:
        raise ValueError("run path/seed mismatch")
    digest = hashlib.sha256(_canonical_bytes(record)).hexdigest()
    before = input_fingerprints(record, row)
    repository = record["repository"]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository["root"], text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=repository["root"], text=True).strip()
    if head != repository["commit"] or dirty:
        raise ValueError("qualification requires the exact clean repository commit")
    completion = validate_completion(run, row, digest, processes=_processes())
    identity = _json(run / "run_identity.json")
    if (identity.get("git_commit") != head or identity.get("git_dirty") is not False
        or identity.get("seed") != seed or identity.get("argv") != row["training_argv"]
        or identity.get("cwd") != repository["root"]):
        raise ValueError("run identity mismatch")
    config = _json(run / "resolved_config.json")
    if config != expected_training_config(row):
        raise ValueError("resolved configuration mismatch")
    _no_gt([identity, config])
    source, snapshot = _snapshot_binding(record, row)
    from reliability.g1_timeline import load_d0_timeline
    timeline = load_d0_timeline(run)
    counts = timeline.point_counts.tolist()
    if len(set(counts)) < 2:
        raise ValueError("real topology change is missing")
    checkpoints = [_checkpoint(run, iteration, config, timeline.events[iteration // 1000 - 1], counts[iteration // 1000 - 1])
                   for iteration in (3000, 7000)]
    if input_fingerprints(record, row) != before:
        raise ValueError("immutable inputs changed during qualification")
    validate_completion(run, row, digest, processes=_processes())
    return {"schema_version": 1, "kind": "prior_transfer_run_qualification", "outcome": "QUALIFIED",
            "gt_access": "NONE", "training_started": False, "c1_authorized": False,
            "confirmation_sha256": digest, "completion": completion, "point_counts": counts,
            "source_sha256": source["source_sha256"], "checkpoints": checkpoints,
            "input_fingerprints": before,
            "run_binding": {"seed": seed, "run_dir": row["run_dir"], "view_dir": row["view_dir"],
                "repository_commit": head, "snapshot_sha256": snapshot["snapshot_sha256"],
                "evidence_version": 4, "qualification_path": row["qualification_path"]}}
