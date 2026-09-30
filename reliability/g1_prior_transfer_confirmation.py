"""Canonical post-DA3, pre-run confirmation for Utility transfer."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path, PurePath
import re
import uuid
from typing import Mapping

from reliability.g1_complementarity import ProbeConfig
from reliability.g1_prior_transfer import _GATES, TRANSFER_OUTCOMES


_SHA40 = re.compile(r"[0-9a-f]{40}")
_SHA64 = re.compile(r"[0-9a-f]{64}")


def _absolute(value, name):
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute")
    return path.resolve()


def _safe_id(value):
    value = str(value)
    path = PurePath(value)
    if (
        not value
        or path.is_absolute()
        or len(path.parts) != 1
        or value in (".", "..")
        or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for character in value
        )
    ):
        raise ValueError("confirmation ID must be one safe path component")
    return value


def _canonical_bytes(value):
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_record_file(identity, name):
    if not isinstance(identity, dict) or not {"path", "sha256"} <= set(identity):
        raise ValueError(f"{name} identity mismatch")
    path = _absolute(identity["path"], f"{name} path")
    if not path.is_file() or _SHA64.fullmatch(str(identity["sha256"])) is None:
        raise ValueError(f"{name} identity mismatch")
    if _file_sha(path) != identity["sha256"]:
        raise ValueError(f"{name} SHA256 mismatch")


def _probe_contract():
    config = asdict(ProbeConfig())
    config["interval_percentiles"] = list(config["interval_percentiles"])
    config["bootstrap_seed"] = [20260930, "iteration", "training_seed"]
    return {
        "models": {"M0": ["A", "1-S"], "M1": ["A", "1-S", "1-r_p"]},
        "training_seeds": [0, 1, 2],
        "resolution": 2,
        "training_iterations": 7000,
        "refresh_interval": 1000,
        "evaluation_iterations": [1000, 2000, 3000, 4000, 5000, 6000, 7000],
        "checkpoint_iterations": [3000, 7000],
        "performance_iteration": 7000,
        "evidence_version": 4,
        "enabled_features": ["shadow_diagnostics"],
        "sample_domain": "all_finite_centers_then_V_p_true",
        "label": "distance_to_complete_valid_mesh_strictly_greater_than_0.05_m",
        "frozen_probe_module": "scripts.diagnostics.probe_g1_prior_complementarity",
        "probe_configuration": config,
        "gates": dict(_GATES),
        "outcomes": list(TRANSFER_OUTCOMES),
    }


def _mesh_contract():
    return {
        "coordinate_transform": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "world_unit": "meter",
        "sparse_point_limit": 50000,
        "sparse_point_selection": "ascending_sha256_of_point_id_and_xyz_bytes",
        "point_distance_median_at_most_m": 0.05,
        "point_distance_p90_at_most_m": 0.15,
        "point_fraction_within_0.10_m_at_least": 0.80,
        "camera_ray_grid": [8, 6],
        "aggregate_ray_hit_fraction_at_least": 0.80,
        "camera_fraction_with_hit_fraction_at_least_0.50": 0.90,
        "registered_camera_count": 147,
        "triangle_policy": "all_finite_non_degenerate",
        "evaluation_filtering": "NONE",
    }


def _expected_argv(row):
    argv = row["training_argv"]
    if not isinstance(argv, list) or len(argv) < 2:
        raise ValueError("training argv mismatch")
    return [
        str(argv[0]),
        "train.py",
        "--source_path",
        row["view_dir"],
        "--model_path",
        row["run_dir"],
        "-r",
        "2",
        "--iterations",
        "7000",
        "--seed",
        str(row["seed"]),
        "--core_shadow_mode",
        "--d0_refresh_interval",
        "1000",
        "--test_iterations",
        "1000",
        "2000",
        "3000",
        "4000",
        "5000",
        "6000",
        "7000",
        "--save_iterations",
        "7000",
        "--checkpoint_iterations",
        "3000",
        "7000",
    ]


def _validate_record(record, *, require_targets_absent, verify_record_files):
    required = {
        "schema_version",
        "kind",
        "confirmation_id",
        "created_utc",
        "repository",
        "source_record",
        "snapshot_record",
        "gt_mesh",
        "protocol",
        "mesh_admission",
        "accepted_geometry_releases",
        "runs",
        "probe_targets",
        "firewall",
    }
    if not isinstance(record, dict) or set(record) != required:
        raise ValueError("prior transfer confirmation fields mismatch")
    if record["schema_version"] != 1 or record["kind"] != "g1_prior_transfer_confirmation":
        raise ValueError("prior transfer confirmation schema mismatch")
    _safe_id(record["confirmation_id"])
    try:
        datetime.fromisoformat(str(record["created_utc"]).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("confirmation timestamp mismatch") from exc
    repository = record["repository"]
    if set(repository) != {"root", "commit", "clean"}:
        raise ValueError("repository identity mismatch")
    _absolute(repository["root"], "repository root")
    if _SHA40.fullmatch(str(repository["commit"])) is None or repository["clean"] is not True:
        raise ValueError("confirmation requires an exact clean commit")
    for name in ("source_record", "snapshot_record"):
        identity = record[name]
        expected_fields = {"path", "sha256"}
        if name == "snapshot_record":
            expected_fields.add("snapshot_sha256")
        if not isinstance(identity, dict) or set(identity) != expected_fields:
            raise ValueError(f"{name} identity mismatch")
        if _SHA64.fullmatch(str(identity["sha256"])) is None:
            raise ValueError(f"{name} SHA256 mismatch")
        if name == "snapshot_record" and _SHA64.fullmatch(str(identity["snapshot_sha256"])) is None:
            raise ValueError("snapshot identity mismatch")
        _absolute(identity["path"], f"{name} path")
        if verify_record_files:
            _verify_record_file(identity, name)
    gt = record["gt_mesh"]
    if set(gt) != {"path", "bytes", "sha256", "content_accessed"}:
        raise ValueError("GT metadata field mismatch")
    _absolute(gt["path"], "GT mesh path")
    if (
        isinstance(gt["bytes"], bool)
        or not isinstance(gt["bytes"], int)
        or gt["bytes"] <= 0
        or _SHA64.fullmatch(str(gt["sha256"])) is None
        or gt["content_accessed"] is not False
    ):
        raise ValueError("GT metadata mismatch")
    if record["protocol"] != _probe_contract():
        raise ValueError("frozen probe protocol mismatch")
    if record["mesh_admission"] != _mesh_contract():
        raise ValueError("mesh admission contract mismatch")
    if record["accepted_geometry_releases"] != [
        "STAGE_G_C_CANDIDATE",
        "NO_ACTION_SPECIFIC_SIGNAL",
    ]:
        raise ValueError("geometry release contract mismatch")
    firewall = record["firewall"]
    if firewall != {
        "gt_before_release": "METADATA_ONLY",
        "requires_prior_confirmation_reload": True,
        "requires_geometry_release_reload": True,
        "first_access_log_precedes_mesh_parse": True,
        "utility_feedback_cannot_reopen_geometry": True,
    }:
        raise ValueError("Utility GT firewall contract mismatch")
    runs = record["runs"]
    if not isinstance(runs, list) or [row.get("seed") for row in runs if isinstance(row, dict)] != [0, 1, 2]:
        raise ValueError("run seed inventory mismatch")
    all_paths = []
    run_fields = {
        "seed",
        "run_dir",
        "view_dir",
        "state_file",
        "launcher_dir",
        "qualification_path",
        "training_argv",
    }
    for row in runs:
        if set(row) != run_fields:
            raise ValueError("run preregistration fields mismatch")
        for name in run_fields - {"seed", "training_argv"}:
            path = _absolute(row[name], f"run {name}")
            row[name] = str(path)
            all_paths.append(path)
        if row["training_argv"] != _expected_argv(row):
            raise ValueError("training argv is outside the frozen protocol")
        if any(
            str(token).startswith("--")
            and ("gt" in str(token).lower() or "mesh" in str(token).lower())
            for token in row["training_argv"]
        ):
            raise ValueError("training argv must not reference GT or mesh")
    probe = record["probe_targets"]
    if set(probe) != {"output_dir", "staging_dir", "access_log_path"}:
        raise ValueError("probe target inventory mismatch")
    for name, value in probe.items():
        path = _absolute(value, f"probe {name}")
        probe[name] = str(path)
        all_paths.append(path)
    if len({str(path) for path in all_paths}) != len(all_paths):
        raise ValueError("preregistered targets must be unique")
    if require_targets_absent:
        existing = [path for path in all_paths if path.exists()]
        if existing:
            raise ValueError(f"preregistered target already exists: {existing[0]}")
    return json.loads(json.dumps(record))


def build_prior_transfer_confirmation(**request):
    record = {
        "schema_version": 1,
        "kind": "g1_prior_transfer_confirmation",
        "confirmation_id": request["confirmation_id"],
        "created_utc": request["created_utc"],
        "repository": dict(request["repository"]),
        "source_record": dict(request["source_record"]),
        "snapshot_record": dict(request["snapshot_record"]),
        "gt_mesh": dict(request["gt_mesh"]),
        "protocol": _probe_contract(),
        "mesh_admission": _mesh_contract(),
        "accepted_geometry_releases": [
            "STAGE_G_C_CANDIDATE",
            "NO_ACTION_SPECIFIC_SIGNAL",
        ],
        "runs": [dict(row) for row in request["runs"]],
        "probe_targets": dict(request["probe_targets"]),
        "firewall": {
            "gt_before_release": "METADATA_ONLY",
            "requires_prior_confirmation_reload": True,
            "requires_geometry_release_reload": True,
            "first_access_log_precedes_mesh_parse": True,
            "utility_feedback_cannot_reopen_geometry": True,
        },
    }
    return _validate_record(
        record, require_targets_absent=True, verify_record_files=True
    )


def validate_preregistration_targets(record):
    _validate_record(
        record, require_targets_absent=True, verify_record_files=True
    )


def write_prior_transfer_confirmation(record, path):
    validated = _validate_record(
        record, require_targets_absent=True, verify_record_files=True
    )
    path = _absolute(path, "confirmation output")
    sha_path = Path(str(path) + ".sha256")
    if not path.parent.is_dir():
        raise ValueError("confirmation parent directory is missing")
    if path.exists() or sha_path.exists():
        raise FileExistsError("confirmation output already exists")
    payload = _canonical_bytes(validated)
    digest = hashlib.sha256(payload).hexdigest()
    token = uuid.uuid4().hex
    temporary = path.parent / f".{path.name}.tmp-{token}"
    temporary_sha = path.parent / f".{sha_path.name}.tmp-{token}"
    try:
        temporary.write_bytes(payload)
        temporary_sha.write_text(digest + "\n", encoding="ascii")
        os.replace(temporary, path)
        os.replace(temporary_sha, sha_path)
    except Exception:
        for candidate in (temporary, temporary_sha):
            try:
                candidate.unlink()
            except FileNotFoundError:
                pass
        raise
    return {"path": path, "sha256_path": sha_path, "sha256": digest}


def load_prior_transfer_confirmation(path, expected_sha256):
    path = _absolute(path, "confirmation path")
    payload = path.read_bytes()
    if (
        _SHA64.fullmatch(str(expected_sha256)) is None
        or hashlib.sha256(payload).hexdigest() != expected_sha256
    ):
        raise ValueError("confirmation SHA256 mismatch")
    try:
        record = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError("confirmation is not valid JSON") from exc
    return _validate_record(
        record, require_targets_absent=False, verify_record_files=True
    )


def validate_completed_run_binding(record, run_records):
    record = _validate_record(
        record, require_targets_absent=False, verify_record_files=True
    )
    if not isinstance(run_records, list) or [row.get("seed") for row in run_records if isinstance(row, dict)] != [0, 1, 2]:
        raise ValueError("completed run seed inventory mismatch")
    frozen = {row["seed"]: row for row in record["runs"]}
    expected_fields = {
        "seed",
        "run_dir",
        "view_dir",
        "repository_commit",
        "snapshot_sha256",
        "evidence_version",
        "qualification_path",
    }
    for row in run_records:
        if set(row) != expected_fields:
            raise ValueError("completed run binding fields mismatch")
        target = frozen[row["seed"]]
        if row["run_dir"] != target["run_dir"] or row["view_dir"] != target["view_dir"] or row["qualification_path"] != target["qualification_path"]:
            raise ValueError("completed run path mismatch")
        if row["repository_commit"] != record["repository"]["commit"]:
            raise ValueError("completed run commit mismatch")
        if row["snapshot_sha256"] != record["snapshot_record"]["snapshot_sha256"]:
            raise ValueError("completed run snapshot mismatch")
        if row["evidence_version"] != 4:
            raise ValueError("completed run evidence version mismatch")
    return True
