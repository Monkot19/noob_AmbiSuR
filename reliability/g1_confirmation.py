"""Immutable preregistration and formal-admission checks for soft G1."""

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping


_CONFIRMATION_SCHEMA_VERSION = 1
_FORMULA = {
    "count": "M/(M+K_c)",
    "angle": "D/(D+D_c)",
    "sufficiency": "sqrt(S_count*S_angle)",
    "need": "1-S*(1-A)",
    "k_c": 5.0,
    "theta_c_degrees": 30.0,
    "d_c": (1.0 - math.cos(math.radians(30.0))) / 2.0,
}
_G1_GATE = {
    "auroc_n_strictly_greater_than": 0.60,
    "gain_over_best_component_at_least": 0.03,
}
_TRAINING_ITERATIONS = 7000
_REFRESH_INTERVAL = 1000
_EVALUATION_ITERATIONS = list(range(1000, 7001, 1000))
_CHECKPOINT_ITERATIONS = [3000, 7000]
_SAVE_ITERATIONS = [7000]
_G1_ITERATIONS = [3000, 7000]
_G1_DECISION_ITERATION = 7000
_TARGET_NAMES = (
    "run_dir",
    "view_dir",
    "report_path",
    "output_dir",
    "archive_path",
)
_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _require_sha(value, *, length, label):
    value = str(value)
    pattern = _HEX_40 if length == 40 else _HEX_64
    if pattern.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase {length}-character SHA")
    return value


def _canonical_bytes(record):
    return (
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _resolved_path(value):
    return str(Path(value).expanduser().resolve())


def _read_json_object(path, label):
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"{label} is missing: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _read_sha_record(path, label):
    path = Path(path)
    try:
        value = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"{label} is missing: {path}") from None
    return _require_sha(value, length=64, label=label)


def _contains_gt_reference(value):
    serialized = json.dumps(value, sort_keys=True).lower()
    return "gt_mesh" in serialized or '"gt"' in serialized or "/gt/" in serialized


def _argument_value(argv, flag):
    positions = [index for index, value in enumerate(argv) if value == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise ValueError(f"training argv must contain exactly one {flag}")
    return argv[positions[0] + 1]


def _validate_record(record):
    if not isinstance(record, Mapping):
        raise ValueError("confirmation record must be a mapping")
    required = {
        "schema_version",
        "confirmation_id",
        "preflight_utc",
        "formula_commit",
        "formula",
        "training",
        "inputs",
        "evaluation",
        "targets",
        "g1_gate",
    }
    if set(record) != required:
        missing = sorted(required - set(record))
        extra = sorted(set(record) - required)
        raise ValueError(
            f"confirmation fields mismatch: missing={missing}, extra={extra}"
        )
    if record["schema_version"] != _CONFIRMATION_SCHEMA_VERSION:
        raise ValueError("confirmation schema version mismatch")
    confirmation_id = record["confirmation_id"]
    if not isinstance(confirmation_id, str) or _SAFE_ID.fullmatch(
        confirmation_id
    ) is None:
        raise ValueError("confirmation ID is unsafe")
    preflight_utc = record["preflight_utc"]
    if not isinstance(preflight_utc, str) or not preflight_utc.endswith("Z"):
        raise ValueError("preflight UTC must use the canonical Z suffix")
    try:
        timestamp = datetime.fromisoformat(preflight_utc[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError("preflight UTC is malformed") from exc
    if timestamp.utcoffset() != timezone.utc.utcoffset(timestamp):
        raise ValueError("preflight UTC must be UTC")
    _require_sha(record["formula_commit"], length=40, label="formula commit")
    if record["formula"] != _FORMULA:
        raise ValueError("soft-calibration formula contract mismatch")
    if record["g1_gate"] != _G1_GATE:
        raise ValueError("G1 gate contract mismatch")

    training = record["training"]
    if not isinstance(training, Mapping):
        raise ValueError("training contract must be a mapping")
    expected_training = {
        "argv",
        "scene",
        "seed",
        "resolution",
        "iterations",
        "refresh_interval",
        "evaluation_iterations",
        "checkpoint_iterations",
        "save_iterations",
        "resolved_config",
    }
    if set(training) != expected_training:
        raise ValueError("training contract fields mismatch")
    if not isinstance(training["argv"], list) or not all(
        isinstance(value, str) and value for value in training["argv"]
    ):
        raise ValueError("training argv must be a nonempty token list")
    if _contains_gt_reference(training["argv"]) or _contains_gt_reference(
        training["resolved_config"]
    ):
        raise ValueError("training contract must not reference GT")
    if not isinstance(training["scene"], str) or not training["scene"]:
        raise ValueError("training scene is invalid")
    fixed_training = {
        "seed": 0,
        "resolution": 2,
        "iterations": _TRAINING_ITERATIONS,
        "refresh_interval": _REFRESH_INTERVAL,
        "evaluation_iterations": _EVALUATION_ITERATIONS,
        "checkpoint_iterations": _CHECKPOINT_ITERATIONS,
        "save_iterations": _SAVE_ITERATIONS,
    }
    for name, expected in fixed_training.items():
        if training[name] != expected:
            raise ValueError(f"training {name} contract mismatch")
    if not isinstance(training["resolved_config"], Mapping):
        raise ValueError("resolved config contract must be a mapping")

    inputs = record["inputs"]
    if not isinstance(inputs, Mapping) or set(inputs) != {
        "dataset_sha256",
        "aligned_prior_sha256",
    }:
        raise ValueError("input hash contract fields mismatch")
    _require_sha(inputs["dataset_sha256"], length=64, label="dataset SHA256")
    _require_sha(
        inputs["aligned_prior_sha256"],
        length=64,
        label="aligned-prior SHA256",
    )

    evaluation = record["evaluation"]
    if not isinstance(evaluation, Mapping) or set(evaluation) != {
        "iterations",
        "decision_iteration",
        "gt_mesh_sha256",
    }:
        raise ValueError("evaluation contract fields mismatch")
    if evaluation["iterations"] != _G1_ITERATIONS:
        raise ValueError("G1 iteration contract mismatch")
    if evaluation["decision_iteration"] != _G1_DECISION_ITERATION:
        raise ValueError("G1 decision iteration contract mismatch")
    _require_sha(
        evaluation["gt_mesh_sha256"], length=64, label="GT mesh SHA256"
    )

    targets = record["targets"]
    if not isinstance(targets, Mapping) or set(targets) != set(_TARGET_NAMES):
        raise ValueError("target contract fields mismatch")
    for name, target in targets.items():
        if not isinstance(target, Mapping) or set(target) != {"path", "absent"}:
            raise ValueError(f"target contract is malformed: {name}")
        if not isinstance(target["path"], str) or not Path(
            target["path"]
        ).is_absolute():
            raise ValueError(f"target path must be absolute: {name}")
        if target["absent"] is not True:
            raise ValueError(f"target absence proof is missing: {name}")

    argv = training["argv"]
    if _resolved_path(_argument_value(argv, "--source_path")) != targets[
        "view_dir"
    ]["path"]:
        raise ValueError("training source path does not match frozen view")
    if _resolved_path(_argument_value(argv, "--model_path")) != targets[
        "run_dir"
    ]["path"]:
        raise ValueError("training model path does not match frozen run")
    return record


def build_confirmation_record(
    *,
    confirmation_id,
    preflight_utc,
    formula_commit,
    training_argv,
    scene,
    seed,
    resolution,
    iterations,
    refresh_interval,
    evaluation_iterations,
    checkpoint_iterations,
    save_iterations,
    g1_iterations,
    g1_decision_iteration,
    dataset_sha256,
    prior_sha256,
    gt_sha256,
    run_dir,
    view_dir,
    report_path,
    output_dir,
    archive_path,
    expected_resolved_config,
    auroc_threshold,
    gain_threshold,
):
    paths = {
        "run_dir": Path(run_dir).expanduser().resolve(),
        "view_dir": Path(view_dir).expanduser().resolve(),
        "report_path": Path(report_path).expanduser().resolve(),
        "output_dir": Path(output_dir).expanduser().resolve(),
        "archive_path": Path(archive_path).expanduser().resolve(),
    }
    for name, path in paths.items():
        if path.exists():
            raise FileExistsError(f"confirmation target already exists: {name}")

    record = {
        "schema_version": _CONFIRMATION_SCHEMA_VERSION,
        "confirmation_id": confirmation_id,
        "preflight_utc": preflight_utc,
        "formula_commit": formula_commit,
        "formula": dict(_FORMULA),
        "training": {
            "argv": list(training_argv),
            "scene": scene,
            "seed": seed,
            "resolution": resolution,
            "iterations": iterations,
            "refresh_interval": refresh_interval,
            "evaluation_iterations": list(evaluation_iterations),
            "checkpoint_iterations": list(checkpoint_iterations),
            "save_iterations": list(save_iterations),
            "resolved_config": json.loads(
                json.dumps(expected_resolved_config)
            ),
        },
        "inputs": {
            "dataset_sha256": dataset_sha256,
            "aligned_prior_sha256": prior_sha256,
        },
        "evaluation": {
            "iterations": list(g1_iterations),
            "decision_iteration": g1_decision_iteration,
            "gt_mesh_sha256": gt_sha256,
        },
        "targets": {
            name: {"path": str(path), "absent": True}
            for name, path in paths.items()
        },
        "g1_gate": {
            "auroc_n_strictly_greater_than": auroc_threshold,
            "gain_over_best_component_at_least": gain_threshold,
        },
    }
    return dict(_validate_record(record))


def write_confirmation_record(record: Mapping, path: Path):
    _validate_record(record)
    path = Path(path).expanduser().resolve()
    sidecar = Path(f"{path}.sha256")
    if path.exists() or sidecar.exists():
        raise FileExistsError("confirmation record or sidecar already exists")
    payload = _canonical_bytes(record)
    digest = hashlib.sha256(payload).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    record_created = False
    sidecar_created = False
    try:
        with path.open("xb") as stream:
            record_created = True
            stream.write(payload)
        with sidecar.open("x", encoding="utf-8", newline="\n") as stream:
            sidecar_created = True
            stream.write(f"{digest}  {path.name}\n")
    except Exception:
        if sidecar_created:
            sidecar.unlink()
        if record_created:
            path.unlink()
        raise
    return path, sidecar, digest


def load_confirmation_record(path: Path, expected_sha256: str):
    expected_sha256 = _require_sha(
        expected_sha256,
        length=64,
        label="expected confirmation SHA256",
    )
    path = Path(path).expanduser().resolve()
    try:
        payload = path.read_bytes()
    except FileNotFoundError:
        raise FileNotFoundError(f"confirmation record is missing: {path}") from None
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise ValueError("confirmation SHA256 mismatch")
    sidecar = Path(f"{path}.sha256")
    try:
        sidecar_digest = sidecar.read_text(encoding="utf-8").split()[0]
    except FileNotFoundError:
        raise FileNotFoundError(
            f"confirmation SHA256 sidecar is missing: {sidecar}"
        ) from None
    except IndexError:
        raise ValueError("confirmation SHA256 sidecar is empty") from None
    if sidecar_digest != expected_sha256:
        raise ValueError("confirmation sidecar SHA256 mismatch")
    try:
        record = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError("confirmation record is not valid JSON") from exc
    _validate_record(record)
    if payload != _canonical_bytes(record):
        raise ValueError("confirmation record is not canonically serialized")
    return record


def validate_formal_admission(
    record: Mapping,
    *,
    run_dir: Path,
    confirmation_id: str,
    evaluator_commit: str,
    dataset_sha256: str,
    prior_sha256: str,
    gt_sha256: str,
):
    _validate_record(record)
    evaluator_commit = _require_sha(
        evaluator_commit, length=40, label="evaluator commit"
    )
    dataset_sha256 = _require_sha(
        dataset_sha256, length=64, label="dataset SHA256"
    )
    prior_sha256 = _require_sha(
        prior_sha256, length=64, label="aligned-prior SHA256"
    )
    gt_sha256 = _require_sha(gt_sha256, length=64, label="GT mesh SHA256")
    run_dir = Path(run_dir).expanduser().resolve()
    if confirmation_id != record["confirmation_id"]:
        raise ValueError("confirmation ID mismatch")
    if evaluator_commit != record["formula_commit"]:
        raise ValueError("evaluator/formula commit mismatch")
    if str(run_dir) != record["targets"]["run_dir"]["path"]:
        raise ValueError("formal run path mismatch")
    if dataset_sha256 != record["inputs"]["dataset_sha256"]:
        raise ValueError("dataset SHA256 mismatch")
    if prior_sha256 != record["inputs"]["aligned_prior_sha256"]:
        raise ValueError("aligned-prior SHA256 mismatch")
    if gt_sha256 != record["evaluation"]["gt_mesh_sha256"]:
        raise ValueError("GT mesh SHA256 mismatch")

    identity = _read_json_object(run_dir / "run_identity.json", "run identity")
    if identity.get("git_commit") != record["formula_commit"]:
        raise ValueError("training/formula commit mismatch")
    if identity.get("git_dirty") is not False:
        raise ValueError("formal training worktree was dirty")
    if identity.get("argv") != record["training"]["argv"]:
        raise ValueError("formal training argv mismatch")
    if identity.get("seed") != record["training"]["seed"]:
        raise ValueError("formal training identity seed mismatch")

    config = _read_json_object(
        run_dir / "resolved_config.json", "resolved configuration"
    )
    if config != record["training"]["resolved_config"]:
        raise ValueError("formal resolved configuration mismatch")
    if config.get("training_path") != "core":
        raise ValueError("formal training path must be core")
    core = config.get("core")
    if not isinstance(core, Mapping):
        raise ValueError("formal Core configuration is missing")
    if core.get("core_shadow_mode") is not True:
        raise ValueError("formal Core shadow mode is disabled")
    if core.get("enabled_features") != ["shadow_diagnostics"]:
        raise ValueError("formal Core feature set mismatch")
    if _contains_gt_reference(identity.get("argv")) or _contains_gt_reference(
        config
    ):
        raise ValueError("formal training metadata references GT")

    hash_contracts = (
        ("dataset_manifest_before.sha256", dataset_sha256, "dataset before SHA256"),
        ("dataset_manifest_after.sha256", dataset_sha256, "dataset after SHA256"),
        ("aligned_prior_before.sha256", prior_sha256, "prior before SHA256"),
        ("aligned_prior_after.sha256", prior_sha256, "prior after SHA256"),
    )
    for filename, expected, label in hash_contracts:
        if _read_sha_record(run_dir / filename, label) != expected:
            raise ValueError(f"{label} mismatch")
