"""Read-only hard and diagnostic gates for explicit schema-3 G0 audits."""

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re

from scripts.diagnostics.compare_feature_off import (
    evaluate_scalar_triplet,
    evaluate_triplet_report,
)


_SUMMARY_NAME = re.compile(
    r"^(?P<field>capture\.[A-Za-z0-9_]+|"
    r"optimizer\.[A-Za-z0-9_]+\.(?:exp_avg|exp_avg_sq))\."
    r"(?:channel_[0-9]{3}|row_l2)\."
    r"(?P<statistic>mean|std|q01|q05|q25|q50|q75|q95|q99)$"
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ROLES = ("b1", "b2", "e0")


def group_by_validated_field_and_statistic(outliers, expected_diagnostic_names):
    """Count outliers by family, field and statistic without fuzzy matching."""
    grouped = {}
    for result in sorted(outliers, key=lambda item: item["name"]):
        name = result["name"]
        match = _SUMMARY_NAME.fullmatch(name)
        if name not in expected_diagnostic_names or match is None:
            raise ValueError(f"unrecognized diagnostic summary: {name!r}")
        field = match.group("field")
        family = field.partition(".")[0]
        statistic = match.group("statistic")
        counts = grouped.setdefault(family, {}).setdefault(field, {})
        counts[statistic] = counts.get(statistic, 0) + 1
    return grouped


def evaluate_behavioral_report(report, expected_diagnostic_names):
    """Keep observable failures hard and Gaussian numeric outliers diagnostic."""
    if report.get("schema_version") != 3:
        raise ValueError("behavioral G0 requires schema version 3")
    expected = set(expected_diagnostic_names)
    if not expected or any(_SUMMARY_NAME.fullmatch(name) is None for name in expected):
        raise ValueError("expected diagnostic names must be nonempty and valid")
    metrics = report.get("diagnostic_scalar_metrics")
    if not isinstance(metrics, list):
        raise ValueError("diagnostic scalar metrics must be a list")

    names = []
    for metric in metrics:
        if not isinstance(metric, dict) or not isinstance(metric.get("name"), str):
            raise ValueError("diagnostic scalar metric requires a name")
        names.append(metric["name"])
        for role in ("b1", "b2", "e0"):
            value = metric.get(role)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(
                    f"nonfinite diagnostic summary: {metric['name']} {role}"
                )
    if len(names) != len(expected) or len(names) != len(set(names)) or set(names) != expected:
        raise ValueError("missing, duplicate or unexpected diagnostic summaries")

    hard = evaluate_triplet_report(report, factor=2.0)
    diagnostic_results = [
        evaluate_scalar_triplet(
            metric["name"], metric["b1"], metric["b2"], metric["e0"], factor=2.0
        )
        for metric in metrics
    ]
    outliers = sorted(
        (result for result in diagnostic_results if not result["passed"]),
        key=lambda result: result["name"],
    )
    return {
        **hard,
        "hard_equivalent": hard["equivalent"],
        "diagnostic_results": diagnostic_results,
        "diagnostic_outliers": outliers,
        "diagnostic_outlier_counts": group_by_validated_field_and_statistic(
            outliers, expected
        ),
    }


def load_confirmation_contract(path, expected_sha256):
    """Read an independently hash-pinned, pre-launch JSON contract."""
    if not isinstance(expected_sha256, str) or _SHA256.fullmatch(expected_sha256) is None:
        raise ValueError("a recorded lowercase SHA256 is required")
    try:
        raw = Path(path).read_bytes()
    except OSError as error:
        raise ValueError(f"confirmation contract cannot be read: {error}") from error
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("confirmation contract SHA256 mismatch")
    try:
        contract = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("confirmation contract is not valid JSON") from error
    if not isinstance(contract, dict):
        raise ValueError("confirmation contract must be a JSON object")
    return contract


def _hash_file_into(digest, path):
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)


def _hash_tree(root):
    root = Path(root)
    if not root.is_dir() or root.resolve() != root:
        raise ValueError(f"canonical root must be an existing resolved directory: {root}")
    digest = hashlib.sha256()
    files = sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix())
    if not files:
        raise ValueError(f"canonical tree is empty: {root}")
    file_count = 0
    for path in files:
        resolved = path.resolve()
        if path.is_symlink() and (resolved == root or root not in resolved.parents):
            raise ValueError(f"canonical link escapes root: {path}")
        if path.is_symlink() and path.is_dir():
            raise ValueError(f"canonical directory links are unsupported: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"unsupported canonical tree entry: {path}")
        if resolved != root and root not in resolved.parents:
            raise ValueError(f"canonical file escapes root: {path}")
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        _hash_file_into(digest, path)
        digest.update(b"\0")
        file_count += 1
    if file_count == 0:
        raise ValueError(f"canonical tree has no files: {root}")
    return digest.hexdigest()


def _run_artifact_paths(iteration, role):
    paths = [
        "train.log",
        f"chkpnt{iteration}.pth",
        f"point_cloud/iteration_{iteration}/point_cloud.ply",
        f"app_model/iteration_{iteration}/app.pth",
        "cfg_args",
        "cfg_opts",
        "g0_run_contract.json",
        "exit_code.txt",
        "start_utc.txt",
        "end_utc.txt",
        "gpu_peak_mib.txt",
        "launcher.log",
    ]
    if role == "e0":
        paths.extend(("resolved_config.json", "run_identity.json"))
    return sorted(paths)


def _hash_run_artifacts(directory, iteration, role):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"{role} run directory is missing: {directory}")
    digest = hashlib.sha256()
    for relative in _run_artifact_paths(iteration, role):
        path = directory / relative
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"missing or linked {role} artifact: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"empty {role} artifact: {path}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        _hash_file_into(digest, path)
        digest.update(b"\0")
    return digest.hexdigest()


def fingerprint_immutable_inputs(
    contract, run_directories, iteration, *, roles=("b1", "b2", "e0")
):
    """Hash canonical inputs and selected immutable run artifacts, without writes."""
    if not roles or len(roles) != len(set(roles)) or any(role not in _ROLES for role in roles):
        raise ValueError("roles must be a nonempty, unique subset of b1/b2/e0")
    result = {
        "canonical_source_tree_sha256": _hash_tree(contract["canonical_source_root"]),
        "canonical_prior_tree_sha256": _hash_tree(
            contract["canonical_aligned_prior_root"]
        ),
    }
    for role in roles:
        result[f"runs.{role}"] = _hash_run_artifacts(
            run_directories[role], iteration, role
        )
    return result


def _role_exact(name, actual, expected):
    return {
        "name": name,
        **{role: actual[role] for role in _ROLES},
        "expected": {role: expected[role] for role in _ROLES},
    }


def _same(value):
    return {role: value for role in _ROLES}


def _normalized_command(value):
    return " ".join(value.split()) if isinstance(value, str) and value.strip() else None


def _valid_utc(value):
    if not isinstance(value, str):
        return None
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return result if result.tzinfo is not None else None


def confirmation_invariants(
    contract,
    run_directories,
    run_contracts,
    resolved_configs,
    fingerprints_before,
    *,
    iteration,
    evaluation_iterations,
    expected_baseline_commit,
    expected_e0_commit,
    expected_dataset_sha,
    expected_prior_sha,
):
    """Return named exact checks tying an unseen E0 to its frozen preflight."""
    if not isinstance(contract, dict):
        raise ValueError("confirmation contract must be a mapping")
    checks = []

    def add(name, actual, expected):
        checks.append(_role_exact(name, actual, expected))

    add("confirmation.contract_version", _same(contract.get("contract_version")), _same(1))
    confirmation_id = contract.get("confirmation_id")
    add(
        "confirmation.id_present",
        _same(isinstance(confirmation_id, str) and bool(confirmation_id.strip())),
        _same(True),
    )
    add(
        "confirmation.e0_absent_at_preflight",
        _same(contract.get("e0_output_absent_at_preflight") is True),
        _same(True),
    )
    paths = contract.get("paths") if isinstance(contract.get("paths"), dict) else {}
    add(
        "confirmation.run_paths",
        {role: str(Path(run_directories[role]).resolve()) for role in _ROLES},
        {role: paths.get(role) for role in _ROLES},
    )
    commits = contract.get("commits") if isinstance(contract.get("commits"), dict) else {}
    expected_commits = {
        "b1": expected_baseline_commit,
        "b2": expected_baseline_commit,
        "e0": expected_e0_commit,
    }
    add("confirmation.role_commits", {role: commits.get(role) for role in _ROLES}, expected_commits)
    add(
        "confirmation.launcher_commits",
        {role: (run_contracts.get(role) or {}).get("commit") for role in _ROLES},
        {role: commits.get(role) for role in _ROLES},
    )
    commands = contract.get("normalized_commands")
    commands = commands if isinstance(commands, dict) else {}
    add(
        "confirmation.normalized_commands",
        {role: _normalized_command((run_contracts.get(role) or {}).get("command")) for role in _ROLES},
        {role: _normalized_command(commands.get(role)) for role in _ROLES},
    )
    add("confirmation.dataset_sha256", _same(contract.get("dataset_sha256")), _same(expected_dataset_sha))
    add("confirmation.prior_sha256", _same(contract.get("prior_sha256")), _same(expected_prior_sha))
    add(
        "confirmation.launcher_dataset_sha256",
        {role: (run_contracts.get(role) or {}).get("dataset_manifest_sha256") for role in _ROLES},
        _same(contract.get("dataset_sha256")),
    )
    add(
        "confirmation.launcher_prior_sha256",
        {role: (run_contracts.get(role) or {}).get("aligned_prior_sha256") for role in _ROLES},
        _same(contract.get("prior_sha256")),
    )
    for name, contract_key, launcher_key, expected in (
        ("seed", "seed", "semantic_seed", 0),
        ("resolution", "resolution", "resolution", 2),
        ("iteration", "iteration", "iterations", iteration),
        ("evaluation_iterations", "evaluation_iterations", "evaluation_iterations", list(evaluation_iterations)),
    ):
        add(f"confirmation.{name}", _same(contract.get(contract_key)), _same(expected))
        add(
            f"confirmation.launcher_{name}",
            {role: ((run_contracts.get(role) or {}).get("training_config") or {}).get(launcher_key) for role in _ROLES},
            _same(contract.get(contract_key)),
        )
    e0_config = resolved_configs.get("e0") or {}
    e0_core = e0_config.get("core") or {}
    e0_model = e0_config.get("model") or {}
    e0_optimization = e0_config.get("optimization") or {}
    for name, actual, expected in (
        ("legacy_path", e0_config.get("training_path"), "legacy"),
        ("seed", e0_core.get("seed"), contract.get("seed")),
        ("resolution", e0_model.get("resolution"), contract.get("resolution")),
        ("iteration", e0_optimization.get("iterations"), contract.get("iteration")),
    ):
        add(f"confirmation.e0_resolved_{name}", {"b1": True, "b2": True, "e0": actual}, {"b1": True, "b2": True, "e0": expected})
    for name, contract_key, fingerprint_key in (
        ("source_tree", "canonical_source_tree_sha256", "canonical_source_tree_sha256"),
        ("prior_tree", "canonical_prior_tree_sha256", "canonical_prior_tree_sha256"),
    ):
        add(
            f"confirmation.{name}_sha256",
            _same(fingerprints_before.get(fingerprint_key)),
            _same(contract.get(contract_key)),
        )
    baseline_hashes = contract.get("baseline_artifact_fingerprints")
    baseline_hashes = baseline_hashes if isinstance(baseline_hashes, dict) else {}
    add(
        "confirmation.baseline_artifact_fingerprints",
        {"b1": fingerprints_before.get("runs.b1"), "b2": fingerprints_before.get("runs.b2"), "e0": True},
        {"b1": baseline_hashes.get("b1"), "b2": baseline_hashes.get("b2"), "e0": True},
    )
    preflight_time = _valid_utc(contract.get("preflight_utc"))
    try:
        e0_start_time = _valid_utc(
            (Path(run_directories["e0"]) / "start_utc.txt").read_text(encoding="utf-8").strip()
        )
    except OSError:
        e0_start_time = None
    add(
        "confirmation.preflight_before_e0_start",
        _same(
            preflight_time is not None
            and e0_start_time is not None
            and preflight_time < e0_start_time
        ),
        _same(True),
    )
    return checks
