"""Read-only G0 audit for baseline/baseline-repeat/E0 checkpoint triplets."""

from argparse import ArgumentParser
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import torch


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from scripts.diagnostics.compare_feature_off import (  # noqa: E402
    evaluate_triplet_report,
    load_run,
)


ROLES = ("b1", "b2", "e0")
CAPTURE_NAMES = (
    "active_sh_degree",
    "xyz",
    "knn_f",
    "features_dc",
    "features_rest",
    "scaling",
    "rotation",
    "opacity",
    "max_radii2D",
    "max_weight",
    "xyz_gradient_accum",
    "xyz_gradient_accum_abs",
    "denom",
    "denom_abs",
    "optimizer",
    "spatial_lr_scale",
)
_ERROR_PATTERNS = {
    "traceback": re.compile(r"Traceback", re.IGNORECASE),
    "runtime_error": re.compile(r"RuntimeError", re.IGNORECASE),
    "cuda_error": re.compile(r"CUDA\s+error", re.IGNORECASE),
    "device_side_assert": re.compile(r"device-side\s+assert", re.IGNORECASE),
    "assertion_error": re.compile(r"AssertionError", re.IGNORECASE),
    "nonfinite_token": re.compile(r"(?<![A-Za-z0-9_])(?:nan|inf)(?![A-Za-z0-9_])", re.IGNORECASE),
}


def _json_safe(value):
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError("only scalar tensors can be embedded in exact metadata")
        return value.detach().cpu().item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise ValueError(f"metadata value is not JSON-safe: {type(value).__name__}")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_pair_stats(left, right, *, chunk_size=1_000_000):
    """Return exact and float64 distance statistics without broadcasting."""
    if not torch.is_tensor(left) or not torch.is_tensor(right):
        raise TypeError("tensor_pair_stats requires two Torch tensors")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    shape_equal = tuple(left.shape) == tuple(right.shape)
    dtype_equal = left.dtype == right.dtype
    result = {
        "shape_equal": shape_equal,
        "dtype_equal": dtype_equal,
        "finite": False,
        "exact": False,
        "element_count": left.numel() if shape_equal else None,
        "mismatch_count": None,
        "max_abs": None,
        "mean_abs": None,
        "rmse": None,
    }
    if not shape_equal:
        return result

    left_flat = left.detach().reshape(-1).cpu()
    right_flat = right.detach().reshape(-1).cpu()
    result["exact"] = dtype_equal and torch.equal(left_flat, right_flat)
    if left_flat.numel() == 0:
        result.update(
            finite=True,
            mismatch_count=0,
            max_abs=0.0,
            mean_abs=0.0,
            rmse=0.0,
        )
        return result

    sum_abs = 0.0
    sum_squared = 0.0
    max_abs = 0.0
    mismatch_count = 0
    for start in range(0, left_flat.numel(), chunk_size):
        stop = min(start + chunk_size, left_flat.numel())
        left_chunk = left_flat[start:stop]
        right_chunk = right_flat[start:stop]
        if not bool(torch.isfinite(left_chunk).all()) or not bool(
            torch.isfinite(right_chunk).all()
        ):
            return result
        difference = left_chunk.to(torch.float64) - right_chunk.to(torch.float64)
        absolute = difference.abs()
        sum_abs += float(absolute.sum().item())
        sum_squared += float((difference * difference).sum().item())
        max_abs = max(max_abs, float(absolute.max().item()))
        mismatch_count += int(torch.count_nonzero(left_chunk != right_chunk).item())

    count = left_flat.numel()
    result.update(
        finite=True,
        mismatch_count=mismatch_count,
        max_abs=max_abs,
        mean_abs=sum_abs / count,
        rmse=math.sqrt(sum_squared / count),
    )
    return result


def unpack_legacy_checkpoint(payload, expected_iteration):
    """Validate the frozen legacy ``(capture16, iteration)`` checkpoint schema."""
    if not isinstance(payload, tuple) or len(payload) != 2:
        raise ValueError("outer checkpoint must be a two-field tuple")
    capture, iteration = payload
    if not isinstance(capture, tuple) or len(capture) != len(CAPTURE_NAMES):
        raise ValueError("capture must be a 16-field tuple")
    if iteration != expected_iteration:
        raise ValueError(
            f"checkpoint iteration {iteration!r} does not match {expected_iteration}"
        )
    return capture


def name_capture(capture):
    if not isinstance(capture, tuple) or len(capture) != len(CAPTURE_NAMES):
        raise ValueError("capture must be a 16-field tuple")
    return dict(zip(CAPTURE_NAMES, capture))


def _state_for_parameter(state, parameter_id):
    if parameter_id in state:
        return state[parameter_id]
    string_id = str(parameter_id)
    if string_id in state:
        return state[string_id]
    raise ValueError(f"optimizer state missing parameter id {parameter_id!r}")


def normalize_optimizer(optimizer_state):
    """Separate exact optimizer structure from numeric tensor state."""
    if not isinstance(optimizer_state, dict):
        raise ValueError("optimizer state must be a mapping")
    state = optimizer_state.get("state")
    groups = optimizer_state.get("param_groups")
    if not isinstance(state, dict) or not isinstance(groups, list):
        raise ValueError("optimizer state requires state and param_groups")

    structure = {"group_names": [], "groups": {}}
    tensors = {}
    for group in groups:
        if not isinstance(group, dict) or not isinstance(group.get("name"), str):
            raise ValueError("every optimizer group requires a string name")
        name = group["name"]
        if name in structure["groups"]:
            raise ValueError(f"duplicate optimizer group name: {name}")
        parameter_ids = group.get("params")
        if not isinstance(parameter_ids, list) or len(parameter_ids) != 1:
            raise ValueError(f"optimizer group {name!r} must contain exactly one parameter")
        parameter_state = _state_for_parameter(state, parameter_ids[0])
        if not isinstance(parameter_state, dict):
            raise ValueError(f"optimizer state for group {name!r} is not a mapping")

        group_structure = {
            key: _json_safe(value)
            for key, value in group.items()
            if key not in ("name", "params")
        }
        group_structure["state_keys"] = sorted(str(key) for key in parameter_state)
        step = parameter_state.get("step")
        if step is None:
            raise ValueError(f"optimizer group {name!r} has no step counter")
        if torch.is_tensor(step):
            if step.numel() != 1:
                raise ValueError(f"optimizer group {name!r} has a non-scalar step")
            group_structure["step"] = float(step.detach().cpu().item())
        elif isinstance(step, (int, float)) and not isinstance(step, bool):
            group_structure["step"] = float(step)
        else:
            raise ValueError(f"optimizer group {name!r} has an invalid step counter")

        structure["group_names"].append(name)
        structure["groups"][name] = group_structure
        for state_name, value in parameter_state.items():
            if torch.is_tensor(value):
                tensors[f"optimizer.{name}.{state_name}"] = value.detach().cpu()
    return structure, tensors


def finalize_gate(gate, *, exploratory):
    """Attach audit status without allowing a 500 replay to promote G0."""
    finalized = dict(gate)
    equivalent = gate.get("equivalent") is True
    finalized.update(
        audit_completed=True,
        exploratory=bool(exploratory),
        numerical_equivalent=equivalent,
        g0_equivalent=equivalent and not exploratory,
        exit_code=0 if equivalent else 1,
    )
    return finalized


def load_checkpoint(run_directory, iteration):
    path = Path(run_directory) / f"chkpnt{iteration}.pth"
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    capture = unpack_legacy_checkpoint(payload, iteration)
    return name_capture(capture), {"path": str(path.resolve()), "sha256": _sha256(path)}


def _load_tensor_mapping(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"tensor artifact is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) and torch.is_tensor(value)
        for key, value in payload.items()
    ):
        raise ValueError(f"tensor artifact must be a string-to-tensor mapping: {path}")
    return {key: value.detach().cpu() for key, value in payload.items()}


def _exact(name, values, expected=None):
    invariant = {"name": name, **{role: _json_safe(values[role]) for role in ROLES}}
    if expected is not None:
        invariant["expected"] = {
            role: _json_safe(expected[role]) for role in ROLES
        }
    return invariant


def _tensor_field(name, tensors):
    return {
        "name": name,
        "dtype": {role: str(tensors[role].dtype) for role in ROLES},
        "shape": {role: list(tensors[role].shape) for role in ROLES},
        "b1_b2": tensor_pair_stats(tensors["b1"], tensors["b2"]),
        "b1_e0": tensor_pair_stats(tensors["b1"], tensors["e0"]),
        "b2_e0": tensor_pair_stats(tensors["b2"], tensors["e0"]),
    }


def _read_json(path):
    path = Path(path)
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _log_safety(run_directory):
    path = Path(run_directory) / "train.log"
    text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    return {
        "log_present": path.is_file(),
        **{name: len(pattern.findall(text)) for name, pattern in _ERROR_PATTERNS.items()},
    }


def _evaluation_map(snapshot):
    return {
        (entry["iteration"], entry["split"]): entry
        for entry in snapshot["evaluations"]
    }


def _point_cloud_diagnostic(snapshot, iteration):
    entry = snapshot["point_clouds"].get(str(iteration))
    if entry is None:
        return None
    return {
        "vertices": entry["vertices"],
        "sha256": entry["sha256"],
        "bytes": entry["bytes"],
    }


def _feature_off_metadata_valid(resolved_config, run_identity, contract):
    if resolved_config is None or run_identity is None or contract is None:
        return False
    core = resolved_config.get("core")
    if not isinstance(core, dict):
        return False
    enabled_flags = (
        "core_shadow_mode",
        "enable_observation_calibration",
        "enable_dual_reliability",
        "enable_abstention",
        "enable_parameter_routing",
        "enable_gradient_projection",
        "enable_reliability_lifecycle",
    )
    if any(core.get(flag) is not False for flag in enabled_flags):
        return False
    return (
        run_identity.get("commit") == contract.get("commit")
        and run_identity.get("dirty") is False
        and run_identity.get("seed") == core.get("seed")
    )


def _add_contract_invariants(
    exact_invariants,
    contracts,
    resolved_configs,
    run_identities,
    *,
    exploratory,
    expected_baseline_commit,
    expected_e0_commit,
    expected_dataset_sha,
    expected_prior_sha,
):
    if exploratory:
        return
    if not expected_baseline_commit or not expected_e0_commit:
        raise ValueError("confirmation audit requires both expected commit arguments")

    present = {role: contracts[role] is not None for role in ROLES}
    exact_invariants.append(_exact("launcher_contract.present", present, {role: True for role in ROLES}))
    if not all(present.values()):
        return

    expected_roles = {"b1": "b1", "b2": "b2", "e0": "e0"}
    expected_commits = {
        "b1": expected_baseline_commit,
        "b2": expected_baseline_commit,
        "e0": expected_e0_commit,
    }
    exact_invariants.extend(
        [
            _exact("launcher_contract.role", {r: contracts[r].get("role") for r in ROLES}, expected_roles),
            _exact("launcher_contract.commit", {r: contracts[r].get("commit") for r in ROLES}, expected_commits),
            _exact("launcher_contract.clean", {r: contracts[r].get("dirty") for r in ROLES}, {r: False for r in ROLES}),
            _exact("launcher_contract.command_present", {r: bool(contracts[r].get("command")) for r in ROLES}, {r: True for r in ROLES}),
        ]
    )
    for field in (
        "python",
        "torch",
        "gpu",
        "dataset_manifest_sha256",
        "aligned_prior_sha256",
        "training_config",
    ):
        values = {role: contracts[role].get(field) for role in ROLES}
        exact_invariants.append(_exact(f"launcher_contract.{field}", values))
    if expected_dataset_sha:
        exact_invariants.append(
            _exact(
                "launcher_contract.expected_dataset",
                {r: contracts[r].get("dataset_manifest_sha256") for r in ROLES},
                {r: expected_dataset_sha for r in ROLES},
            )
        )
    if expected_prior_sha:
        exact_invariants.append(
            _exact(
                "launcher_contract.expected_prior",
                {r: contracts[r].get("aligned_prior_sha256") for r in ROLES},
                {r: expected_prior_sha for r in ROLES},
            )
        )
    exact_invariants.append(
        _exact(
            "e0_metadata_agrees_with_contract",
            {
                "b1": True,
                "b2": True,
                "e0": _feature_off_metadata_valid(
                    resolved_configs["e0"], run_identities["e0"], contracts["e0"]
                ),
            },
            {r: True for r in ROLES},
        )
    )


def build_report(
    run_directories,
    iteration,
    *,
    exploratory=False,
    evaluation_iterations=None,
    expected_baseline_commit=None,
    expected_e0_commit=None,
    expected_dataset_sha=None,
    expected_prior_sha=None,
):
    """Normalize three immutable run directories into the versioned audit schema."""
    run_directories = {role: Path(run_directories[role]).resolve() for role in ROLES}
    for role, directory in run_directories.items():
        if not directory.is_dir():
            raise FileNotFoundError(f"{role} run directory is missing: {directory}")

    captures = {}
    checkpoint_diagnostics = {}
    optimizer_structures = {}
    optimizer_tensors = {}
    app_tensors = {}
    snapshots = {}
    safety = {}
    contracts = {}
    resolved_configs = {}
    run_identities = {}
    for role in ROLES:
        directory = run_directories[role]
        captures[role], checkpoint_diagnostics[role] = load_checkpoint(directory, iteration)
        optimizer_structures[role], optimizer_tensors[role] = normalize_optimizer(
            captures[role]["optimizer"]
        )
        app_path = directory / "app_model" / f"iteration_{iteration}" / "app.pth"
        app_tensors[role] = _load_tensor_mapping(app_path)
        snapshots[role] = load_run(directory)
        safety[role] = _log_safety(directory)
        contracts[role] = _read_json(directory / "g0_run_contract.json")
        resolved_configs[role] = _read_json(directory / "resolved_config.json")
        run_identities[role] = _read_json(directory / "run_identity.json")

    exact_invariants = []
    expected_iteration = {role: iteration for role in ROLES}
    exact_invariants.extend(
        [
            _exact("checkpoint.schema", {role: "legacy_capture16" for role in ROLES}),
            _exact("checkpoint.iteration", expected_iteration, expected_iteration),
            _exact(
                "active_sh_degree",
                {role: captures[role]["active_sh_degree"] for role in ROLES},
            ),
            _exact(
                "spatial_lr_scale",
                {role: captures[role]["spatial_lr_scale"] for role in ROLES},
            ),
            _exact(
                "gaussian_count",
                {role: int(captures[role]["xyz"].shape[0]) for role in ROLES},
            ),
            _exact("optimizer.structure", optimizer_structures),
            _exact(
                "optimizer.tensor_keys",
                {role: sorted(optimizer_tensors[role]) for role in ROLES},
            ),
            _exact(
                "application.tensor_keys",
                {role: sorted(app_tensors[role]) for role in ROLES},
            ),
        ]
    )

    capture_tensor_names = CAPTURE_NAMES[1:14]
    for name in capture_tensor_names:
        if not all(torch.is_tensor(captures[role][name]) for role in ROLES):
            raise ValueError(f"capture field {name!r} is not a tensor in every run")
        exact_invariants.append(
            _exact(
                f"capture.{name}.dtype",
                {role: str(captures[role][name].dtype) for role in ROLES},
            )
        )
        exact_invariants.append(
            _exact(
                f"capture.{name}.shape",
                {role: list(captures[role][name].shape) for role in ROLES},
            )
        )

    for family_name, family in (
        ("optimizer", optimizer_tensors),
        ("application", app_tensors),
    ):
        common_keys = sorted(set.intersection(*(set(family[role]) for role in ROLES)))
        for key in common_keys:
            exact_invariants.append(
                _exact(
                    f"{family_name}.{key}.dtype",
                    {role: str(family[role][key].dtype) for role in ROLES},
                )
            )
            exact_invariants.append(
                _exact(
                    f"{family_name}.{key}.shape",
                    {role: list(family[role][key].shape) for role in ROLES},
                )
            )

    required_artifacts = {
        role: {
            "checkpoint": (run_directories[role] / f"chkpnt{iteration}.pth").is_file(),
            "application": (
                run_directories[role]
                / "app_model"
                / f"iteration_{iteration}"
                / "app.pth"
            ).is_file(),
            "log": (run_directories[role] / "train.log").is_file(),
            "exit_code": (run_directories[role] / "exit_code.txt").is_file(),
            "point_cloud": (
                run_directories[role]
                / "point_cloud"
                / f"iteration_{iteration}"
                / "point_cloud.ply"
            ).is_file(),
            "cfg_args": (run_directories[role] / "cfg_args").is_file(),
            "cfg_opts": (run_directories[role] / "cfg_opts").is_file(),
        }
        for role in ROLES
    }
    for artifact in next(iter(required_artifacts.values())):
        exact_invariants.append(
            _exact(
                f"artifact.{artifact}.present",
                {role: required_artifacts[role][artifact] for role in ROLES},
                {role: True for role in ROLES},
            )
        )
    exact_invariants.extend(
        [
            _exact(
                "run.exit_code",
                {role: snapshots[role]["exit_code"] for role in ROLES},
                {role: 0 for role in ROLES},
            ),
            _exact(
                "run.final_points",
                {role: snapshots[role]["final_points"] for role in ROLES},
            ),
        ]
    )
    for safety_key in _ERROR_PATTERNS:
        exact_invariants.append(
            _exact(
                f"log.{safety_key}_count",
                {role: safety[role][safety_key] for role in ROLES},
                {role: 0 for role in ROLES},
            )
        )

    _add_contract_invariants(
        exact_invariants,
        contracts,
        resolved_configs,
        run_identities,
        exploratory=exploratory,
        expected_baseline_commit=expected_baseline_commit,
        expected_e0_commit=expected_e0_commit,
        expected_dataset_sha=expected_dataset_sha,
        expected_prior_sha=expected_prior_sha,
    )

    numeric_fields = [
        _tensor_field(
            f"capture.{name}",
            {role: captures[role][name] for role in ROLES},
        )
        for name in capture_tensor_names
    ]
    for family in (optimizer_tensors, app_tensors):
        common_keys = sorted(set.intersection(*(set(family[role]) for role in ROLES)))
        numeric_fields.extend(
            _tensor_field(key, {role: family[role][key] for role in ROLES})
            for key in common_keys
        )

    evaluation_iterations = (
        sorted(set(evaluation_iterations)) if evaluation_iterations else [iteration]
    )
    evaluation_maps = {role: _evaluation_map(snapshots[role]) for role in ROLES}
    requested_keys = sorted(
        {
            key
            for role in ROLES
            for key in evaluation_maps[role]
            if key[0] in evaluation_iterations
        }
    )
    scalar_metrics = []
    for eval_iteration in evaluation_iterations:
        keys_at_iteration = [key for key in requested_keys if key[0] == eval_iteration]
        exact_invariants.append(
            _exact(
                f"evaluation.iteration_{eval_iteration}.present",
                {
                    role: any(key[0] == eval_iteration for key in evaluation_maps[role])
                    for role in ROLES
                },
                {role: True for role in ROLES},
            )
        )
        for key in keys_at_iteration:
            for metric in ("l1", "psnr"):
                scalar_metrics.append(
                    {
                        "name": f"evaluation[{key[0]}:{key[1]}].{metric}",
                        **{
                            role: (
                                evaluation_maps[role].get(key, {}).get(metric)
                            )
                            for role in ROLES
                        },
                    }
                )

    learned_ply = {
        role: _point_cloud_diagnostic(snapshots[role], iteration) for role in ROLES
    }
    report = {
        "schema_version": 1,
        "iteration": iteration,
        "runs": {
            role: {
                "directory": str(run_directories[role]),
                "snapshot": snapshots[role],
                "launcher_contract": contracts[role],
                "resolved_config_present": resolved_configs[role] is not None,
                "run_identity_present": run_identities[role] is not None,
            }
            for role in ROLES
        },
        "exact_invariants": exact_invariants,
        "numeric_fields": numeric_fields,
        "scalar_metrics": scalar_metrics,
        "diagnostics": {
            "checkpoint": checkpoint_diagnostics,
            "learned_ply": learned_ply,
            "application_sha256": {
                role: _sha256(
                    run_directories[role]
                    / "app_model"
                    / f"iteration_{iteration}"
                    / "app.pth"
                )
                for role in ROLES
            },
            "cfg_args_sha256": {
                role: _sha256(run_directories[role] / "cfg_args") for role in ROLES
            },
            "cfg_opts_sha256": {
                role: _sha256(run_directories[role] / "cfg_opts") for role in ROLES
            },
            "required_artifacts": required_artifacts,
            "log_safety": safety,
            "contract_files_present": {
                role: contracts[role] is not None for role in ROLES
            },
        },
    }
    return report


def _output_is_outside_runs(output, run_directories):
    output = output.resolve()
    return all(
        output != directory.resolve() and directory.resolve() not in output.parents
        for directory in run_directories.values()
    )


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("baseline_1", type=Path)
    parser.add_argument("baseline_2", type=Path)
    parser.add_argument("e0", type=Path)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation-iterations", type=int, nargs="*")
    parser.add_argument("--exploratory", action="store_true")
    parser.add_argument("--expected-baseline-commit")
    parser.add_argument("--expected-e0-commit")
    parser.add_argument("--expected-dataset-sha")
    parser.add_argument("--expected-prior-sha")
    args = parser.parse_args(argv)

    run_directories = {
        "b1": args.baseline_1.resolve(),
        "b2": args.baseline_2.resolve(),
        "e0": args.e0.resolve(),
    }
    try:
        if args.iteration <= 0:
            raise ValueError("iteration must be positive")
        if not _output_is_outside_runs(args.output, run_directories):
            raise ValueError("output must be outside all input run directories")
        report = build_report(
            run_directories,
            args.iteration,
            exploratory=args.exploratory,
            evaluation_iterations=args.evaluation_iterations,
            expected_baseline_commit=args.expected_baseline_commit,
            expected_e0_commit=args.expected_e0_commit,
            expected_dataset_sha=args.expected_dataset_sha,
            expected_prior_sha=args.expected_prior_sha,
        )
        report["gate"] = finalize_gate(
            evaluate_triplet_report(report), exploratory=args.exploratory
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except Exception as error:
        print(f"AUDIT_MALFORMED: {type(error).__name__}: {error}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "exploratory": report["gate"]["exploratory"],
                "numerical_equivalent": report["gate"]["numerical_equivalent"],
                "g0_equivalent": report["gate"]["g0_equivalent"],
                "exact_failure_count": len(report["gate"]["exact_failures"]),
                "numeric_failure_count": len(report["gate"]["numeric_failures"]),
                "exit_code": report["gate"]["exit_code"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return report["gate"]["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
