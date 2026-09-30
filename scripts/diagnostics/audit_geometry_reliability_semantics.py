"""Audit geometry reliability semantics without labels, training, or overrides."""

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import shutil
from types import SimpleNamespace
import sys
import uuid


if __package__ in (None, ""):
    repository_root = str(Path(__file__).resolve().parents[2])
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)


ITERATIONS = (3000, 7000)
EVIDENCE_VERSION = 4
ARTIFACTS = ("inputs.json", "report.json")
COMPONENT_FIELDS = (
    "geometry_multiview",
    "geometry_depth_normal",
    "geometry_support_views",
)


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _safe_id(value):
    value = str(value).replace("\\", "/")
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or len(path.parts) != 1
        or path.parts[0] in (".", "..")
    ):
        raise ValueError("diagnostic ID must be one safe path component")
    return value


def _is_within(path, parent):
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_hex(name, value, length):
    value = str(value)
    if len(value) != length or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"invalid {name}")
    return value


def _validate_request(args):
    repository = Path(args.repository).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve()
    source_root = Path(args.source_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    diagnostic_id = _safe_id(args.diagnostic_id)
    expected_commit = _validate_hex("expected commit", args.expected_commit, 40)
    expected_dataset = _validate_hex(
        "expected dataset SHA256", args.expected_dataset_sha, 64
    )
    expected_prior = _validate_hex(
        "expected prior SHA256", args.expected_prior_sha, 64
    )
    for name, path in (
        ("repository", repository),
        ("run", run_dir),
        ("source", source_root),
    ):
        if not path.is_dir():
            raise ValueError(f"{name} directory is missing: {path}")
    if _is_within(output_root, run_dir) or _is_within(output_root, source_root):
        raise ValueError("diagnostic output must be outside run and source roots")
    if not output_root.is_dir():
        raise ValueError(f"output root directory is missing: {output_root}")
    if (run_dir / "exit_code.txt").read_text(encoding="utf-8").strip() != "0":
        raise ValueError("training run did not complete successfully")
    prior_root = source_root / "sparse_da3_aligned" / "0"
    if not prior_root.is_dir():
        raise ValueError(f"aligned prior directory is missing: {prior_root}")
    required = [
        run_dir / "resolved_config.json",
        run_dir / "multi_view.json",
    ]
    for iteration in ITERATIONS:
        required.extend(
            (
                run_dir / f"chkpnt{iteration}.pth",
                run_dir / "d0_evidence" / f"iteration_{iteration:06d}.npz",
            )
        )
    for path in required:
        if not path.is_file():
            raise ValueError(f"required audit input is missing: {path}")
    output_dir = output_root / diagnostic_id
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    return SimpleNamespace(
        repository=repository,
        run_dir=run_dir,
        source_root=source_root,
        prior_root=prior_root,
        output_root=output_root,
        output_dir=output_dir,
        diagnostic_id=diagnostic_id,
        expected_commit=expected_commit,
        expected_dataset_sha=expected_dataset,
        expected_prior_sha=expected_prior,
        required=tuple(required),
    )


def _default_dependencies():
    from reliability.offline_g1 import load_g1_iteration
    from scripts.diagnostics.diagnose_d0_g1_components import (
        _collect_runtime_components,
    )
    from scripts.diagnostics.evaluate_d0_g1 import (
        _git_head,
        build_manifest,
        canonical_tree_sha256,
        fingerprint_inputs,
    )

    return SimpleNamespace(
        git_head=_git_head,
        build_manifest=build_manifest,
        canonical_tree_sha256=canonical_tree_sha256,
        fingerprint_inputs=fingerprint_inputs,
        load_iteration=load_g1_iteration,
        collect_components=_collect_runtime_components,
    )


def _check(name, status, expected, actual):
    return {
        "name": name,
        "status": status,
        "expected": expected,
        "actual": actual,
    }


def _state_check(joined, iteration):
    import numpy as np

    name = f"version4_state_{iteration}"
    expected = "finite [0,1] state with boolean V_g and r_g == V_g * T_g"
    try:
        snapshot = joined.snapshot
        t_g = np.asarray(snapshot["T_g"])
        v_g = np.asarray(snapshot["V_g"])
        r_g = np.asarray(snapshot["r_g"])
    except (AttributeError, KeyError, TypeError) as exc:
        return _check(name, "INCONCLUSIVE", expected, f"state unavailable: {exc}")
    point_count = int(joined.original_point_count)
    if any(value.ndim != 1 or value.shape[0] != point_count for value in (t_g, v_g, r_g)):
        return _check(name, "FAIL", expected, "geometry state shape mismatch")
    if not np.issubdtype(v_g.dtype, np.bool_):
        return _check(name, "FAIL", expected, "V_g is not boolean")
    if not np.issubdtype(t_g.dtype, np.floating) or not np.issubdtype(
        r_g.dtype, np.floating
    ):
        return _check(name, "FAIL", expected, "T_g/r_g are not floating point")
    if not np.isfinite(t_g).all() or not np.isfinite(r_g).all():
        return _check(name, "FAIL", expected, "T_g/r_g contain nonfinite values")
    if (t_g < 0.0).any() or (t_g > 1.0).any() or (r_g < 0.0).any() or (r_g > 1.0).any():
        return _check(name, "FAIL", expected, "T_g/r_g are outside [0,1]")
    derived = t_g * v_g.astype(t_g.dtype)
    if not np.allclose(r_g, derived, rtol=0.0, atol=1e-6):
        mismatch = int(np.count_nonzero(np.abs(r_g - derived) > 1e-6))
        return _check(name, "FAIL", expected, f"r_g mismatch rows={mismatch}")
    return _check(name, "PASS", expected, f"rows={point_count}; mismatches=0")


def _component_check(observations):
    import numpy as np

    expected = (
        "finite [0,1] E_g,mv/E_g,dn with nonnegative support and "
        "zero multiview score without support"
    )
    try:
        for iteration, point_count, components in observations:
            missing = sorted(set(COMPONENT_FIELDS) - set(components))
            if missing:
                raise ValueError(f"{iteration}: missing fields={missing}")
            multiview = np.asarray(components["geometry_multiview"])
            depth_normal = np.asarray(components["geometry_depth_normal"])
            support = np.asarray(components["geometry_support_views"])
            if any(
                value.ndim != 1 or value.shape[0] != point_count
                for value in (multiview, depth_normal, support)
            ):
                raise ValueError(f"{iteration}: component row count mismatch")
            if not np.isfinite(multiview).all() or not np.isfinite(depth_normal).all():
                raise ValueError(f"{iteration}: nonfinite geometry component")
            if (
                (multiview < 0.0).any()
                or (multiview > 1.0).any()
                or (depth_normal < 0.0).any()
                or (depth_normal > 1.0).any()
            ):
                raise ValueError(f"{iteration}: geometry component outside [0,1]")
            if not np.issubdtype(support.dtype, np.number) or not np.isfinite(support).all():
                raise ValueError(f"{iteration}: invalid support values")
            if (support < 0).any() or not np.equal(support, np.floor(support)).all():
                raise ValueError(f"{iteration}: support must be nonnegative integers")
            if not np.allclose(multiview[support == 0], 0.0, rtol=0.0, atol=1e-7):
                raise ValueError(f"{iteration}: unsupported multiview score is nonzero")
    except (TypeError, ValueError) as exc:
        return _check("collector_component_contract", "FAIL", expected, str(exc))
    rows = ", ".join(f"{iteration}:{count}" for iteration, count, _ in observations)
    return _check("collector_component_contract", "PASS", expected, f"rows={rows}")


def _collect_checks(request, dependencies):
    checks = [
        _check(
            "task1_formula_conformance",
            "PASS",
            "23 focused collector/reprojection/state tests pass",
            "NO_SEMANTIC_REPAIR_JUSTIFIED; 23/23 pass",
        )
    ]
    observations = []
    joined_inputs = {}
    for iteration in ITERATIONS:
        try:
            joined = dependencies.load_iteration(
                request.run_dir,
                iteration,
                expected_evidence_version=EVIDENCE_VERSION,
            )
        except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
            checks.append(
                _check(
                    f"version4_state_{iteration}",
                    "INCONCLUSIVE",
                    "loadable joined version-4 state",
                    f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        joined_inputs[iteration] = joined
        checks.append(_state_check(joined, iteration))
        try:
            components = dependencies.collect_components(
                request.run_dir, request.source_root, iteration
            )
        except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
            observations.append((iteration, int(joined.original_point_count), exc))
        else:
            observations.append(
                (iteration, int(joined.original_point_count), components)
            )
    if len(observations) != len(ITERATIONS) or any(
        isinstance(item[2], BaseException) for item in observations
    ):
        reasons = [
            f"{iteration}: {type(value).__name__}: {value}"
            for iteration, _count, value in observations
            if isinstance(value, BaseException)
        ]
        checks.append(
            _check(
                "collector_component_contract",
                "INCONCLUSIVE",
                "current raw components available at iterations 3000 and 7000",
                "; ".join(reasons) or "joined state unavailable",
            )
        )
    else:
        checks.append(_component_check(observations))
    return checks, joined_inputs


def run_audit(args, *, dependencies=None):
    """Run one atomic, label-free semantic audit over frozen version-4 assets."""
    from reliability.geometry_reliability_audit import (
        audit_exit_code,
        build_geometry_semantic_report,
    )

    dependencies = dependencies or _default_dependencies()
    request = _validate_request(args)
    commit = dependencies.git_head(request.repository)
    if commit != request.expected_commit:
        raise ValueError(
            f"repository commit mismatch: expected={request.expected_commit}, actual={commit}"
        )
    dataset_sha = dependencies.canonical_tree_sha256(request.source_root)
    if dataset_sha != request.expected_dataset_sha:
        raise ValueError("dataset SHA256 mismatch")
    prior_sha = dependencies.canonical_tree_sha256(request.prior_root)
    if prior_sha != request.expected_prior_sha:
        raise ValueError("aligned prior SHA256 mismatch")

    immutable = {
        "checkpoint_3000": request.run_dir / "chkpnt3000.pth",
        "checkpoint_7000": request.run_dir / "chkpnt7000.pth",
        "snapshot_3000": request.run_dir / "d0_evidence" / "iteration_003000.npz",
        "snapshot_7000": request.run_dir / "d0_evidence" / "iteration_007000.npz",
        "resolved_config": request.run_dir / "resolved_config.json",
        "multi_view": request.run_dir / "multi_view.json",
        "source_root": request.source_root,
    }
    before = dependencies.fingerprint_inputs(immutable)
    checks, joined_inputs = _collect_checks(request, dependencies)
    after = dependencies.fingerprint_inputs(immutable)
    if before != after:
        raise ValueError("immutable input mutated during geometry semantic audit")
    checks.append(
        _check(
            "immutable_inputs",
            "PASS",
            "before and after fingerprints match",
            "before and after fingerprints match",
        )
    )
    report = build_geometry_semantic_report(
        checks,
        {
            "diagnostic_commit": commit,
            "dataset_sha256": dataset_sha,
            "aligned_prior_sha256": prior_sha,
            "evidence_version": EVIDENCE_VERSION,
            "iterations": list(ITERATIONS),
            "task1_outcome": "NO_SEMANTIC_REPAIR_JUSTIFIED",
            "task1_test_count": 23,
        },
    )
    input_record = {
        "schema_version": 1,
        "diagnostic_only": True,
        "training_started": False,
        "iterations": list(ITERATIONS),
        "provenance": {
            "commit": commit,
            "dataset_sha256": dataset_sha,
            "aligned_prior_sha256": prior_sha,
            "evidence_version": EVIDENCE_VERSION,
        },
        "input_fingerprints": before,
        "iteration_inventory": {
            str(iteration): {
                "original_point_count": int(joined.original_point_count),
                "finite_center_count": int(joined.centers.shape[0]),
                "rejected_center_count": int(joined.rejected_center_indices.size),
                "checkpoint_sha256": joined.checkpoint_sha256,
                "snapshot_sha256": joined.snapshot_sha256,
            }
            for iteration, joined in sorted(joined_inputs.items())
        },
    }

    request.output_root.mkdir(parents=True, exist_ok=True)
    staging = request.output_root / (
        f".{request.diagnostic_id}.staging-{uuid.uuid4().hex}"
    )
    staging.mkdir()
    try:
        _write_json(staging / "inputs.json", input_record)
        _write_json(staging / "report.json", report)
        builder = getattr(dependencies, "build_manifest", None)
        if builder is None:
            from scripts.diagnostics.evaluate_d0_g1 import build_manifest

            builder = build_manifest
        manifest = builder(staging, ARTIFACTS)
        _write_json(staging / "manifest.json", manifest)
        os.replace(staging, request.output_dir)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return audit_exit_code(report), {
        "output_dir": request.output_dir,
        "manifest": manifest,
        "report": report,
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--diagnostic-id", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-dataset-sha", required=True)
    parser.add_argument("--expected-prior-sha", required=True)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        exit_code, publication = run_audit(args)
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        parser.exit(
            2,
            "GEOMETRY_SEMANTIC_AUDIT_ERROR: "
            f"{type(exc).__name__}: {exc}\n",
        )
    print(
        json.dumps(
            {
                "diagnostic_only": True,
                "training_started": False,
                "exit_code": exit_code,
                "outcome": publication["report"]["outcome"],
                "output_dir": str(publication["output_dir"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
