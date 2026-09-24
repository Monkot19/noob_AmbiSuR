"""Diagnose iteration-7000 D0/G1 components without revising the G1 gate."""

import argparse
import csv
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


DIAGNOSTIC_FILES = ("inputs.json", "report.json", "risk_bins.csv")


def _jsonable(value):
    import numpy as np

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _safe_confirmation_id(value):
    value = str(value).replace("\\", "/")
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or len(path.parts) != 1
        or path.parts[0] in (".", "..")
    ):
        raise ValueError("confirmation ID must be one safe path component")
    return value


def _is_within(path, parent):
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _assemble_component_arrays(refresh_inputs, sufficiency, consistency):
    """Convert the frozen collector component contract to independent arrays."""
    import numpy as np

    def as_numpy(value):
        return np.asarray(value.detach().cpu().numpy()).copy()

    return {
        "view_count": as_numpy(sufficiency.M_obs),
        "s_count": as_numpy(sufficiency.S_count),
        "s_angle": as_numpy(sufficiency.S_angle),
        "s_raw": as_numpy(sufficiency.S),
        "prior_confidence": as_numpy(refresh_inputs.prior_confidence),
        "prior_multiview": as_numpy(refresh_inputs.prior_multiview),
        "prior_support_views": as_numpy(refresh_inputs.prior_support_views),
        "geometry_multiview": as_numpy(refresh_inputs.geometry_multiview),
        "geometry_depth_normal": as_numpy(
            refresh_inputs.geometry_depth_normal
        ),
        "geometry_support_views": as_numpy(
            refresh_inputs.geometry_support_views
        ),
        "pg_raw_k": as_numpy(consistency.K_raw),
    }


def collect_component_arrays(refresh_inputs):
    """Expose current raw collector components; history stability is unavailable."""
    import torch

    from reliability.evidence import (
        compute_observation_sufficiency,
        compute_pg_consistency,
    )

    with torch.no_grad():
        sufficiency = compute_observation_sufficiency(
            refresh_inputs.pixel_hits,
            refresh_inputs.camera_centers,
            refresh_inputs.centers,
        )
        consistency = compute_pg_consistency(
            refresh_inputs.pg_weighted_support,
            refresh_inputs.pg_depth_error_sum,
            refresh_inputs.pg_normal_error_sum,
        )
    return _assemble_component_arrays(refresh_inputs, sufficiency, consistency)


def _restore_training_neighbors(cameras, multi_view_path):
    """Restore the frozen training neighbor graph in offline camera order."""
    cameras = list(cameras)
    name_to_index = {}
    for index, camera in enumerate(cameras):
        name = getattr(camera, "image_name", None)
        if not isinstance(name, str) or not name:
            raise ValueError("offline camera name must be a non-empty string")
        if name in name_to_index:
            raise ValueError(f"duplicate offline camera name: {name}")
        name_to_index[name] = index

    records = {}
    path = Path(multi_view_path)
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            raise ValueError(f"blank multi-view record at line {line_number}")
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"multi-view record must be an object at line {line_number}")
        ref_name = record.get("ref_name")
        nearest_names = record.get("nearest_name")
        if not isinstance(ref_name, str) or not ref_name:
            raise ValueError(f"invalid reference camera at line {line_number}")
        if ref_name not in name_to_index:
            raise ValueError(f"unknown reference camera: {ref_name}")
        if ref_name in records:
            raise ValueError(f"duplicate reference camera: {ref_name}")
        if not isinstance(nearest_names, list) or not all(
            isinstance(name, str) and name for name in nearest_names
        ):
            raise ValueError(f"invalid neighbor camera list for: {ref_name}")
        if len(nearest_names) != len(set(nearest_names)):
            raise ValueError(f"duplicate neighbor camera for: {ref_name}")
        unknown = [name for name in nearest_names if name not in name_to_index]
        if unknown:
            raise ValueError(f"unknown neighbor camera: {unknown[0]}")
        records[ref_name] = list(nearest_names)

    missing = [name for name in name_to_index if name not in records]
    if missing:
        raise ValueError(f"missing camera record: {missing[0]}")

    for camera in cameras:
        nearest_names = records[camera.image_name]
        camera.nearest_names = list(nearest_names)
        camera.nearest_id = [name_to_index[name] for name in nearest_names]


def _collect_runtime_components(run_dir, source_root, iteration):
    import torch

    from gaussian_renderer import render, render_normal
    from reliability.collector import D0EvidenceCollector
    from scripts.diagnostics.evaluate_d0_g1 import _load_render_runtime

    run_dir = Path(run_dir)
    config = json.loads(
        (run_dir / "resolved_config.json").read_text(encoding="utf-8")
    )
    optimization = SimpleNamespace(**config["optimization"])
    gaussians, cameras, pipeline, background = _load_render_runtime(
        run_dir, source_root, iteration
    )
    _restore_training_neighbors(cameras, run_dir / "multi_view.json")
    collector = D0EvidenceCollector(
        cameras,
        gaussians,
        render,
        pipeline,
        background,
        optimization,
        normal_from_depth_fn=render_normal,
    )
    try:
        with torch.no_grad():
            refresh_inputs = collector()
        return collect_component_arrays(refresh_inputs)
    finally:
        del collector, gaussians, cameras, pipeline, background
        torch.cuda.empty_cache()


def _default_dependencies():
    from reliability.offline_g1 import (
        closest_triangle_distances,
        load_g1_iteration,
        load_valid_mesh,
    )
    from scripts.diagnostics.evaluate_d0_g1 import (
        _git_head,
        _sha256_file,
        canonical_tree_sha256,
        fingerprint_inputs,
    )

    return SimpleNamespace(
        git_head=_git_head,
        canonical_tree_sha256=canonical_tree_sha256,
        sha256_file=_sha256_file,
        fingerprint_inputs=fingerprint_inputs,
        load_iteration=load_g1_iteration,
        load_mesh=load_valid_mesh,
        distance_query=closest_triangle_distances,
        collect_components=_collect_runtime_components,
    )


def _validate_request(args):
    run_dir = Path(args.run_dir).expanduser().resolve()
    source_root = Path(args.source_root).expanduser().resolve()
    gt_mesh = Path(args.gt_mesh).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    confirmation_id = _safe_confirmation_id(args.confirmation_id)
    if not run_dir.is_dir():
        raise ValueError(f"run directory is missing: {run_dir}")
    if not source_root.is_dir():
        raise ValueError(f"source root is missing: {source_root}")
    if not gt_mesh.is_file():
        raise ValueError(f"GT mesh is missing: {gt_mesh}")
    if _is_within(gt_mesh, run_dir):
        raise ValueError("GT mesh must be outside the run directory")
    if _is_within(output_root, run_dir) or _is_within(output_root, source_root):
        raise ValueError("diagnostic output must be outside run and source roots")
    if (run_dir / "exit_code.txt").read_text(encoding="utf-8").strip() != "0":
        raise ValueError("training run did not complete successfully")
    required = (
        run_dir / "resolved_config.json",
        run_dir / "multi_view.json",
        run_dir / "chkpnt7000.pth",
        run_dir / "d0_evidence" / "iteration_007000.npz",
    )
    for path in required:
        if not path.is_file():
            raise ValueError(f"required iteration-7000 input is missing: {path}")
    output_dir = output_root / confirmation_id
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    return run_dir, source_root, gt_mesh, output_root, output_dir


def _write_risk_csv(path, rows):
    fieldnames = (
        "field",
        "bin",
        "count",
        "score_min",
        "score_max",
        "mean_distance_m",
        "high_error_count",
        "high_error_rate",
    )
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_diagnostic(args, *, dependencies=None):
    """Run one atomic, diagnostic-only iteration-7000 component audit."""
    import numpy as np

    from reliability.g1_component_diagnostics import (
        RAW_COMPONENTS,
        SNAPSHOT_COMPONENTS,
        build_component_report,
        risk_bin_rows,
    )
    from scripts.diagnostics.evaluate_d0_g1 import (
        assert_inputs_unchanged,
        build_manifest,
        validate_provenance,
    )

    dependencies = dependencies or _default_dependencies()
    run_dir, source_root, gt_mesh, output_root, output_dir = _validate_request(args)
    repository = Path(__file__).resolve().parents[2]
    commit = dependencies.git_head(repository)
    dataset_sha = dependencies.canonical_tree_sha256(source_root)
    gt_sha = dependencies.sha256_file(gt_mesh)
    validate_provenance(
        commit=commit,
        dataset_sha=dataset_sha,
        gt_sha=gt_sha,
        expected_commit=args.expected_commit,
        expected_dataset_sha=args.expected_dataset_sha,
        expected_gt_sha=args.expected_gt_sha,
    )
    immutable = {
        "checkpoint_7000": run_dir / "chkpnt7000.pth",
        "snapshot_7000": run_dir / "d0_evidence" / "iteration_007000.npz",
        "resolved_config": run_dir / "resolved_config.json",
        "multi_view": run_dir / "multi_view.json",
        "source_root": source_root,
        "gt_mesh": gt_mesh,
    }
    before = dependencies.fingerprint_inputs(immutable)

    joined = dependencies.load_iteration(run_dir, 7000)
    mesh = dependencies.load_mesh(gt_mesh)
    distances = np.asarray(
        dependencies.distance_query(joined.centers, mesh), dtype=np.float64
    )
    full_components = dependencies.collect_components(run_dir, source_root, 7000)
    if set(full_components) != set(RAW_COMPONENTS):
        raise ValueError("collector component fields do not match fixed contract")
    finite_rows = np.asarray(joined.finite_row_indices, dtype=np.int64)
    if finite_rows.ndim != 1:
        raise ValueError("finite row indices must be a vector")
    components = {}
    for name in RAW_COMPONENTS:
        values = np.asarray(full_components[name])
        if values.ndim != 1 or values.shape[0] != joined.original_point_count:
            raise ValueError(f"collector component row count mismatch: {name}")
        components[name] = values[finite_rows].copy()
    snapshot = {
        name: np.asarray(joined.snapshot[name])[finite_rows].copy()
        for name in SNAPSHOT_COMPONENTS
    }
    metadata = {
        "commit": commit,
        "dataset_sha256": dataset_sha,
        "gt_mesh_sha256": gt_sha,
        "checkpoint_sha256": joined.checkpoint_sha256,
        "snapshot_sha256": joined.snapshot_sha256,
        "original_point_count": int(joined.original_point_count),
        "evaluated_center_count": int(joined.centers.shape[0]),
        "rejected_center_count": int(joined.rejected_center_indices.size),
        "current_component_observation": (
            "post_training_recomputation_at_iteration_7000"
        ),
        "historical_geometry_stability_note": (
            "post-refresh checkpoint history cannot reconstruct the pre-refresh "
            "stability value used at iteration 7000"
        ),
    }
    report = build_component_report(
        snapshot,
        components,
        distances,
        iteration=7000,
        metadata=metadata,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{output_dir.name}.tmp-{uuid.uuid4().hex}"
    if staging.exists():
        raise FileExistsError(f"staging output already exists: {staging}")
    staging.mkdir()
    try:
        _write_json(staging / "report.json", report)
        _write_risk_csv(staging / "risk_bins.csv", risk_bin_rows(report))
        _write_json(
            staging / "inputs.json",
            {
                "schema_version": 1,
                "diagnostic_only": True,
                "iteration": 7000,
                "provenance": {
                    "commit": commit,
                    "dataset_sha256": dataset_sha,
                    "gt_mesh_sha256": gt_sha,
                },
                "input_fingerprints": before,
                "original_point_count": int(joined.original_point_count),
                "evaluated_center_count": int(joined.centers.shape[0]),
                "rejected_center_count": int(joined.rejected_center_indices.size),
            },
        )
        manifest = build_manifest(staging, DIAGNOSTIC_FILES)
        _write_json(staging / "manifest.json", manifest)
        after = dependencies.fingerprint_inputs(immutable)
        assert_inputs_unchanged(before, after)
        os.replace(staging, output_dir)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return 0, {"output_dir": output_dir, "manifest": manifest, "report": report}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--gt-mesh", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--confirmation-id", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-dataset-sha", required=True)
    parser.add_argument("--expected-gt-sha", required=True)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        exit_code, publication = run_diagnostic(args)
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"D0_G1_COMPONENT_DIAGNOSTIC_ERROR: {type(exc).__name__}: {exc}\n")
    print(
        json.dumps(
            {
                "diagnostic_only": True,
                "exit_code": exit_code,
                "g1_decision": None,
                "output_dir": str(publication["output_dir"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
