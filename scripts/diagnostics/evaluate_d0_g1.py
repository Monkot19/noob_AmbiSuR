"""Fail-closed publication helpers for the offline D0/G1 evaluator."""

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import uuid


if __package__ in (None, ""):
    repository_root = str(Path(__file__).resolve().parents[2])
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)


FORMAL_ITERATIONS = (3000, 7000)
EXPLORATORY_ITERATIONS = (500,)
TIMELINE_ITERATIONS = tuple(range(1000, 7001, 1000))


def _resolved(path):
    return Path(path).expanduser().resolve()


def _is_within(path, parent):
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _safe_relative_name(name):
    name = str(name).replace("\\", "/")
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"unsafe relative path: {name}")
    return path.as_posix()


def validate_publication_request(
    run_dir,
    source_root,
    gt_mesh,
    output_root,
    confirmation_id,
    iterations,
    *,
    exploratory,
):
    run_dir = _resolved(run_dir)
    source_root = _resolved(source_root)
    gt_mesh = _resolved(gt_mesh)
    output_root = _resolved(output_root)
    if not run_dir.is_dir():
        raise ValueError(f"run directory is missing: {run_dir}")
    if not source_root.is_dir():
        raise ValueError(f"source root is missing: {source_root}")
    if not gt_mesh.is_file():
        raise ValueError(f"GT mesh is missing: {gt_mesh}")
    if _is_within(gt_mesh, run_dir):
        raise ValueError("GT mesh must be outside the run directory")
    confirmation_id = _safe_relative_name(confirmation_id)
    if len(PurePosixPath(confirmation_id).parts) != 1:
        raise ValueError("confirmation ID must be a single path component")
    iterations = tuple(int(value) for value in iterations)
    expected = EXPLORATORY_ITERATIONS if exploratory else FORMAL_ITERATIONS
    if iterations != expected:
        kind = "exploratory" if exploratory else "formal"
        raise ValueError(f"{kind} iterations must be {expected}")
    if not exploratory:
        evidence = run_dir / "d0_evidence"
        for iteration in TIMELINE_ITERATIONS:
            path = evidence / f"iteration_{iteration:06d}.npz"
            if not path.is_file():
                raise ValueError(f"missing D0 timeline snapshot: {iteration}")
    output_dir = output_root / confirmation_id
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    return output_dir


def validate_provenance(
    *,
    commit,
    dataset_sha,
    gt_sha,
    expected_commit,
    expected_dataset_sha,
    expected_gt_sha,
):
    if commit != expected_commit:
        raise ValueError("commit mismatch")
    if dataset_sha != expected_dataset_sha:
        raise ValueError("dataset SHA256 mismatch")
    if gt_sha != expected_gt_sha:
        raise ValueError("GT SHA256 mismatch")


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint_path(path):
    path = _resolved(path)
    if path.is_file():
        return {"kind": "file", "bytes": path.stat().st_size, "sha256": _sha256_file(path)}
    if not path.is_dir():
        raise ValueError(f"input path is missing: {path}")
    digest = hashlib.sha256()
    total_bytes = 0
    files = 0
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = child.relative_to(path).as_posix()
        size = child.stat().st_size
        child_sha = _sha256_file(child)
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(str(size).encode("ascii") + b"\0")
        digest.update(child_sha.encode("ascii") + b"\n")
        total_bytes += size
        files += 1
    return {
        "kind": "directory",
        "files": files,
        "bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }


def fingerprint_inputs(paths):
    return {
        str(name): _fingerprint_path(path)
        for name, path in sorted(dict(paths).items(), key=lambda item: str(item[0]))
    }


def assert_inputs_unchanged(before, after):
    if before != after:
        raise RuntimeError("input mutated during evaluation")


def build_manifest(root, required):
    root = _resolved(root)
    required = tuple(sorted(_safe_relative_name(name) for name in required))
    if len(required) != len(set(required)):
        raise ValueError("required artifact inventory contains duplicates")
    actual = tuple(
        sorted(
            child.relative_to(root).as_posix()
            for child in root.rglob("*")
            if child.is_file() and child.name != "manifest.json"
        )
    )
    if actual != required:
        missing = sorted(set(required) - set(actual))
        extra = sorted(set(actual) - set(required))
        raise ValueError(
            f"required artifact inventory mismatch: missing={missing}, extra={extra}"
        )
    files = []
    for name in required:
        path = root / Path(name)
        files.append(
            {"path": name, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}
        )
    return {"schema_version": 1, "files": files}


def _normalized_tar_info(name, size):
    info = tarfile.TarInfo(_safe_relative_name(name))
    info.size = int(size)
    info.mtime = 0
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def write_deterministic_archive(root, archive_path, manifest):
    root = _resolved(root)
    archive_path = Path(archive_path)
    expected = tuple(entry["path"] for entry in manifest["files"])
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("manifest.json is missing")
    members = ("manifest.json",) + expected
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with archive_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for name in sorted(members):
                    payload = (root / Path(name)).read_bytes()
                    archive.addfile(_normalized_tar_info(name, len(payload)), io.BytesIO(payload))
    return archive_path


def validate_archive(archive_path, manifest):
    expected_files = tuple(entry["path"] for entry in manifest["files"])
    expected = tuple(sorted(("manifest.json",) + expected_files))
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        names = []
        payloads = {}
        for member in members:
            try:
                name = _safe_relative_name(member.name)
            except ValueError as exc:
                raise ValueError(f"unsafe archive member: {member.name}") from exc
            if not member.isfile():
                raise ValueError(f"unsafe archive member: {member.name}")
            if name in payloads:
                raise ValueError(f"duplicate archive member: {name}")
            stream = archive.extractfile(member)
            payloads[name] = stream.read() if stream is not None else b""
            names.append(name)
    if tuple(sorted(names)) != expected:
        raise ValueError("archive inventory mismatch")
    embedded = json.loads(payloads["manifest.json"].decode("utf-8"))
    if embedded != manifest:
        raise ValueError("archive manifest mismatch")
    for entry in manifest["files"]:
        payload = payloads[entry["path"]]
        if len(payload) != entry["bytes"] or hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise ValueError(f"archive artifact mismatch: {entry['path']}")
    return tuple(sorted(names))


def publication_exit_code(report, *, exploratory=False):
    if exploratory:
        if report.get("g1_evaluable") is not None or report.get("g1_pass") is not None:
            raise ValueError("exploratory report must not carry a formal G1 decision")
        return 0
    evaluable = report.get("g1_evaluable")
    passed = report.get("g1_pass")
    if evaluable is not True:
        return 2
    if not isinstance(passed, bool):
        raise ValueError("formal report must carry a boolean G1 decision")
    return 0 if passed else 1


def publish_atomically(
    output_root,
    confirmation_id,
    immutable_inputs,
    producer,
    *,
    required=None,
):
    from reliability.g1_visualization import required_artifacts

    output_root = _resolved(output_root)
    required = required_artifacts() if required is None else tuple(required)
    confirmation_id = _safe_relative_name(confirmation_id)
    if len(PurePosixPath(confirmation_id).parts) != 1:
        raise ValueError("confirmation ID must be a single path component")
    output_dir = output_root / confirmation_id
    archive_path = output_root / f"{confirmation_id}.tar.gz"
    archive_sha256_path = output_root / f"{confirmation_id}.tar.gz.sha256"
    for path in (output_dir, archive_path, archive_sha256_path):
        if path.exists():
            raise FileExistsError(f"output already exists: {path}")

    output_root.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    staging = output_root / f".{confirmation_id}.tmp-{token}"
    temporary_archive = output_root / f".{confirmation_id}.tmp-{token}.tar.gz"
    temporary_sha = output_root / f".{confirmation_id}.tmp-{token}.tar.gz.sha256"
    before = fingerprint_inputs(immutable_inputs)
    published = []
    try:
        staging.mkdir()
        producer(staging)
        after = fingerprint_inputs(immutable_inputs)
        assert_inputs_unchanged(before, after)
        manifest = build_manifest(staging, required)
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_deterministic_archive(staging, temporary_archive, manifest)
        validate_archive(temporary_archive, manifest)
        archive_sha256 = _sha256_file(temporary_archive)
        temporary_sha.write_text(
            f"{archive_sha256}  {archive_path.name}\n", encoding="ascii"
        )

        os.replace(staging, output_dir)
        published.append(output_dir)
        os.replace(temporary_archive, archive_path)
        published.append(archive_path)
        os.replace(temporary_sha, archive_sha256_path)
        published.append(archive_sha256_path)
        return {
            "output_dir": output_dir,
            "archive_path": archive_path,
            "archive_sha256_path": archive_sha256_path,
            "archive_sha256": archive_sha256,
            "manifest": manifest,
            "inputs_before": before,
            "inputs_after": after,
        }
    except BaseException:
        for path in reversed(published):
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        if staging.exists():
            shutil.rmtree(staging)
        for path in (temporary_archive, temporary_sha):
            if path.exists():
                path.unlink()
        raise


def evaluate_iteration(inputs, mesh, *, distance_fn=None):
    """Evaluate one joined checkpoint/snapshot without changing its row domain."""
    import numpy as np

    from reliability.g1_metrics import evaluate_g1_gate
    from reliability.offline_g1 import closest_triangle_distances

    if distance_fn is None:
        distance_fn = closest_triangle_distances
    distances = np.asarray(distance_fn(inputs.centers, mesh), dtype=np.float64)
    if distances.ndim != 1 or distances.shape[0] != inputs.centers.shape[0]:
        raise ValueError("distance row count must match finite Gaussian centers")
    if not np.isfinite(distances).all() or np.any(distances < 0.0):
        raise ValueError("distances must be finite and nonnegative")

    finite_rows = np.asarray(inputs.finite_row_indices, dtype=np.int64)
    snapshot = {
        name: np.asarray(value)[finite_rows].copy()
        for name, value in inputs.snapshot.items()
    }
    report = evaluate_g1_gate(snapshot, distances, inputs.iteration)
    return {
        "iteration": int(inputs.iteration),
        "centers": np.asarray(inputs.centers).copy(),
        "snapshot": snapshot,
        "distances": distances,
        "report": report,
        "original_point_count": int(inputs.original_point_count),
        "evaluated_center_count": int(inputs.centers.shape[0]),
        "rejected_center_count": int(inputs.rejected_center_indices.size),
        "checkpoint_sha256": inputs.checkpoint_sha256,
        "snapshot_sha256": inputs.snapshot_sha256,
    }


def _jsonable(value):
    import numpy as np

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
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


def canonical_tree_sha256(root):
    """Match the frozen `find | sort -z | xargs sha256sum | sha256sum` contract."""
    root = _resolved(root)
    if not root.is_dir():
        raise ValueError(f"tree is missing: {root}")
    digest = hashlib.sha256()
    for child in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = "./" + child.relative_to(root).as_posix()
        digest.update(f"{_sha256_file(child)}  {relative}\n".encode("utf-8"))
    return digest.hexdigest()


def _git_head(repository):
    import subprocess

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _load_render_runtime(run_dir, source_root, iteration):
    """Restore checkpoint tensors and cameras without writing into the run."""
    from types import SimpleNamespace

    import torch

    from scene.dataset_readers import sceneLoadTypeCallbacks
    from scene.gaussian_model import GaussianModel
    from utils.camera_utils import cameraList_from_camInfos

    run_dir = Path(run_dir)
    config = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    model_values = dict(config["model"])
    optimization = SimpleNamespace(**config["optimization"])
    pipeline = SimpleNamespace(**config["pipeline"])
    model_values["source_path"] = str(source_root)
    dataset = SimpleNamespace(**model_values)

    payload = torch.load(
        run_dir / f"chkpnt{int(iteration)}.pth",
        map_location="cuda",
        weights_only=False,
    )
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("G1 rendering requires a versioned Core checkpoint")
    if int(payload.get("iteration", -1)) != int(iteration):
        raise ValueError("render checkpoint iteration mismatch")

    gaussians = GaussianModel(int(dataset.sh_degree))
    gaussians.disable_trunc = bool(dataset.disable_trunc)
    gaussians.trunc_sigma = float(dataset.trunc_sigma)
    gaussians.restore(payload["gaussian_state"], optimization)

    scene_info = sceneLoadTypeCallbacks["Colmap"](
        str(source_root), dataset.images, bool(dataset.eval)
    )
    cameras = cameraList_from_camInfos(scene_info.train_cameras, 1.0, dataset)
    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )
    return gaussians, cameras, pipeline, background


def _save_rgb(path, image):
    import numpy as np
    from PIL import Image

    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)
    if image.ndim != 3 or image.shape[-1] != 3 or not np.isfinite(image).all():
        raise ValueError("rendered RGB image is invalid")
    rgb = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(path)


def _field_colors(evaluation):
    from reliability.g1_visualization import scalar_colors, state_colors

    snapshot = evaluation["snapshot"]
    return {
        "A": scalar_colors(snapshot["A"], "A"),
        "S": scalar_colors(snapshot["S"], "S"),
        "N": scalar_colors(snapshot["N"], "N"),
        "T_p": scalar_colors(snapshot["T_p"], "T_p"),
        "T_g": scalar_colors(snapshot["T_g"], "T_g"),
        "K": scalar_colors(snapshot["K"], "K"),
        "state": state_colors(snapshot["stable"]),
        "gt_distance": scalar_colors(evaluation["distances"], "gt_distance"),
    }


def produce_exploratory_500(
    staging,
    *,
    run_dir,
    source_root,
    gt_mesh,
    provenance=None,
    input_fingerprints=None,
):
    """Produce the approved 55-file exploratory-500 bundle."""
    import numpy as np
    import torch

    from gaussian_renderer import render
    from reliability.g1_visualization import (
        render_override_color,
        select_static_cameras,
        write_colored_ply,
        write_gt_overlay,
        write_metric_figures,
    )
    from reliability.offline_g1 import load_g1_iteration, load_valid_mesh

    iteration = 500
    joined = load_g1_iteration(run_dir, iteration)
    mesh = load_valid_mesh(gt_mesh)
    evaluation = evaluate_iteration(joined, mesh)
    colors_by_field = _field_colors(evaluation)
    root = Path(staging) / f"iteration_{iteration:06d}"
    for field, colors in colors_by_field.items():
        write_colored_ply(
            root / "fields" / f"{field}.ply",
            evaluation["centers"],
            colors,
        )

    gaussians, cameras, pipeline, background = _load_render_runtime(
        run_dir, source_root, iteration
    )
    if gaussians.get_xyz.shape[0] != joined.original_point_count:
        raise ValueError("render checkpoint row count mismatch")
    selected = select_static_cameras(cameras)
    view_names = ("q25", "q50", "q75")
    finite_rows = torch.as_tensor(
        joined.finite_row_indices, device="cuda", dtype=torch.long
    )
    for field, finite_colors in colors_by_field.items():
        full_colors = torch.full(
            (joined.original_point_count, 3),
            127.0 / 255.0,
            device="cuda",
            dtype=gaussians.get_xyz.dtype,
        )
        full_colors[finite_rows] = torch.as_tensor(
            finite_colors / 255.0,
            device="cuda",
            dtype=gaussians.get_xyz.dtype,
        )
        for view_name, camera in zip(view_names, selected):
            image = render_override_color(
                camera, gaussians, pipeline, background, full_colors
            )
            _save_rgb(root / "views" / f"{field}_{view_name}.png", image)

    for view_name, camera in zip(view_names, selected):
        with torch.no_grad():
            gaussian_rgb = render(
                camera,
                gaussians,
                pipeline,
                background,
                return_plane=False,
            )["render"]
        write_gt_overlay(
            camera,
            mesh,
            gaussian_rgb,
            root / "overlays" / f"gt_{view_name}.png",
        )

    write_metric_figures(
        evaluation["report"],
        Path(staging) / "metrics",
        prefix="iteration_000500",
    )
    report = dict(evaluation["report"])
    report.update(
        exploratory=True,
        evaluated_center_count=evaluation["evaluated_center_count"],
        rejected_center_count=evaluation["rejected_center_count"],
        mesh={
            "source_vertex_count": mesh.source_vertex_count,
            "source_triangle_count": mesh.source_triangle_count,
            "valid_vertex_count": int(mesh.vertices.shape[0]),
            "valid_triangle_count": int(mesh.triangles.shape[0]),
            "rejected_triangle_count": mesh.rejected_triangle_count,
        },
    )
    _write_json(Path(staging) / "report.json", report)
    _write_json(
        Path(staging) / "inputs.json",
        {
            "schema_version": 1,
            "exploratory": True,
            "iterations": [500],
            "provenance": dict(provenance or {}),
            "input_fingerprints": dict(input_fingerprints or {}),
            "checkpoint_sha256": joined.checkpoint_sha256,
            "snapshot_sha256": joined.snapshot_sha256,
            "finite_row_indices": joined.finite_row_indices,
            "rejected_center_indices": joined.rejected_center_indices,
        },
    )
    return report


def produce_formal_3000_7000(
    staging,
    *,
    run_dir,
    source_root,
    gt_mesh,
    provenance=None,
    input_fingerprints=None,
):
    """Produce the frozen 108-file formal 3000/7000 evaluation bundle."""
    import torch

    from gaussian_renderer import render
    from reliability.g1_timeline import (
        load_d0_timeline,
        write_timeline_artifacts,
    )
    from reliability.g1_visualization import (
        render_override_color,
        select_static_cameras,
        write_colored_ply,
        write_gt_overlay,
        write_metric_figures,
    )
    from reliability.offline_g1 import load_g1_iteration, load_valid_mesh

    staging = Path(staging)
    mesh = load_valid_mesh(gt_mesh)
    iteration_reports = {}
    iteration_inputs = {}
    for iteration in FORMAL_ITERATIONS:
        joined = load_g1_iteration(run_dir, iteration)
        evaluation = evaluate_iteration(joined, mesh)
        colors_by_field = _field_colors(evaluation)
        root = staging / f"iteration_{iteration:06d}"
        for field, colors in colors_by_field.items():
            write_colored_ply(
                root / "fields" / f"{field}.ply",
                evaluation["centers"],
                colors,
            )

        gaussians, cameras, pipeline, background = _load_render_runtime(
            run_dir, source_root, iteration
        )
        if gaussians.get_xyz.shape[0] != joined.original_point_count:
            raise ValueError("render checkpoint row count mismatch")
        selected = select_static_cameras(cameras)
        view_names = ("q25", "q50", "q75")
        finite_rows = torch.as_tensor(
            joined.finite_row_indices, device="cuda", dtype=torch.long
        )
        for field, finite_colors in colors_by_field.items():
            full_colors = torch.full(
                (joined.original_point_count, 3),
                127.0 / 255.0,
                device="cuda",
                dtype=gaussians.get_xyz.dtype,
            )
            full_colors[finite_rows] = torch.as_tensor(
                finite_colors / 255.0,
                device="cuda",
                dtype=gaussians.get_xyz.dtype,
            )
            for view_name, camera in zip(view_names, selected):
                image = render_override_color(
                    camera, gaussians, pipeline, background, full_colors
                )
                _save_rgb(root / "views" / f"{field}_{view_name}.png", image)

        for view_name, camera in zip(view_names, selected):
            with torch.no_grad():
                gaussian_rgb = render(
                    camera,
                    gaussians,
                    pipeline,
                    background,
                    return_plane=False,
                )["render"]
            write_gt_overlay(
                camera,
                mesh,
                gaussian_rgb,
                root / "overlays" / f"gt_{view_name}.png",
            )

        current_report = dict(evaluation["report"])
        current_report.update(
            evaluated_center_count=evaluation["evaluated_center_count"],
            rejected_center_count=evaluation["rejected_center_count"],
        )
        iteration_reports[str(iteration)] = current_report
        iteration_inputs[str(iteration)] = {
            "checkpoint_sha256": joined.checkpoint_sha256,
            "snapshot_sha256": joined.snapshot_sha256,
            "finite_row_indices": joined.finite_row_indices,
            "rejected_center_indices": joined.rejected_center_indices,
        }
        del gaussians, cameras, pipeline, background
        torch.cuda.empty_cache()

    decision = iteration_reports["7000"]
    write_metric_figures(
        decision,
        staging / "metrics",
        prefix="iteration_007000",
    )
    timeline = load_d0_timeline(run_dir)
    write_timeline_artifacts(timeline, staging)

    report = {
        "schema_version": 1,
        "exploratory": False,
        "evaluated_iterations": list(FORMAL_ITERATIONS),
        "decision_iteration": 7000,
        "g1_evaluable": decision["g1_evaluable"],
        "g1_pass": decision["g1_pass"],
        "iterations": iteration_reports,
        "mesh": {
            "source_vertex_count": mesh.source_vertex_count,
            "source_triangle_count": mesh.source_triangle_count,
            "valid_vertex_count": int(mesh.vertices.shape[0]),
            "valid_triangle_count": int(mesh.triangles.shape[0]),
            "rejected_triangle_count": mesh.rejected_triangle_count,
        },
    }
    _write_json(staging / "report.json", report)
    _write_json(
        staging / "inputs.json",
        {
            "schema_version": 1,
            "exploratory": False,
            "iterations": list(FORMAL_ITERATIONS),
            "provenance": dict(provenance or {}),
            "input_fingerprints": dict(input_fingerprints or {}),
            "iteration_inputs": iteration_inputs,
        },
    )
    return report


def run_evaluator(args):
    from reliability.g1_visualization import required_artifacts

    run_dir = _resolved(args.run_dir)
    source_root = _resolved(args.source_root)
    gt_mesh = _resolved(args.gt_mesh)
    output_root = _resolved(args.output_root)
    iterations = tuple(args.iterations)
    validate_publication_request(
        run_dir,
        source_root,
        gt_mesh,
        output_root,
        args.confirmation_id,
        iterations,
        exploratory=args.exploratory,
    )
    commit = _git_head(Path(__file__).resolve().parents[2])
    dataset_sha = canonical_tree_sha256(source_root)
    gt_sha = _sha256_file(gt_mesh)
    validate_provenance(
        commit=commit,
        dataset_sha=dataset_sha,
        gt_sha=gt_sha,
        expected_commit=args.expected_commit,
        expected_dataset_sha=args.expected_dataset_sha,
        expected_gt_sha=args.expected_gt_sha,
    )
    evidence = run_dir / "d0_evidence"
    if args.exploratory:
        immutable = {
            "checkpoint_500": run_dir / "chkpnt500.pth",
            "snapshot_500": evidence / "iteration_000500.npz",
            "resolved_config": run_dir / "resolved_config.json",
            "source_root": source_root,
            "gt_mesh": gt_mesh,
        }
    else:
        immutable = {
            "checkpoint_3000": run_dir / "chkpnt3000.pth",
            "checkpoint_7000": run_dir / "chkpnt7000.pth",
            "events": evidence / "events.jsonl",
            "resolved_config": run_dir / "resolved_config.json",
            "source_root": source_root,
            "gt_mesh": gt_mesh,
        }
        immutable.update(
            {
                f"snapshot_{iteration}": (
                    evidence / f"iteration_{iteration:06d}.npz"
                )
                for iteration in TIMELINE_ITERATIONS
            }
        )
    provenance = {
        "commit": commit,
        "dataset_sha256": dataset_sha,
        "gt_mesh_sha256": gt_sha,
    }
    input_fingerprints = fingerprint_inputs(immutable)
    required = required_artifacts(
        iterations=iterations, exploratory=args.exploratory
    )
    report_box = {}

    def producer(staging):
        produce = (
            produce_exploratory_500
            if args.exploratory
            else produce_formal_3000_7000
        )
        report_box["report"] = produce(
            staging,
            run_dir=run_dir,
            source_root=source_root,
            gt_mesh=gt_mesh,
            provenance=provenance,
            input_fingerprints=input_fingerprints,
        )

    publication = publish_atomically(
        output_root,
        args.confirmation_id,
        immutable,
        producer,
        required=required,
    )
    return publication_exit_code(
        report_box["report"], exploratory=args.exploratory
    ), publication


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--gt-mesh", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--confirmation-id", required=True)
    parser.add_argument("--iterations", nargs="+", type=int, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-dataset-sha", required=True)
    parser.add_argument("--expected-gt-sha", required=True)
    parser.add_argument("--exploratory", action="store_true")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        exit_code, publication = run_evaluator(args)
    except (FileNotFoundError, ValueError, RuntimeError, NotImplementedError) as exc:
        parser.exit(2, f"G1_EVALUATION_ERROR: {type(exc).__name__}: {exc}\n")
    print(
        json.dumps(
            {
                "exit_code": exit_code,
                "output_dir": str(publication["output_dir"]),
                "archive_path": str(publication["archive_path"]),
                "archive_sha256_path": str(publication["archive_sha256_path"]),
                "archive_sha256": publication["archive_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
