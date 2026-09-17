"""Deterministic, offline-only visualization primitives for D0/G1."""

import csv
import json
from pathlib import Path

import numpy as np


STATE_PALETTE = np.array(
    [
        [127, 127, 127],  # Bypass
        [44, 160, 44],  # Consensus
        [31, 119, 180],  # Prior-led
        [255, 127, 14],  # Geometry-led
        [214, 39, 40],  # Abstain
    ],
    dtype=np.uint8,
)

_FALLBACK_MAPS = {
    "viridis": np.array(
        [
            [68, 1, 84],
            [59, 82, 139],
            [33, 145, 140],
            [94, 201, 98],
            [253, 231, 37],
        ],
        dtype=np.float64,
    ),
    "magma": np.array(
        [
            [0, 0, 4],
            [81, 18, 124],
            [183, 55, 121],
            [252, 137, 97],
            [252, 253, 191],
        ],
        dtype=np.float64,
    ),
}

_UNIT_INTERVAL_FIELDS = {"A", "S", "N", "T_p", "T_g", "K", "r_p", "r_g"}
_STATE_NAMES = ("Bypass", "Consensus", "Prior-led", "Geometry-led", "Abstain")
_ARTIFACT_ITERATIONS = (3000, 7000)
_ARTIFACT_FIELDS = ("A", "S", "N", "T_p", "T_g", "K", "state", "gt_distance")
_ARTIFACT_VIEWS = ("q25", "q50", "q75")
_FIGURE_SUFFIXES = ("png", "svg", "pdf", "csv", "json")


def required_artifacts():
    """Return the frozen relative-file inventory required before publication."""
    names = {"inputs.json", "report.json"}
    for iteration in _ARTIFACT_ITERATIONS:
        root = f"iteration_{iteration:06d}"
        for field in _ARTIFACT_FIELDS:
            names.add(f"{root}/fields/{field}.ply")
            for view in _ARTIFACT_VIEWS:
                names.add(f"{root}/views/{field}_{view}.png")
        for view in _ARTIFACT_VIEWS:
            names.add(f"{root}/overlays/gt_{view}.png")
            names.add(f"{root}/overlays/gt_{view}.json")
    for stem in ("primary_curves", "risk_coverage", "state_error"):
        for suffix in _FIGURE_SUFFIXES:
            names.add(f"metrics/iteration_007000_{stem}.{suffix}")
    for stem in ("state_proportion", "state_transition", "joint_coverage"):
        for suffix in _FIGURE_SUFFIXES:
            names.add(f"timeline/{stem}.{suffix}")
    return tuple(sorted(names))


def camera_quartile_indices(count):
    count = int(count)
    if count < 4:
        raise ValueError("at least four cameras are required")
    return tuple(((count - 1) * quarter) // 4 for quarter in (1, 2, 3))


def select_static_cameras(cameras):
    ordered = sorted(
        tuple(cameras),
        key=lambda camera: (int(camera.colmap_id), str(camera.image_name)),
    )
    return tuple(ordered[index] for index in camera_quartile_indices(len(ordered)))


def _fallback_colors(values, name):
    anchors = _FALLBACK_MAPS[name]
    position = values * (anchors.shape[0] - 1)
    lower = np.floor(position).astype(np.int64)
    upper = np.minimum(lower + 1, anchors.shape[0] - 1)
    fraction = (position - lower)[:, None]
    return np.rint(anchors[lower] * (1.0 - fraction) + anchors[upper] * fraction)


def _mapped_colors(values, name):
    try:
        from matplotlib import colormaps
    except ImportError:
        return _fallback_colors(values, name)
    rgba = colormaps[name](values, bytes=True)
    return np.asarray(rgba[:, :3], dtype=np.uint8)


def scalar_colors(values, field):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("scalar values must be a non-empty vector")
    if not np.isfinite(values).all():
        raise ValueError("scalar values must be finite")

    field = str(field)
    if field == "gt_distance":
        normalized = np.clip(values, 0.0, 0.10) / 0.10
        colormap = "magma"
    elif field in _UNIT_INTERVAL_FIELDS:
        normalized = np.clip(values, 0.0, 1.0)
        colormap = "viridis"
    else:
        raise ValueError(f"unsupported scalar visualization field: {field}")

    return np.asarray(_mapped_colors(normalized, colormap), dtype=np.uint8)


def state_colors(states):
    states = np.asarray(states)
    if states.ndim != 1 or states.size == 0:
        raise ValueError("state values must be a non-empty vector")
    if not np.issubdtype(states.dtype, np.integer):
        raise ValueError("state values must be integers")
    indices = states.astype(np.int64, copy=False)
    if np.any(indices < 0) or np.any(indices >= STATE_PALETTE.shape[0]):
        raise ValueError("state value is outside the frozen five-state domain")
    return STATE_PALETTE[indices].copy()


def write_colored_ply(path, centers, colors):
    path = Path(path)
    centers = np.asarray(centers, dtype=np.float64)
    colors = np.asarray(colors)
    if centers.ndim != 2 or centers.shape[1:] != (3,):
        raise ValueError("centers must have shape [N,3]")
    if colors.ndim != 2 or colors.shape[1:] != (3,):
        raise ValueError("colors must have shape [N,3]")
    if centers.shape[0] != colors.shape[0]:
        raise ValueError("center/color row count mismatch")
    if not np.isfinite(centers).all():
        raise ValueError("centers must be finite")
    if not np.issubdtype(colors.dtype, np.integer):
        raise ValueError("colors must use integer RGB values")
    if np.any(colors < 0) or np.any(colors > 255):
        raise ValueError("colors must lie in [0,255]")

    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {centers.shape[0]}\n"
        "property double x\n"
        "property double y\n"
        "property double z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write(header)
        for center, color in zip(centers, colors):
            xyz = " ".join(format(float(value), ".17g") for value in center)
            rgb = " ".join(str(int(value)) for value in color)
            stream.write(f"{xyz} {rgb}\n")
    return {"path": str(path), "vertex_count": int(centers.shape[0])}


def render_override_color(camera, gaussians, pipeline, background, colors):
    import torch

    from gaussian_renderer import render

    if not isinstance(colors, torch.Tensor):
        raise ValueError("colors must be a torch Tensor")
    if colors.ndim != 2 or colors.shape[1] != 3:
        raise ValueError("colors must have shape [N,3]")
    if colors.shape[0] != gaussians.get_xyz.shape[0]:
        raise ValueError("colors row count must match Gaussian count")
    if not bool(torch.isfinite(colors).all()):
        raise ValueError("colors must be finite")

    colors = colors.to(
        device=gaussians.get_xyz.device,
        dtype=gaussians.get_xyz.dtype,
    )
    with torch.no_grad():
        result = render(
            camera,
            gaussians,
            pipeline,
            background,
            override_color=colors,
            return_plane=False,
        )
    image = result["render"].detach()
    if not bool(torch.isfinite(image).all()):
        raise ValueError("override-color render produced nonfinite pixels")
    return image


def _as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def cast_gt_depth(camera, mesh):
    """Cast full-frame pixel-center camera rays against a validated GT mesh."""
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("Open3D is required for GT depth casting") from exc

    height = int(camera.image_height)
    width = int(camera.image_width)
    if height <= 0 or width <= 0:
        raise ValueError("camera image dimensions must be positive")
    intrinsic, camera_to_world = camera.get_calib_matrix_nerf(scale=1.0)
    intrinsic = _as_numpy(intrinsic).astype(np.float64, copy=False)
    camera_to_world = _as_numpy(camera_to_world).astype(np.float64, copy=False)
    if intrinsic.shape != (3, 3) or camera_to_world.shape != (4, 4):
        raise ValueError("camera calibration matrices have invalid shapes")
    if not np.isfinite(intrinsic).all() or not np.isfinite(camera_to_world).all():
        raise ValueError("camera calibration matrices must be finite")
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("camera focal lengths must be positive")

    x, y = np.meshgrid(
        np.arange(width, dtype=np.float64) + 0.5,
        np.arange(height, dtype=np.float64) + 0.5,
        indexing="xy",
    )
    camera_directions = np.stack(
        ((x - cx) / fx, (y - cy) / fy, np.ones_like(x)), axis=-1
    )
    rotation = camera_to_world[:3, :3]
    origin = camera_to_world[:3, 3]
    world_directions = camera_directions @ rotation.T
    origins = np.broadcast_to(origin, world_directions.shape)
    rays = np.concatenate((origins, world_directions), axis=-1).astype(np.float32)

    triangle_mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32)),
        o3d.core.Tensor(np.asarray(mesh.triangles, dtype=np.int32)),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(triangle_mesh)
    depth = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().astype(
        np.float64, copy=False
    )
    finite = np.isfinite(depth)
    finite_depth = depth[finite]
    metadata = {
        "camera_image_name": str(camera.image_name),
        "camera_colmap_id": int(camera.colmap_id),
        "image_shape": [height, width],
        "intrinsic_matrix": intrinsic.tolist(),
        "camera_to_world_matrix": camera_to_world.tolist(),
        "hit_fraction": float(finite.mean()),
        "depth_min_m": float(finite_depth.min()) if finite_depth.size else None,
        "depth_max_m": float(finite_depth.max()) if finite_depth.size else None,
    }
    return depth, metadata


def _rgb_uint8(image, *, expected_shape=None):
    image = _as_numpy(image)
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.moveaxis(image, 0, -1)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("RGB image must have shape [H,W,3] or [3,H,W]")
    if expected_shape is not None and image.shape[:2] != tuple(expected_shape):
        raise ValueError("RGB image shape does not match the frozen camera")
    if not np.isfinite(image).all():
        raise ValueError("RGB image must be finite")
    if np.issubdtype(image.dtype, np.floating):
        image = np.rint(np.clip(image, 0.0, 1.0) * 255.0)
    return np.clip(image, 0, 255).astype(np.uint8)


def write_gt_overlay(camera, mesh, gaussian_rgb, path):
    from PIL import Image

    path = Path(path)
    depth, metadata = cast_gt_depth(camera, mesh)
    gaussian_rgb = _rgb_uint8(
        gaussian_rgb, expected_shape=(camera.image_height, camera.image_width)
    )
    finite = np.isfinite(depth)
    depth_color = np.zeros_like(gaussian_rgb)
    if finite.any():
        finite_depth = depth[finite]
        lower, upper = float(finite_depth.min()), float(finite_depth.max())
        if upper > lower:
            normalized = (finite_depth - lower) / (upper - lower)
        else:
            normalized = np.full(finite_depth.shape, 0.5, dtype=np.float64)
        depth_color[finite] = _mapped_colors(normalized, "magma")
    overlay = gaussian_rgb.copy()
    overlay[finite] = np.rint(
        0.60 * gaussian_rgb[finite] + 0.40 * depth_color[finite]
    ).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay, mode="RGB").save(path)
    metadata = dict(metadata)
    metadata["image_shape"] = [int(value) for value in overlay.shape]
    sidecar = path.with_suffix(".json")
    sidecar.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def _matplotlib_pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    return plt


def _write_csv(path, header, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _save_figure_bundle(figure, output_dir, stem, header, rows, metadata):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix, options in (
        ("png", {"dpi": 300}),
        ("svg", {}),
        ("pdf", {}),
    ):
        path = output_dir / f"{stem}.{suffix}"
        figure.savefig(path, bbox_inches="tight", **options)
        paths.append(path)
    csv_path = output_dir / f"{stem}.csv"
    _write_csv(csv_path, header, rows)
    paths.append(csv_path)
    json_path = output_dir / f"{stem}.json"
    json_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    paths.append(json_path)
    return paths


def write_metric_figures(report, output_dir, *, prefix):
    """Export the frozen primary, risk, and state evidence with source data."""
    plt = _matplotlib_pyplot()
    output_dir = Path(output_dir)
    written = []

    curves = report["primary"]["curves"]
    figure, axes = plt.subplots(1, 2, figsize=(6.8, 2.7), constrained_layout=True)
    curve_rows = []
    labels = {"N": "Need N", "A": "Ambiguity A", "one_minus_S": "1 - S"}
    colors = {"N": "#D62728", "A": "#7F7F7F", "one_minus_S": "#1F77B4"}
    for name in ("N", "A", "one_minus_S"):
        curve = curves[name]
        axes[0].plot(curve["fpr"], curve["tpr"], label=labels[name], color=colors[name])
        axes[1].plot(
            curve["recall"], curve["precision"], label=labels[name], color=colors[name]
        )
        curve_rows.extend(
            (name, "roc", float(x), float(y))
            for x, y in zip(curve["fpr"], curve["tpr"])
        )
        curve_rows.extend(
            (name, "pr", float(x), float(y))
            for x, y in zip(curve["recall"], curve["precision"])
        )
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#BBBBBB", linewidth=0.8)
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1))
    axes[1].set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
    axes[0].legend()
    written.extend(
        _save_figure_bundle(
            figure,
            output_dir,
            f"{prefix}_primary_curves",
            ("series", "curve", "x", "y"),
            curve_rows,
            {"iteration": int(report["iteration"]), "panels": ["ROC", "PR"]},
        )
    )
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(6.8, 2.7), constrained_layout=True)
    risk_rows = []
    for name in ("N", "r_p", "r_g"):
        risk = report["risk_coverage"][name]
        if risk is None:
            continue
        points = risk["points"]
        coverage = [point["coverage"] for point in points]
        mean_distance = [point["mean_distance"] for point in points]
        high_error_rate = [point["high_error_rate"] for point in points]
        axes[0].plot(coverage, mean_distance, marker="o", markersize=2.5, label=name)
        axes[1].plot(coverage, high_error_rate, marker="o", markersize=2.5, label=name)
        risk_rows.extend(
            (
                name,
                risk["direction"],
                float(point["coverage"]),
                int(point["count"]),
                float(point["mean_distance"]),
                float(point["high_error_rate"]),
            )
            for point in points
        )
    axes[0].set(xlabel="Coverage", ylabel="Mean GT distance (m)", xlim=(0, 1))
    axes[1].set(xlabel="Coverage", ylabel="High-error rate", xlim=(0, 1), ylim=(0, 1))
    axes[0].legend()
    written.extend(
        _save_figure_bundle(
            figure,
            output_dir,
            f"{prefix}_risk_coverage",
            (
                "series",
                "direction",
                "coverage",
                "count",
                "mean_distance_m",
                "high_error_rate",
            ),
            risk_rows,
            {"iteration": int(report["iteration"]), "coverage_range": [0.0, 1.0]},
        )
    )
    plt.close(figure)

    per_state = report["state"]["per_state"]
    state_rows = []
    mean_distance = []
    high_error_rate = []
    for name in _STATE_NAMES:
        values = per_state[name]
        mean_distance.append(np.nan if values["mean_distance"] is None else values["mean_distance"])
        high_error_rate.append(
            np.nan if values["high_error_rate"] is None else values["high_error_rate"]
        )
        state_rows.append(
            (
                name,
                int(values["count"]),
                float(values["fraction"]),
                values["mean_distance"],
                values["high_error_rate"],
            )
        )
    figure, axes = plt.subplots(1, 2, figsize=(7.4, 2.7), constrained_layout=True)
    positions = np.arange(len(_STATE_NAMES))
    axes[0].bar(positions, mean_distance, color=STATE_PALETTE / 255.0)
    axes[1].bar(positions, high_error_rate, color=STATE_PALETTE / 255.0)
    for axis in axes:
        axis.set_xticks(positions, _STATE_NAMES, rotation=25, ha="right")
    axes[0].set_ylabel("Mean GT distance (m)")
    axes[1].set_ylabel("High-error rate")
    axes[1].set_ylim(0, 1)
    written.extend(
        _save_figure_bundle(
            figure,
            output_dir,
            f"{prefix}_state_error",
            ("state", "count", "fraction", "mean_distance_m", "high_error_rate"),
            state_rows,
            {"iteration": int(report["iteration"]), "state_order": list(_STATE_NAMES)},
        )
    )
    plt.close(figure)
    return tuple(written)
