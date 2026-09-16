"""Deterministic, offline-only visualization primitives for D0/G1."""

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
