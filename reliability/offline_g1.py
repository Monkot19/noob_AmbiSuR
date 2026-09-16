from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np

from reliability.diagnostics import SNAPSHOT_FIELDS

try:
    import torch
except ModuleNotFoundError:
    torch = None


_STATE_FIELDS = {
    "A": "a_value",
    "S": "s_value",
    "T_p": "t_p_value",
    "V_p": "t_p_current_valid",
    "T_g": "t_g_value",
    "V_g": "t_g_current_valid",
    "K": "k_value",
    "candidate": "candidate_state",
    "stable": "stable_state",
}
_BOOLEAN_FIELDS = {"V_p", "V_g", "V_pg"}
_STATE_ID_FIELDS = {"candidate", "stable"}


@dataclass(frozen=True)
class G1IterationInputs:
    iteration: int
    original_point_count: int
    centers: np.ndarray
    finite_row_indices: np.ndarray
    rejected_center_indices: np.ndarray
    snapshot: Mapping[str, np.ndarray]
    checkpoint_sha256: str
    snapshot_sha256: str


@dataclass(frozen=True)
class ValidatedMesh:
    vertices: np.ndarray
    triangles: np.ndarray
    source_vertex_count: int
    source_triangle_count: int
    nonfinite_vertex_count: int
    rejected_nonfinite_triangle_count: int
    rejected_degenerate_triangle_count: int

    @property
    def rejected_triangle_count(self):
        return (
            self.rejected_nonfinite_triangle_count
            + self.rejected_degenerate_triangle_count
        )


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_mesh_arrays(vertices, triangles):
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("GT vertices must have shape [V,3]")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("GT triangles must have shape [F,3]")
    if not np.issubdtype(triangles.dtype, np.integer):
        raise ValueError("GT triangle indices must be integers")
    triangles = triangles.astype(np.int64, copy=False)
    if triangles.shape[0] == 0:
        raise ValueError("GT mesh has no valid GT triangles")
    if triangles.min() < 0 or triangles.max() >= vertices.shape[0]:
        raise ValueError("triangle index out of range")

    finite_vertices = np.isfinite(vertices).all(axis=1)
    finite_triangles = finite_vertices[triangles].all(axis=1)
    finite_faces = triangles[finite_triangles]
    degenerate = np.zeros(finite_faces.shape[0], dtype=np.bool_)
    if finite_faces.shape[0]:
        a = vertices[finite_faces[:, 0]]
        b = vertices[finite_faces[:, 1]]
        c = vertices[finite_faces[:, 2]]
        cross = np.cross(b - a, c - a)
        degenerate = np.einsum("ij,ij->i", cross, cross) == 0.0
    valid_faces = finite_faces[~degenerate]
    if valid_faces.shape[0] == 0:
        raise ValueError("GT mesh has no valid GT triangles")

    used_vertices, compact_indices = np.unique(
        valid_faces.reshape(-1), return_inverse=True
    )
    compact_vertices = np.ascontiguousarray(vertices[used_vertices])
    compact_triangles = np.ascontiguousarray(
        compact_indices.reshape(-1, 3).astype(np.int64, copy=False)
    )
    compact_vertices.setflags(write=False)
    compact_triangles.setflags(write=False)
    return ValidatedMesh(
        vertices=compact_vertices,
        triangles=compact_triangles,
        source_vertex_count=int(vertices.shape[0]),
        source_triangle_count=int(triangles.shape[0]),
        nonfinite_vertex_count=int((~finite_vertices).sum()),
        rejected_nonfinite_triangle_count=int((~finite_triangles).sum()),
        rejected_degenerate_triangle_count=int(degenerate.sum()),
    )


def _require_open3d():
    try:
        import open3d as o3d
    except ModuleNotFoundError as exc:
        raise RuntimeError("Open3D is required for GT mesh evaluation") from exc
    return o3d


def load_valid_mesh(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"GT mesh is missing: {path}")
    o3d = _require_open3d()
    mesh = o3d.io.read_triangle_mesh(str(path))
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if vertices.size == 0 or triangles.size == 0:
        raise ValueError(f"GT mesh has no triangle surface: {path}")
    return validate_mesh_arrays(vertices, triangles)


def closest_triangle_distances(points, mesh, chunk_size=65536):
    points64 = np.asarray(points, dtype=np.float64)
    if points64.ndim != 2 or points64.shape[1] != 3:
        raise ValueError("query points must have shape [P,3]")
    if points64.shape[0] == 0:
        raise ValueError("query points must not be empty")
    if not np.isfinite(points64).all():
        raise ValueError("query points must be finite")
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not isinstance(mesh, ValidatedMesh):
        raise TypeError("mesh must be a ValidatedMesh")

    o3d = _require_open3d()
    legacy = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices),
        o3d.utility.Vector3iVector(mesh.triangles),
    )
    tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(
        legacy,
        vertex_dtype=o3d.core.Dtype.Float32,
        triangle_dtype=o3d.core.Dtype.Int64,
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tensor_mesh)

    distances = np.empty(points64.shape[0], dtype=np.float64)
    for start in range(0, points64.shape[0], chunk_size):
        stop = min(start + chunk_size, points64.shape[0])
        query = o3d.core.Tensor(
            points64[start:stop].astype(np.float32, copy=False)
        )
        result = scene.compute_closest_points(query)
        closest = result["points"].numpy().astype(np.float64, copy=False)
        distances[start:stop] = np.linalg.norm(
            points64[start:stop] - closest, axis=1
        )
    if not np.isfinite(distances).all():
        raise ValueError("nonfinite GT distance")
    return distances


def _as_numpy(value):
    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _require_snapshot(snapshot):
    if set(snapshot) != set(SNAPSHOT_FIELDS):
        missing = sorted(set(SNAPSHOT_FIELDS) - set(snapshot))
        extra = sorted(set(snapshot) - set(SNAPSHOT_FIELDS))
        raise ValueError(
            f"snapshot fields mismatch: missing={missing}, extra={extra}"
        )
    arrays = {name: _as_numpy(snapshot[name]) for name in SNAPSHOT_FIELDS}
    point_count = None
    for name, value in arrays.items():
        if value.ndim != 1:
            raise ValueError(f"snapshot field must be one-dimensional: {name}")
        if name in _BOOLEAN_FIELDS and not np.issubdtype(value.dtype, np.bool_):
            raise ValueError(f"snapshot dtype mismatch: {name}")
        if name in _STATE_ID_FIELDS and not np.issubdtype(
            value.dtype, np.integer
        ):
            raise ValueError(f"snapshot dtype mismatch: {name}")
        if name not in _BOOLEAN_FIELDS | _STATE_ID_FIELDS and not np.issubdtype(
            value.dtype, np.floating
        ):
            raise ValueError(f"snapshot dtype mismatch: {name}")
        if not np.isfinite(value).all():
            raise ValueError(f"nonfinite snapshot field: {name}")
        if point_count is None:
            point_count = int(value.shape[0])
        elif value.shape[0] != point_count:
            raise ValueError(f"snapshot field row count mismatch: {name}")
    if point_count is None or point_count <= 0:
        raise ValueError("snapshot must contain at least one Gaussian")
    return arrays, point_count


def _require_exact(name, actual, expected):
    actual = _as_numpy(actual)
    expected = _as_numpy(expected)
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        raise ValueError(f"snapshot/core_state row contract mismatch: {name}")


def validate_snapshot_core_state(snapshot, core_state):
    arrays, point_count = _require_snapshot(snapshot)
    if not isinstance(core_state, dict):
        raise ValueError("missing Core runtime state")
    evidence = core_state.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("missing checkpoint evidence state")
    if int(evidence.get("point_count", -1)) != point_count:
        raise ValueError("snapshot/core_state point count mismatch")

    for snapshot_name, state_name in _STATE_FIELDS.items():
        if state_name not in evidence:
            raise ValueError(f"missing checkpoint evidence field: {state_name}")
        _require_exact(snapshot_name, arrays[snapshot_name], evidence[state_name])

    expected_need = np.clip(
        1.0 - arrays["S"] * (1.0 - arrays["A"]), 0.0, 1.0
    ).astype(arrays["N"].dtype, copy=False)
    expected_prior = (
        arrays["T_p"] * arrays["V_p"].astype(arrays["T_p"].dtype)
    ).astype(arrays["r_p"].dtype, copy=False)
    expected_geometry = (
        arrays["T_g"] * arrays["V_g"].astype(arrays["T_g"].dtype)
    ).astype(arrays["r_g"].dtype, copy=False)
    expected_delta = (expected_prior - expected_geometry).astype(
        arrays["delta"].dtype, copy=False
    )
    for name, expected in (
        ("N", expected_need),
        ("r_p", expected_prior),
        ("r_g", expected_geometry),
        ("delta", expected_delta),
    ):
        if not np.array_equal(arrays[name], expected):
            raise ValueError(f"snapshot/core_state derived mismatch: {name}")
    return arrays


def build_g1_iteration_inputs(
    *,
    iteration,
    centers,
    snapshot,
    core_state,
    checkpoint_sha256,
    snapshot_sha256,
):
    iteration = int(iteration)
    if iteration <= 0:
        raise ValueError("iteration must be positive")
    centers = _as_numpy(centers)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("checkpoint centers must have shape [P,3]")
    arrays, point_count = _require_snapshot(snapshot)
    if centers.shape[0] != point_count:
        raise ValueError(
            "snapshot/checkpoint row count mismatch: "
            f"snapshot={point_count}, checkpoint={centers.shape[0]}"
        )
    if not isinstance(core_state, dict) or int(
        core_state.get("last_refresh_iteration", -1)
    ) != iteration:
        raise ValueError("D0 refresh iteration mismatch")
    arrays = validate_snapshot_core_state(arrays, core_state)
    centers64 = np.asarray(centers, dtype=np.float64)
    finite = np.isfinite(centers64).all(axis=1)
    finite_rows = np.flatnonzero(finite).astype(np.int64, copy=False)
    rejected_rows = np.flatnonzero(~finite).astype(np.int64, copy=False)
    if finite_rows.size == 0:
        raise ValueError("checkpoint contains no finite Gaussian centers")
    return G1IterationInputs(
        iteration=iteration,
        original_point_count=point_count,
        centers=centers64[finite_rows].copy(),
        finite_row_indices=finite_rows,
        rejected_center_indices=rejected_rows,
        snapshot={name: value.copy() for name, value in arrays.items()},
        checkpoint_sha256=str(checkpoint_sha256),
        snapshot_sha256=str(snapshot_sha256),
    )


def load_g1_iteration(run_dir, iteration):
    if torch is None:
        raise RuntimeError("Torch is required to read a Core checkpoint")
    run_dir = Path(run_dir)
    iteration = int(iteration)
    checkpoint_path = run_dir / f"chkpnt{iteration}.pth"
    snapshot_path = (
        run_dir / "d0_evidence" / f"iteration_{iteration:06d}.npz"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint is missing: {checkpoint_path}")
    if not snapshot_path.is_file():
        raise FileNotFoundError(f"snapshot is missing: {snapshot_path}")

    payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("G1 requires a versioned Core checkpoint")
    if int(payload.get("iteration", -1)) != iteration:
        raise ValueError("checkpoint iteration mismatch")
    capture = payload.get("gaussian_state")
    if not isinstance(capture, tuple) or len(capture) != 16:
        raise ValueError("invalid Gaussian capture schema")
    centers = capture[1]
    if not isinstance(centers, torch.Tensor):
        raise ValueError("Gaussian centers are missing from checkpoint")

    with np.load(snapshot_path, allow_pickle=False) as archive:
        if set(archive.files) != set(SNAPSHOT_FIELDS):
            snapshot = {name: archive[name] for name in archive.files}
        else:
            snapshot = {name: archive[name] for name in SNAPSHOT_FIELDS}

    return build_g1_iteration_inputs(
        iteration=iteration,
        centers=centers,
        snapshot=snapshot,
        core_state=payload.get("core_state"),
        checkpoint_sha256=sha256_file(checkpoint_path),
        snapshot_sha256=sha256_file(snapshot_path),
    )
