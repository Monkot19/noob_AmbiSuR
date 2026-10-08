"""Immutable information firewall and post-token, fail-closed mesh admission.

This is a prerequisite, not human execution approval. A caller must separately
verify completed runs and obtain GT-probe approval (Task 8). Inputs to
authorize_first_gt_access are {path, sha256} identities, NOT trusted payloads.
There is deliberately no geometry release/approval generator here.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from reliability.g1_prior_transfer_confirmation import (
    _canonical_bytes, _validate_record,
)


_SHA = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_CONTRACT_FIELDS = {
    "action", "cluster_unit", "affected_parameters", "outcome", "horizon",
    "formula", "constants", "validity", "state", "topology_lineage",
    "thresholds", "metrics", "coverage", "compute_budget", "stop_rules",
}
_RELEASE_FIELDS = {
    "schema_version", "kind", "release_id", "created_utc", "repository", "stage",
    "outcome", "specifications", "evidence", "frozen_contract", "approval",
    "candidate_admitted_to_utility", "utility_cannot_reopen", "utility_gt_access",
}
_APPROVED_FIELDS = (
    "release_id", "stage", "outcome", "repository", "specifications", "evidence",
    "candidate_admitted_to_utility", "utility_cannot_reopen", "utility_gt_access",
)


def _absolute(value):
    if not isinstance(value, (str, Path)) or not Path(value).is_absolute():
        raise ValueError("firewall paths must be absolute")
    return Path(value).resolve()


def _identity(value):
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError("expected an exact disk identity handle {path, sha256}")
    path = _absolute(value["path"])
    if _SHA.fullmatch(str(value["sha256"])) is None:
        raise ValueError("invalid identity SHA256")
    return {"path": str(path), "sha256": value["sha256"]}


def _time(value):
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.tzinfo is None or result.utcoffset().total_seconds() != 0:
            raise ValueError("timestamp must be UTC")
        return result
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("invalid UTC chronology") from exc


def _read_bytes_guarded(path, protected):
    _reject_aliases([{"path": str(path)}], protected)
    with path.open("rb") as stream:
        # Check the opened object too: a path may be replaced after its initial
        # metadata check. No bytes are read until the descriptor is admitted.
        opened = os.fstat(stream.fileno())
        for target in protected:
            try:
                info = target.stat()
            except FileNotFoundError:
                continue
            if (opened.st_dev, opened.st_ino) == (info.st_dev, info.st_ino):
                raise ValueError("opened firewall artifact aliases protected GT/target")
        return stream.read()


def _read_verified(identity, *, detached=False, protected=()):
    identity = _identity(identity)
    path = Path(identity["path"])
    payload = _read_bytes_guarded(path, protected)
    if hashlib.sha256(payload).hexdigest() != identity["sha256"]:
        raise ValueError("firewall artifact SHA256 mismatch")
    if detached:
        # Only the canonical detached digest format is admitted (not shell input).
        accepted = [(identity["sha256"] + ending).encode("ascii") for ending in ("\n", "\r\n")]
        if _read_bytes_guarded(Path(str(path) + ".sha256"), protected) not in accepted:
            raise ValueError("detached geometry release SHA256 mismatch")
    return payload


def _nonempty(value):
    """Reject empty placeholders; scientific content is bound to reviewed bytes."""
    if value is None or value == "" or value == {} or value == []:
        return False
    if isinstance(value, Mapping):
        return all(isinstance(key, str) and key and _nonempty(item) for key, item in value.items())
    if isinstance(value, list):
        return all(_nonempty(item) for item in value)
    return True


def _release_shape(record):
    if not isinstance(record, Mapping) or set(record) != _RELEASE_FIELDS:
        raise ValueError("geometry release fields mismatch")
    if type(record["schema_version"]) is not int or record["schema_version"] != 1 or record["kind"] != "geometry_release":
        raise ValueError("geometry release schema mismatch")
    if not isinstance(record["release_id"], str) or re.fullmatch(r"[A-Za-z0-9_-]+", record["release_id"]) is None:
        raise ValueError("unsafe geometry release ID")
    _time(record["created_utc"])
    repository = record["repository"]
    if not isinstance(repository, Mapping) or set(repository) != {"root", "commit", "clean"}:
        raise ValueError("geometry repository identity mismatch")
    _absolute(repository["root"])
    if _COMMIT.fullmatch(str(repository["commit"])) is None or repository["clean"] is not True:
        raise ValueError("geometry release requires an exact clean commit")
    if record["stage"] not in ("G-A", "G-B", "G-C"):
        raise ValueError("unknown geometry stage")
    outcome = record["outcome"]
    if outcome not in ("STAGE_G_C_CANDIDATE", "NO_ACTION_SPECIFIC_SIGNAL"):
        raise ValueError("not an approved geometry release outcome")
    candidate = outcome == "STAGE_G_C_CANDIDATE"
    if (record["candidate_admitted_to_utility"] is not candidate
            or record["utility_cannot_reopen"] is not True
            or record["utility_gt_access"] != "METADATA_ONLY"):
        raise ValueError("geometry release violates Utility isolation/termination")
    contract = record["frozen_contract"]
    if candidate:
        if record["stage"] != "G-C" or not isinstance(contract, Mapping) or set(contract) != _CONTRACT_FIELDS:
            raise ValueError("candidate requires complete Stage G-C contract")
        if not _nonempty(contract):
            raise ValueError("candidate has empty contract definitions")
    elif contract is not None:
        raise ValueError("terminal release may not admit a candidate contract")
    # allow_nan=False also rejects nonfinite scientific constants recursively.
    _canonical_bytes(contract)
    specifications = record["specifications"]
    if not isinstance(specifications, Mapping) or set(specifications) != {"charter", "stage_specification"}:
        raise ValueError("geometry specification identities missing")
    evidence = record["evidence"]
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("geometry release requires stage evidence")
    identities = [*specifications.values(), *evidence, record["approval"]]
    paths = [str(_absolute(_identity(item)["path"])) for item in identities]
    if len(set(paths)) != len(paths):
        raise ValueError("geometry evidence/specification/approval must be distinct")
    return outcome


def validate_geometry_release(record: Mapping) -> str:
    """Shape-only classification; no referenced artifact is opened/authorized.

    Full review/evidence verification is private to the protected first-access
    boundary. An outcome string from this function is NOT an access token.
    """
    return _release_shape(record)


def _verify_geometry_references(record, protected):
    identities = [*record["specifications"].values(), *record["evidence"], record["approval"]]
    for identity in identities:
        _read_verified(identity, protected=protected)
    approval = json.loads(_read_verified(record["approval"], protected=protected))
    expected = {"schema_version": 1, "kind": "geometry_release_approval",
                **{key: record[key] for key in _APPROVED_FIELDS},
                "frozen_contract_sha256": hashlib.sha256(_canonical_bytes(record["frozen_contract"])).hexdigest()}
    if approval != expected:
        raise ValueError("geometry release differs from reviewed approval")


def load_geometry_release(path: Path, expected_sha256: str) -> dict:
    """Hash-check release JSON + detached SHA, without opening its references.

    Use authorize_first_gt_access for full review/evidence verification.
    """
    payload = _read_verified({"path": str(path), "sha256": expected_sha256}, detached=True)
    record = json.loads(payload)
    if _canonical_bytes(record) != payload:
        raise ValueError("geometry release must be canonical JSON")
    validate_geometry_release(record)
    return record


def _exists(path):
    return os.path.lexists(path)


def _reloaded(prior_identity, geometry_identity):
    # Inspect only the small confirmation itself before loading any referenced
    # artifact: a disguised evidence/source reference must never open GT.
    prior = json.loads(_read_verified(prior_identity))
    prior = _validate_record(prior, require_targets_absent=False, verify_record_files=False)
    protected = [_absolute(prior["gt_mesh"]["path"]),
                 *[_absolute(path) for path in prior["probe_targets"].values()]]
    _reject_aliases([prior_identity, geometry_identity, prior["source_record"], prior["snapshot_record"]], protected)
    geometry_payload = _read_verified(geometry_identity, detached=True, protected=protected)
    geometry = json.loads(geometry_payload)
    if _canonical_bytes(geometry) != geometry_payload:
        raise ValueError("geometry release must be canonical JSON")
    validate_geometry_release(geometry)
    _reject_aliases([*geometry["specifications"].values(), *geometry["evidence"], geometry["approval"]], protected)
    # Reuse the production schema validator, with guarded I/O for its two file
    # identities. The general loader's unguarded opens cannot enforce this
    # boundary's protected-descriptor contract against replacement races.
    for name in ("source_record", "snapshot_record"):
        identity = {key: prior[name][key] for key in ("path", "sha256")}
        _read_verified(identity, protected=protected)
    _verify_geometry_references(geometry, protected)
    return prior, geometry


def _reject_aliases(identities, protected):
    for item in identities:
        for path in (_absolute(item["path"]), _absolute(str(item["path"]) + ".sha256")):
            if any(path == target or path in target.parents or target in path.parents for target in protected):
                raise ValueError("firewall artifact aliases GT or a probe target")
            if path.exists():
                info = path.stat()
                for target in protected:
                    try:
                        other = target.stat()
                    except FileNotFoundError:
                        continue
                    if (info.st_dev, info.st_ino) == (other.st_dev, other.st_ino):
                        raise ValueError("firewall artifact hardlinks protected GT/target")


def authorize_first_gt_access(prior_record, geometry_record, *, access_log_path: Path) -> dict:
    """Reload both identities and publish a one-shot record before any mesh parse.

    Any existing access log is fail-closed, even if identical. A failed attempt
    after log publication stays recorded; never delete it to obtain a retry.
    The future evaluator must call this immediately before its mesh boundary.
    """
    prior_identity, geometry_identity = _identity(prior_record), _identity(geometry_record)
    prior, geometry = _reloaded(prior_identity, geometry_identity)
    log = _absolute(access_log_path)
    targets = prior["probe_targets"]
    if log != _absolute(targets["access_log_path"]):
        raise ValueError("first-access target differs from preregistration")
    if _exists(log):
        raise FileExistsError("first Utility GT access is already recorded")
    for name in ("output_dir", "staging_dir"):
        if _exists(targets[name]):
            raise ValueError("Utility probe artifact exists before first access")
    now = datetime.now(timezone.utc)
    if _time(prior["created_utc"]) > now or _time(geometry["created_utc"]) > now:
        raise ValueError("confirmation/release is later than first GT access")
    if _absolute(geometry["repository"]["root"]) != _absolute(prior["repository"]["root"]):
        raise ValueError("geometry release belongs to another project")
    record = {
        "schema_version": 1, "kind": "utility_first_gt_access", "created_utc": now.isoformat(),
        "prior_confirmation": prior_identity, "geometry_release": geometry_identity,
        "geometry_outcome": geometry["outcome"], "gt_mesh": prior["gt_mesh"],
        "access_log_path": str(log), "mesh_parse_preceded_by_this_record": True,
        "utility_feedback_cannot_reopen_geometry": True,
    }
    payload = _canonical_bytes(record)
    temporary = log.parent / f".{log.name}.tmp-{uuid.uuid4().hex}"
    if not log.parent.is_dir():
        raise ValueError("first-access log parent is missing")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _reloaded(prior_identity, geometry_identity)
        for name in ("output_dir", "staging_dir"):
            if _exists(targets[name]):
                raise ValueError("Utility probe target appeared before publication")
        # Hardlink publication is exclusive. os.replace would overwrite a racer.
        os.link(temporary, log)
        _reloaded(prior_identity, geometry_identity)
        for name in ("output_dir", "staging_dir"):
            if _exists(targets[name]):
                raise ValueError("Utility probe target appeared during publication")
        if log.read_bytes() != payload:
            raise ValueError("first-access record changed during publication")
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(log), "sha256": hashlib.sha256(payload).hexdigest(), "record": record}


@dataclass(frozen=True)
class UtilityMeshAdmission:
    """Compact admission evidence only: never a filter or repaired surface."""

    outcome: str
    reasons: tuple[str, ...]
    summary: dict


def _verify_mesh_access(confirmation, token):
    if not isinstance(token, Mapping) or set(token) != {"path", "sha256", "record"}:
        raise ValueError("verified first-access token required before mesh parsing")
    prior = _validate_record(confirmation, require_targets_absent=False, verify_record_files=False)
    identity = {key: token[key] for key in ("path", "sha256")}
    log = _absolute(prior["probe_targets"]["access_log_path"])
    if _absolute(identity["path"]) != log:
        raise ValueError("access token target mismatch")
    payload = _read_verified(identity, protected=[_absolute(prior["gt_mesh"]["path"])])
    record = json.loads(payload)
    fields = {"schema_version", "kind", "created_utc", "prior_confirmation", "geometry_release",
              "geometry_outcome", "gt_mesh", "access_log_path", "mesh_parse_preceded_by_this_record",
              "utility_feedback_cannot_reopen_geometry"}
    if set(record) != fields or payload != _canonical_bytes(record) or record != token["record"]:
        raise ValueError("first-access record payload mismatch")
    if (type(record["schema_version"]) is not int or record["schema_version"] != 1
            or record["kind"] != "utility_first_gt_access"
            or record["mesh_parse_preceded_by_this_record"] is not True
            or record["utility_feedback_cannot_reopen_geometry"] is not True
            or record["access_log_path"] != str(log) or record["gt_mesh"] != prior["gt_mesh"]):
        raise ValueError("first-access record contract mismatch")
    reloaded, geometry = _reloaded(record["prior_confirmation"], record["geometry_release"])
    if reloaded != prior or record["geometry_outcome"] != geometry["outcome"]:
        raise ValueError("first-access confirmation/release binding mismatch")
    when = _time(record["created_utc"])
    if not max(_time(prior["created_utc"]), _time(geometry["created_utc"])) <= when <= datetime.now(timezone.utc):
        raise ValueError("first-access chronology mismatch")
    return prior


def select_admission_points(points):
    """Frozen SHA-minhash: LE uint64 ID || LE float64 XYZ; at most 50,000."""
    ranked = []
    for point_id, point in points.items():
        if not isinstance(point_id, (int, np.integer)) or not 0 <= point_id < 2**64:
            raise ValueError("invalid COLMAP point identity")
        xyz = np.asarray(point.xyz, dtype="<f8")
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError("nonfinite/invalid registered sparse point")
        payload = np.asarray([point_id], dtype="<u8").tobytes() + xyz.tobytes()
        ranked.append((hashlib.sha256(payload).digest(), int(point_id), xyz))
    if not ranked:
        raise ValueError("no registered sparse points for mesh admission")
    ranked.sort(key=lambda row: (row[0], row[1]))
    selected = ranked[:50000]
    return np.array([row[1] for row in selected], dtype=np.uint64), np.array([row[2] for row in selected])


def utility_camera_rays(camera, image):
    """8x6 full-frame cell-center rays, no crop or image resampling."""
    from reliability.g1_visualization import camera_to_world_for_ray_cast
    from scripts.preprocess.read_write_model import qvec2rotmat
    params = np.asarray(camera.params, dtype=np.float64)
    if camera.model == "PINHOLE" and params.shape == (4,):
        fx, fy, cx, cy = params
    elif camera.model == "SIMPLE_PINHOLE" and params.shape == (3,):
        fx, cx, cy = params
        fy = fx
    else:
        raise ValueError("unsupported admission camera model")
    if not np.isfinite(params).all() or min(fx, fy, camera.width, camera.height) <= 0:
        raise ValueError("invalid admission camera intrinsics")
    qvec, tvec = np.asarray(image.qvec), np.asarray(image.tvec)
    if (qvec.shape != (4,) or tvec.shape != (3,) or not np.isfinite(qvec).all()
            or not np.isfinite(tvec).all() or not np.isclose(np.linalg.norm(qvec), 1., rtol=0., atol=1e-6)):
        raise ValueError("invalid admission camera pose")
    intrinsic = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
    w2c = np.eye(4)
    w2c[:3, :3], w2c[:3, 3] = qvec2rotmat(qvec), tvec
    # Reuse the established W2C inversion boundary, without constructing Scene
    # or any training/rendering runtime.
    class Calibration:
        def get_calib_matrix_nerf(self, scale):
            return intrinsic, w2c
    _, c2w = camera_to_world_for_ray_cast(Calibration())
    x, y = np.meshgrid((np.arange(8) + .5) * camera.width / 8.,
                       (np.arange(6) + .5) * camera.height / 6., indexing="xy")
    directions = np.stack(((x-cx)/fx, (y-cy)/fy, np.ones_like(x)), axis=-1).reshape(48, 3)
    directions = directions @ c2w[:3, :3].T
    rays = np.concatenate((np.broadcast_to(c2w[:3, 3], directions.shape), directions), axis=1)
    if not np.isfinite(rays).all() or np.any(np.linalg.norm(directions, axis=1) <= 0.):
        raise ValueError("nonfinite/zero admission camera rays")
    return rays


def _load_admission_mesh(path):
    from reliability.offline_g1 import load_valid_mesh
    return load_valid_mesh(path)


def _admission_distances(points, mesh):
    from reliability.offline_g1 import closest_triangle_distances
    return closest_triangle_distances(points, mesh)


def _admission_ray_depths(rays, mesh):
    # Same Open3D full-surface raycast backend as cast_gt_depth; batch only the
    # fixed 7,056 admission rays so the full triangle scene is built once.
    import open3d as o3d
    surface = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32)),
        o3d.core.Tensor(np.asarray(mesh.triangles, dtype=np.int32)))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(surface)
    return scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))["t_hit"].numpy().astype(np.float64)


def _file_identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _source_identities(root):
    return {str(path.relative_to(root)): _file_identity(path.stat())
            for path in root.rglob("*") if path.is_file()}


def _mesh_identity(path, expected):
    # Do not silently admit another lexical path/symlink to the frozen asset.
    path = Path(path)
    if not path.is_absolute() or str(path) != expected["path"] or path.is_symlink() or not path.is_file():
        raise ValueError("GT mesh path identity mismatch")
    before = path.stat()
    if before.st_size != expected["bytes"]:
        raise ValueError("GT mesh byte size mismatch")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if (before.st_dev, before.st_ino, before.st_size) != (opened.st_dev, opened.st_ino, opened.st_size):
            raise ValueError("GT mesh changed at open")
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    after = path.stat()
    if _file_identity(before) != _file_identity(after) or digest.hexdigest() != expected["sha256"]:
        raise ValueError("GT mesh SHA256/mutation mismatch")
    return _file_identity(before)


def _admission_source(source_root, prior):
    from reliability.utility_snapshot import audit_utility_source, source_manifest
    protected = [_absolute(prior["gt_mesh"]["path"]),
                 *[_absolute(path) for path in prior["probe_targets"].values()]]
    source = json.loads(_read_verified(prior["source_record"], protected=protected))
    root = _absolute(source_root)
    if (root != _absolute(source["source_root"]) or source.get("audit_kind") != "utility_source"
            or source.get("gt_access") != "NONE"):
        raise ValueError("mesh admission source binding mismatch")
    identities = _source_identities(root)
    _reject_aliases([{"path": str(path)} for path in root.rglob("*") if path.is_file()], protected)
    current = source_manifest(audit_utility_source(root, expected_count=147))
    if any(source.get(key) != value for key, value in current.items()):
        raise ValueError("mesh admission source manifest changed")
    from scripts.preprocess.read_write_model import read_model
    cameras, images, points = read_model(str(root / "sparse/0"), ext=".txt")
    if len(images) != 147 or sorted(image.name for image in images.values()) != current["image_names"]:
        raise ValueError("mesh admission registered camera inventory mismatch")
    rays = np.stack([utility_camera_rays(cameras[image.camera_id], image)
                     for _, image in sorted(images.items())])
    ids, xyz = select_admission_points(points)
    if _source_identities(root) != identities:
        raise ValueError("mesh admission source changed during parsing")
    return current, ids, xyz, rays, identities


def audit_utility_mesh(mesh_path: Path, source_root: Path, confirmation: Mapping, *, access_token) -> UtilityMeshAdmission:
    """Post-firewall, read-only admission; any failure stops as INCONCLUSIVE.

    Not a geometry release generator, execution approval, or model fitter.
    No admission statistic selects Gaussian rows, faces, or a new transform.
    """
    summary = {}
    try:
        prior = _verify_mesh_access(confirmation, access_token)
        source, ids, points, rays, source_identities = _admission_source(source_root, prior)
        mesh_identity = _mesh_identity(mesh_path, prior["gt_mesh"])
        mesh = _load_admission_mesh(mesh_path)
        if mesh.nonfinite_vertex_count or len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
            raise ValueError("mesh contains nonfinite vertices or no valid surface")
        summary.update(gt_mesh=dict(prior["gt_mesh"]), coordinate_transform=prior["mesh_admission"]["coordinate_transform"],
                       world_unit="meter", source_sha256=source["source_sha256"],
                       surface={"source_vertex_count": mesh.source_vertex_count,
                                "source_triangle_count": mesh.source_triangle_count,
                                "valid_vertex_count": len(mesh.vertices), "valid_triangle_count": len(mesh.triangles),
                                "rejected_nonfinite_triangle_count": mesh.rejected_nonfinite_triangle_count,
                                "rejected_degenerate_triangle_count": mesh.rejected_degenerate_triangle_count})
        distances = np.asarray(_admission_distances(points, mesh), dtype=np.float64)
        if distances.shape != (len(points),) or not np.isfinite(distances).all() or np.any(distances < 0):
            raise ValueError("invalid admission sparse-point distances")
        median, p90 = np.quantile(distances, [.5, .9])
        fraction = float(np.mean(distances <= .10))
        summary["alignment"] = {"sample_count": len(points), "registered_point_count": source["point_count"],
                                "sample_ids_sha256": hashlib.sha256(ids.astype("<u8").tobytes()).hexdigest(),
                                "median_distance_m": float(median), "p90_distance_m": float(p90),
                                "fraction_within_0_10_m": fraction}
        limits = prior["mesh_admission"]
        if (median > limits["point_distance_median_at_most_m"] or p90 > limits["point_distance_p90_at_most_m"]
                or fraction < limits["point_fraction_within_0.10_m_at_least"]):
            raise ValueError("mesh sparse-point alignment gate failed; no repair permitted")
        depths = np.asarray(_admission_ray_depths(rays, mesh), dtype=np.float64)
        if depths.shape != (147, 48) or np.isnan(depths).any() or (depths <= 0).any():
            raise ValueError("invalid admission ray-depth inventory")
        hit = np.isfinite(depths)
        per_camera = hit.mean(axis=1)
        aggregate, camera_fraction = float(hit.mean()), float(np.mean(per_camera >= .50))
        summary["coverage"] = {"registered_camera_count": 147, "rays_per_camera": 48,
                               "aggregate_hit_fraction": aggregate, "per_camera_hit_fraction": per_camera.tolist(),
                               "camera_fraction_at_least_half": camera_fraction}
        if (aggregate < limits["aggregate_ray_hit_fraction_at_least"]
                or camera_fraction < limits["camera_fraction_with_hit_fraction_at_least_0.50"]):
            raise ValueError("mesh camera coverage gate failed; no crop permitted")
        # Recheck every consumed input, not just the mesh, before admission.
        if _mesh_identity(mesh_path, prior["gt_mesh"]) != mesh_identity:
            raise ValueError("GT mesh changed during parsing/query")
        _verify_mesh_access(confirmation, access_token)
        final_source, _, _, _, final_identities = _admission_source(source_root, prior)
        if final_source != source or final_identities != source_identities:
            raise ValueError("mesh admission source changed during query")
        return UtilityMeshAdmission("ADMITTED", (), summary)
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, ImportError) as exc:
        return UtilityMeshAdmission("INCONCLUSIVE", (f"{type(exc).__name__}: {exc}",), summary)
