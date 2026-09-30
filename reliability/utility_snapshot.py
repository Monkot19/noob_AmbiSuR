"""Fail-closed Utility Room source and preprocessing snapshot contracts."""

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
from typing import Mapping
import uuid

import numpy as np
from PIL import Image, UnidentifiedImageError


_SOURCE_SCHEMA_VERSION = 1
_SUPPORTED_CAMERA_MODELS = {"PINHOLE": 4, "SIMPLE_PINHOLE": 3}
_DERIVED_NAMES = (
    "estimated_depths",
    "estimated_confs",
    "sparse_da3",
    "sparse_da3_aligned",
)
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class UtilitySourceAudit:
    source_root: str
    expected_count: int
    image_count: int
    registered_image_count: int
    camera_count: int
    point_count: int
    image_names: tuple[str, ...]
    files: tuple[Mapping[str, object], ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, root: Path) -> dict:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _data_lines(path: Path):
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            yield line


def _parse_cameras(path: Path) -> dict[int, dict]:
    cameras = {}
    for line in _data_lines(path):
        fields = line.split()
        if len(fields) < 4:
            raise ValueError("malformed COLMAP camera record")
        camera_id = int(fields[0])
        if camera_id in cameras:
            raise ValueError("duplicate COLMAP camera ID")
        model = fields[1]
        if model not in _SUPPORTED_CAMERA_MODELS:
            raise ValueError(f"unsupported COLMAP camera model: {model}")
        width, height = int(fields[2]), int(fields[3])
        params = tuple(float(value) for value in fields[4:])
        if width <= 0 or height <= 0 or len(params) != _SUPPORTED_CAMERA_MODELS[model]:
            raise ValueError("invalid COLMAP camera dimensions or parameter count")
        if not all(math.isfinite(value) for value in params):
            raise ValueError("non-finite COLMAP camera parameters")
        cameras[camera_id] = {
            "model": model,
            "width": width,
            "height": height,
        }
    if not cameras:
        raise ValueError("no COLMAP cameras")
    return cameras


def _parse_images(path: Path) -> tuple[dict[int, dict], set[str]]:
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    images = {}
    names = set()
    index = 0
    while index < len(raw_lines):
        line = raw_lines[index].strip()
        index += 1
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 10:
            raise ValueError("malformed COLMAP image record")
        image_id = int(fields[0])
        if image_id in images:
            raise ValueError("duplicate COLMAP image ID")
        values = tuple(float(value) for value in fields[1:8])
        if not all(math.isfinite(value) for value in values):
            raise ValueError("non-finite COLMAP image pose")
        quaternion = values[:4]
        if not math.isclose(
            math.sqrt(sum(value * value for value in quaternion)),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError("COLMAP quaternion is not normalized")
        camera_id = int(fields[8])
        name = fields[9]
        if name in names or Path(name).name != name:
            raise ValueError("duplicate or unsafe COLMAP image name")
        names.add(name)
        images[image_id] = {"camera_id": camera_id, "name": name}
        if index >= len(raw_lines):
            raise ValueError("COLMAP image record is missing its observation line")
        observations = raw_lines[index].split()
        index += 1
        if len(observations) % 3 != 0:
            raise ValueError("malformed COLMAP observation record")
        for offset in range(0, len(observations), 3):
            if not all(math.isfinite(float(value)) for value in observations[offset : offset + 2]):
                raise ValueError("non-finite COLMAP observation")
            int(observations[offset + 2])
    if not images:
        raise ValueError("no registered COLMAP images")
    return images, names


def _parse_points(path: Path) -> int:
    seen = set()
    for line in _data_lines(path):
        fields = line.split()
        if len(fields) < 8:
            raise ValueError("malformed COLMAP point record")
        point_id = int(fields[0])
        if point_id in seen:
            raise ValueError("duplicate COLMAP point ID")
        seen.add(point_id)
        numeric = tuple(float(value) for value in fields[1:4]) + (float(fields[7]),)
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("non-finite COLMAP point")
        if (len(fields) - 8) % 2 != 0:
            raise ValueError("malformed COLMAP point track")
    if not seen:
        raise ValueError("no COLMAP points")
    return len(seen)


def audit_utility_source(
    source_root: Path,
    expected_count: int = 147,
) -> UtilitySourceAudit:
    source_root = Path(source_root).expanduser().resolve()
    if not source_root.is_dir() or expected_count <= 0:
        raise ValueError("invalid Utility source root or expected count")
    for name in _DERIVED_NAMES:
        if (source_root / name).exists():
            raise ValueError(f"pre-existing derived path is forbidden: {name}")

    images_root = source_root / "images"
    sparse_root = source_root / "sparse" / "0"
    required = tuple(
        sparse_root / name
        for name in ("cameras.txt", "images.txt", "points3D.txt")
    )
    if not images_root.is_dir() or not all(path.is_file() for path in required):
        raise ValueError("Utility source inventory is incomplete")
    image_paths = tuple(sorted(
        (path for path in images_root.iterdir() if path.is_file()),
        key=lambda path: path.name,
    ))
    if any(path.is_dir() for path in images_root.iterdir()):
        raise ValueError("nested Utility image directories are forbidden")
    if len(image_paths) != expected_count:
        raise ValueError("Utility image count mismatch")

    cameras = _parse_cameras(required[0])
    registered, registered_names = _parse_images(required[1])
    point_count = _parse_points(required[2])
    disk_names = {path.name for path in image_paths}
    if len(registered) != expected_count or disk_names != registered_names:
        raise ValueError("Utility disk/registration filename mismatch")

    by_name = {record["name"]: record for record in registered.values()}
    for path in image_paths:
        record = by_name[path.name]
        camera_id = record["camera_id"]
        if camera_id not in cameras:
            raise ValueError("COLMAP image references an unknown camera")
        try:
            with Image.open(path) as image:
                image.load()
                size = image.size
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"unreadable Utility image: {path.name}") from exc
        camera = cameras[camera_id]
        if size != (camera["width"], camera["height"]):
            raise ValueError("image dimensions do not match COLMAP camera")

    files = tuple(
        _file_record(path, source_root)
        for path in sorted((*image_paths, *required), key=lambda item: item.relative_to(source_root).as_posix())
    )
    return UtilitySourceAudit(
        source_root=str(source_root),
        expected_count=int(expected_count),
        image_count=len(image_paths),
        registered_image_count=len(registered),
        camera_count=len(cameras),
        point_count=point_count,
        image_names=tuple(sorted(disk_names)),
        files=files,
    )


def source_manifest(audit: UtilitySourceAudit) -> dict:
    files = [dict(entry) for entry in audit.files]
    payload = (json.dumps(files, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    return {
        "schema_version": _SOURCE_SCHEMA_VERSION,
        "source_root": audit.source_root,
        "expected_count": audit.expected_count,
        "image_count": audit.image_count,
        "registered_image_count": audit.registered_image_count,
        "camera_count": audit.camera_count,
        "point_count": audit.point_count,
        "image_names": list(audit.image_names),
        "files": files,
        "source_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _canonical_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _absolute_path(value, label) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    return path.resolve()


def _contains_gt(value) -> bool:
    serialized = json.dumps(value, sort_keys=True).replace("\\", "/").lower()
    return "/gt/" in serialized or "gt_mesh" in serialized or '"gt"' in serialized


def _verify_source_manifest(manifest: Mapping, *, root_override=None) -> None:
    root = _absolute_path(root_override or manifest["source_root"], "source root")
    current = []
    for entry in manifest["files"]:
        path = root / PurePosixPath(entry["path"])
        if not path.is_file():
            raise ValueError(f"source file is missing: {entry['path']}")
        actual = _file_record(path, root)
        if actual != entry:
            raise ValueError(f"source file changed: {entry['path']}")
        current.append(actual)
    digest = hashlib.sha256(_canonical_bytes(current)).hexdigest()
    if digest != manifest["source_sha256"]:
        raise ValueError("source manifest SHA256 mismatch")


def _verify_da3_checkpoint(checkpoint: Mapping) -> None:
    path = _absolute_path(checkpoint["path"], "DA3 checkpoint path")
    if not path.is_file():
        raise FileNotFoundError(f"DA3 checkpoint is missing: {path}")
    if path.stat().st_size != int(checkpoint["bytes"]):
        raise ValueError("DA3 checkpoint byte count mismatch")
    if _sha256_file(path) != str(checkpoint["sha256"]):
        raise ValueError("DA3 checkpoint SHA256 mismatch")


def _validate_da3_confirmation(record: Mapping, *, require_targets_absent: bool) -> dict:
    required = {
        "schema_version",
        "kind",
        "created_utc",
        "repository",
        "source",
        "da3_checkpoint",
        "environment",
        "preprocessing",
        "command",
        "targets",
        "gt_access",
    }
    if set(record) != required or record["schema_version"] != 1 or record["kind"] != "utility_da3_confirmation":
        raise ValueError("DA3 confirmation schema mismatch")
    try:
        datetime.fromisoformat(str(record["created_utc"]).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("invalid DA3 confirmation timestamp") from exc
    repository = record["repository"]
    if set(repository) != {"root", "commit", "clean"} or _SHA40.fullmatch(str(repository["commit"])) is None or repository["clean"] is not True:
        raise ValueError("DA3 repository identity must be exact and clean")
    _absolute_path(repository["root"], "repository root")
    source = record["source"]
    if source.get("schema_version") != _SOURCE_SCHEMA_VERSION or _SHA64.fullmatch(str(source.get("source_sha256"))) is None:
        raise ValueError("DA3 source manifest mismatch")
    checkpoint = record["da3_checkpoint"]
    if set(checkpoint) != {"path", "bytes", "sha256"} or _SHA64.fullmatch(str(checkpoint["sha256"])) is None or int(checkpoint["bytes"]) <= 0:
        raise ValueError("DA3 checkpoint identity mismatch")
    _absolute_path(checkpoint["path"], "DA3 checkpoint path")
    environment = record["environment"]
    if set(environment) != {"python", "python_version", "torch_version", "cuda_version"} or not all(str(value) for value in environment.values()):
        raise ValueError("DA3 environment mismatch")
    preprocessing = record["preprocessing"]
    if set(preprocessing) != {"max_points", "ransac_thresh"}:
        raise ValueError("DA3 preprocessing field mismatch")
    max_points = preprocessing["max_points"]
    threshold = preprocessing["ransac_thresh"]
    if isinstance(max_points, bool) or not isinstance(max_points, int) or max_points <= 0:
        raise ValueError("max_points must be an explicit positive integer")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(float(threshold)) or float(threshold) <= 0:
        raise ValueError("ransac_thresh must be an explicit positive finite number")
    command = record["command"]
    if not isinstance(command, list) or len(command) != 5 or command[0] != "bash" or Path(command[1]).name != "run_da3_single.sh":
        raise ValueError("DA3 command mismatch")
    if command[3:] != [str(max_points), str(threshold)]:
        raise ValueError("DA3 command/preprocessing mismatch")
    targets = record["targets"]
    if set(targets) != {"staging_root", "snapshot_record_path"}:
        raise ValueError("DA3 target inventory mismatch")
    staging = _absolute_path(targets["staging_root"], "DA3 staging root")
    snapshot = _absolute_path(targets["snapshot_record_path"], "snapshot record path")
    source_root = _absolute_path(source["source_root"], "source root")
    if staging == source_root or source_root in staging.parents or snapshot == source_root or source_root in snapshot.parents:
        raise ValueError("DA3 outputs must be outside the source tree")
    if str(staging) != str(Path(command[2]).expanduser().resolve()):
        raise ValueError("DA3 command does not target the frozen staging root")
    if require_targets_absent and (staging.exists() or snapshot.exists() or snapshot.with_suffix(snapshot.suffix + ".sha256").exists()):
        raise ValueError("DA3 target already exists")
    if record["gt_access"] != "NONE" or _contains_gt(record):
        raise ValueError("DA3 confirmation must not reference GT")
    return json.loads(json.dumps(record))


def build_da3_confirmation(
    *,
    repository_root: Path,
    repository_commit: str,
    repository_clean: bool,
    source_manifest: Mapping,
    da3_checkpoint: Mapping,
    environment: Mapping,
    command,
    max_points: int,
    ransac_thresh: float,
    staging_root: Path,
    snapshot_record_path: Path,
    created_utc: str,
) -> dict:
    record = {
        "schema_version": 1,
        "kind": "utility_da3_confirmation",
        "created_utc": str(created_utc),
        "repository": {
            "root": str(_absolute_path(repository_root, "repository root")),
            "commit": str(repository_commit),
            "clean": repository_clean,
        },
        "source": json.loads(json.dumps(source_manifest)),
        "da3_checkpoint": dict(da3_checkpoint),
        "environment": dict(environment),
        "preprocessing": {
            "max_points": max_points,
            "ransac_thresh": ransac_thresh,
        },
        "command": [str(value) for value in command],
        "targets": {
            "staging_root": str(_absolute_path(staging_root, "DA3 staging root")),
            "snapshot_record_path": str(_absolute_path(snapshot_record_path, "snapshot record path")),
        },
        "gt_access": "NONE",
    }
    return _validate_da3_confirmation(record, require_targets_absent=True)


def _atomic_record_write(record: Mapping, path: Path) -> dict:
    path = _absolute_path(path, "record path")
    sha_path = path.with_suffix(path.suffix + ".sha256")
    if not path.parent.is_dir():
        raise ValueError("record parent directory is missing")
    if path.exists() or sha_path.exists():
        raise FileExistsError("record already exists")
    payload = _canonical_bytes(record)
    digest = hashlib.sha256(payload).hexdigest()
    token = uuid.uuid4().hex
    temporary = path.parent / f".{path.name}.tmp-{token}"
    temporary_sha = path.parent / f".{sha_path.name}.tmp-{token}"
    try:
        temporary.write_bytes(payload)
        temporary_sha.write_text(digest + "\n", encoding="ascii")
        os.replace(temporary, path)
        os.replace(temporary_sha, sha_path)
    except Exception:
        for candidate in (temporary, temporary_sha):
            try:
                candidate.unlink()
            except FileNotFoundError:
                pass
        raise
    return {"path": path, "sha256_path": sha_path, "sha256": digest}


def write_da3_confirmation(record: Mapping, path: Path) -> dict:
    validated = _validate_da3_confirmation(record, require_targets_absent=True)
    _verify_source_manifest(validated["source"])
    _verify_da3_checkpoint(validated["da3_checkpoint"])
    return _atomic_record_write(validated, path)


def load_da3_confirmation(
    path: Path,
    expected_sha256: str,
    *,
    verify_source: bool = False,
    verify_checkpoint: bool = False,
    expected_kind: str = "utility_da3_confirmation",
) -> dict:
    path = _absolute_path(path, "record path")
    payload = path.read_bytes()
    if _SHA64.fullmatch(str(expected_sha256)) is None or hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("record SHA256 mismatch")
    try:
        record = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError("record is not valid JSON") from exc
    if expected_kind == "utility_da3_confirmation":
        record = _validate_da3_confirmation(record, require_targets_absent=False)
        if verify_source:
            _verify_source_manifest(record["source"])
        if verify_checkpoint:
            _verify_da3_checkpoint(record["da3_checkpoint"])
    elif expected_kind == "utility_da3_snapshot":
        if record.get("schema_version") != 1 or record.get("kind") != expected_kind or _SHA64.fullmatch(str(record.get("snapshot_sha256"))) is None:
            raise ValueError("snapshot record schema mismatch")
    else:
        raise ValueError("unsupported record kind")
    return record


def _validate_derived_model(model_root: Path, expected_names: set[str]) -> None:
    text = tuple(model_root / f"{name}.txt" for name in ("cameras", "images", "points3D"))
    binary = tuple(model_root / f"{name}.bin" for name in ("cameras", "images", "points3D"))
    if all(path.is_file() for path in text):
        cameras = _parse_cameras(text[0])
        images, names = _parse_images(text[1])
        _parse_points(text[2])
        if names != expected_names or any(value["camera_id"] not in cameras for value in images.values()):
            raise ValueError("derived COLMAP image/camera inventory mismatch")
        return
    if all(path.is_file() for path in binary):
        from scripts.preprocess.read_write_model import read_model

        cameras, images, points = read_model(str(model_root), ext=".bin")
        if {image.name for image in images.values()} != expected_names or not cameras or not points:
            raise ValueError("derived binary COLMAP inventory mismatch")
        arrays = [camera.params for camera in cameras.values()]
        arrays.extend(image.qvec for image in images.values())
        arrays.extend(image.tvec for image in images.values())
        arrays.extend(point.xyz for point in points.values())
        if not all(np.isfinite(value).all() for value in arrays):
            raise ValueError("non-finite derived COLMAP model")
        return
    raise ValueError("derived COLMAP model is incomplete")


def audit_da3_snapshot(snapshot_root: Path, confirmation: Mapping) -> dict:
    confirmation = _validate_da3_confirmation(confirmation, require_targets_absent=False)
    snapshot_root = _absolute_path(snapshot_root, "snapshot root")
    if str(snapshot_root) != confirmation["targets"]["staging_root"] or not snapshot_root.is_dir():
        raise ValueError("completed snapshot root does not match confirmation")
    _verify_source_manifest(confirmation["source"], root_override=snapshot_root)
    names = tuple(confirmation["source"]["image_names"])
    expected = {f"{name}.npy" for name in names}
    depth_root = snapshot_root / "estimated_depths"
    conf_root = snapshot_root / "estimated_confs"
    depth_files = {path.name: path for path in depth_root.glob("*.npy")}
    conf_files = {path.name: path for path in conf_root.glob("*.npy")}
    if set(depth_files) != expected or set(conf_files) != expected:
        raise ValueError("DA3 depth/confidence array inventory mismatch")
    shapes = {}
    scales = {}
    source_root = snapshot_root / "images"
    for name in names:
        depth = np.load(depth_files[f"{name}.npy"], allow_pickle=False)
        confidence = np.load(conf_files[f"{name}.npy"], allow_pickle=False)
        if depth.ndim != 2 or confidence.ndim != 2 or depth.shape != confidence.shape or min(depth.shape) <= 0:
            raise ValueError("DA3 depth/confidence shape mismatch")
        if not np.issubdtype(depth.dtype, np.number) or not np.issubdtype(confidence.dtype, np.number) or not np.isfinite(depth).all() or not np.isfinite(confidence).all():
            raise ValueError("DA3 arrays must be finite numeric values")
        with Image.open(source_root / name) as image:
            width, height = image.size
        if depth.shape[0] > height or depth.shape[1] > width:
            raise ValueError("DA3 array shape exceeds registered image")
        shapes[name] = [int(depth.shape[0]), int(depth.shape[1])]
        scales[name] = [depth.shape[0] / height, depth.shape[1] / width]

    raw = snapshot_root / "sparse_da3" / "0"
    aligned = snapshot_root / "sparse_da3_aligned" / "0"
    _validate_derived_model(raw, set(names))
    _validate_derived_model(aligned, set(names))
    transform_path = aligned / "trans.json"
    try:
        transform = json.loads(transform_path.read_text(encoding="utf-8"))
        scale = float(transform["scale"])
        matrix = np.asarray(transform["T_matrix_scene2_from_scene1"], dtype=np.float64)
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("invalid DA3 alignment metadata") from exc
    if not math.isfinite(scale) or scale <= 0 or matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("invalid DA3 alignment transform")

    derived_roots = (depth_root, conf_root, snapshot_root / "sparse_da3", snapshot_root / "sparse_da3_aligned")
    derived_files = sorted(
        (path for root in derived_roots for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(snapshot_root).as_posix(),
    )
    manifest = [_file_record(path, snapshot_root) for path in derived_files]
    identity = {
        "source_sha256": confirmation["source"]["source_sha256"],
        "repository_commit": confirmation["repository"]["commit"],
        "da3_checkpoint": confirmation["da3_checkpoint"],
        "environment": confirmation["environment"],
        "preprocessing": confirmation["preprocessing"],
        "command": confirmation["command"],
        "array_shapes": shapes,
        "array_scales": scales,
        "alignment": {"scale": scale, "matrix": matrix.tolist()},
        "derived_manifest": manifest,
    }
    return {
        **identity,
        "snapshot_root": str(snapshot_root),
        "array_count": {"depth": len(depth_files), "confidence": len(conf_files)},
        "snapshot_sha256": hashlib.sha256(_canonical_bytes(identity)).hexdigest(),
    }


def write_snapshot_record(audit: Mapping, confirmation: Mapping, path: Path) -> dict:
    confirmation = _validate_da3_confirmation(confirmation, require_targets_absent=False)
    path = _absolute_path(path, "snapshot record path")
    if str(path) != confirmation["targets"]["snapshot_record_path"]:
        raise ValueError("snapshot record path does not match confirmation")
    record = {
        "schema_version": 1,
        "kind": "utility_da3_snapshot",
        "confirmation_sha256": hashlib.sha256(_canonical_bytes(confirmation)).hexdigest(),
        **json.loads(json.dumps(audit)),
        "gt_access": "NONE",
    }
    if _contains_gt(record):
        raise ValueError("snapshot record must not reference GT")
    return _atomic_record_write(record, path)
