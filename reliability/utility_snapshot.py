"""Fail-closed Utility Room source and preprocessing snapshot contracts."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

from PIL import Image, UnidentifiedImageError


_SOURCE_SCHEMA_VERSION = 1
_SUPPORTED_CAMERA_MODELS = {"PINHOLE": 4, "SIMPLE_PINHOLE": 3}
_DERIVED_NAMES = (
    "estimated_depths",
    "estimated_confs",
    "sparse_da3",
    "sparse_da3_aligned",
)


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
