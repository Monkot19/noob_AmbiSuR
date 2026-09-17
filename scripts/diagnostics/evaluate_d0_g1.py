"""Fail-closed publication helpers for the offline D0/G1 evaluator."""

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile


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
    parser.parse_args(argv)
    parser.error("offline G1 evaluation orchestration is not implemented yet")


if __name__ == "__main__":
    main()
