"""Fail-closed publication helpers for the offline D0/G1 evaluator."""

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import uuid


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


def publish_atomically(output_root, confirmation_id, immutable_inputs, producer):
    from reliability.g1_visualization import required_artifacts

    output_root = _resolved(output_root)
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
        manifest = build_manifest(staging, required_artifacts())
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
