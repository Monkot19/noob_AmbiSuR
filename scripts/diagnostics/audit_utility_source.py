"""Audit the uploaded Utility source without DA3 or GT access."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import uuid


if __package__ in (None, ""):
    repository_root = str(Path(__file__).resolve().parents[2])
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)

from reliability.utility_snapshot import audit_utility_source, source_manifest


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_request(args):
    source_text = str(args.source_root)
    output_text = str(args.output)
    source = Path(source_text).expanduser()
    output = Path(output_text).expanduser()
    if not source.is_absolute() or not output.is_absolute():
        raise ValueError("source root and output must be absolute paths")
    source = source.resolve()
    output = output.resolve()
    if not source.is_dir() or not output.parent.is_dir():
        raise ValueError("source root or output parent is missing")
    if _within(output, source):
        raise ValueError("audit output must be outside the source tree")
    sha = str(args.expected_source_sha)
    if _SHA256.fullmatch(sha) is None:
        raise ValueError("expected source SHA256 is malformed")
    sha_path = output.with_suffix(output.suffix + ".sha256")
    if output.exists() or sha_path.exists():
        raise FileExistsError("audit output already exists")
    return source, output, sha_path, sha


def run_audit(args):
    temporary = []
    try:
        source, output, sha_path, expected_sha = _validate_request(args)
        before = source_manifest(audit_utility_source(source, int(args.expected_count)))
        if before["source_sha256"] != expected_sha:
            raise ValueError("source SHA256 mismatch")
        record = dict(before)
        record["audit_kind"] = "utility_source"
        record["gt_access"] = "NONE"
        payload = _canonical_bytes(record)
        token = uuid.uuid4().hex
        temporary_json = output.parent / f".{output.name}.tmp-{token}"
        temporary_sha = output.parent / f".{sha_path.name}.tmp-{token}"
        temporary.extend((temporary_json, temporary_sha))
        temporary_json.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        temporary_sha.write_text(digest + "\n", encoding="ascii")
        after = source_manifest(audit_utility_source(source, int(args.expected_count)))
        if after != before:
            raise ValueError("Utility source mutated during audit")
        os.replace(temporary_json, output)
        os.replace(temporary_sha, sha_path)
        return 0, {
            "output": str(output),
            "output_sha256": digest,
            "source_sha256": expected_sha,
            "gt_access": "NONE",
        }
    except Exception as exc:
        for path in temporary:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        return 2, {"error": f"{type(exc).__name__}: {exc}"}


def build_parser():
    parser = argparse.ArgumentParser(description="Audit one Utility Room source upload")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--expected-count", required=True, type=int)
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main():
    code, report = run_audit(build_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
