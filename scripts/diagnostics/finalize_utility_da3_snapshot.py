"""Audit completed DA3 bytes and publish an immutable snapshot record."""

import argparse
import json
from pathlib import Path
import sys


if __package__ in (None, ""):
    repository_root = str(Path(__file__).resolve().parents[2])
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)

from reliability.utility_snapshot import (
    audit_da3_snapshot,
    load_da3_confirmation,
    write_snapshot_record,
)


def run_finalizer(args):
    try:
        confirmation = load_da3_confirmation(
            Path(args.confirmation),
            str(args.expected_confirmation_sha),
            verify_source=True,
        )
        output = Path(args.output).expanduser()
        if not output.is_absolute():
            raise ValueError("output must be absolute")
        audit = audit_da3_snapshot(Path(args.snapshot_root), confirmation)
        published = write_snapshot_record(audit, confirmation, output)
        result = {key: str(value) for key, value in published.items()}
        result["output"] = result["path"]
        return 0, result
    except Exception as exc:
        return 2, {"error": f"{type(exc).__name__}: {exc}"}


def build_parser():
    parser = argparse.ArgumentParser(description="Finalize one completed Utility DA3 snapshot")
    parser.add_argument("--confirmation", required=True)
    parser.add_argument("--expected-confirmation-sha", required=True)
    parser.add_argument("--snapshot-root", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main():
    code, report = run_finalizer(build_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
