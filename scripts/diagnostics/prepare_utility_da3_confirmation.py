"""Publish a reviewed pre-DA3 confirmation without invoking DA3."""

import argparse
import json
from pathlib import Path
import sys


if __package__ in (None, ""):
    repository_root = str(Path(__file__).resolve().parents[2])
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)

from reliability.utility_snapshot import write_da3_confirmation


def run_prepare(args):
    try:
        request = Path(args.request).expanduser()
        output = Path(args.output).expanduser()
        if not request.is_absolute() or not output.is_absolute():
            raise ValueError("request and output must be absolute paths")
        record = json.loads(request.read_text(encoding="utf-8"))
        published = write_da3_confirmation(record, output)
        return 0, {key: str(value) for key, value in published.items()}
    except Exception as exc:
        return 2, {"error": f"{type(exc).__name__}: {exc}"}


def build_parser():
    parser = argparse.ArgumentParser(description="Publish one frozen Utility DA3 preprocessing confirmation")
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main():
    code, report = run_prepare(build_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
