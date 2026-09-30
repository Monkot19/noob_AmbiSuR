"""Write the canonical Utility prior-transfer confirmation; never reads GT."""

import argparse
import json
from pathlib import Path
import sys


if __package__ in (None, ""):
    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)

from reliability.g1_prior_transfer_confirmation import (
    build_prior_transfer_confirmation,
    write_prior_transfer_confirmation,
)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    return parser


def run_create(args):
    try:
        request_path = Path(args.request).expanduser()
        output = Path(args.output).expanduser()
        if not request_path.is_absolute() or not output.is_absolute():
            raise ValueError("request and output paths must be absolute")
        request = json.loads(request_path.read_text(encoding="utf-8"))
        record = build_prior_transfer_confirmation(**request)
        published = write_prior_transfer_confirmation(record, output)
        return 0, {
            "output": str(published["path"]),
            "sha256_path": str(published["sha256_path"]),
            "sha256": published["sha256"],
        }
    except Exception as exc:
        return 2, {"error": f"{type(exc).__name__}: {exc}"}


def main():
    code, result = run_create(build_parser().parse_args())
    print(json.dumps(result, sort_keys=True, indent=2))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
