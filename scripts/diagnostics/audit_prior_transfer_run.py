"""Publish one immutable GT-free Utility run qualification, never run training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reliability.g1_prior_transfer_confirmation import load_prior_transfer_confirmation, _canonical_bytes
from reliability.prior_transfer_assets import input_fingerprints, qualify_prior_transfer_run


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation", required=True)
    parser.add_argument("--confirmation-sha", required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    return parser


def run_audit(args):
    temporary, published = [], []
    try:
        confirmation = load_prior_transfer_confirmation(args.confirmation, args.confirmation_sha)
        row = confirmation["runs"][args.seed]
        target = Path(row["qualification_path"])
        sha_path = Path(str(target) + ".sha256")
        snapshot = json.loads(Path(confirmation["snapshot_record"]["path"]).read_text(encoding="utf-8"))
        source = json.loads(Path(confirmation["source_record"]["path"]).read_text(encoding="utf-8"))
        protected = [Path(row[key]) for key in ("run_dir", "view_dir", "launcher_dir")]
        protected += [Path(snapshot["snapshot_root"]), Path(source["source_root"]), Path(confirmation["repository"]["root"])]
        for path in (target, sha_path):
            if not path.is_absolute() or any(path.resolve().is_relative_to(root.resolve()) for root in protected):
                raise ValueError("qualification output must be outside immutable input trees")
            if path.exists():
                raise FileExistsError("qualification already exists; never replace completed seed")
        if not target.parent.is_dir():
            raise ValueError("qualification parent missing")
        report = qualify_prior_transfer_run(Path(row["run_dir"]), args.seed, confirmation)
        payload = _canonical_bytes(report)
        digest = hashlib.sha256(payload).hexdigest()
        token = uuid.uuid4().hex
        for path, content in ((target, payload), (sha_path, (digest + "\n").encode("ascii"))):
            tmp = path.parent / f".{path.name}.tmp-{token}"
            temporary.append(tmp)
            with tmp.open("xb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        load_prior_transfer_confirmation(args.confirmation, args.confirmation_sha)
        if report["input_fingerprints"] != input_fingerprints(confirmation, row):
            raise ValueError("immutable input changed before publication")
        # Hard-link exclusive publication: never replace a concurrently created result.
        for tmp, path in zip(temporary, (target, sha_path)):
            os.link(tmp, path)
            published.append(path)
        return 0, {"outcome": "QUALIFIED", "path": str(target), "sha256": digest,
                   "gt_access": "NONE", "training_started": False}
    except Exception as exc:
        for path in published:
            path.unlink()
        return 2, {"outcome": "NOT_QUALIFIED", "error": f"{type(exc).__name__}: {exc}",
                   "gt_access": "NONE", "training_started": False}
    finally:
        for path in temporary:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def main():
    code, result = run_audit(build_parser().parse_args())
    print(json.dumps(result, sort_keys=True, indent=2))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
