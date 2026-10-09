"""One-shot operational wrapper: original assets + approved prior-only access.

No statistical implementation lives here. It invokes the existing evaluator.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reliability.prior_transfer_access_amendment import (
    _json, approval_record, build_amendment, git_identity, publish_record,
    validate_pre_gt_recovery,
)
from reliability.utility_gt_firewall import _identity, _validate_record
from scripts.diagnostics.audit_utility_source import _canonical_bytes as source_bytes
from scripts.diagnostics.evaluate_g1_prior_transfer import _fingerprint, run_evaluator


def prepare_and_run(args):
    started = time.monotonic()
    repository = Path(args.repository).resolve()
    if git_identity(repository) != {"commit": args.expected_commit, "clean": True}:
        raise ValueError("exact clean evaluation commit required")
    prior_handle = _identity({"path": args.confirmation, "sha256": args.confirmation_sha})
    protected = [Path(args.gt_mesh).resolve()]
    prior = _validate_record(_json(prior_handle, protected), require_targets_absent=False, verify_record_files=False)
    if prior["gt_mesh"]["path"] != str(protected[0]):
        raise ValueError("explicit GT path differs from frozen identity")
    protected.extend(Path(p) for p in prior["probe_targets"].values())
    if any(os.path.lexists(path) for path in protected[1:]):
        raise FileExistsError("one-shot probe output/staging/access log already exists")
    handles = [_identity({"path": pair[0], "sha256": pair[1]}) for pair in args.qualification_record]
    specification = repository / "docs/superpowers/specs/2026-10-09-utility-prior-transfer-gt-decoupling-amendment.md"
    spec_handle = {"path": str(specification), "sha256": _fingerprint(specification, protected)[0]["sha256"]}
    current_repository = {"root": str(repository), "commit": args.expected_commit, "clean": True}
    recovery_pair = getattr(args, "pre_gt_recovery_from", None)
    failure_pair = getattr(args, "pre_gt_failure_receipt", None)
    failure = _identity({"path": failure_pair[0], "sha256": failure_pair[1]}) if failure_pair else None
    if failure is not None and not recovery_pair:
        raise ValueError("failure receipt requires explicit preserved recovery amendment")
    recovery = None
    suffix = ""
    if recovery_pair:
        recovery = validate_pre_gt_recovery(
            {"path": recovery_pair[0], "sha256": recovery_pair[1]}, prior_handle,
            handles, current_repository, spec_handle, protected=protected, failure_receipt=failure)
        suffix = ".id-recovery1" if failure else ".format-recovery1"
    directory = Path(prior_handle["path"]).parent
    approval_path = directory / (prior["confirmation_id"] + suffix + ".prior-only-approval.json")
    amendment_path = directory / (prior["confirmation_id"] + suffix + ".access-amendment.json")
    receipt_path = directory / (prior["confirmation_id"] + suffix + ".evaluation-receipt.json")
    for path in (approval_path, amendment_path, receipt_path):
        if os.path.lexists(path) or os.path.lexists(str(path) + ".sha256"):
            raise FileExistsError("prior-only execution record already exists; do not retry")
    approval = approval_record(prior_handle, handles,
        current_repository, spec_handle, recovery_from=recovery, failure_receipt=failure)
    approval_handle = publish_record(approval_path, approval)
    amendment = build_amendment(prior_handle, handles, approval_handle, protected=protected)
    # Only the original target parents are created; run/view/state stay read-only.
    for path in protected[1:]:
        path.parent.mkdir(parents=True, exist_ok=True)
    amendment_handle = publish_record(amendment_path, amendment)
    print("PRIOR_ONLY_AMENDMENT=PASS", json.dumps(amendment_handle), flush=True)
    print("REVALIDATING_ALL_THREE_QUALIFIED_ASSETS", flush=True)
    request = SimpleNamespace(repository=str(repository), expected_commit=args.expected_commit,
        confirmation=prior_handle["path"], confirmation_sha=prior_handle["sha256"],
        qualification_record=args.qualification_record, geometry_release=None, geometry_release_sha=None,
        access_amendment=amendment_handle["path"], access_amendment_sha=amendment_handle["sha256"],
        source_root=_json(prior["source_record"], protected, serialize=source_bytes)["source_root"],
        gt_mesh=prior["gt_mesh"]["path"], output_root=str(Path(prior["probe_targets"]["output_dir"]).parent),
        diagnostic_id=Path(prior["probe_targets"]["output_dir"]).name)
    try:
        code, publication = run_evaluator(request)
    except Exception as exc:
        publish_record(receipt_path, {"kind": "prior_transfer_operational_failure",
            "error": f"{type(exc).__name__}: {exc}", "access_amendment": amendment_handle,
            "wall_seconds": round(time.monotonic() - started, 3), "automatic_retry": False})
        raise
    receipt = {"kind": "prior_transfer_evaluation_completion", "exit_code": code,
        "outcome": publication["report"]["outcome"], "output_dir": publication["output_dir"],
        "manifest_sha256": hashlib.sha256((Path(publication["output_dir"]) / "manifest.json").read_bytes()).hexdigest(),
        "access_amendment": amendment_handle, "evaluation_commit": args.expected_commit,
        "training_commit": prior["repository"]["commit"], "end_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": round(time.monotonic() - started, 3), "training_started": False,
        "geometry_experiment_started": False, "c1_started": False}
    publish_record(receipt_path, receipt)
    print(json.dumps(receipt, indent=2), flush=True)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("repository", "expected-commit", "confirmation", "confirmation-sha", "gt-mesh"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--qualification-record", nargs=2, action="append", required=True)
    parser.add_argument("--pre-gt-recovery-from", nargs=2, metavar=("PATH", "SHA256"),
                        help="explicit preserved format-failure amendment; never an automatic retry")
    parser.add_argument("--pre-gt-failure-receipt", nargs=2, metavar=("PATH", "SHA256"),
                        help="exact unsafe-ID operational receipt from the preserved format recovery")
    parser.add_argument("--execute-approved-prior-transfer", action="store_true", required=True)
    try:
        return prepare_and_run(parser.parse_args(argv))
    except Exception as exc:
        print(f"STOP: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
