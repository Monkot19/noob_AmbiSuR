"""Frozen three-seed Utility probe; never trains, renders, or repairs assets."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reliability.g1_prior_transfer import build_transfer_report, evaluate_transfer_iteration, transfer_exit_code
from reliability.g1_prior_transfer_confirmation import _validate_record, validate_completed_run_binding
from reliability.prior_transfer_assets import _file_inventory
from reliability.utility_gt_firewall import (
    _absolute, _canonical_bytes, _file_identity, _identity, _read_verified,
    _reject_aliases, _verify_mesh_access, authorize_first_gt_access,
    audit_utility_mesh, UtilityMeshAdmission,
)
from scripts.diagnostics.evaluate_d0_g1 import build_manifest, assert_inputs_unchanged
from scripts.diagnostics.probe_g1_prior_complementarity import (
    ProductionDependencies as FrozenDependencies, _write_csv,
)

ARTIFACTS = ("inputs.json", "mesh_admission.json", "report.json", "seed_folds.csv", "risk_bins.csv", "bootstrap.csv")


class ProductionDependencies:
    git_identity = staticmethod(FrozenDependencies.git_identity)
    load_iteration = staticmethod(FrozenDependencies.load_iteration)
    load_mesh = staticmethod(FrozenDependencies.load_mesh)
    distances = staticmethod(FrozenDependencies.distances)
    admit_mesh = staticmethod(audit_utility_mesh)
    evaluate_iteration = staticmethod(evaluate_transfer_iteration)


def _fingerprint(path, protected):
    """Streaming integrity adapter; reject protected inode before any byte read."""
    path = Path(path)
    _reject_aliases([{"path": str(path)}], protected)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        # Windows path-stat ctime is creation time, while descriptor fstat may
        # report change time. Compare their common identity fields at open;
        # path-stat generations still include ctime across the whole interval.
        if _file_identity(opened)[:4] != _file_identity(before)[:4]:
            raise ValueError("input changed at open")
        for target in protected:
            if target.exists():
                info = target.stat()
                if (info.st_dev, info.st_ino) == (opened.st_dev, opened.st_ino):
                    raise ValueError("input descriptor aliases protected GT/target")
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if _file_identity(path.stat()) != _file_identity(before):
        raise ValueError("input changed while hashing")
    return {"resolved_path": str(path.resolve()), "bytes": before.st_size,
            "sha256": digest.hexdigest()}, _file_identity(before)


def _read_json(identity, protected, *, detached=False, serialize=_canonical_bytes):
    payload = _read_verified(identity, protected=protected, detached=detached)
    record = json.loads(payload)
    if payload != serialize(record):
        raise ValueError("input record is not canonical")
    return record


def _trees(prior, source, snapshot):
    roots = [Path(source["source_root"]), Path(snapshot["snapshot_root"])]
    paths = {Path(prior[name]["path"]) for name in ("source_record", "snapshot_record")}
    for row in prior["runs"]:
        roots.extend(Path(row[name]) for name in ("run_dir", "view_dir", "launcher_dir"))
        paths.add(Path(row["state_file"]))
    for root in roots:
        if not root.is_dir():
            raise ValueError("qualified input tree missing")
        paths.update(root / name for name in _file_inventory(root))
    return paths, roots


def _admit_request(args, dependencies):
    repository = _absolute(args.repository)
    if (re.fullmatch(r"[0-9a-f]{40}", args.expected_commit) is None
            or dependencies.git_identity(repository) != {"commit": args.expected_commit, "clean": True}):
        raise ValueError("exact clean evaluator commit required")
    # The explicit GT/output paths let even a disguised confirmation be rejected
    # before its first byte read. Later require exact binding to its payload.
    target = _absolute(args.output_root) / args.diagnostic_id
    # The frozen training recovery preregisters <confirmation_id>.probe.
    # Admit that single basename suffix, not arbitrary dotted/traversal paths;
    # the full target must still equal the SHA-bound confirmation below.
    if re.fullmatch(r"[A-Za-z0-9_-]+(?:[.]probe)?", args.diagnostic_id) is None:
        raise ValueError("unsafe diagnostic ID")
    gt = _absolute(args.gt_mesh)
    protected = [gt, target]
    prior_identity = _identity({"path": args.confirmation, "sha256": args.confirmation_sha})
    prior = _read_json(prior_identity, protected)
    prior = _validate_record(prior, require_targets_absent=False, verify_record_files=False)
    amended = bool(getattr(args, "access_amendment", None))
    if not amended and prior["repository"] != {"root": str(repository), "commit": args.expected_commit, "clean": True}:
        raise ValueError("confirmation/evaluator repository binding mismatch")
    targets = prior["probe_targets"]
    if target != Path(targets["output_dir"]) or gt != Path(prior["gt_mesh"]["path"]):
        raise ValueError("exact preregistered output/GT target required")
    staging = Path(targets["staging_dir"])
    log = Path(targets["access_log_path"])
    for path in (target, staging, log):
        if os.path.lexists(path):
            raise FileExistsError("one-shot probe target already exists")
    protected.extend((staging, log))
    from scripts.diagnostics.audit_utility_source import _canonical_bytes as source_bytes
    from reliability.utility_snapshot import _canonical_bytes as snapshot_bytes
    source = _read_json(prior["source_record"], protected, serialize=source_bytes)
    snapshot_identity = {key: prior["snapshot_record"][key] for key in ("path", "sha256")}
    snapshot = _read_json(snapshot_identity, protected, serialize=snapshot_bytes)
    if (_absolute(args.source_root) != _absolute(source["source_root"])
            or snapshot["snapshot_sha256"] != prior["snapshot_record"]["snapshot_sha256"]
            or snapshot["source_sha256"] != source["source_sha256"] or snapshot["gt_access"] != "NONE"):
        raise ValueError("source/snapshot binding mismatch")
    paths, roots = _trees(prior, source, snapshot)
    roots.append(repository)
    # Outputs may not contain, equal, or be contained in any immutable input.
    immutable = [*roots, *paths, Path(prior_identity["path"]), gt]
    for output in (target, staging, log):
        for item in immutable:
            item = item.resolve()
            if output == item or output in item.parents or item in output.parents:
                raise ValueError("probe output overlaps immutable input")
        if not output.parent.is_dir():
            raise ValueError("probe target parent must already exist")
    handles = [_identity({"path": pair[0], "sha256": pair[1]}) for pair in args.qualification_record]
    if len(handles) != 3 or [h["path"] for h in handles] != [r["qualification_path"] for r in prior["runs"]]:
        raise ValueError("exact three qualification handles required in seed order")
    qualifications = [_read_json(handle, protected, detached=True) for handle in handles]
    bindings = []
    expected_fingerprints = {}
    for qualification in qualifications:
        if (qualification.get("schema_version") != 1
                or qualification.get("kind") != "prior_transfer_run_qualification"
                or qualification.get("outcome") != "QUALIFIED"
                or qualification.get("gt_access") != "NONE"
                or qualification.get("training_started") is not False
                or qualification.get("c1_authorized") is not False
                or qualification.get("confirmation_sha256") != args.confirmation_sha
                or qualification.get("source_sha256") != source["source_sha256"]):
            raise ValueError("qualification is not bound to the original confirmation/source")
        bindings.append(qualification["run_binding"])
        for path, fingerprint in qualification["input_fingerprints"].items():
            if path in expected_fingerprints and expected_fingerprints[path] != fingerprint:
                raise ValueError("seeds disagree on shared immutable input")
            expected_fingerprints[path] = fingerprint
    # Guarded source/snapshot reads above replace this validator's general I/O.
    validate_completed_run_binding(prior, bindings, verify_record_files=False)
    if {str(path) for path in paths} != set(expected_fingerprints):
        raise ValueError("qualified input file inventory changed")
    for row in prior["runs"]:
        for iteration in (3000, 7000):
            required = [Path(row["run_dir"]) / f"chkpnt{iteration}.pth",
                        Path(row["run_dir"]) / "d0_evidence" / f"iteration_{iteration:06d}.npz"]
            if any(str(path) not in expected_fingerprints for path in required):
                raise ValueError("qualification omits required checkpoint/snapshot")
    if amended:
        if getattr(args, "geometry_release", None) or getattr(args, "geometry_release_sha", None):
            raise ValueError("access amendment and geometry release are mutually exclusive")
        from reliability.prior_transfer_access_amendment import load_amendment
        geometry_identity = _identity({"path": args.access_amendment, "sha256": args.access_amendment_sha})
        amendment = load_amendment(geometry_identity, protected=protected)
        if (amendment["prior_confirmation"] != prior_identity
                or amendment["qualification_records"] != handles
                or amendment["repository"] != {"root": str(repository), "commit": args.expected_commit, "clean": True}):
            raise ValueError("access amendment/evaluator binding mismatch")
        records = [prior_identity, geometry_identity, *handles, amendment["specification"], amendment["approval"]]
        if "pre_gt_recovery_from" in amendment:
            from reliability.prior_transfer_access_amendment import _json
            previous = amendment["pre_gt_recovery_from"]
            preserved = _json(previous, protected, detached=True)
            records.extend((previous, preserved["approval"]))
            paths.update(Path(h["path"] + ".sha256") for h in (previous, preserved["approval"]))
        paths.add(Path(amendment["approval"]["path"] + ".sha256"))
        paths.update(repository / name for name in amendment["core_sha256"] if "/" in name)
        paths.add(repository / "reliability/utility_gt_firewall.py")
    else:
        geometry_identity = _identity({"path": args.geometry_release, "sha256": args.geometry_release_sha})
        geometry = _read_json(geometry_identity, protected, detached=True)
        from reliability.utility_gt_firewall import validate_geometry_release
        validate_geometry_release(geometry)
        records = [prior_identity, geometry_identity, *handles, *geometry["specifications"].values(),
                   *geometry["evidence"], geometry["approval"]]
    for identity in records:
        _reject_aliases([identity], protected)
        paths.add(Path(identity["path"]))
    for identity in (geometry_identity, *handles):
        paths.add(Path(identity["path"] + ".sha256"))
    before, generations = {}, {}
    for path in sorted(paths):
        fingerprint, generation = _fingerprint(path, protected)
        if str(path) in expected_fingerprints and fingerprint != expected_fingerprints[str(path)]:
            raise ValueError("qualified input changed before evaluation")
        before[str(path)], generations[str(path)] = fingerprint, generation
    return prior, prior_identity, geometry_identity, source, snapshot, paths, roots[:-1], before, generations


def _rename_exclusive(staging, target):
    if sys.platform == "linux":
        # Atomic directory publication without overwriting an empty raced target.
        libc = ctypes.CDLL(None, use_errno=True)
        rename = getattr(libc, "renameat2", None)
        if rename is None:
            raise RuntimeError("exclusive atomic directory rename unavailable")
        if rename(-100, os.fsencode(staging), -100, os.fsencode(target), 1) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), str(target))
    elif os.name == "nt":
        os.rename(staging, target)  # Windows refuses an existing target.
    else:
        raise RuntimeError("exclusive atomic directory publication unsupported")


def _csv_artifacts(root, report):
    folds, risks, bootstraps = [], [], []
    for seed in report["seeds"]:
        for row in seed["iterations"]:
            prefix = {"training_seed": seed["training_seed"], "iteration": row["iteration"]}
            folds.extend({**prefix, **fold} for fold in row["crossfit"].get("folds", []))
            risks.extend({**prefix, **risk} for risk in row["direction"].get("risk_bins", []))
            bootstraps.extend({**prefix, "replicate": b, "auroc_gain": value}
                              for b, value in enumerate(row["bootstrap"]["auroc_gain_replicates"]))
    if report["macro_7000"] is not None:
        bootstraps.extend({"training_seed": "macro", "iteration": 7000, "replicate": b, "auroc_gain": value}
                          for b, value in enumerate(report["macro_7000"]["bootstrap"]["auroc_gain_replicates"]))
    fold_fields = ["training_seed", "iteration", "fold", "training_count", "validation_count",
        "training_positive_count", "training_negative_count", "validation_positive_count", "validation_negative_count",
        "baseline_auroc", "augmented_auroc", "auroc_gain", "candidate_relative_residual", "positive_weight", "negative_weight",
        "baseline_a_mean", "baseline_one_minus_s_mean", "baseline_a_scale", "baseline_one_minus_s_scale",
        "candidate_mean", "candidate_scale", "baseline_iterations", "augmented_iterations", "baseline_converged", "augmented_converged"]
    _write_csv(root / "seed_folds.csv", folds, fold_fields)
    _write_csv(root / "risk_bins.csv", risks, ["training_seed", "iteration", "bin", "count", "risk_min", "risk_max", "mean_distance_m", "high_error_rate"])
    _write_csv(root / "bootstrap.csv", bootstraps, ["training_seed", "iteration", "replicate", "auroc_gain"])


def run_evaluator(args, dependencies=None):
    dependencies = dependencies or ProductionDependencies()
    (prior, prior_identity, geometry_identity, source, snapshot, paths, roots,
     before, generations) = _admit_request(args, dependencies)
    protected = [Path(args.gt_mesh), *(Path(path) for path in prior["probe_targets"].values())]
    authorizer = authorize_first_gt_access
    if getattr(args, "access_amendment", None):
        from reliability.prior_transfer_access_amendment import authorize_prior_transfer_gt_access
        authorizer = authorize_prior_transfer_gt_access
    token = authorizer(prior_identity, geometry_identity, protected=protected,
                                      access_log_path=Path(prior["probe_targets"]["access_log_path"]))
    _verify_mesh_access(prior, token)
    # From this point a failed attempt remains logged. No retry deletes the log.
    # Never reset the GT baseline after admission: the very same bytes and
    # generation must span admission, evaluation, and final publication.
    gt_before = None
    print("FIRST_GT_ACCESS_LOGGED; checking frozen mesh admission", flush=True)
    try:
        gt_before = _fingerprint(Path(args.gt_mesh), [])
    except OSError as exc:
        admission = UtilityMeshAdmission("INCONCLUSIVE", (f"{type(exc).__name__}: {exc}",), {})
    else:
        expected_gt = {"resolved_path": prior["gt_mesh"]["path"],
                       "bytes": prior["gt_mesh"]["bytes"], "sha256": prior["gt_mesh"]["sha256"]}
        if gt_before[0] != expected_gt:
            admission = UtilityMeshAdmission("INCONCLUSIVE", ("preregistered GT mesh identity mismatch",), {})
        else:
            admission = dependencies.admit_mesh(Path(args.gt_mesh), Path(args.source_root), prior, access_token=token)
        assert_inputs_unchanged(gt_before, _fingerprint(Path(args.gt_mesh), []))
    provenance = {"diagnostic_commit": args.expected_commit, "confirmation_id": prior["confirmation_id"],
                  "confirmation_sha256": args.confirmation_sha,
                  "training_commit": prior["repository"]["commit"],
                  ("access_amendment" if getattr(args, "access_amendment", None) else "geometry_release"): geometry_identity,
                  "first_access_log": {key: token[key] for key in ("path", "sha256")},
                  "snapshot_sha256": snapshot["snapshot_sha256"], "source_sha256": source["source_sha256"],
                  "gt_mesh": prior["gt_mesh"], "qualification_records": args.qualification_record}
    results, reasons = {}, list(admission.reasons)
    print(f"MESH_ADMISSION={admission.outcome}", flush=True)
    if admission.outcome != "ADMITTED" and not reasons:
        reasons = ["mesh admission did not succeed"]
    if not reasons:
        try:
            mesh = dependencies.load_mesh(Path(args.gt_mesh))
            for row in prior["runs"]:
                results[row["seed"]] = {}
                for iteration in (3000, 7000):
                    print(f"EVALUATING seed={row['seed']} iteration={iteration}", flush=True)
                    joined = dependencies.load_iteration(Path(row["run_dir"]), iteration, expected_evidence_version=4)
                    distances = dependencies.distances(joined.centers, mesh)
                    results[row["seed"]][iteration] = dependencies.evaluate_iteration(
                        joined, distances, training_seed=row["seed"], iteration=iteration)
                    print(f"COMPLETED seed={row['seed']} iteration={iteration}", flush=True)
        except (ValueError, OSError, RuntimeError, ImportError, KeyError) as exc:
            reasons = [f"{type(exc).__name__}: {exc}"]
    try:
        report = build_transfer_report(results, provenance=provenance, inconclusive_reasons=reasons)
    except (ValueError, KeyError, TypeError) as exc:
        report = build_transfer_report({}, provenance=provenance,
                                      inconclusive_reasons=[f"{type(exc).__name__}: {exc}"])
    target, staging = (Path(prior["probe_targets"][key]) for key in ("output_dir", "staging_dir"))
    owned_staging = False
    try:
        staging.mkdir()
        owned_staging = True
        inputs = {"schema_version": 1, "diagnostic_only": True, "provenance": provenance,
                  "input_fingerprints": before, "gt_fingerprint": gt_before[0] if gt_before else None,
                  "protocol": prior["protocol"]}
        for name, record in (("inputs.json", inputs), ("mesh_admission.json", {
                "outcome": admission.outcome, "reasons": list(admission.reasons), "summary": admission.summary}),
                ("report.json", report)):
            (staging / name).write_bytes(_canonical_bytes(record))
        _csv_artifacts(staging, report)
        manifest = build_manifest(staging, ARTIFACTS)
        (staging / "manifest.json").write_bytes(_canonical_bytes(manifest))
        protected = [Path(args.gt_mesh), target, staging, Path(token["path"])]
        after, final_generations = {}, {}
        for path in sorted(paths):
            after[str(path)], final_generations[str(path)] = _fingerprint(path, protected)
        assert_inputs_unchanged(before, after)
        assert_inputs_unchanged(generations, final_generations)
        actual_paths, _ = _trees(prior, source, snapshot)
        if not actual_paths.issubset(paths):
            raise ValueError("immutable input inventory changed during evaluation")
        if gt_before is not None:
            assert_inputs_unchanged(gt_before, _fingerprint(Path(args.gt_mesh), []))
        _verify_mesh_access(prior, token)
        if dependencies.git_identity(Path(args.repository)) != {"commit": args.expected_commit, "clean": True}:
            raise ValueError("evaluator code identity changed")
        _rename_exclusive(staging, target)
        owned_staging = False
    finally:
        if owned_staging:
            shutil.rmtree(staging)
    return transfer_exit_code(report), {"output_dir": str(target), "report": report, "manifest": manifest}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("repository", "expected-commit", "confirmation", "confirmation-sha",
                 "source-root", "gt-mesh", "output-root", "diagnostic-id"):
        parser.add_argument("--" + flag, required=True)
    access = parser.add_mutually_exclusive_group(required=True)
    access.add_argument("--geometry-release")
    access.add_argument("--access-amendment")
    parser.add_argument("--geometry-release-sha")
    parser.add_argument("--access-amendment-sha")
    parser.add_argument("--qualification-record", nargs=2, action="append", required=True, metavar=("PATH", "SHA256"))
    return parser


def main(argv=None):
    try:
        code, publication = run_evaluator(build_parser().parse_args(argv))
    except Exception as exc:
        print(f"UTILITY_TRANSFER_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"diagnostic_only": True, "exit_code": code,
                      "outcome": publication["report"]["outcome"], "output_dir": publication["output_dir"]}, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
