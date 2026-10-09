"""Post-training, prior-transfer-only GT approval; never a geometry release.

The caller records the human approval. Hashes establish integrity, not a digital
signature. No pause, outcome string or command-line boolean alone grants access.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import uuid

from reliability.utility_gt_firewall import (
    _absolute, _canonical_bytes, _identity, _read_verified, _reject_aliases,
    _time, _validate_record, _read_bytes_guarded,
)

CORE_PATHS = (
    "reliability/g1_complementarity.py", "reliability/g1_prior_transfer.py",
    "reliability/g1_prior_transfer_confirmation.py", "reliability/g1_metrics.py",
    "reliability/offline_g1.py", "reliability/g1_visualization.py",
    "reliability/diagnostics.py", "reliability/utility_snapshot.py",
    "scripts/preprocess/read_write_model.py",
    "scripts/diagnostics/probe_g1_prior_complementarity.py",
)
POLICY = {
    "scope": "FROZEN_ONE_MINUS_R_P_UTILITY_TRANSFER_ONLY",
    "geometry_status": "PAUSED_SCOPE_BUDGET",
    "geometry_candidate_admitted": False,
    "utility_feedback_prohibited": True,
    "geometry_release_created": False,
    "training_authorized": False, "c1_authorized": False,
}
AUTHORIZATION = "User explicitly authorizes the minimal access amendment and one frozen Utility GT evaluation; no Geometry or training."


def git_blob(repository, commit, name):
    return subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=repository)


def git_identity(repository):
    def command(*args):
        return subprocess.check_output(["git", *args], cwd=repository, text=True).strip()
    return {"commit": command("rev-parse", "HEAD"),
            "clean": not command("status", "--porcelain", "--untracked-files=all")}


def _json(identity, protected, detached=False):
    payload = _read_verified(identity, protected=protected, detached=detached)
    record = json.loads(payload)
    if payload != _canonical_bytes(record):
        raise ValueError("access artifact must be canonical")
    return record


def approval_record(prior_identity, qualifications, repository, specification):
    return {"schema_version": 1, "kind": "utility_prior_transfer_execution_approval",
            "prior_confirmation": _identity(prior_identity),
            "qualification_records": [_identity(q) for q in qualifications],
            "repository": dict(repository), "specification": _identity(specification),
            "authorization": AUTHORIZATION, **POLICY}


def _core_hashes(prior, repository, protected):
    if (not isinstance(repository, dict) or set(repository) != {"root", "commit", "clean"}
            or repository["clean"] is not True
            or re.fullmatch(r"[0-9a-f]{40}", str(repository["commit"])) is None):
        raise ValueError("access amendment requires exact clean repository identity")
    root = _absolute(repository["root"])
    if repository != {"root": str(root), **git_identity(root)}:
        raise ValueError("exact clean evaluator commit required by access amendment")
    if root != _absolute(prior["repository"]["root"]):
        raise ValueError("access amendment belongs to another repository")
    hashes = {}
    for name in CORE_PATHS:
        original = git_blob(root, prior["repository"]["commit"], name)
        digest = hashlib.sha256(original).hexdigest()
        _read_verified({"path": str(root / name), "sha256": digest}, protected=protected)
        hashes[name] = digest
    # Only permission dispatch changes are allowed in the firewall. Bind the
    # actual mesh admission algorithms against the original Git object too.
    import ast
    name = "reliability/utility_gt_firewall.py"
    old = ast.parse(git_blob(root, prior["repository"]["commit"], name))
    _reject_aliases([{"path": str(root / name)}], protected)
    current = ast.parse(_read_bytes_guarded(root / name, protected))
    for function in ("select_admission_points", "utility_camera_rays", "audit_utility_mesh",
                     "_load_admission_mesh", "_admission_distances", "_admission_ray_depths",
                     "_admission_source", "_mesh_identity", "_source_identities", "_file_identity"):
        def body(tree):
            return ast.dump(next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function), include_attributes=False)
        if body(old) != body(current):
            raise ValueError("frozen mesh admission algorithm changed")
        hashes[function] = hashlib.sha256(body(old).encode()).hexdigest()
    return hashes


def build_amendment(prior_identity, qualifications, approval_identity, *, protected=()):
    prior_identity = _identity(prior_identity)
    prior = _json(prior_identity, protected)
    prior = _validate_record(prior, require_targets_absent=False, verify_record_files=False)
    protected = [*protected, Path(prior["gt_mesh"]["path"]),
                 *(Path(p) for p in prior["probe_targets"].values())]
    if _json(prior_identity, protected) != prior:
        raise ValueError("confirmation changed during access reload")
    approval = _json(approval_identity, protected, detached=True)
    repository = approval.get("repository", {})
    specification = approval.get("specification", {})
    qualifications = [_identity(q) for q in qualifications]
    if approval != approval_record(prior_identity, qualifications, repository, specification):
        raise ValueError("explicit reviewed prior-only execution approval required")
    _read_verified(specification, protected=protected)
    if [q["path"] for q in qualifications] != [r["qualification_path"] for r in prior["runs"]]:
        raise ValueError("access amendment qualification inventory mismatch")
    for q, row in zip(qualifications, prior["runs"]):
        record = _json(q, protected, detached=True)
        if (record.get("outcome") != "QUALIFIED" or record.get("gt_access") != "NONE"
                or record.get("confirmation_sha256") != prior_identity["sha256"]
                or record.get("run_binding", {}).get("seed") != row["seed"]):
            raise ValueError("access amendment requires three original qualified runs")
    for name in ("source_record", "snapshot_record"):
        _read_verified({k: prior[name][k] for k in ("path", "sha256")}, protected=protected)
    return {"schema_version": 1, "kind": "utility_prior_transfer_access_amendment",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "prior_confirmation": prior_identity, "qualification_records": qualifications,
            "approval": _identity(approval_identity), "specification": specification,
            "repository": repository, "training_commit": prior["repository"]["commit"],
            "core_sha256": _core_hashes(prior, repository, protected),
            **{k: prior[k] for k in ("source_record", "snapshot_record", "gt_mesh", "protocol", "mesh_admission", "probe_targets")},
            **POLICY}


def load_amendment(identity, *, protected=()):
    record = _json(identity, protected, detached=True)
    expected = build_amendment(record.get("prior_confirmation"), record.get("qualification_records", []),
                               record.get("approval"), protected=protected)
    when = _time(record.get("created_utc"))
    prior = _json(expected["prior_confirmation"], protected)
    if not _time(prior["created_utc"]) <= when <= datetime.now(timezone.utc):
        raise ValueError("access amendment chronology mismatch")
    expected["created_utc"] = record["created_utc"]
    if record != expected:
        raise ValueError("access amendment changed frozen contract or authority")
    return record


def publish_record(path, record):
    """Exclusive immutable JSON and detached digest; never replaces user files."""
    path = Path(path)
    sidecar = Path(str(path) + ".sha256")
    if os.path.lexists(path) or os.path.lexists(sidecar):
        raise FileExistsError("access record already exists")
    payload = _canonical_bytes(record)
    digest = hashlib.sha256(payload).hexdigest()
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    with sidecar.open("xb") as stream:
        stream.write((digest + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": str(path), "sha256": digest}


def authorize_prior_transfer_gt_access(prior_identity, amendment_identity, *, access_log_path, protected=()):
    amendment = load_amendment(amendment_identity, protected=protected)
    if amendment["prior_confirmation"] != _identity(prior_identity):
        raise ValueError("access amendment binds another confirmation")
    log = _absolute(access_log_path)
    if str(log) != amendment["probe_targets"]["access_log_path"]:
        raise ValueError("first-access target mismatch")
    if any(os.path.lexists(p) for p in amendment["probe_targets"].values()):
        raise FileExistsError("one-shot probe target already exists")
    record = {"schema_version": 1, "kind": "utility_prior_transfer_first_gt_access",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "prior_confirmation": _identity(prior_identity), "access_amendment": _identity(amendment_identity),
              "gt_mesh": amendment["gt_mesh"], "access_log_path": str(log),
              "mesh_parse_preceded_by_this_record": True, **POLICY}
    # Same exclusive hardlink publication as the original authorizer.
    temporary = log.parent / ("." + log.name + ".tmp-" + uuid.uuid4().hex)
    payload = _canonical_bytes(record)
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        load_amendment(amendment_identity, protected=protected)
        if any(os.path.lexists(p) for p in amendment["probe_targets"].values()):
            raise FileExistsError("one-shot target appeared before access")
        os.link(temporary, log)
        for name in ("output_dir", "staging_dir"):
            if os.path.lexists(amendment["probe_targets"][name]):
                raise ValueError("probe target appeared during first-access publication")
    finally:
        temporary.unlink(missing_ok=True)
    token = {"path": str(log), "sha256": hashlib.sha256(payload).hexdigest(), "record": record}
    verify_prior_transfer_mesh_access(_json(prior_identity, protected), token)
    return token


def verify_prior_transfer_mesh_access(prior, token):
    protected = [Path(prior["gt_mesh"]["path"]), *(Path(p) for p in prior["probe_targets"].values())]
    identity = _identity({k: token[k] for k in ("path", "sha256")})
    if identity["path"] != prior["probe_targets"]["access_log_path"]:
        raise ValueError("first-access token path mismatch")
    # The log itself is the object being verified, not a protected reference.
    record = _json(identity, [protected[0]])
    amendment = load_amendment(record.get("access_amendment"), protected=protected)
    expected = {"schema_version": 1, "kind": "utility_prior_transfer_first_gt_access",
                "created_utc": record.get("created_utc"),
                "prior_confirmation": amendment["prior_confirmation"],
                "access_amendment": record["access_amendment"], "gt_mesh": amendment["gt_mesh"],
                "access_log_path": identity["path"], "mesh_parse_preceded_by_this_record": True, **POLICY}
    when = _time(record["created_utc"])
    if (record != expected or record != token["record"]
            or _json(amendment["prior_confirmation"], protected) != prior
            or not _time(amendment["created_utc"]) <= when <= datetime.now(timezone.utc)):
        raise ValueError("prior-transfer access token binding mismatch")
    return prior
