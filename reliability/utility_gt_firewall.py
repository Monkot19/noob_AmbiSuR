"""Immutable information-firewall records; no mesh parsing or experiment launch.

This is a prerequisite, not human execution approval. A caller must separately
verify completed runs and obtain GT-probe approval (Task 8). Inputs to
authorize_first_gt_access are {path, sha256} identities, NOT trusted payloads.
There is deliberately no geometry release/approval generator here.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import uuid
from collections.abc import Mapping

from reliability.g1_prior_transfer_confirmation import (
    _canonical_bytes, _validate_record,
)


_SHA = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_CONTRACT_FIELDS = {
    "action", "cluster_unit", "affected_parameters", "outcome", "horizon",
    "formula", "constants", "validity", "state", "topology_lineage",
    "thresholds", "metrics", "coverage", "compute_budget", "stop_rules",
}
_RELEASE_FIELDS = {
    "schema_version", "kind", "release_id", "created_utc", "repository", "stage",
    "outcome", "specifications", "evidence", "frozen_contract", "approval",
    "candidate_admitted_to_utility", "utility_cannot_reopen", "utility_gt_access",
}
_APPROVED_FIELDS = (
    "release_id", "stage", "outcome", "repository", "specifications", "evidence",
    "candidate_admitted_to_utility", "utility_cannot_reopen", "utility_gt_access",
)


def _absolute(value):
    if not isinstance(value, (str, Path)) or not Path(value).is_absolute():
        raise ValueError("firewall paths must be absolute")
    return Path(value).resolve()


def _identity(value):
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError("expected an exact disk identity handle {path, sha256}")
    path = _absolute(value["path"])
    if _SHA.fullmatch(str(value["sha256"])) is None:
        raise ValueError("invalid identity SHA256")
    return {"path": str(path), "sha256": value["sha256"]}


def _time(value):
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.tzinfo is None or result.utcoffset().total_seconds() != 0:
            raise ValueError("timestamp must be UTC")
        return result
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("invalid UTC chronology") from exc


def _read_bytes_guarded(path, protected):
    _reject_aliases([{"path": str(path)}], protected)
    with path.open("rb") as stream:
        # Check the opened object too: a path may be replaced after its initial
        # metadata check. No bytes are read until the descriptor is admitted.
        opened = os.fstat(stream.fileno())
        for target in protected:
            try:
                info = target.stat()
            except FileNotFoundError:
                continue
            if (opened.st_dev, opened.st_ino) == (info.st_dev, info.st_ino):
                raise ValueError("opened firewall artifact aliases protected GT/target")
        return stream.read()


def _read_verified(identity, *, detached=False, protected=()):
    identity = _identity(identity)
    path = Path(identity["path"])
    payload = _read_bytes_guarded(path, protected)
    if hashlib.sha256(payload).hexdigest() != identity["sha256"]:
        raise ValueError("firewall artifact SHA256 mismatch")
    if detached:
        # Only the canonical detached digest format is admitted (not shell input).
        accepted = [(identity["sha256"] + ending).encode("ascii") for ending in ("\n", "\r\n")]
        if _read_bytes_guarded(Path(str(path) + ".sha256"), protected) not in accepted:
            raise ValueError("detached geometry release SHA256 mismatch")
    return payload


def _nonempty(value):
    """Reject empty placeholders; scientific content is bound to reviewed bytes."""
    if value is None or value == "" or value == {} or value == []:
        return False
    if isinstance(value, Mapping):
        return all(isinstance(key, str) and key and _nonempty(item) for key, item in value.items())
    if isinstance(value, list):
        return all(_nonempty(item) for item in value)
    return True


def _release_shape(record):
    if not isinstance(record, Mapping) or set(record) != _RELEASE_FIELDS:
        raise ValueError("geometry release fields mismatch")
    if type(record["schema_version"]) is not int or record["schema_version"] != 1 or record["kind"] != "geometry_release":
        raise ValueError("geometry release schema mismatch")
    if not isinstance(record["release_id"], str) or re.fullmatch(r"[A-Za-z0-9_-]+", record["release_id"]) is None:
        raise ValueError("unsafe geometry release ID")
    _time(record["created_utc"])
    repository = record["repository"]
    if not isinstance(repository, Mapping) or set(repository) != {"root", "commit", "clean"}:
        raise ValueError("geometry repository identity mismatch")
    _absolute(repository["root"])
    if _COMMIT.fullmatch(str(repository["commit"])) is None or repository["clean"] is not True:
        raise ValueError("geometry release requires an exact clean commit")
    if record["stage"] not in ("G-A", "G-B", "G-C"):
        raise ValueError("unknown geometry stage")
    outcome = record["outcome"]
    if outcome not in ("STAGE_G_C_CANDIDATE", "NO_ACTION_SPECIFIC_SIGNAL"):
        raise ValueError("not an approved geometry release outcome")
    candidate = outcome == "STAGE_G_C_CANDIDATE"
    if (record["candidate_admitted_to_utility"] is not candidate
            or record["utility_cannot_reopen"] is not True
            or record["utility_gt_access"] != "METADATA_ONLY"):
        raise ValueError("geometry release violates Utility isolation/termination")
    contract = record["frozen_contract"]
    if candidate:
        if record["stage"] != "G-C" or not isinstance(contract, Mapping) or set(contract) != _CONTRACT_FIELDS:
            raise ValueError("candidate requires complete Stage G-C contract")
        if not _nonempty(contract):
            raise ValueError("candidate has empty contract definitions")
    elif contract is not None:
        raise ValueError("terminal release may not admit a candidate contract")
    # allow_nan=False also rejects nonfinite scientific constants recursively.
    _canonical_bytes(contract)
    specifications = record["specifications"]
    if not isinstance(specifications, Mapping) or set(specifications) != {"charter", "stage_specification"}:
        raise ValueError("geometry specification identities missing")
    evidence = record["evidence"]
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("geometry release requires stage evidence")
    identities = [*specifications.values(), *evidence, record["approval"]]
    paths = [str(_absolute(_identity(item)["path"])) for item in identities]
    if len(set(paths)) != len(paths):
        raise ValueError("geometry evidence/specification/approval must be distinct")
    return outcome


def validate_geometry_release(record: Mapping) -> str:
    """Shape-only classification; no referenced artifact is opened/authorized.

    Full review/evidence verification is private to the protected first-access
    boundary. An outcome string from this function is NOT an access token.
    """
    return _release_shape(record)


def _verify_geometry_references(record, protected):
    identities = [*record["specifications"].values(), *record["evidence"], record["approval"]]
    for identity in identities:
        _read_verified(identity, protected=protected)
    approval = json.loads(_read_verified(record["approval"], protected=protected))
    expected = {"schema_version": 1, "kind": "geometry_release_approval",
                **{key: record[key] for key in _APPROVED_FIELDS},
                "frozen_contract_sha256": hashlib.sha256(_canonical_bytes(record["frozen_contract"])).hexdigest()}
    if approval != expected:
        raise ValueError("geometry release differs from reviewed approval")


def load_geometry_release(path: Path, expected_sha256: str) -> dict:
    """Hash-check release JSON + detached SHA, without opening its references.

    Use authorize_first_gt_access for full review/evidence verification.
    """
    payload = _read_verified({"path": str(path), "sha256": expected_sha256}, detached=True)
    record = json.loads(payload)
    if _canonical_bytes(record) != payload:
        raise ValueError("geometry release must be canonical JSON")
    validate_geometry_release(record)
    return record


def _exists(path):
    return os.path.lexists(path)


def _reloaded(prior_identity, geometry_identity):
    # Inspect only the small confirmation itself before loading any referenced
    # artifact: a disguised evidence/source reference must never open GT.
    prior = json.loads(_read_verified(prior_identity))
    prior = _validate_record(prior, require_targets_absent=False, verify_record_files=False)
    protected = [_absolute(prior["gt_mesh"]["path"]),
                 *[_absolute(path) for path in prior["probe_targets"].values()]]
    _reject_aliases([prior_identity, geometry_identity, prior["source_record"], prior["snapshot_record"]], protected)
    geometry_payload = _read_verified(geometry_identity, detached=True, protected=protected)
    geometry = json.loads(geometry_payload)
    if _canonical_bytes(geometry) != geometry_payload:
        raise ValueError("geometry release must be canonical JSON")
    validate_geometry_release(geometry)
    _reject_aliases([*geometry["specifications"].values(), *geometry["evidence"], geometry["approval"]], protected)
    # Reuse the production schema validator, with guarded I/O for its two file
    # identities. The general loader's unguarded opens cannot enforce this
    # boundary's protected-descriptor contract against replacement races.
    for name in ("source_record", "snapshot_record"):
        identity = {key: prior[name][key] for key in ("path", "sha256")}
        _read_verified(identity, protected=protected)
    _verify_geometry_references(geometry, protected)
    return prior, geometry


def _reject_aliases(identities, protected):
    for item in identities:
        for path in (_absolute(item["path"]), _absolute(str(item["path"]) + ".sha256")):
            if any(path == target or path in target.parents or target in path.parents for target in protected):
                raise ValueError("firewall artifact aliases GT or a probe target")
            if path.exists():
                info = path.stat()
                for target in protected:
                    try:
                        other = target.stat()
                    except FileNotFoundError:
                        continue
                    if (info.st_dev, info.st_ino) == (other.st_dev, other.st_ino):
                        raise ValueError("firewall artifact hardlinks protected GT/target")


def authorize_first_gt_access(prior_record, geometry_record, *, access_log_path: Path) -> dict:
    """Reload both identities and publish a one-shot record before any mesh parse.

    Any existing access log is fail-closed, even if identical. A failed attempt
    after log publication stays recorded; never delete it to obtain a retry.
    The future evaluator must call this immediately before its mesh boundary.
    """
    prior_identity, geometry_identity = _identity(prior_record), _identity(geometry_record)
    prior, geometry = _reloaded(prior_identity, geometry_identity)
    log = _absolute(access_log_path)
    targets = prior["probe_targets"]
    if log != _absolute(targets["access_log_path"]):
        raise ValueError("first-access target differs from preregistration")
    if _exists(log):
        raise FileExistsError("first Utility GT access is already recorded")
    for name in ("output_dir", "staging_dir"):
        if _exists(targets[name]):
            raise ValueError("Utility probe artifact exists before first access")
    now = datetime.now(timezone.utc)
    if _time(prior["created_utc"]) > now or _time(geometry["created_utc"]) > now:
        raise ValueError("confirmation/release is later than first GT access")
    if _absolute(geometry["repository"]["root"]) != _absolute(prior["repository"]["root"]):
        raise ValueError("geometry release belongs to another project")
    record = {
        "schema_version": 1, "kind": "utility_first_gt_access", "created_utc": now.isoformat(),
        "prior_confirmation": prior_identity, "geometry_release": geometry_identity,
        "geometry_outcome": geometry["outcome"], "gt_mesh": prior["gt_mesh"],
        "access_log_path": str(log), "mesh_parse_preceded_by_this_record": True,
        "utility_feedback_cannot_reopen_geometry": True,
    }
    payload = _canonical_bytes(record)
    temporary = log.parent / f".{log.name}.tmp-{uuid.uuid4().hex}"
    if not log.parent.is_dir():
        raise ValueError("first-access log parent is missing")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _reloaded(prior_identity, geometry_identity)
        for name in ("output_dir", "staging_dir"):
            if _exists(targets[name]):
                raise ValueError("Utility probe target appeared before publication")
        # Hardlink publication is exclusive. os.replace would overwrite a racer.
        os.link(temporary, log)
        _reloaded(prior_identity, geometry_identity)
        for name in ("output_dir", "staging_dir"):
            if _exists(targets[name]):
                raise ValueError("Utility probe target appeared during publication")
        if log.read_bytes() != payload:
            raise ValueError("first-access record changed during publication")
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(log), "sha256": hashlib.sha256(payload).hexdigest(), "record": record}
