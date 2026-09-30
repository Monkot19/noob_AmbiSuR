"""Fail-closed, label-free audit decisions for geometry reliability semantics."""

from collections.abc import Mapping, Sequence
import re


SCHEMA_VERSION = 1
OUTCOMES = (
    "SEMANTIC_DEFECT_CONFIRMED",
    "NO_SEMANTIC_REPAIR_JUSTIFIED",
    "INCONCLUSIVE",
)
CHECK_NAMES = (
    "task1_formula_conformance",
    "version4_state_3000",
    "version4_state_7000",
    "collector_component_contract",
    "immutable_inputs",
)
REPAIRABLE_CHECKS = frozenset(CHECK_NAMES[:-1])
CHECK_FIELDS = frozenset(("name", "status", "expected", "actual"))
CHECK_STATUSES = frozenset(("PASS", "FAIL", "INCONCLUSIVE"))
PROVENANCE_FIELDS = frozenset(
    (
        "diagnostic_commit",
        "dataset_sha256",
        "aligned_prior_sha256",
        "evidence_version",
        "iterations",
        "task1_outcome",
        "task1_test_count",
    )
)
REPORT_FIELDS = frozenset(
    (
        "schema_version",
        "diagnostic_only",
        "training_started",
        "outcome",
        "repair_authorized",
        "authorized_evidence_version",
        "utility_authorized",
        "c1_authorized",
        "tasks_4_to_13_stopped",
        "checks",
        "failed_checks",
        "inconclusive_reasons",
        "provenance",
    )
)
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")


def _plain_mapping(name, value, fields):
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise ValueError(f"{name} field inventory mismatch")
    return dict(value)


def _validate_provenance(value):
    result = _plain_mapping("provenance", value, PROVENANCE_FIELDS)
    if not _HEX40.fullmatch(str(result["diagnostic_commit"])):
        raise ValueError("invalid diagnostic commit")
    for name in ("dataset_sha256", "aligned_prior_sha256"):
        if not _HEX64.fullmatch(str(result[name])):
            raise ValueError(f"invalid provenance digest: {name}")
    if result["evidence_version"] != 4:
        raise ValueError("geometry semantic audit requires evidence version 4")
    if result["iterations"] != [3000, 7000]:
        raise ValueError("geometry semantic audit requires iterations 3000 and 7000")
    if result["task1_outcome"] != "NO_SEMANTIC_REPAIR_JUSTIFIED":
        raise ValueError("Task 1 outcome is not the frozen semantic result")
    if result["task1_test_count"] != 23:
        raise ValueError("Task 1 test inventory mismatch")
    return result


def _validate_checks(checks):
    if isinstance(checks, (str, bytes)) or not isinstance(checks, Sequence):
        raise ValueError("checks must be a sequence")
    normalized = []
    for check in checks:
        item = _plain_mapping("check", check, CHECK_FIELDS)
        if item["status"] not in CHECK_STATUSES:
            raise ValueError(f"invalid check status: {item['status']}")
        if not isinstance(item["expected"], str) or not item["expected"]:
            raise ValueError("check expected value must be non-empty text")
        if not isinstance(item["actual"], str) or not item["actual"]:
            raise ValueError("check actual value must be non-empty text")
        normalized.append(item)
    if tuple(item["name"] for item in normalized) != CHECK_NAMES:
        raise ValueError("check inventory mismatch")
    return normalized


def _decision(checks):
    inconclusive = [
        item["actual"] for item in checks if item["status"] == "INCONCLUSIVE"
    ]
    failed = [item["name"] for item in checks if item["status"] == "FAIL"]
    if inconclusive or any(name not in REPAIRABLE_CHECKS for name in failed):
        return "INCONCLUSIVE", failed, inconclusive
    if failed:
        return "SEMANTIC_DEFECT_CONFIRMED", failed, []
    return "NO_SEMANTIC_REPAIR_JUSTIFIED", [], []


def build_geometry_semantic_report(checks, provenance):
    """Build the exhaustive outcome without consulting labels or metrics."""
    checks = _validate_checks(checks)
    provenance = _validate_provenance(provenance)
    outcome, failed, inconclusive = _decision(checks)
    defect = outcome == "SEMANTIC_DEFECT_CONFIRMED"
    report = {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_only": True,
        "training_started": False,
        "outcome": outcome,
        "repair_authorized": defect,
        "authorized_evidence_version": 5 if defect else None,
        "utility_authorized": False,
        "c1_authorized": False,
        "tasks_4_to_13_stopped": not defect,
        "checks": checks,
        "failed_checks": failed,
        "inconclusive_reasons": inconclusive,
        "provenance": provenance,
    }
    validate_geometry_semantic_report(report)
    return report


def validate_geometry_semantic_report(report):
    """Reject extra fields, overrides, and outcomes inconsistent with checks."""
    result = _plain_mapping("report", report, REPORT_FIELDS)
    if result["schema_version"] != SCHEMA_VERSION:
        raise ValueError("report schema version mismatch")
    if result["diagnostic_only"] is not True:
        raise ValueError("geometry audit must remain diagnostic-only")
    if result["training_started"] is not False:
        raise ValueError("geometry audit cannot start training")
    checks = _validate_checks(result["checks"])
    provenance = _validate_provenance(result["provenance"])
    outcome, failed, inconclusive = _decision(checks)
    defect = outcome == "SEMANTIC_DEFECT_CONFIRMED"
    expected = {
        "outcome": outcome,
        "repair_authorized": defect,
        "authorized_evidence_version": 5 if defect else None,
        "utility_authorized": False,
        "c1_authorized": False,
        "tasks_4_to_13_stopped": not defect,
        "failed_checks": failed,
        "inconclusive_reasons": inconclusive,
    }
    for name, value in expected.items():
        if result[name] != value:
            raise ValueError(f"report decision contract mismatch: {name}")
    return None


def audit_exit_code(report):
    validate_geometry_semantic_report(report)
    outcome = report["outcome"]
    return {
        "NO_SEMANTIC_REPAIR_JUSTIFIED": 0,
        "SEMANTIC_DEFECT_CONFIRMED": 1,
        "INCONCLUSIVE": 2,
    }[outcome]
