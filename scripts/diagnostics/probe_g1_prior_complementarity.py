"""Read-only, single-candidate G1 prior-complementarity feasibility probe."""

import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePath
import shutil
import subprocess
import sys
import uuid

import numpy as np


if __package__ in (None, ""):
    repository_root = str(Path(__file__).resolve().parents[2])
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)

from reliability.g1_complementarity import (
    ProbeConfig,
    ProbeInconclusiveError,
    bootstrap_rows,
    build_probe_domain,
    build_probe_report,
    crossfit_comparison,
    fold_rows,
    make_spatial_folds,
    paired_voxel_bootstrap,
    probe_exit_code,
    raw_risk_direction,
    risk_bin_rows,
)


FORMAL_ITERATIONS = (3000, 7000)
FORMAL_EVIDENCE_VERSION = 4
REQUIRED_ARTIFACTS = (
    "bootstrap.csv",
    "folds.csv",
    "inputs.json",
    "report.json",
    "risk_bins.csv",
)


def _resolved(path):
    return Path(path).expanduser().resolve()


def _is_within(path, parent):
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_id(value):
    value = str(value)
    path = PurePath(value)
    if not value or path.is_absolute() or len(path.parts) != 1 or value in (".", ".."):
        raise ValueError("diagnostic ID must be one safe path component")
    if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in value):
        raise ValueError("diagnostic ID contains unsafe characters")
    return value


def _canonical_json(path, value):
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    Path(path).write_text(payload, encoding="utf-8", newline="\n")


def _write_csv(path, rows, fieldnames):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _risk_bins(domain, bin_count=20):
    order = np.lexsort((domain.row_indices, domain.prior_risk))
    rows = []
    for index, selected in enumerate(np.array_split(order, bin_count)):
        if selected.size == 0:
            raise ProbeInconclusiveError("raw-risk bin is empty")
        rows.append(
            {
                "bin": index,
                "count": int(selected.size),
                "risk_min": float(domain.prior_risk[selected].min()),
                "risk_max": float(domain.prior_risk[selected].max()),
                "mean_distance_m": float(domain.distances[selected].mean()),
                "high_error_rate": float(domain.labels[selected].mean()),
            }
        )
    return rows


def evaluate_iteration(joined, distances, *, iteration, config):
    domain = build_probe_domain(joined, distances, iteration=iteration)
    folds = make_spatial_folds(
        domain.centers, domain.row_indices, fold_count=config.fold_count
    )
    direction = raw_risk_direction(
        domain.prior_risk, domain.distances, domain.row_indices
    )
    from reliability.g1_metrics import binary_curves

    marginal = binary_curves(domain.prior_risk, domain.labels)
    direction["marginal"] = {
        "auroc": float(marginal["auroc"]),
        "auprc": float(marginal["auprc"]),
    }
    direction["risk_bins"] = _risk_bins(domain)
    comparison = crossfit_comparison(domain, folds, config=config)
    bootstrap = paired_voxel_bootstrap(
        domain.centers,
        domain.labels,
        comparison.baseline_oof,
        comparison.augmented_oof,
        config=config,
    )
    normalized_folds = []
    residuals = []
    for row in comparison.folds:
        residual = float(row["independence_relative_residual"])
        residuals.append(residual)
        normalized_folds.append(
            {
                "fold": int(row["fold"]),
                "training_count": int(row["training_count"]),
                "validation_count": int(row["validation_count"]),
                "training_positive_count": int(row["training_positive_count"]),
                "training_negative_count": int(row["training_negative_count"]),
                "validation_positive_count": int(row["validation_positive_count"]),
                "validation_negative_count": int(row["validation_negative_count"]),
                "baseline_auroc": float(row["baseline_auroc"]),
                "augmented_auroc": float(row["augmented_auroc"]),
                "auroc_gain": float(row["auroc_gain"]),
                "candidate_relative_residual": residual,
                "positive_weight": float(row["positive_weight"]),
                "negative_weight": float(row["negative_weight"]),
                "baseline_a_mean": float(row["baseline_feature_mean"][0]),
                "baseline_one_minus_s_mean": float(row["baseline_feature_mean"][1]),
                "baseline_a_scale": float(row["baseline_feature_scale"][0]),
                "baseline_one_minus_s_scale": float(row["baseline_feature_scale"][1]),
                "candidate_mean": float(row["candidate_mean"]),
                "candidate_scale": float(row["candidate_scale"]),
                "baseline_iterations": int(row["baseline_iterations"]),
                "augmented_iterations": int(row["augmented_iterations"]),
                "baseline_converged": True,
                "augmented_converged": True,
            }
        )
    return {
        "iteration": int(iteration),
        "role": "primary" if iteration == 7000 else "direction_stability",
        "domain": {
            "original_point_count": int(domain.original_point_count),
            "finite_center_count": int(domain.finite_center_count),
            "eligible_count": int(domain.labels.size),
            "positive_count": int(domain.labels.sum()),
            "negative_count": int((~domain.labels).sum()),
            "coverage": float(domain.coverage),
            "label": "distance_gt_0.05_m",
        },
        "direction": direction,
        "numerical_independence": {
            "threshold_strictly_greater_than": float(
                config.independence_threshold
            ),
            "fold_relative_residuals": residuals,
        },
        "crossfit": {
            "folds": normalized_folds,
            "baseline": {
                "auroc": float(comparison.baseline["auroc"]),
                "auprc": float(comparison.baseline["auprc"]),
            },
            "augmented": {
                "auroc": float(comparison.augmented["auroc"]),
                "auprc": float(comparison.augmented["auprc"]),
            },
            "pooled_auroc_gain": float(comparison.pooled_auroc_gain),
        },
        "bootstrap": bootstrap,
    }


class ProductionDependencies:
    @staticmethod
    def git_identity(repository):
        repository = Path(repository)
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=repository,
            text=True,
        )
        return {"commit": commit, "clean": status == ""}

    @staticmethod
    def load_confirmation(path, expected_sha):
        from reliability.g1_confirmation import load_confirmation_record

        return load_confirmation_record(path, expected_sha)

    @staticmethod
    def validate_admission(**kwargs):
        from reliability.g1_confirmation import validate_formal_admission
        from scripts.diagnostics.evaluate_d0_g1 import canonical_tree_sha256

        record = kwargs["record"]
        run_dir = kwargs["run_dir"]
        source_root = kwargs["source_root"]
        gt_mesh = kwargs["gt_mesh"]
        args = kwargs["args"]
        if (run_dir / "exit_code.txt").read_text(encoding="utf-8").strip() != "0":
            raise ValueError("formal training did not exit successfully")
        if canonical_tree_sha256(source_root) != args.expected_dataset_sha:
            raise ValueError("dataset SHA256 mismatch")
        if _sha256_file(gt_mesh) != args.expected_gt_sha:
            raise ValueError("GT mesh SHA256 mismatch")
        for name, expected in (
            ("dataset_manifest_before.sha256", args.expected_dataset_sha),
            ("dataset_manifest_after.sha256", args.expected_dataset_sha),
            ("aligned_prior_before.sha256", args.expected_prior_sha),
            ("aligned_prior_after.sha256", args.expected_prior_sha),
        ):
            if (run_dir / name).read_text(encoding="utf-8").strip() != expected:
                raise ValueError(f"formal hash record mismatch: {name}")
        targets = record["targets"]
        validate_formal_admission(
            record,
            run_dir=run_dir,
            confirmation_id=record["confirmation_id"],
            evaluator_commit=record["formula_commit"],
            dataset_sha256=args.expected_dataset_sha,
            prior_sha256=args.expected_prior_sha,
            gt_sha256=args.expected_gt_sha,
            report_path=targets["report_path"]["path"],
            output_dir=targets["output_dir"]["path"],
            archive_path=targets["archive_path"]["path"],
        )

    @staticmethod
    def load_iteration(run_dir, iteration, *, expected_evidence_version):
        from reliability.offline_g1 import load_g1_iteration

        return load_g1_iteration(
            run_dir,
            iteration,
            expected_evidence_version=expected_evidence_version,
        )

    @staticmethod
    def load_mesh(path):
        from reliability.offline_g1 import load_valid_mesh

        return load_valid_mesh(path)

    @staticmethod
    def distances(centers, mesh):
        from reliability.offline_g1 import closest_triangle_distances

        return closest_triangle_distances(centers, mesh)

    evaluate_iteration = staticmethod(evaluate_iteration)

    @staticmethod
    def fingerprint(paths):
        from scripts.diagnostics.evaluate_d0_g1 import fingerprint_inputs

        return fingerprint_inputs(paths)

    @staticmethod
    def assert_unchanged(before, after):
        from scripts.diagnostics.evaluate_d0_g1 import assert_inputs_unchanged

        return assert_inputs_unchanged(before, after)


def _validate_request(args):
    run_dir = _resolved(args.run_dir)
    source_root = _resolved(args.source_root)
    gt_mesh = _resolved(args.gt_mesh)
    confirmation = _resolved(args.confirmation_contract)
    output_root = _resolved(args.output_root)
    diagnostic_id = _safe_id(args.diagnostic_id)
    for path, label, kind in (
        (run_dir, "run directory", "directory"),
        (source_root, "source root", "directory"),
        (gt_mesh, "GT mesh", "file"),
        (confirmation, "confirmation contract", "file"),
    ):
        valid = path.is_dir() if kind == "directory" else path.is_file()
        if not valid:
            raise ValueError(f"{label} is missing: {path}")
    for immutable in (run_dir, source_root, gt_mesh):
        if _is_within(output_root, immutable) or output_root == immutable:
            raise ValueError("output root must remain outside immutable inputs")
    target = output_root / diagnostic_id
    if target.exists():
        raise FileExistsError(f"output already exists: {target}")
    return run_dir, source_root, gt_mesh, confirmation, output_root, diagnostic_id


def _immutable_paths(run_dir, source_root, gt_mesh, confirmation):
    evidence = run_dir / "d0_evidence"
    paths = {
        "checkpoint_3000": run_dir / "chkpnt3000.pth",
        "checkpoint_7000": run_dir / "chkpnt7000.pth",
        "snapshot_3000": evidence / "iteration_003000.npz",
        "snapshot_7000": evidence / "iteration_007000.npz",
        "confirmation_contract": confirmation,
        "confirmation_sha256": Path(f"{confirmation}.sha256"),
        "run_identity": run_dir / "run_identity.json",
        "resolved_config": run_dir / "resolved_config.json",
        "dataset_manifest_before": run_dir / "dataset_manifest_before.sha256",
        "dataset_manifest_after": run_dir / "dataset_manifest_after.sha256",
        "aligned_prior_before": run_dir / "aligned_prior_before.sha256",
        "aligned_prior_after": run_dir / "aligned_prior_after.sha256",
        "source_root": source_root,
        "gt_mesh": gt_mesh,
    }
    return paths


def _provenance(args, record, run_dir):
    identity = run_dir / "run_identity.json"
    return {
        "diagnostic_commit": args.expected_diagnostic_commit,
        "formula_commit": record.get("formula_commit", "0" * 40),
        "confirmation_id": record.get("confirmation_id", "unknown"),
        "confirmation_sha256": args.expected_confirmation_sha,
        "dataset_sha256": args.expected_dataset_sha,
        "aligned_prior_sha256": args.expected_prior_sha,
        "gt_mesh_sha256": args.expected_gt_sha,
        "run_identity_sha256": _sha256_file(identity) if identity.is_file() else "0" * 64,
    }


def _inconclusive_iterations(reason):
    return [
        {
            "iteration": iteration,
            "role": "primary" if iteration == 7000 else "direction_stability",
            "status": "INCONCLUSIVE",
            "reason": str(reason),
        }
        for iteration in FORMAL_ITERATIONS
    ]


def _publish(output_root, diagnostic_id, inputs, report, immutable, dependencies):
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / diagnostic_id
    staging = output_root / f".{diagnostic_id}.tmp-{uuid.uuid4().hex}"
    if target.exists():
        raise FileExistsError(f"output already exists: {target}")
    staging.mkdir(parents=False)
    try:
        _canonical_json(staging / "inputs.json", inputs)
        _canonical_json(staging / "report.json", report)
        _write_csv(
            staging / "folds.csv",
            fold_rows(report),
            [
                "iteration", "fold", "training_count", "validation_count",
                "training_positive_count", "training_negative_count",
                "validation_positive_count", "validation_negative_count",
                "baseline_auroc", "augmented_auroc", "auroc_gain",
                "candidate_relative_residual", "positive_weight", "negative_weight",
                "baseline_a_mean", "baseline_one_minus_s_mean",
                "baseline_a_scale", "baseline_one_minus_s_scale",
                "candidate_mean", "candidate_scale", "baseline_iterations",
                "augmented_iterations", "baseline_converged", "augmented_converged",
            ],
        )
        _write_csv(
            staging / "risk_bins.csv",
            risk_bin_rows(report),
            ["iteration", "bin", "count", "risk_min", "risk_max", "mean_distance_m", "high_error_rate"],
        )
        _write_csv(
            staging / "bootstrap.csv",
            bootstrap_rows(report),
            ["iteration", "replicate", "auroc_gain"],
        )
        after = dependencies.fingerprint(immutable)
        dependencies.assert_unchanged(inputs["input_fingerprints"], after)
        from scripts.diagnostics.evaluate_d0_g1 import build_manifest

        manifest = build_manifest(staging, REQUIRED_ARTIFACTS)
        _canonical_json(staging / "manifest.json", manifest)
        staging.replace(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"output_dir": str(target), "manifest": manifest, "report": report}


def run_probe(args, dependencies=None):
    dependencies = dependencies or ProductionDependencies()
    (
        run_dir,
        source_root,
        gt_mesh,
        confirmation,
        output_root,
        diagnostic_id,
    ) = _validate_request(args)
    repository = Path(__file__).resolve().parents[2]
    identity = dependencies.git_identity(repository)
    if identity != {"commit": args.expected_diagnostic_commit, "clean": True}:
        raise ValueError("diagnostic code identity must be exact and clean")
    immutable = _immutable_paths(run_dir, source_root, gt_mesh, confirmation)
    before = dependencies.fingerprint(immutable)
    config = ProbeConfig()
    record = {}
    try:
        record = dependencies.load_confirmation(
            confirmation, args.expected_confirmation_sha
        )
        dependencies.validate_admission(
            record=record,
            run_dir=run_dir,
            source_root=source_root,
            gt_mesh=gt_mesh,
            args=args,
        )
        mesh = dependencies.load_mesh(gt_mesh)
        summaries = []
        for iteration in FORMAL_ITERATIONS:
            joined = dependencies.load_iteration(
                run_dir,
                iteration,
                expected_evidence_version=FORMAL_EVIDENCE_VERSION,
            )
            distances = dependencies.distances(joined.centers, mesh)
            summaries.append(
                dependencies.evaluate_iteration(
                    joined,
                    distances,
                    iteration=iteration,
                    config=config,
                )
            )
        report = build_probe_report(
            summaries,
            provenance=_provenance(args, record, run_dir),
            config=config,
        )
    except (ProbeInconclusiveError, ValueError, FileNotFoundError) as error:
        reason = f"{type(error).__name__}: {error}"
        report = build_probe_report(
            _inconclusive_iterations(reason),
            provenance=_provenance(args, record, run_dir),
            config=config,
            inconclusive_reasons=[reason],
        )
    inputs = {
        "schema_version": 1,
        "diagnostic_id": diagnostic_id,
        "iterations": list(FORMAL_ITERATIONS),
        "expected_evidence_version": FORMAL_EVIDENCE_VERSION,
        "diagnostic_commit": args.expected_diagnostic_commit,
        "input_fingerprints": before,
    }
    publication = _publish(
        output_root,
        diagnostic_id,
        inputs,
        report,
        immutable,
        dependencies,
    )
    return probe_exit_code(report), publication


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run the frozen read-only G1 prior-complementarity probe."
    )
    for flag in (
        "run-dir", "source-root", "gt-mesh", "confirmation-contract",
        "output-root", "diagnostic-id", "expected-diagnostic-commit",
        "expected-confirmation-sha", "expected-dataset-sha",
        "expected-prior-sha", "expected-gt-sha",
    ):
        parser.add_argument(f"--{flag}", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        exit_code, publication = run_probe(args)
    except Exception as error:
        print(f"G1_COMPLEMENTARITY_ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "diagnostic_only": True,
                "exit_code": exit_code,
                "outcome": publication["report"]["outcome"],
                "output_dir": publication["output_dir"],
            },
            sort_keys=True,
            indent=2,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
