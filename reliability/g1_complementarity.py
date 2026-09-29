"""Pure statistics for the read-only G1 prior-complementarity probe."""

from dataclasses import dataclass
import math

import numpy as np

from reliability.g1_metrics import PRIMARY_THRESHOLD_M, binary_curves


class ProbeInconclusiveError(ValueError):
    """Raised when the frozen probe cannot produce a valid conclusion."""


@dataclass(frozen=True)
class ProbeConfig:
    fold_count: int = 5
    l2_penalty: float = 1e-4
    max_iterations: int = 100
    gradient_tolerance: float = 1e-8
    armijo_constant: float = 1e-4
    backtracking_factor: float = 0.5
    minimum_step: float = 2.0 ** -20
    independence_threshold: float = 1e-8
    voxel_size_m: float = 0.5
    bootstrap_seed: int = 20260928
    bootstrap_replicates: int = 2000
    interval_percentiles: tuple = (2.5, 97.5)


@dataclass(frozen=True)
class ProbeDomain:
    iteration: int
    original_point_count: int
    finite_center_count: int
    row_indices: np.ndarray
    centers: np.ndarray
    distances: np.ndarray
    labels: np.ndarray
    a: np.ndarray
    one_minus_s: np.ndarray
    prior_risk: np.ndarray
    coverage: float


@dataclass(frozen=True)
class LogisticModel:
    coefficients: np.ndarray
    iterations: int
    gradient_inf_norm: float

    def predict_proba(self, features):
        features = _matrix("features", features)
        if features.shape[1] + 1 != self.coefficients.size:
            raise ProbeInconclusiveError("prediction feature count mismatch")
        design = np.column_stack(
            (np.ones(features.shape[0], dtype=np.float64), features)
        )
        probabilities = _sigmoid(design @ self.coefficients)
        if not np.isfinite(probabilities).all():
            raise ProbeInconclusiveError("predictions must be finite")
        return probabilities


@dataclass(frozen=True)
class CrossfitResult:
    fold_assignments: np.ndarray
    baseline_oof: np.ndarray
    augmented_oof: np.ndarray
    folds: tuple
    baseline: dict
    augmented: dict
    pooled_auroc_gain: float


def _vector(name, values, *, dtype=np.float64, nonempty=True):
    values = np.asarray(values, dtype=dtype)
    if values.ndim != 1 or (nonempty and values.size == 0):
        raise ValueError(f"{name} must be a nonempty vector")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains nonfinite values")
    return values


def _matrix(name, values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ProbeInconclusiveError(f"{name} must be a nonempty matrix")
    if not np.isfinite(values).all():
        raise ProbeInconclusiveError(f"{name} must be finite")
    return values


def _boolean_vector(name, values, rows=None):
    values = np.asarray(values)
    if values.ndim != 1 or (rows is not None and values.shape[0] != rows):
        raise ValueError(f"{name} row count mismatch")
    if not np.issubdtype(values.dtype, np.bool_):
        raise ValueError(f"{name} must be boolean")
    return values


def _metric_summary(scores, labels):
    curves = binary_curves(scores, labels)
    return {"auroc": curves["auroc"], "auprc": curves["auprc"]}


def _average_ranks(values):
    values = _vector("rank values", values)
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and ordered[stop] == ordered[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    return ranks


def _spearman(left, right):
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    left_rank -= left_rank.mean()
    right_rank -= right_rank.mean()
    denominator = math.sqrt(
        float(np.dot(left_rank, left_rank))
        * float(np.dot(right_rank, right_rank))
    )
    return 0.0 if denominator == 0.0 else float(
        np.dot(left_rank, right_rank) / denominator
    )


def build_probe_domain(joined, distances, *, iteration):
    iteration = int(iteration)
    if iteration not in (3000, 7000):
        raise ValueError("probe iteration must be 3000 or 7000")
    if int(getattr(joined, "iteration", iteration)) != iteration:
        raise ValueError("joined iteration mismatch")
    original_count = int(joined.original_point_count)
    if original_count <= 0:
        raise ValueError("original point count must be positive")
    centers = np.asarray(joined.centers, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3 or centers.shape[0] == 0:
        raise ValueError("finite checkpoint centers must have shape [P,3]")
    if not np.isfinite(centers).all():
        raise ValueError("finite checkpoint centers contain nonfinite values")
    finite_rows = np.asarray(joined.finite_row_indices)
    if (
        finite_rows.ndim != 1
        or finite_rows.shape[0] != centers.shape[0]
        or not np.issubdtype(finite_rows.dtype, np.integer)
    ):
        raise ValueError("finite row map mismatch")
    finite_rows = finite_rows.astype(np.int64, copy=False)
    if (
        np.any(finite_rows < 0)
        or np.any(finite_rows >= original_count)
        or np.unique(finite_rows).size != finite_rows.size
    ):
        raise ValueError("finite row map is invalid")
    distances = _vector("GT distances", distances)
    if distances.shape[0] != centers.shape[0] or np.any(distances < 0.0):
        raise ValueError("GT distances must match finite centers and be nonnegative")
    snapshot = joined.snapshot
    required = ("A", "S", "r_p", "V_p")
    missing = [name for name in required if name not in snapshot]
    if missing:
        raise ValueError(f"missing probe snapshot fields: {missing}")
    validity = _boolean_vector("V_p", snapshot["V_p"], original_count)
    numeric = {}
    for name in ("A", "S", "r_p"):
        values = _vector(name, snapshot[name])
        if values.shape[0] != original_count:
            raise ValueError(f"{name} row count mismatch")
        numeric[name] = values
    selected = validity[finite_rows]
    if not bool(selected.any()):
        raise ProbeInconclusiveError("probe domain has no V_p=True rows")
    rows = finite_rows[selected]
    return ProbeDomain(
        iteration=iteration,
        original_point_count=original_count,
        finite_center_count=int(centers.shape[0]),
        row_indices=rows.copy(),
        centers=centers[selected].copy(),
        distances=distances[selected].copy(),
        labels=(distances[selected] > PRIMARY_THRESHOLD_M),
        a=numeric["A"][rows].copy(),
        one_minus_s=(1.0 - numeric["S"][rows]).copy(),
        prior_risk=(1.0 - numeric["r_p"][rows]).copy(),
        coverage=float(selected.mean()),
    )


def make_spatial_folds(centers, row_indices, *, fold_count=5):
    centers = np.asarray(centers, dtype=np.float64)
    row_indices = np.asarray(row_indices)
    fold_count = int(fold_count)
    if centers.ndim != 2 or centers.shape[1] != 3 or centers.shape[0] == 0:
        raise ProbeInconclusiveError("fold centers must have shape [P,3]")
    if not np.isfinite(centers).all():
        raise ProbeInconclusiveError("fold centers must be finite")
    if (
        row_indices.ndim != 1
        or row_indices.shape[0] != centers.shape[0]
        or not np.issubdtype(row_indices.dtype, np.integer)
    ):
        raise ProbeInconclusiveError("fold row indices mismatch")
    if fold_count < 2 or centers.shape[0] < fold_count:
        raise ProbeInconclusiveError("insufficient rows for spatial folds")
    axis = int(np.argmax(np.ptp(centers, axis=0)))
    order = np.lexsort((row_indices.astype(np.int64), centers[:, axis]))
    assignments = np.empty(centers.shape[0], dtype=np.int64)
    for fold, indices in enumerate(np.array_split(order, fold_count)):
        if indices.size == 0:
            raise ProbeInconclusiveError("empty spatial fold")
        assignments[indices] = fold
    return assignments


def raw_risk_direction(risk, distances, row_indices):
    risk = _vector("raw prior risk", risk)
    distances = _vector("GT distances", distances)
    row_indices = np.asarray(row_indices)
    if (
        distances.shape != risk.shape
        or row_indices.ndim != 1
        or row_indices.shape != risk.shape
        or not np.issubdtype(row_indices.dtype, np.integer)
    ):
        raise ProbeInconclusiveError("raw direction row mismatch")
    if risk.size < 5:
        raise ProbeInconclusiveError("raw direction requires five quintiles")
    order = np.lexsort((row_indices.astype(np.int64), risk))
    quintiles = np.array_split(order, 5)
    if any(indices.size == 0 for indices in quintiles):
        raise ProbeInconclusiveError("raw direction has an empty quintile")
    labels = distances > PRIMARY_THRESHOLD_M
    low = quintiles[0]
    high = quintiles[-1]
    low_rate = float(labels[low].mean())
    high_rate = float(labels[high].mean())
    return {
        "lowest_quintile_count": int(low.size),
        "highest_quintile_count": int(high.size),
        "lowest_quintile_high_error_rate": low_rate,
        "highest_quintile_high_error_rate": high_rate,
        "high_minus_low_error_rate": high_rate - low_rate,
        "spearman_risk_distance": _spearman(risk, distances),
    }


def column_independence(baseline, candidate):
    baseline = _matrix("baseline features", baseline)
    candidate = _vector("candidate feature", candidate)
    if candidate.shape[0] != baseline.shape[0]:
        raise ProbeInconclusiveError("candidate row count mismatch")
    design = np.column_stack(
        (np.ones(baseline.shape[0], dtype=np.float64), baseline)
    )
    try:
        coefficients, _residuals, _rank, _singular = np.linalg.lstsq(
            design, candidate, rcond=None
        )
    except np.linalg.LinAlgError as exc:
        raise ProbeInconclusiveError("column projection failed") from exc
    residual = candidate - design @ coefficients
    denominator = max(float(np.linalg.norm(candidate)), np.finfo(np.float64).tiny)
    ratio = float(np.linalg.norm(residual) / denominator)
    if not np.isfinite(ratio):
        raise ProbeInconclusiveError("column residual must be finite")
    return ratio


def _sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _class_weights(labels):
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.bool_):
        raise ProbeInconclusiveError("labels must be a boolean vector")
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        raise ProbeInconclusiveError("model requires both classes")
    positive_weight = labels.size / (2.0 * positives)
    negative_weight = labels.size / (2.0 * negatives)
    return (
        np.where(labels, positive_weight, negative_weight).astype(np.float64),
        float(positive_weight),
        float(negative_weight),
    )


def _loss_gradient_hessian(beta, design, labels_float, weights, config):
    logits = design @ beta
    normalization = float(weights.sum())
    loss = float(
        np.dot(weights, np.logaddexp(0.0, logits) - labels_float * logits)
        / normalization
        + 0.5 * config.l2_penalty * np.dot(beta[1:], beta[1:])
    )
    probability = _sigmoid(logits)
    residual = weights * (probability - labels_float) / normalization
    gradient = design.T @ residual
    gradient[1:] += config.l2_penalty * beta[1:]
    curvature = weights * probability * (1.0 - probability) / normalization
    hessian = design.T @ (design * curvature[:, None])
    diagonal = np.zeros(beta.size, dtype=np.float64)
    diagonal[1:] = config.l2_penalty
    hessian += np.diag(diagonal)
    return loss, gradient, hessian


def fit_logistic(features, labels, *, config):
    features = _matrix("features", features)
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ProbeInconclusiveError("labels must match features")
    if not np.issubdtype(labels.dtype, np.bool_):
        raise ProbeInconclusiveError("labels must be boolean")
    weights, _positive_weight, _negative_weight = _class_weights(labels)
    if int(config.max_iterations) <= 0:
        raise ProbeInconclusiveError("logistic solver did not converge")
    design = np.column_stack(
        (np.ones(features.shape[0], dtype=np.float64), features)
    )
    beta = np.zeros(design.shape[1], dtype=np.float64)
    labels_float = labels.astype(np.float64)
    for iteration in range(1, int(config.max_iterations) + 1):
        loss, gradient, hessian = _loss_gradient_hessian(
            beta, design, labels_float, weights, config
        )
        if not (
            np.isfinite(loss)
            and np.isfinite(gradient).all()
            and np.isfinite(hessian).all()
        ):
            raise ProbeInconclusiveError("logistic solver produced nonfinite values")
        gradient_norm = float(np.max(np.abs(gradient)))
        if gradient_norm <= config.gradient_tolerance:
            return LogisticModel(beta.copy(), iteration - 1, gradient_norm)
        try:
            newton_step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as exc:
            raise ProbeInconclusiveError("logistic Hessian solve failed") from exc
        directional = float(np.dot(gradient, newton_step))
        if not np.isfinite(directional) or directional <= 0.0:
            raise ProbeInconclusiveError("logistic Hessian direction is invalid")
        step = 1.0
        accepted = False
        while step >= config.minimum_step:
            candidate = beta - step * newton_step
            candidate_loss = _loss_gradient_hessian(
                candidate, design, labels_float, weights, config
            )[0]
            if np.isfinite(candidate_loss) and candidate_loss <= (
                loss - config.armijo_constant * step * directional
            ):
                beta = candidate
                accepted = True
                break
            step *= config.backtracking_factor
        if not accepted:
            raise ProbeInconclusiveError("logistic line search found no accepted step")
    _loss, gradient, _hessian = _loss_gradient_hessian(
        beta, design, labels_float, weights, config
    )
    gradient_norm = float(np.max(np.abs(gradient)))
    if gradient_norm > config.gradient_tolerance:
        raise ProbeInconclusiveError("logistic solver did not converge")
    return LogisticModel(beta.copy(), int(config.max_iterations), gradient_norm)


def _standardization(values):
    values = _matrix("training features", values)
    mean = values.mean(axis=0)
    scale = values.std(axis=0, ddof=0)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ProbeInconclusiveError("feature standardization must be finite")
    if np.any(scale == 0.0):
        raise ProbeInconclusiveError("feature standard deviation must be positive")
    return mean, scale


def summarize_oof(labels, baseline, augmented, folds):
    labels = np.asarray(labels)
    baseline = _vector("baseline OOF predictions", baseline)
    augmented = _vector("augmented OOF predictions", augmented)
    folds = np.asarray(folds)
    if (
        labels.ndim != 1
        or not np.issubdtype(labels.dtype, np.bool_)
        or labels.shape != baseline.shape
        or augmented.shape != baseline.shape
        or folds.ndim != 1
        or folds.shape != baseline.shape
        or not np.issubdtype(folds.dtype, np.integer)
    ):
        raise ProbeInconclusiveError("OOF inputs must share rows")
    try:
        baseline_summary = _metric_summary(baseline, labels)
        augmented_summary = _metric_summary(augmented, labels)
    except ValueError as exc:
        raise ProbeInconclusiveError(str(exc)) from exc
    fold_rows = []
    for fold in sorted(np.unique(folds).tolist()):
        selected = folds == fold
        try:
            baseline_fold = _metric_summary(baseline[selected], labels[selected])
            augmented_fold = _metric_summary(augmented[selected], labels[selected])
        except ValueError as exc:
            raise ProbeInconclusiveError(
                f"validation fold {fold} requires both classes"
            ) from exc
        fold_rows.append(
            {
                "fold": int(fold),
                "count": int(selected.sum()),
                "baseline_auroc": baseline_fold["auroc"],
                "augmented_auroc": augmented_fold["auroc"],
                "auroc_gain": (
                    augmented_fold["auroc"] - baseline_fold["auroc"]
                ),
            }
        )
    return {
        "baseline": baseline_summary,
        "augmented": augmented_summary,
        "pooled_auroc_gain": (
            augmented_summary["auroc"] - baseline_summary["auroc"]
        ),
        "folds": fold_rows,
    }


def crossfit_comparison(domain, folds, *, config):
    folds = np.asarray(folds)
    rows = domain.labels.size
    if (
        folds.ndim != 1
        or folds.shape[0] != rows
        or not np.issubdtype(folds.dtype, np.integer)
        or set(np.unique(folds).tolist()) != set(range(config.fold_count))
    ):
        raise ProbeInconclusiveError("crossfit fold assignment mismatch")
    baseline_values = np.column_stack((domain.a, domain.one_minus_s)).astype(
        np.float64, copy=False
    )
    candidate_values = np.asarray(domain.prior_risk, dtype=np.float64)
    if (
        baseline_values.shape != (rows, 2)
        or candidate_values.shape != (rows,)
        or not np.isfinite(baseline_values).all()
        or not np.isfinite(candidate_values).all()
    ):
        raise ProbeInconclusiveError("crossfit features must be finite and aligned")
    baseline_oof = np.full(rows, np.nan, dtype=np.float64)
    augmented_oof = np.full(rows, np.nan, dtype=np.float64)
    fold_reports = []
    for fold in range(config.fold_count):
        validation = folds == fold
        training = ~validation
        _class_weights(domain.labels[training])
        _class_weights(domain.labels[validation])
        baseline_mean, baseline_scale = _standardization(
            baseline_values[training]
        )
        baseline_train = (
            baseline_values[training] - baseline_mean
        ) / baseline_scale
        baseline_validation = (
            baseline_values[validation] - baseline_mean
        ) / baseline_scale
        candidate_train_raw = candidate_values[training, None]
        candidate_mean, candidate_scale = _standardization(candidate_train_raw)
        candidate_train = (
            candidate_train_raw[:, 0] - candidate_mean[0]
        ) / candidate_scale[0]
        candidate_validation = (
            candidate_values[validation] - candidate_mean[0]
        ) / candidate_scale[0]
        independence = column_independence(baseline_train, candidate_train)
        augmented_train = np.column_stack(
            (baseline_train, candidate_train)
        )
        augmented_validation = np.column_stack(
            (baseline_validation, candidate_validation)
        )
        baseline_model = fit_logistic(
            baseline_train, domain.labels[training], config=config
        )
        augmented_model = fit_logistic(
            augmented_train, domain.labels[training], config=config
        )
        baseline_oof[validation] = baseline_model.predict_proba(
            baseline_validation
        )
        augmented_oof[validation] = augmented_model.predict_proba(
            augmented_validation
        )
        _weights, positive_weight, negative_weight = _class_weights(
            domain.labels[training]
        )
        validation_rows = domain.row_indices[validation].astype(int).tolist()
        fold_reports.append(
            {
                "fold": fold,
                "training_count": int(training.sum()),
                "validation_count": int(validation.sum()),
                "positive_weight": positive_weight,
                "negative_weight": negative_weight,
                "baseline_feature_mean": baseline_mean.tolist(),
                "baseline_feature_scale": baseline_scale.tolist(),
                "candidate_mean": float(candidate_mean[0]),
                "candidate_scale": float(candidate_scale[0]),
                "independence_relative_residual": independence,
                "baseline_iterations": baseline_model.iterations,
                "augmented_iterations": augmented_model.iterations,
                "baseline_validation_rows": validation_rows,
                "augmented_validation_rows": list(validation_rows),
            }
        )
    if not np.isfinite(baseline_oof).all() or not np.isfinite(augmented_oof).all():
        raise ProbeInconclusiveError("OOF predictions must be finite and complete")
    summary = summarize_oof(
        domain.labels, baseline_oof, augmented_oof, folds
    )
    by_fold = {row["fold"]: row for row in summary["folds"]}
    for report in fold_reports:
        report.update(by_fold[report["fold"]])
    return CrossfitResult(
        fold_assignments=folds.copy(),
        baseline_oof=baseline_oof,
        augmented_oof=augmented_oof,
        folds=tuple(fold_reports),
        baseline=summary["baseline"],
        augmented=summary["augmented"],
        pooled_auroc_gain=summary["pooled_auroc_gain"],
    )


def _voxel_ids(centers, voxel_size):
    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3 or centers.shape[0] == 0:
        raise ProbeInconclusiveError("bootstrap centers must have shape [P,3]")
    if not np.isfinite(centers).all():
        raise ProbeInconclusiveError("bootstrap centers must be finite")
    if not np.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ProbeInconclusiveError("voxel size must be positive")
    return np.floor(centers / voxel_size).astype(np.int64)


def _pair_contribution_matrix(scores, labels, inverse, voxel_count):
    positives = [scores[(inverse == index) & labels] for index in range(voxel_count)]
    negatives = [
        np.sort(scores[(inverse == index) & ~labels])
        for index in range(voxel_count)
    ]
    contributions = np.zeros((voxel_count, voxel_count), dtype=np.float64)
    for positive_index, positive_scores in enumerate(positives):
        if positive_scores.size == 0:
            continue
        for negative_index, negative_scores in enumerate(negatives):
            if negative_scores.size == 0:
                continue
            left = np.searchsorted(negative_scores, positive_scores, side="left")
            right = np.searchsorted(negative_scores, positive_scores, side="right")
            contributions[positive_index, negative_index] = float(
                np.sum(left + 0.5 * (right - left))
            )
    return contributions


def _bootstrap_auc(multiplicity, contributions, positives, negatives):
    replicate_count = multiplicity.shape[0]
    result = np.empty(replicate_count, dtype=np.float64)
    for start in range(0, replicate_count, 32):
        stop = min(start + 32, replicate_count)
        current = multiplicity[start:stop].astype(np.float64, copy=False)
        positive_total = current @ positives
        negative_total = current @ negatives
        if np.any(positive_total == 0.0) or np.any(negative_total == 0.0):
            raise ProbeInconclusiveError(
                "bootstrap replicate requires both classes"
            )
        numerator = np.einsum(
            "bi,ij,bj->b", current, contributions, current, optimize=True
        )
        result[start:stop] = numerator / (positive_total * negative_total)
    return result


def paired_voxel_bootstrap(
    centers,
    labels,
    baseline_predictions,
    augmented_predictions,
    *,
    config,
):
    labels = np.asarray(labels)
    try:
        baseline = _vector("baseline bootstrap predictions", baseline_predictions)
        augmented = _vector("augmented bootstrap predictions", augmented_predictions)
    except ValueError as error:
        raise ProbeInconclusiveError(str(error)) from error
    if (
        labels.ndim != 1
        or not np.issubdtype(labels.dtype, np.bool_)
        or labels.shape != baseline.shape
        or augmented.shape != baseline.shape
    ):
        raise ProbeInconclusiveError("bootstrap inputs must share rows")
    _class_weights(labels)
    voxels = _voxel_ids(centers, config.voxel_size_m)
    if voxels.shape[0] != labels.size:
        raise ProbeInconclusiveError("bootstrap center row count mismatch")
    _unique, inverse = np.unique(voxels, axis=0, return_inverse=True)
    voxel_count = int(inverse.max()) + 1
    positives = np.bincount(
        inverse, weights=labels.astype(np.float64), minlength=voxel_count
    )
    negatives = np.bincount(
        inverse, weights=(~labels).astype(np.float64), minlength=voxel_count
    )
    rng = np.random.Generator(np.random.PCG64(config.bootstrap_seed))
    multiplicity = rng.multinomial(
        voxel_count,
        np.full(voxel_count, 1.0 / voxel_count, dtype=np.float64),
        size=config.bootstrap_replicates,
    )
    baseline_contribution = _pair_contribution_matrix(
        baseline, labels, inverse, voxel_count
    )
    augmented_contribution = _pair_contribution_matrix(
        augmented, labels, inverse, voxel_count
    )
    baseline_auc = _bootstrap_auc(
        multiplicity, baseline_contribution, positives, negatives
    )
    augmented_auc = _bootstrap_auc(
        multiplicity, augmented_contribution, positives, negatives
    )
    gains = augmented_auc - baseline_auc
    if not np.isfinite(gains).all():
        raise ProbeInconclusiveError("bootstrap AUROC gains must be finite")
    lower, upper = np.percentile(gains, config.interval_percentiles)
    return {
        "seed": int(config.bootstrap_seed),
        "replicate_count": int(config.bootstrap_replicates),
        "interval_percentiles": [
            float(config.interval_percentiles[0]),
            float(config.interval_percentiles[1]),
        ],
        "voxel_size_m": float(config.voxel_size_m),
        "voxel_origin": [0.0, 0.0, 0.0],
        "voxel_count": voxel_count,
        "lower": float(lower),
        "upper": float(upper),
        "auroc_gain_replicates": gains,
    }
