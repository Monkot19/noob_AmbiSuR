"""Pure statistics for the read-only G1 prior-complementarity probe."""

from dataclasses import asdict, dataclass
import copy
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
        training_positive_count = int(domain.labels[training].sum())
        validation_positive_count = int(domain.labels[validation].sum())
        fold_reports.append(
            {
                "fold": fold,
                "training_count": int(training.sum()),
                "validation_count": int(validation.sum()),
                "training_positive_count": training_positive_count,
                "training_negative_count": int(training.sum()) - training_positive_count,
                "validation_positive_count": validation_positive_count,
                "validation_negative_count": int(validation.sum()) - validation_positive_count,
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


PROBE_OUTCOMES = (
    "INDEPENDENT_EVIDENCE_FEASIBLE",
    "NO_CLEAR_COMPLEMENT",
    "INCONCLUSIVE",
)

_REPORT_FIELDS = {
    "schema_version",
    "diagnostic_only",
    "training_started",
    "candidate",
    "baseline",
    "configuration",
    "gates",
    "provenance",
    "iterations",
    "outcome",
    "failed_gates",
    "inconclusive_reasons",
    "c1_authorized",
    "causal_claim",
    "cross_scene_claim",
}

_ITERATION_FIELDS = {
    "iteration",
    "role",
    "domain",
    "direction",
    "numerical_independence",
    "crossfit",
    "bootstrap",
}

_DOMAIN_FIELDS = {
    "original_point_count",
    "finite_center_count",
    "eligible_count",
    "positive_count",
    "negative_count",
    "coverage",
    "label",
}

_DIRECTION_FIELDS = {
    "lowest_quintile_count",
    "highest_quintile_count",
    "lowest_quintile_high_error_rate",
    "highest_quintile_high_error_rate",
    "high_minus_low_error_rate",
    "spearman_risk_distance",
    "marginal",
    "risk_bins",
}

_BOOTSTRAP_FIELDS = {
    "seed",
    "replicate_count",
    "interval_percentiles",
    "voxel_size_m",
    "voxel_origin",
    "voxel_count",
    "lower",
    "upper",
    "auroc_gain_replicates",
}

_INCONCLUSIVE_ITERATION_FIELDS = {"iteration", "role", "status", "reason"}

_PROVENANCE_FIELDS = {
    "diagnostic_commit",
    "formula_commit",
    "confirmation_id",
    "confirmation_sha256",
    "dataset_sha256",
    "aligned_prior_sha256",
    "gt_mesh_sha256",
    "run_identity_sha256",
}

_FROZEN_GATES = {
    "coverage_at_7000_at_least": 0.80,
    "direction_spearman_at_both_at_least": 0.0,
    "direction_high_minus_low_at_both_at_least": 0.0,
    "direction_high_minus_low_at_7000_at_least": 0.05,
    "fold_relative_residual_strictly_greater_than": 1e-8,
    "pooled_auroc_gain_at_3000_at_least": 0.0,
    "pooled_auroc_gain_at_7000_at_least": 0.02,
    "bootstrap_95pct_lower_at_7000_strictly_greater_than": 0.005,
}


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _exact_fields(value, expected, name):
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{name} fields mismatch")


def _finite_scalar(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _config_record(config):
    record = _json_safe(asdict(config))
    record["interval_percentiles"] = list(record["interval_percentiles"])
    return record


def _failed_gates(iterations):
    by_iteration = {row["iteration"]: row for row in iterations}
    early = by_iteration[3000]
    primary = by_iteration[7000]
    failures = []
    if primary["domain"]["coverage"] < 0.80:
        failures.append("7000_COVERAGE_BELOW_0.80")
    for row in (early, primary):
        iteration = row["iteration"]
        direction = row["direction"]
        if direction["highest_quintile_high_error_rate"] < direction["lowest_quintile_high_error_rate"]:
            failures.append(f"{iteration}_RAW_RISK_QUINTILE_DIRECTION_NEGATIVE")
        if direction["spearman_risk_distance"] < 0.0:
            failures.append(f"{iteration}_RAW_RISK_SPEARMAN_NEGATIVE")
        residuals = row["numerical_independence"]["fold_relative_residuals"]
        if any(value <= 1e-8 for value in residuals):
            failures.append(f"{iteration}_CANDIDATE_COLUMN_NOT_INDEPENDENT")
    if primary["direction"]["high_minus_low_error_rate"] < 0.05:
        failures.append("7000_RAW_RISK_SEPARATION_BELOW_0.05")
    if early["crossfit"]["pooled_auroc_gain"] < 0.0:
        failures.append("3000_POOLED_AUROC_GAIN_NEGATIVE")
    if primary["crossfit"]["pooled_auroc_gain"] < 0.02:
        failures.append("7000_POOLED_AUROC_GAIN_BELOW_0.02")
    if primary["bootstrap"]["lower"] <= 0.005:
        failures.append("7000_BOOTSTRAP_LOWER_NOT_GREATER_THAN_0.005")
    return failures


def build_probe_report(iterations, *, provenance, config, inconclusive_reasons=None):
    """Build the frozen, diagnostic-only three-state complementarity report."""
    iterations = _json_safe(copy.deepcopy(list(iterations)))
    reasons = [str(reason) for reason in (inconclusive_reasons or [])]
    report = {
        "schema_version": 1,
        "diagnostic_only": True,
        "training_started": False,
        "candidate": {"name": "one_minus_r_p", "expression": "1-r_p"},
        "baseline": ["A", "1-S"],
        "configuration": _config_record(config),
        "gates": dict(_FROZEN_GATES),
        "provenance": _json_safe(copy.deepcopy(provenance)),
        "iterations": iterations,
        "outcome": "INCONCLUSIVE",
        "failed_gates": [],
        "inconclusive_reasons": reasons,
        "c1_authorized": False,
        "causal_claim": None,
        "cross_scene_claim": None,
    }
    if not reasons:
        report["failed_gates"] = _failed_gates(iterations)
        report["outcome"] = (
            "NO_CLEAR_COMPLEMENT"
            if report["failed_gates"]
            else "INDEPENDENT_EVIDENCE_FEASIBLE"
        )
    return validate_probe_report(report)


def _validate_risk_bins(bins, iteration):
    if not isinstance(bins, list) or len(bins) != 20:
        raise ValueError(f"{iteration} risk bin count mismatch")
    expected = {
        "bin", "count", "risk_min", "risk_max", "mean_distance_m", "high_error_rate"
    }
    for index, row in enumerate(bins):
        _exact_fields(row, expected, f"{iteration} risk bin")
        if row["bin"] != index:
            raise ValueError(f"{iteration} risk bin order mismatch")
        if (
            isinstance(row["count"], bool)
            or not isinstance(row["count"], int)
            or row["count"] < 0
        ):
            raise ValueError(f"{iteration} risk bin count mismatch")
        for name in expected - {"bin", "count"}:
            _finite_scalar(row[name], f"{iteration} risk bin {name}")


def _validate_iteration(row, expected_iteration, expected_role, config):
    _exact_fields(row, _ITERATION_FIELDS, f"{expected_iteration} iteration")
    if row["iteration"] != expected_iteration or row["role"] != expected_role:
        raise ValueError("iteration role mismatch")
    _exact_fields(row["domain"], _DOMAIN_FIELDS, f"{expected_iteration} domain")
    domain = row["domain"]
    if domain["label"] != "distance_gt_0.05_m":
        raise ValueError("label contract mismatch")
    for name in ("original_point_count", "finite_center_count", "eligible_count", "positive_count", "negative_count"):
        if isinstance(domain[name], bool) or not isinstance(domain[name], int) or domain[name] < 0:
            raise ValueError(f"{expected_iteration} domain count mismatch")
    coverage = _finite_scalar(domain["coverage"], "coverage")
    if not 0.0 <= coverage <= 1.0:
        raise ValueError("coverage outside unit interval")
    if domain["eligible_count"] != domain["positive_count"] + domain["negative_count"]:
        raise ValueError("domain class counts mismatch")
    if (
        domain["finite_center_count"] <= 0
        or domain["finite_center_count"] > domain["original_point_count"]
        or domain["eligible_count"] > domain["finite_center_count"]
        or not math.isclose(
            coverage,
            domain["eligible_count"] / domain["finite_center_count"],
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise ValueError("domain coverage/count contract mismatch")
    _exact_fields(row["direction"], _DIRECTION_FIELDS, f"{expected_iteration} direction")
    for name in _DIRECTION_FIELDS - {"risk_bins", "marginal"}:
        _finite_scalar(row["direction"][name], f"direction {name}")
    _exact_fields(row["direction"]["marginal"], {"auroc", "auprc"}, "marginal metrics")
    _finite_scalar(row["direction"]["marginal"]["auroc"], "marginal AUROC")
    _finite_scalar(row["direction"]["marginal"]["auprc"], "marginal AUPRC")
    if not math.isclose(
        row["direction"]["high_minus_low_error_rate"],
        row["direction"]["highest_quintile_high_error_rate"]
        - row["direction"]["lowest_quintile_high_error_rate"],
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("direction separation mismatch")
    _validate_risk_bins(row["direction"]["risk_bins"], expected_iteration)
    risk_bins = row["direction"]["risk_bins"]
    if sum(bin_row["count"] for bin_row in risk_bins) != domain["eligible_count"]:
        raise ValueError("risk bin inventory mismatch")
    if (
        row["direction"]["lowest_quintile_count"]
        != sum(bin_row["count"] for bin_row in risk_bins[:4])
        or row["direction"]["highest_quintile_count"]
        != sum(bin_row["count"] for bin_row in risk_bins[-4:])
    ):
        raise ValueError("risk quintile inventory mismatch")
    independence = row["numerical_independence"]
    _exact_fields(independence, {"threshold_strictly_greater_than", "fold_relative_residuals"}, "numerical independence")
    if independence["threshold_strictly_greater_than"] != config.independence_threshold:
        raise ValueError("independence threshold mismatch")
    residuals = independence["fold_relative_residuals"]
    if not isinstance(residuals, list) or len(residuals) != config.fold_count:
        raise ValueError("independence fold count mismatch")
    for value in residuals:
        _finite_scalar(value, "independence residual")
    crossfit = row["crossfit"]
    _exact_fields(crossfit, {"folds", "baseline", "augmented", "pooled_auroc_gain"}, "crossfit")
    _finite_scalar(crossfit["pooled_auroc_gain"], "pooled AUROC gain")
    for name in ("baseline", "augmented"):
        _exact_fields(crossfit[name], {"auroc", "auprc"}, f"{name} metrics")
        _finite_scalar(crossfit[name]["auroc"], f"{name} AUROC")
        _finite_scalar(crossfit[name]["auprc"], f"{name} AUPRC")
    if not math.isclose(
        crossfit["pooled_auroc_gain"],
        crossfit["augmented"]["auroc"] - crossfit["baseline"]["auroc"],
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("pooled AUROC gain mismatch")
    folds = crossfit["folds"]
    if not isinstance(folds, list) or len(folds) != config.fold_count:
        raise ValueError("crossfit fold count mismatch")
    fold_fields = {
        "fold", "training_count", "validation_count",
        "training_positive_count", "training_negative_count",
        "validation_positive_count", "validation_negative_count",
        "baseline_auroc", "augmented_auroc", "auroc_gain",
        "candidate_relative_residual", "positive_weight", "negative_weight",
        "baseline_a_mean", "baseline_one_minus_s_mean",
        "baseline_a_scale", "baseline_one_minus_s_scale",
        "candidate_mean", "candidate_scale", "baseline_iterations",
        "augmented_iterations", "baseline_converged", "augmented_converged",
    }
    validation_count = 0
    validation_positive_count = 0
    validation_negative_count = 0
    for fold, fold_record in enumerate(folds):
        _exact_fields(fold_record, fold_fields, "fold")
        if fold_record["fold"] != fold:
            raise ValueError("fold order mismatch")
        count_fields = {
            "fold", "training_count", "validation_count",
            "training_positive_count", "training_negative_count",
            "validation_positive_count", "validation_negative_count",
            "baseline_iterations", "augmented_iterations",
        }
        boolean_fields = {"baseline_converged", "augmented_converged"}
        for name in fold_fields - count_fields - boolean_fields:
            _finite_scalar(fold_record[name], f"fold {name}")
        for name in count_fields - {"fold"}:
            if isinstance(fold_record[name], bool) or not isinstance(fold_record[name], int) or fold_record[name] < 0:
                raise ValueError(f"fold {name} mismatch")
        if fold_record["baseline_converged"] is not True or fold_record["augmented_converged"] is not True:
            raise ValueError("fold solver convergence mismatch")
        if (
            fold_record["training_count"]
            != fold_record["training_positive_count"] + fold_record["training_negative_count"]
            or fold_record["validation_count"]
            != fold_record["validation_positive_count"] + fold_record["validation_negative_count"]
        ):
            raise ValueError("fold class counts mismatch")
        if (
            fold_record["training_count"]
            != domain["eligible_count"] - fold_record["validation_count"]
            or fold_record["training_positive_count"]
            != domain["positive_count"] - fold_record["validation_positive_count"]
            or fold_record["training_negative_count"]
            != domain["negative_count"] - fold_record["validation_negative_count"]
        ):
            raise ValueError("fold training/validation complement mismatch")
        if (
            fold_record["training_positive_count"] == 0
            or fold_record["training_negative_count"] == 0
            or fold_record["validation_positive_count"] == 0
            or fold_record["validation_negative_count"] == 0
        ):
            raise ValueError("fold requires both classes")
        expected_positive_weight = fold_record["training_count"] / (2 * fold_record["training_positive_count"])
        expected_negative_weight = fold_record["training_count"] / (2 * fold_record["training_negative_count"])
        if not math.isclose(fold_record["positive_weight"], expected_positive_weight, rel_tol=0.0, abs_tol=1e-15) or not math.isclose(fold_record["negative_weight"], expected_negative_weight, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("fold class weight mismatch")
        if not math.isclose(
            fold_record["candidate_relative_residual"],
            residuals[fold],
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("fold independence residual mismatch")
        validation_count += fold_record["validation_count"]
        validation_positive_count += fold_record["validation_positive_count"]
        validation_negative_count += fold_record["validation_negative_count"]
    if (
        validation_count != domain["eligible_count"]
        or validation_positive_count != domain["positive_count"]
        or validation_negative_count != domain["negative_count"]
    ):
        raise ValueError("fold validation class inventory mismatch")
    bootstrap = row["bootstrap"]
    _exact_fields(bootstrap, _BOOTSTRAP_FIELDS, "bootstrap")
    if bootstrap["seed"] != config.bootstrap_seed or bootstrap["replicate_count"] != config.bootstrap_replicates:
        raise ValueError("bootstrap seed or replicate count mismatch")
    if bootstrap["interval_percentiles"] != list(config.interval_percentiles):
        raise ValueError("bootstrap interval mismatch")
    if bootstrap["voxel_size_m"] != config.voxel_size_m or bootstrap["voxel_origin"] != [0.0, 0.0, 0.0]:
        raise ValueError("bootstrap voxel contract mismatch")
    gains = bootstrap["auroc_gain_replicates"]
    if not isinstance(gains, list) or len(gains) != config.bootstrap_replicates:
        raise ValueError("bootstrap replicate inventory mismatch")
    for value in [bootstrap["lower"], bootstrap["upper"], *gains]:
        _finite_scalar(value, "bootstrap value")
    expected_lower, expected_upper = np.percentile(
        np.asarray(gains, dtype=np.float64), config.interval_percentiles
    )
    if not math.isclose(bootstrap["lower"], float(expected_lower), rel_tol=0.0, abs_tol=1e-15) or not math.isclose(bootstrap["upper"], float(expected_upper), rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("bootstrap interval mismatch")


def validate_probe_report(report):
    """Fail closed unless *report* exactly matches the frozen JSON contract."""
    _exact_fields(report, _REPORT_FIELDS, "report")
    if report["schema_version"] != 1:
        raise ValueError("report schema version mismatch")
    if report["diagnostic_only"] is not True or report["training_started"] is not False:
        raise ValueError("diagnostic-only scope mismatch")
    if report["candidate"] != {"name": "one_minus_r_p", "expression": "1-r_p"}:
        raise ValueError("candidate contract mismatch")
    if report["baseline"] != ["A", "1-S"]:
        raise ValueError("baseline contract mismatch")
    config = ProbeConfig()
    if report["configuration"] != _config_record(config):
        raise ValueError("configuration constants mismatch")
    if report["gates"] != _FROZEN_GATES:
        raise ValueError("gate constants mismatch")
    _exact_fields(report["provenance"], _PROVENANCE_FIELDS, "provenance")
    provenance = report["provenance"]
    for name in ("diagnostic_commit", "formula_commit"):
        value = provenance[name]
        if not isinstance(value, str) or len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"provenance {name} mismatch")
    for name in _PROVENANCE_FIELDS - {"diagnostic_commit", "formula_commit", "confirmation_id"}:
        value = provenance[name]
        if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"provenance {name} mismatch")
    if not isinstance(provenance["confirmation_id"], str) or not provenance["confirmation_id"]:
        raise ValueError("provenance confirmation ID mismatch")
    if not isinstance(report["iterations"], list) or len(report["iterations"]) != 2:
        raise ValueError("iteration inventory mismatch")
    expected_iterations = ((3000, "direction_stability"), (7000, "primary"))
    placeholder_iterations = all(
        isinstance(row, dict) and set(row) == _INCONCLUSIVE_ITERATION_FIELDS
        for row in report["iterations"]
    )
    if placeholder_iterations:
        for row, (iteration, role) in zip(report["iterations"], expected_iterations):
            if (
                row["iteration"] != iteration
                or row["role"] != role
                or row["status"] != "INCONCLUSIVE"
                or not isinstance(row["reason"], str)
                or not row["reason"]
            ):
                raise ValueError("inconclusive iteration record mismatch")
    else:
        _validate_iteration(report["iterations"][0], 3000, "direction_stability", config)
        _validate_iteration(report["iterations"][1], 7000, "primary", config)
    if report["c1_authorized"] is not False:
        raise ValueError("C1 authorization is forbidden")
    if report["causal_claim"] is not None:
        raise ValueError("causal claim is forbidden")
    if report["cross_scene_claim"] is not None:
        raise ValueError("cross-scene claim is forbidden")
    if report["outcome"] not in PROBE_OUTCOMES:
        raise ValueError("outcome mismatch")
    if not isinstance(report["failed_gates"], list) or not all(isinstance(value, str) for value in report["failed_gates"]):
        raise ValueError("failed gates mismatch")
    reasons = report["inconclusive_reasons"]
    if not isinstance(reasons, list) or not all(isinstance(value, str) and value for value in reasons):
        raise ValueError("inconclusive reasons mismatch")
    expected_failures = [] if placeholder_iterations else _failed_gates(report["iterations"])
    if reasons:
        if report["outcome"] != "INCONCLUSIVE" or report["failed_gates"]:
            raise ValueError("inconclusive outcome mismatch")
    else:
        if placeholder_iterations:
            raise ValueError("inconclusive iteration records require reasons")
        expected_outcome = "NO_CLEAR_COMPLEMENT" if expected_failures else "INDEPENDENT_EVIDENCE_FEASIBLE"
        if report["outcome"] != expected_outcome or report["failed_gates"] != expected_failures:
            raise ValueError("decision outcome mismatch")
    return report


def probe_exit_code(report):
    validate_probe_report(report)
    return {
        "INDEPENDENT_EVIDENCE_FEASIBLE": 0,
        "NO_CLEAR_COMPLEMENT": 1,
        "INCONCLUSIVE": 2,
    }[report["outcome"]]


def fold_rows(report):
    validate_probe_report(report)
    rows = []
    for iteration in report["iterations"]:
        if iteration.get("status") == "INCONCLUSIVE":
            continue
        for fold in iteration["crossfit"]["folds"]:
            rows.append({"iteration": iteration["iteration"], **copy.deepcopy(fold)})
    return rows


def risk_bin_rows(report):
    validate_probe_report(report)
    rows = []
    for iteration in report["iterations"]:
        if iteration.get("status") == "INCONCLUSIVE":
            continue
        for row in iteration["direction"]["risk_bins"]:
            rows.append({"iteration": iteration["iteration"], **copy.deepcopy(row)})
    return rows


def bootstrap_rows(report):
    validate_probe_report(report)
    rows = []
    for iteration in report["iterations"]:
        if iteration.get("status") == "INCONCLUSIVE":
            continue
        for replicate, value in enumerate(iteration["bootstrap"]["auroc_gain_replicates"]):
            rows.append({
                "iteration": iteration["iteration"],
                "replicate": replicate,
                "auroc_gain": value,
            })
    return rows
