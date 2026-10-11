from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def equal_error_rate(labels: np.ndarray, scores: np.ndarray) -> float:
    false_positive, true_positive, _ = roc_curve(labels, scores, drop_intermediate=False)
    false_negative = 1.0 - true_positive
    index = int(np.nanargmin(np.abs(false_positive - false_negative)))
    return float((false_positive[index] + false_negative[index]) / 2.0)


def select_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Maximize validation balanced accuracy; ties use the smallest threshold.

    Sort once and count positives/negatives below each candidate instead of
    evaluating a full confusion matrix for every distinct probability.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if len(labels) != len(scores) or not len(labels):
        raise ValueError("Labels and scores must have the same nonzero length.")
    if not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Threshold fitting requires finite scores and binary labels.")
    candidates = np.unique(np.r_[0.0, scores, 1.0])
    order = np.argsort(scores, kind="mergesort")
    ordered_labels = labels[order]
    below = np.searchsorted(scores[order], candidates, side="left")
    positives_below = np.r_[0, np.cumsum(ordered_labels == 1)][below]
    negatives_below = below - positives_below
    positives = int(np.sum(labels == 1))
    negatives = len(labels) - positives
    if positives and negatives:
        values = ((positives - positives_below) / positives + negatives_below / negatives) / 2
    elif positives:
        values = (positives - positives_below) / positives
    else:
        values = negatives_below / negatives
    return float(candidates[int(np.argmax(values))])


def true_positive_rate_at_fpr(
    labels: np.ndarray, scores: np.ndarray, target_fpr: float
) -> float:
    # Keep every realizable threshold: dropping collinear ROC points can remove
    # the best point immediately below a fixed false-positive budget.
    false_positive, true_positive, _ = roc_curve(labels, scores, drop_intermediate=False)
    eligible = true_positive[false_positive <= target_fpr]
    return float(np.max(eligible)) if len(eligible) else 0.0


def positive_rate(scores: np.ndarray, threshold: float) -> float:
    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores >= threshold))


def classification_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    ranking_scores: np.ndarray | None = None,
) -> dict[str, float]:
    """Report ranking and fixed-threshold probability metrics separately.

    ``scores`` remain the probabilities used for Brier score and decisions at
    ``threshold``. Optional ``ranking_scores`` (for example logistic decision
    scores before sigmoid saturation) supply AUC, average precision, EER and
    test-ROC TPR at fixed FPR. Omission preserves probability ranking.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    ranking = scores if ranking_scores is None else np.asarray(ranking_scores)
    if any(array.ndim != 1 for array in (labels, scores, ranking)):
        raise ValueError("Metric arrays must be one-dimensional.")
    if not len(labels) or not (len(labels) == len(scores) == len(ranking)):
        raise ValueError("Labels, probabilities and ranking scores must have the same nonzero length.")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all() or not np.isfinite(ranking).all():
        raise ValueError("Metrics require binary labels and finite probability/ranking scores.")
    prediction = scores >= threshold
    tn, fp, _, _ = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(labels, ranking)),
        "average_precision": float(average_precision_score(labels, ranking)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "precision": float(precision_score(labels, prediction, zero_division=0)),
        "recall": float(recall_score(labels, prediction, zero_division=0)),
        "f1": float(f1_score(labels, prediction, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else float("nan"),
        "equal_error_rate": equal_error_rate(labels, ranking),
        "brier_score": float(brier_score_loss(labels, scores)),
        "tpr_at_fpr_1pct": true_positive_rate_at_fpr(labels, ranking, 0.01),
        "tpr_at_fpr_5pct": true_positive_rate_at_fpr(labels, ranking, 0.05),
        "threshold": float(threshold),
        "n": len(labels),
    }


def _cluster_members(labels: np.ndarray, groups: np.ndarray) -> dict[int, tuple[list[np.ndarray] | np.ndarray, bool]]:
    """Rows belonging to each class, grouped so a source is resampled as a unit."""
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    if len(labels) != len(groups):
        raise ValueError("Labels and cluster ids must have the same length.")
    if len(labels) == 0:
        return {}
    order = np.argsort(groups, kind="mergesort")
    sorted_groups = groups[order]
    boundaries = np.flatnonzero(sorted_groups[1:] != sorted_groups[:-1]) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, len(order)]
    buckets: dict[int, list[np.ndarray]] = {}
    for start, end in zip(starts, ends):
        member_labels = np.unique(labels[order[start:end]])
        if member_labels.size != 1:
            raise ValueError(
                f"Source group {sorted_groups[start]!r} contains multiple class labels."
            )
        buckets.setdefault(int(member_labels[0]), []).append(order[start:end])
    prepared: dict[int, tuple[list[np.ndarray] | np.ndarray, bool]] = {}
    for label, rows in buckets.items():
        singleton = all(len(item) == 1 for item in rows)
        if singleton:
            prepared[label] = (np.array([int(item[0]) for item in rows], dtype=np.int64), True)
        else:
            prepared[label] = ([np.asarray(item, dtype=np.int64) for item in rows], False)
    return prepared


def _draw_cluster_indices(
    members: dict[int, tuple[list[np.ndarray] | np.ndarray, bool]], rng: np.random.Generator
) -> np.ndarray:
    selected: list[np.ndarray] = []
    for label in sorted(members):
        rows, singleton = members[label]
        picked = rng.integers(0, len(rows), size=len(rows))
        if singleton:
            selected.append(np.asarray(rows)[picked])
        else:
            selected.append(np.concatenate([rows[int(index)] for index in picked]))
    return np.concatenate(selected)


def stratified_cluster_bootstrap_ci(
    labels: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float] = roc_auc_score,
    samples: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    members = _cluster_members(labels, groups)
    estimates = []
    for _ in range(samples):
        selected = _draw_cluster_indices(members, rng)
        estimates.append(metric(labels[selected], scores[selected]))
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return float(lower), float(upper)


def source_cluster_bootstrap_ci(
    labels: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float] = roc_auc_score,
    samples: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Resample whole sources, including sources with mixed outcome labels.

    Unlike the origin-label stratified bootstrap, this draws sources without
    class stratification. This preserves clean/edited pairs for attack and
    degradation outcomes. Draws lacking one outcome class are redrawn.
    """
    labels, scores, groups = map(np.asarray, (labels, scores, groups))
    if not len(labels) or not (len(labels) == len(scores) == len(groups)):
        raise ValueError("Labels, scores and cluster ids must have the same nonzero length.")
    if samples <= 0 or np.unique(labels).size < 2:
        raise ValueError("Bootstrap requires positive samples and both outcome classes.")
    order = np.argsort(groups, kind="mergesort")
    boundaries = np.flatnonzero(groups[order][1:] != groups[order][:-1]) + 1
    members = np.split(order, boundaries)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples * 20):
        picked = rng.integers(0, len(members), size=len(members))
        selected = np.concatenate([members[int(index)] for index in picked])
        if np.unique(labels[selected]).size < 2:
            continue
        value = float(metric(labels[selected], scores[selected]))
        if np.isfinite(value):
            estimates.append(value)
        if len(estimates) == samples:
            break
    if len(estimates) != samples:
        raise ValueError("Could not obtain the requested valid source bootstrap draws.")
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return float(lower), float(upper)


def paired_auc_difference_ci(
    labels: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    groups: np.ndarray | None = None,
    samples: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    if groups is None:
        groups = np.arange(len(labels))
    members = _cluster_members(labels, groups)
    differences = []
    for _ in range(samples):
        selected = _draw_cluster_indices(members, rng)
        differences.append(
            roc_auc_score(labels[selected], first[selected])
            - roc_auc_score(labels[selected], second[selected])
        )
    estimate = float(roc_auc_score(labels, first) - roc_auc_score(labels, second))
    lower, upper = np.percentile(differences, [2.5, 97.5])
    return estimate, float(lower), float(upper)
