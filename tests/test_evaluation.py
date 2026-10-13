import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from aiwstudy.evaluation import (
    classification_metrics,
    paired_auc_difference_ci,
    positive_rate,
    select_threshold,
    source_cluster_bootstrap_ci,
    stratified_cluster_bootstrap_ci,
    true_positive_rate_at_fpr,
)


def test_metrics_and_bootstrap_are_finite():
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.05, 0.1, 0.4, 0.6, 0.8, 0.95])
    groups = np.array(["a", "b", "c", "d", "e", "f"])
    threshold = select_threshold(labels, scores)
    metrics = classification_metrics(labels, scores, threshold)
    lower, upper = stratified_cluster_bootstrap_ci(labels, scores, groups, samples=50, seed=3)
    assert metrics["roc_auc"] == 1.0
    assert 0 <= lower <= upper <= 1


def test_cluster_bootstrap_keeps_derivatives_together():
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.2, 0.3, 0.7, 0.8, 0.8, 0.9])
    groups = np.array(["r1", "r1", "r2", "r2", "f1", "f1", "f2", "f2"])
    lower, upper = stratified_cluster_bootstrap_ci(labels, scores, groups, samples=25, seed=9)
    assert lower == upper == 1.0


def test_positive_rate_on_negative_controls():
    assert positive_rate(np.array([0.1, 0.2, 0.9]), 0.5) == 1 / 3


def test_paired_auc_difference():
    labels = np.array([0, 0, 0, 1, 1, 1])
    stronger = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    weaker = np.array([0.1, 0.4, 0.8, 0.2, 0.6, 0.9])
    groups = np.array(["a", "b", "c", "d", "e", "f"])
    estimate, lower, upper = paired_auc_difference_ci(
        labels, stronger, weaker, groups, samples=50, seed=4
    )
    assert estimate > 0
    assert lower <= estimate <= upper


def test_fast_threshold_matches_exhaustive_with_ties():
    rng = np.random.default_rng(12)
    for _ in range(30):
        labels = np.r_[0, 1, rng.integers(0, 2, size=38)]
        scores = rng.integers(0, 17, size=len(labels)) / 16
        candidates = np.unique(np.r_[0.0, scores, 1.0])
        accuracies = [balanced_accuracy_score(labels, scores >= value) for value in candidates]
        expected = candidates[int(np.argmax(accuracies))]
        assert select_threshold(labels, scores) == expected
    # Every candidate ties: the smallest candidate remains the selected one.
    assert select_threshold(np.array([0, 1]), np.array([0.5, 0.5])) == 0.0


def test_mixed_label_source_bootstrap_resamples_complete_pairs():
    labels = np.array([0, 1, 0, 1, 0, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    groups = np.array(["a", "a", "b", "b", "c", "c"])
    draws = []

    def checked_auc(sample_labels, sample_scores):
        for first, second in ((0.1, 0.2), (0.3, 0.4), (0.5, 0.6)):
            assert np.sum(sample_scores == first) == np.sum(sample_scores == second)
        assert len(sample_scores) == len(scores)
        draws.append(sample_scores.copy())
        return roc_auc_score(sample_labels, sample_scores)

    interval = source_cluster_bootstrap_ci(labels, scores, groups, metric=checked_auc, samples=30, seed=7)
    assert len(draws) == 30
    assert 0 <= interval[0] <= interval[1] <= 1
    assert interval == source_cluster_bootstrap_ci(labels, scores, groups, samples=30, seed=7)


def test_stratified_bootstrap_still_rejects_mixed_label_sources():
    try:
        stratified_cluster_bootstrap_ci(
            np.array([0, 1]), np.array([0.1, 0.9]), np.array(["a", "a"]), samples=5
        )
    except ValueError as error:
        assert "multiple class labels" in str(error)
    else:
        raise AssertionError("Origin bootstrap must require homogeneous source labels.")


def test_tpr_budget_keeps_collinear_realizable_roc_points():
    # Each tied score admits one positive and one negative. The interior points
    # are collinear and sklearn removes some by default. At FPR <= 0.5 the
    # actual middle threshold reaches TPR 0.5 rather than the prior point 0.25.
    labels = np.tile([0, 1], 4)
    scores = np.repeat([0.9, 0.7, 0.5, 0.3], 2)
    assert true_positive_rate_at_fpr(labels, scores, 0.5) == 0.5


def test_tpr_budget_matches_all_observed_thresholds():
    rng = np.random.default_rng(31)
    for _ in range(25):
        labels = np.r_[np.zeros(60, dtype=int), np.ones(50, dtype=int)]
        scores = rng.integers(0, 12, size=len(labels)) / 11
        for budget in (0.01, 0.05, 0.1, 0.5):
            expected = 0.0
            for threshold in np.r_[np.inf, np.unique(scores)]:
                predicted = scores >= threshold
                if predicted[labels == 0].mean() <= budget:
                    expected = max(expected, predicted[labels == 1].mean())
            assert true_positive_rate_at_fpr(labels, scores, budget) == expected


def test_logit_ranking_avoids_probability_saturation_without_changing_decisions():
    labels = np.array([0, 0, 1, 1])
    logits = np.array([30.0, 31.0, 32.0, 33.0])
    probabilities = (1 / (1 + np.exp(-logits))).astype(np.float32)
    assert np.all(probabilities == 1.0)
    clipped = classification_metrics(labels, probabilities, 0.5)
    ranked = classification_metrics(labels, probabilities, 0.5, ranking_scores=logits)
    assert clipped["roc_auc"] == 0.5
    assert ranked["roc_auc"] == ranked["average_precision"] == 1.0
    assert ranked["equal_error_rate"] == 0.0
    assert ranked["tpr_at_fpr_1pct"] == ranked["tpr_at_fpr_5pct"] == 1.0
    for key in ("balanced_accuracy", "precision", "recall", "f1", "false_positive_rate", "brier_score", "threshold", "n"):
        assert clipped[key] == ranked[key]


def test_optional_ranking_scores_validate_shape_length_and_finiteness():
    labels, probabilities = np.array([0, 1]), np.array([0.2, 0.8])
    for invalid in (np.array([0.1]), np.array([0.1, np.inf]), np.array([[0.1, 0.9]])):
        try:
            classification_metrics(labels, probabilities, 0.5, ranking_scores=invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid ranking scores must be rejected.")
