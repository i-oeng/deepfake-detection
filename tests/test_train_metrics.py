from __future__ import annotations

import random

import pytest

from deepfake_detection.train import evaluation
from deepfake_detection.train.metrics import (
    aggregate_videos,
    auroc,
    average_precision,
    balanced_accuracy,
    calibration,
    equal_error_rate,
    f1_score,
    metric_report,
    tpr_at_fpr,
    validation_threshold,
)


def test_auroc_handles_ties_and_reversed_ranking() -> None:
    assert auroc([0, 1], [0.5, 0.5]) == 0.5
    assert auroc([0, 1], [0.1, 0.9]) == 1.0
    assert auroc([0, 1], [0.9, 0.1]) == 0.0
    with pytest.raises(ValueError, match="both binary classes"):
        auroc([1, 1], [0.1, 0.9])


def test_video_aggregation_and_validation_threshold() -> None:
    frames = [
        {"video_id": "real", "source_domain": "ff", "fake_method": "", "label": 0, "score": 0.1},
        {"video_id": "real", "source_domain": "ff", "fake_method": "", "label": 0, "score": 0.3},
        {"video_id": "fake", "source_domain": "ff", "fake_method": "sim", "label": 1, "score": 0.8},
    ]
    videos = aggregate_videos(frames)
    assert len(videos) == 2
    assert [row["frames"] for row in videos] == [1, 2]
    labels = [int(row["label"]) for row in videos]
    scores = [float(row["score"]) for row in videos]
    threshold = validation_threshold(labels, scores)
    assert balanced_accuracy(labels, scores, threshold) == 1.0
    assert metric_report(videos, threshold)["fake_recall_by_method"]["sim"]["recall"] == 1.0


def test_average_precision_uses_step_precision_and_ties() -> None:
    assert average_precision([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1]) == pytest.approx((1 + 2 / 3) / 2)
    assert average_precision([1, 0], [0.5, 0.5]) == 0.5
    with pytest.raises(ValueError, match="both classes"):
        average_precision([1, 1], [0.1, 0.9])


def test_tpr_at_fpr_respects_the_false_positive_budget() -> None:
    labels = [0, 0, 0, 0, 1, 1]
    scores = [0.1, 0.2, 0.3, 0.9, 0.8, 0.95]
    assert tpr_at_fpr(labels, scores, 0.0) == 0.5
    assert tpr_at_fpr(labels, scores, 0.25) == 1.0


def test_equal_error_rate_interpolates_the_roc_crossing() -> None:
    assert equal_error_rate([0, 1], [0.4, 0.6]) == 0.0
    assert equal_error_rate([0, 1], [0.6, 0.4]) == 1.0
    assert equal_error_rate([0, 1], [0.5, 0.5]) == 0.5
    assert equal_error_rate([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9]) == 0.5
    # One of three negatives outranks every positive: ROC (0,0) -> (1/3,0) -> (1/3,1) -> (1,1).
    assert equal_error_rate([0, 0, 0, 1], [0.9, 0.1, 0.2, 0.5]) == pytest.approx(1 / 3)


def test_f1_counts_threshold_ties_as_fake() -> None:
    assert f1_score([1, 1, 0, 0], [0.9, 0.4, 0.6, 0.1], 0.5) == 0.5
    assert f1_score([1, 0], [0.5, 0.1], 0.5) == 1.0
    assert f1_score([1, 0], [0.1, 0.2], 0.5) == 0.0


def test_calibration_reports_brier_and_binned_error() -> None:
    result = calibration([0, 1], [0.25, 0.75])
    assert result == pytest.approx({"brier": 0.0625, "ece_10": 0.25})


def _reference_average_precision(labels: list[int], scores: list[float]) -> float:
    """Pre-move implementation from train/evaluation.py, kept as an exact oracle."""
    ordered = sorted(zip(scores, labels, strict=True), reverse=True)
    true_positive = rank = 0
    area = 0.0
    for score in sorted(set(scores), reverse=True):
        group = [label for value, label in ordered if value == score]
        rank += len(group)
        gained = sum(group)
        true_positive += gained
        area += gained * true_positive / rank
    return area / sum(labels)


def _reference_tpr_at_fpr(labels: list[int], scores: list[float], maximum_fpr: float) -> float:
    positives, negatives = sum(labels), len(labels) - sum(labels)
    pairs = list(zip(labels, scores, strict=True))
    best = 0.0
    for threshold in sorted(set(scores), reverse=True):
        fp = sum(score >= threshold and not label for label, score in pairs)
        if fp / negatives <= maximum_fpr:
            tp = sum(score >= threshold and bool(label) for label, score in pairs)
            best = max(best, tp / positives)
    return best


def test_single_pass_metrics_match_previous_implementation_exactly() -> None:
    rng = random.Random(20261006)
    for _ in range(200):
        size = rng.randint(2, 60)
        labels = [rng.randint(0, 1) for _ in range(size)]
        labels[:2] = [0, 1]
        scores = [rng.choice([0.1, 0.25, 0.5, 0.75, rng.random()]) for _ in range(size)]
        assert average_precision(labels, scores) == _reference_average_precision(labels, scores)
        for budget in (0.0, 0.01, 0.05, 0.3):
            assert tpr_at_fpr(labels, scores, budget) == _reference_tpr_at_fpr(
                labels, scores, budget
            )
        eer = equal_error_rate(labels, scores)
        assert 0.0 <= eer <= 1.0


def test_frozen_evaluation_reuses_shared_metrics() -> None:
    assert evaluation.average_precision is average_precision
    assert evaluation.tpr_at_fpr is tpr_at_fpr
    assert evaluation.calibration is calibration


def test_metric_report_includes_threshold_free_and_locked_metrics() -> None:
    rows = [
        {"source_domain": "ff", "fake_method": "", "label": 0, "score": 0.2},
        {"source_domain": "ff", "fake_method": "", "label": 0, "score": 0.4},
        {"source_domain": "ff", "fake_method": "m", "label": 1, "score": 0.3},
        {"source_domain": "ff", "fake_method": "m", "label": 1, "score": 0.9},
    ]
    free = metric_report(rows)
    for key in ("auroc", "average_precision", "equal_error_rate", "tpr_at_1pct_fpr",
                "tpr_at_5pct_fpr", "brier", "ece_10"):
        assert 0.0 <= free[key] <= 1.0
    assert "f1" not in free
    assert metric_report(rows, 0.35)["f1"] == 0.5  # FAKE predicted for 0.4 and 0.9
