from __future__ import annotations

import pytest

from deepfake_detection.train.metrics import (
    aggregate_videos,
    auroc,
    balanced_accuracy,
    metric_report,
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
