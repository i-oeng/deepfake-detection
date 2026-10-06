"""Binary classification metrics without a dependency on scikit-learn."""

from __future__ import annotations

from collections import defaultdict


def auroc(labels: list[int], scores: list[float]) -> float:
    """Tie-aware rank AUROC; both classes must be present."""
    if len(labels) != len(scores) or not labels:
        raise ValueError("labels and scores must be nonempty and equal length")
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0 or any(label not in (0, 1) for label in labels):
        raise ValueError("AUROC requires both binary classes")
    ordered = sorted(zip(scores, labels, strict=True))
    positive_ranks = 0.0
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        average_rank = (start + 1 + end) / 2
        positive_ranks += average_rank * sum(label for _, label in ordered[start:end])
        start = end
    return (positive_ranks - positives * (positives + 1) / 2) / (positives * negatives)


def balanced_accuracy(labels: list[int], scores: list[float], threshold: float) -> float:
    positive = [score >= threshold for label, score in zip(labels, scores, strict=True) if label]
    negative = [score < threshold for label, score in zip(labels, scores, strict=True) if not label]
    if not positive or not negative:
        raise ValueError("balanced accuracy requires both classes")
    return (sum(positive) / len(positive) + sum(negative) / len(negative)) / 2


def validation_threshold(labels: list[int], scores: list[float]) -> float:
    """Select the balanced-accuracy threshold on validation predictions only."""
    candidates = sorted(set(scores))
    if not candidates:
        raise ValueError("empty validation scores")
    return max(candidates, key=lambda value: (balanced_accuracy(labels, scores, value), -value))


def aggregate_videos(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Average frame probabilities for each domain/video/method/label group."""
    grouped: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["source_domain"]), str(row["video_id"]),
            str(row["fake_method"]), int(row["label"]),
        )
        grouped[key].append(float(row["score"]))
    return [
        {
            "source_domain": domain, "video_id": video, "fake_method": method,
            "label": label, "score": sum(scores) / len(scores), "frames": len(scores),
        }
        for (domain, video, method, label), scores in sorted(grouped.items())
    ]


def metric_report(rows: list[dict[str, object]], threshold: float | None = None) -> dict:
    labels = [int(row["label"]) for row in rows]
    scores = [float(row["score"]) for row in rows]
    result: dict = {"samples": len(rows), "auroc": auroc(labels, scores)}
    if threshold is not None:
        result["balanced_accuracy"] = balanced_accuracy(labels, scores, threshold)
        result["threshold"] = threshold
    by_domain: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_method: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_domain[str(row["source_domain"])].append(row)
        if int(row["label"]) == 1:
            by_method[str(row["fake_method"])].append(float(row["score"]))
    result["by_domain"] = {}
    for domain, group in sorted(by_domain.items()):
        local_labels = [int(row["label"]) for row in group]
        local_scores = [float(row["score"]) for row in group]
        result["by_domain"][domain] = {
            "samples": len(group),
            "auroc": auroc(local_labels, local_scores) if len(set(local_labels)) == 2 else None,
        }
    result["fake_recall_by_method"] = {
        method: {"samples": len(values), "recall": (
            sum(score >= threshold for score in values) / len(values)
            if threshold is not None else None
        )}
        for method, values in sorted(by_method.items())
    }
    return result
