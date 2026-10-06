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


def _descending_counts(labels: list[int], scores: list[float]) -> list[tuple[int, int]]:
    """Cumulative (true positive, false positive) counts at each distinct score, highest first.

    Each entry is the confusion state for the threshold ``score >= value``; tied scores
    always move together, so no ordering among ties is assumed.
    """
    ordered = sorted(zip(scores, labels, strict=True), reverse=True)
    counts: list[tuple[int, int]] = []
    true_positive = false_positive = 0
    index = 0
    while index < len(ordered):
        value = ordered[index][0]
        while index < len(ordered) and ordered[index][0] == value:
            if ordered[index][1]:
                true_positive += 1
            else:
                false_positive += 1
            index += 1
        counts.append((true_positive, false_positive))
    return counts


def average_precision(labels: list[int], scores: list[float]) -> float:
    """Tie-aware step-wise AP: mean precision at each distinct score, weighted by new positives."""
    positives = sum(labels)
    if positives == 0 or positives == len(labels):
        raise ValueError("average precision requires both classes")
    area = 0.0
    previous = 0
    for true_positive, false_positive in _descending_counts(labels, scores):
        area += (true_positive - previous) * true_positive / (true_positive + false_positive)
        previous = true_positive
    return area / positives


def tpr_at_fpr(labels: list[int], scores: list[float], maximum_fpr: float = 0.01) -> float:
    """Highest TPR at any score threshold whose FPR does not exceed ``maximum_fpr``."""
    positives, negatives = sum(labels), len(labels) - sum(labels)
    if not positives or not negatives:
        raise ValueError("TPR requires both classes")
    best = 0.0
    for true_positive, false_positive in _descending_counts(labels, scores):
        if false_positive / negatives <= maximum_fpr:
            best = max(best, true_positive / positives)
    return best


def equal_error_rate(labels: list[int], scores: list[float]) -> float:
    """Error rate where FPR equals FNR, linearly interpolated along the ROC curve."""
    positives, negatives = sum(labels), len(labels) - sum(labels)
    if not positives or not negatives:
        raise ValueError("EER requires both classes")
    previous_fpr, previous_gap = 0.0, -1.0  # threshold above every score: FPR 0, FNR 1
    for true_positive, false_positive in _descending_counts(labels, scores):
        fpr = false_positive / negatives
        gap = fpr - (1 - true_positive / positives)
        if gap >= 0:
            # FPR and FNR are both linear along the segment, so they cross where gap is 0.
            fraction = -previous_gap / (gap - previous_gap)
            return previous_fpr + fraction * (fpr - previous_fpr)
        previous_fpr, previous_gap = fpr, gap
    raise AssertionError("ROC curve must end at FPR 1, FNR 0")


def calibration(labels: list[int], scores: list[float], bins: int = 10) -> dict[str, float]:
    """Brier score and equal-width-bin expected calibration error for probability scores."""
    brier = sum((score - label) ** 2 for label, score in zip(labels, scores, strict=True)) / len(
        labels
    )
    groups: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for label, score in zip(labels, scores, strict=True):
        groups[min(int(score * bins), bins - 1)].append((label, score))
    ece = 0.0
    for group in groups.values():
        ece += (
            len(group)
            / len(labels)
            * abs(
                sum(score for _, score in group) / len(group)
                - sum(label for label, _ in group) / len(group)
            )
        )
    return {"brier": brier, "ece_10": ece}


def f1_score(labels: list[int], scores: list[float], threshold: float) -> float:
    """F1 for the FAKE class with ``score >= threshold`` predicted FAKE."""
    positives = sum(labels)
    if not positives:
        raise ValueError("F1 requires positive samples")
    predicted = [score >= threshold for score in scores]
    true_positive = sum(
        flag and bool(label) for flag, label in zip(predicted, labels, strict=True)
    )
    return 2 * true_positive / (sum(predicted) + positives)


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
    result: dict = {
        "samples": len(rows),
        "auroc": auroc(labels, scores),
        "average_precision": average_precision(labels, scores),
        "equal_error_rate": equal_error_rate(labels, scores),
        "tpr_at_1pct_fpr": tpr_at_fpr(labels, scores, 0.01),
        "tpr_at_5pct_fpr": tpr_at_fpr(labels, scores, 0.05),
        **calibration(labels, scores),
    }
    if threshold is not None:
        result["balanced_accuracy"] = balanced_accuracy(labels, scores, threshold)
        result["f1"] = f1_score(labels, scores, threshold)
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
