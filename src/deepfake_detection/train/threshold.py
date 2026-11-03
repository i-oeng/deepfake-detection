"""Round 4's validation-locked threshold and balanced accuracy on fresh v4.

From ``reports/openworld/round4_protocol.md`` ("Threshold", "Final evaluation"):

- The threshold maximizes the mean balanced accuracy over the target validation
  domains, each weighted equally, among thresholds whose false-positive rate on
  validation reals, with real sources weighted equally, is at most ``TARGET_FPR``.
  Of equally good thresholds the highest is kept, the one with the fewest false
  positives.
- Each test slice reports balanced accuracy at that threshold with a bootstrap of
  ``REPEATS`` replicates resampled within each label, optionally by group (for
  example by identity), and the false-positive rate of every real source in it.
- Round 4 succeeds when every primary slice reaches ``BAR`` with a 95% lower bound
  of at least ``LOWER_BAR`` and no real source in it exceeds ``TARGET_FPR``.

Scores are video scores (``evaluation.video_rows``: the mean frame score; a photo
is a one-frame video). A video is called fake when its score is at least the
threshold, as in ``metrics.balanced_accuracy``.
"""

from __future__ import annotations

import math
import random
import statistics
from bisect import bisect_left
from collections import defaultdict
from collections.abc import Callable

from .evaluation import video_rows

TARGET_FPR = 0.10
BAR = 0.85
LOWER_BAR = 0.80
REPEATS = 2000


def _called_fake(ordered: list[float], threshold: float) -> int:
    """How many of the sorted ``ordered`` scores are at least ``threshold``."""
    return len(ordered) - bisect_left(ordered, threshold)


def by_domain(records: list[dict]) -> dict[str, dict[int, list[float]]]:
    """Sorted video scores per domain and label (1 fake, 0 real), from frame records."""
    domains: dict[str, dict[int, list[float]]] = defaultdict(lambda: {0: [], 1: []})
    for video in video_rows(records):
        domains[video["source_domain"]][int(video["label"])].append(float(video["score"]))
    return {name: {label: sorted(scores) for label, scores in labels.items()}
            for name, labels in sorted(domains.items())}


def lock(domains: dict[str, dict[int, list[float]]], real_sources: dict[str, list[float]], *,
         target_fpr: float = TARGET_FPR) -> dict:
    """The protocol threshold from validation video scores.

    ``domains`` maps each target validation domain to its sorted real (0) and fake
    (1) scores; ``real_sources`` maps each validation real source to its sorted
    scores.
    """
    if not domains or any(not labels[0] or not labels[1] for labels in domains.values()):
        raise ValueError("every target domain needs real and fake videos")
    if not real_sources or any(not scores for scores in real_sources.values()):
        raise ValueError("every real source needs at least one video")
    reals = {name: sorted(scores) for name, scores in real_sources.items()}
    candidates = sorted({score for labels in domains.values() for scores in labels.values()
                         for score in scores} | {score for scores in reals.values()
                                                 for score in scores})
    candidates.append(math.inf)  # call nothing fake
    best = None
    for threshold in candidates:
        fpr = statistics.fmean(_called_fake(scores, threshold) / len(scores)
                               for scores in reals.values())
        if fpr > target_fpr:
            continue
        accuracy = statistics.fmean(
            (_called_fake(labels[1], threshold) / len(labels[1])
             + 1 - _called_fake(labels[0], threshold) / len(labels[0])) / 2
            for labels in domains.values())
        if best is None or accuracy >= best[0]:
            best = (accuracy, threshold, fpr)
    accuracy, threshold, fpr = best
    return {"threshold": threshold, "validation_mean_balanced_accuracy": accuracy,
            "validation_real_fpr": fpr, "target_fpr": target_fpr,
            "domains": sorted(domains), "real_sources": sorted(reals)}


def _quantile(values: list[float], share: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(share * (len(ordered) - 1))))]


def balanced_accuracy(videos: list[dict], threshold: float, *, repeats: int = REPEATS,
                      seed: str = "round4", group: Callable[[dict], str] | None = None) -> dict:
    """Balanced accuracy of ``videos`` at ``threshold`` with a within-label bootstrap.

    ``group`` names each video's resampling unit (by default the video itself), so
    videos of one identity are drawn together.
    """
    hits: dict[int, dict[str, list[int]]] = {0: defaultdict(list), 1: defaultdict(list)}
    for video in videos:
        label = int(video["label"])
        fake = float(video["score"]) >= threshold
        key = group(video) if group else video["video_id"]
        hits[label][key].append(int(fake == bool(label)))
    if not hits[0] or not hits[1]:
        raise ValueError("balanced accuracy needs real and fake videos")

    def rate(units: list[list[int]]) -> float:
        return sum(sum(unit) for unit in units) / sum(len(unit) for unit in units)

    units = {label: list(groups.values()) for label, groups in hits.items()}
    tpr, tnr = rate(units[1]), rate(units[0])
    rng = random.Random(seed)
    replicates = [(rate(rng.choices(units[1], k=len(units[1])))
                   + rate(rng.choices(units[0], k=len(units[0])))) / 2
                  for _ in range(repeats)]
    return {"balanced_accuracy": (tpr + tnr) / 2, "tpr": tpr, "tnr": tnr,
            "interval": [_quantile(replicates, 0.025), _quantile(replicates, 0.975)],
            "videos": {"fake": sum(map(len, units[1])), "real": sum(map(len, units[0]))},
            "groups": {"fake": len(units[1]), "real": len(units[0])}, "repeats": repeats}


def source_fprs(videos: list[dict], threshold: float,
                source: Callable[[dict], str]) -> dict[str, dict]:
    """False-positive rate at ``threshold`` of each real source among ``videos``."""
    reals: dict[str, list[float]] = defaultdict(list)
    for video in videos:
        if int(video["label"]) == 0:
            reals[source(video)].append(float(video["score"]))
    return {name: {"videos": len(scores),
                   "fpr": _called_fake(sorted(scores), threshold) / len(scores)}
            for name, scores in sorted(reals.items())}


def verdict(slices: dict[str, dict], primary: tuple[str, ...]) -> dict:
    """Round 4's success rule over the primary slices' ``balanced_accuracy`` and
    ``source_fprs`` results."""
    failures = []
    for name in primary:
        result = slices[name]
        accuracy, lower = result["balanced_accuracy"], result["interval"][0]
        if accuracy < BAR:
            failures.append(f"{name}: balanced accuracy {accuracy:.3f} < {BAR}")
        if lower < LOWER_BAR:
            failures.append(f"{name}: 95% lower bound {lower:.3f} < {LOWER_BAR}")
        for source, value in result["source_fprs"].items():
            if value["fpr"] > TARGET_FPR:
                failures.append(f"{name}: {source} false-positive rate {value['fpr']:.3f} "
                                f"> {TARGET_FPR}")
    return {"succeeded": not failures, "failures": failures, "bar": BAR,
            "lower_bar": LOWER_BAR, "target_fpr": TARGET_FPR, "primary": list(primary)}
