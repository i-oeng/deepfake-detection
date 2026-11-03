import math

import pytest

from deepfake_detection.train import threshold as locked


def _frames(domain: str, label: int, scores: list[float], prefix: str = "v") -> list[dict]:
    return [{"source_domain": domain, "video_id": f"{domain}:{prefix}{i}", "fake_method":
             "m" if label else "", "label": label, "score": score}
            for i, score in enumerate(scores)]


def test_by_domain_averages_frames_into_sorted_video_scores() -> None:
    frames = _frames("a", 1, [0.9, 0.7]) + _frames("a", 0, [0.2]) + [
        {"source_domain": "a", "video_id": "a:v0", "fake_method": "m", "label": 1,
         "score": 0.5}]
    assert locked.by_domain(frames) == {"a": {0: [0.2], 1: [0.7, 0.7]}}


def test_lock_maximizes_mean_balanced_accuracy_under_the_fpr_cap() -> None:
    domains = {"easy": {0: [0.1, 0.2, 0.3, 0.4], 1: [0.6, 0.7, 0.8, 0.9]},
               "hard": {0: [0.1, 0.2, 0.5, 0.55], 1: [0.3, 0.45, 0.65, 0.95]}}
    # Uncapped, 0.6 is best: easy 1.0, hard (0.5 + 1.0) / 2, mean 0.875.
    loose = locked.lock(domains, {"r": [0.1]}, target_fpr=1.0)
    assert loose["threshold"] == 0.6
    assert loose["validation_mean_balanced_accuracy"] == pytest.approx(0.875)
    # Source x calls 0.62 and 0.7 fake below 0.8; with y at 0, the mean FPR tops 10%.
    capped = locked.lock(domains, {"x": [0.3, 0.5, 0.62, 0.7], "y": [0.2, 0.4]})
    assert capped["threshold"] == 0.8 and capped["validation_real_fpr"] == 0
    assert capped["validation_mean_balanced_accuracy"] == pytest.approx(0.6875)
    assert locked.lock({"d": {0: [0.5], 1: [0.4]}}, {"r": [0.9]})["threshold"] == math.inf


def test_balanced_accuracy_bootstraps_within_labels_and_groups() -> None:
    videos = ([{"video_id": f"f{i}", "label": 1, "score": 0.9, "who": f"p{i % 3}"}
               for i in range(9)]
              + [{"video_id": f"r{i}", "label": 0, "score": 0.1 if i else 0.95, "who": "q"}
                 for i in range(10)])
    result = locked.balanced_accuracy(videos, 0.5, repeats=200)
    assert result["balanced_accuracy"] == pytest.approx((1 + 0.9) / 2)
    assert result["interval"][0] <= 0.95 <= result["interval"][1] <= 1
    grouped = locked.balanced_accuracy(videos, 0.5, repeats=200, group=lambda v: v["who"])
    assert grouped["groups"] == {"fake": 3, "real": 1}
    assert grouped["interval"] == [0.95, 0.95]  # one real group: no resampling spread


def test_source_fprs_and_verdict() -> None:
    videos = [{"label": 0, "score": s, "src": src} for s, src in
              ((0.9, "a"), (0.1, "a"), (0.2, "b"), (0.3, "b"))] + [{"label": 1, "score": 1}]
    fprs = locked.source_fprs(videos, 0.5, lambda v: v["src"])
    assert fprs == {"a": {"videos": 2, "fpr": 0.5}, "b": {"videos": 2, "fpr": 0.0}}
    good = {"balanced_accuracy": 0.9, "interval": [0.86, 0.93],
            "source_fprs": {"b": fprs["b"]}}
    weak = {"balanced_accuracy": 0.84, "interval": [0.79, 0.88], "source_fprs": fprs}
    assert locked.verdict({"s": good}, ("s",))["succeeded"]
    failures = locked.verdict({"s": good, "t": weak}, ("s", "t"))["failures"]
    assert len(failures) == 3 and all(f.startswith("t:") for f in failures)
