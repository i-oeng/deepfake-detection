"""Frozen video-level metrics for unseen-method DF40 evaluation."""

from __future__ import annotations

import math
import random
from collections import defaultdict

from .metrics import auroc, average_precision, calibration, tpr_at_fpr


def _metrics(rows: list[dict]) -> dict:
    labels = [int(row["label"]) for row in rows]
    scores = [float(row["score"]) for row in rows]
    if len(set(labels)) != 2:
        raise ValueError("each reported slice needs real and fake videos")
    return {
        "videos": len(rows),
        "auroc": auroc(labels, scores),
        "average_precision": average_precision(labels, scores),
        "tpr_at_1pct_fpr": tpr_at_fpr(labels, scores),
        **calibration(labels, scores),
    }


def video_rows(frames: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in frames:
        key = (row["source_domain"], row["video_id"], row["fake_method"], int(row["label"]))
        grouped[key].append(row)
    output = []
    for (domain, video, method, label), items in sorted(grouped.items()):
        lineage = {row.get("lineage_video_ids", "") for row in items}
        if len(lineage) != 1:
            raise ValueError(f"inconsistent lineage for {video}")
        output.append(
            {
                "source_domain": domain,
                "video_id": video,
                "fake_method": method,
                "label": label,
                "score": sum(float(row["score"]) for row in items) / len(items),
                "lineage_video_ids": lineage.pop(),
                "frames": len(items),
            }
        )
    return output


def report(frames: list[dict], *, domain: str = "ff") -> dict:
    """Compare each fake method with the same-domain real video pool."""
    return report_videos(video_rows(frames), domain=domain)


def report_videos(video_predictions: list[dict], *, domain: str = "ff") -> dict:
    videos = [row for row in video_predictions if row["source_domain"] == domain]
    real = [row for row in videos if row["label"] == 0]
    methods = sorted({row["fake_method"] for row in videos if row["label"] == 1})
    if not real or not methods:
        raise ValueError(f"missing real or fake videos in domain {domain}")
    by_method = {
        method: _metrics(real + [row for row in videos if row["fake_method"] == method])
        for method in methods
    }
    macro = {
        metric: sum(values[metric] for values in by_method.values()) / len(by_method)
        for metric in ("auroc", "average_precision", "tpr_at_1pct_fpr", "brier", "ece_10")
    }
    return {
        "domain": domain,
        "videos": len(videos),
        "real_videos": len(real),
        "by_method": by_method,
        "macro": macro,
    }


def grouped_bootstrap_difference(
    baseline: list[dict],
    candidate: list[dict],
    *,
    domain: str = "ff",
    repeats: int = 2000,
    seed: int = 20261006,
) -> dict:
    """Resample source-video lineage components and report paired macro-AUROC CI."""
    base = video_rows(baseline)
    cand = video_rows(candidate)

    def key(row: dict) -> tuple:
        return (
            row["source_domain"],
            row["video_id"],
            row["fake_method"],
            row["label"],
        )

    base_map, cand_map = {key(row): row for row in base}, {key(row): row for row in cand}
    if base_map.keys() != cand_map.keys():
        raise ValueError("paired bootstrap requires identical video keys")
    selected = sorted(k for k in base_map if k[0] == domain)
    # A swap with two parent clips joins their lineage components. Resampling
    # these components keeps source-related real and fake videos together.
    parent: dict[str, str] = {}

    def root(value: str) -> str:
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = root(parent[value])
        return parent[value]

    for k in selected:
        tokens = base_map[k]["lineage_video_ids"].split("|")
        tokens = [token for token in tokens if token]
        if not tokens:
            raise ValueError("bootstrap requires source-video lineage for every row")
        for token in tokens[1:]:
            parent[root(token)] = root(tokens[0])
    clusters: dict[str, list[tuple]] = defaultdict(list)
    for k in selected:
        first = base_map[k]["lineage_video_ids"].split("|")[0]
        clusters[root(first)].append(k)
    cluster_keys = sorted(clusters)
    if len(cluster_keys) < 2:
        raise ValueError("fewer than two independent source-video groups")
    observed = (
        report(candidate, domain=domain)["macro"]["auroc"]
        - report(baseline, domain=domain)["macro"]["auroc"]
    )
    rng = random.Random(seed)
    differences = []
    expected_methods = {k[2] for k in selected if k[3] == 1}
    for _ in range(repeats):
        sampled = [rng.choice(cluster_keys) for _ in cluster_keys]
        keys = [k for cluster in sampled for k in clusters[cluster]]
        if {k[2] for k in keys if k[3] == 1} != expected_methods:
            continue
        try:
            a = report_videos([base_map[k] for k in keys], domain=domain)["macro"]["auroc"]
            b = report_videos([cand_map[k] for k in keys], domain=domain)["macro"]["auroc"]
        except ValueError:
            continue
        differences.append(b - a)
    if len(differences) < repeats // 2:
        raise ValueError("too few valid grouped bootstrap replicates")
    differences.sort()
    low = differences[math.floor(0.025 * (len(differences) - 1))]
    high = differences[math.ceil(0.975 * (len(differences) - 1))]
    return {
        "difference": observed,
        "ci95": [low, high],
        "valid_replicates": len(differences),
        "source_groups": len(cluster_keys),
        "clear_improvement": observed >= 0.05 and low > 0,
    }
