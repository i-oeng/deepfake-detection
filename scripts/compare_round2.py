"""Compare selected E8 and E7 seed predictions on the frozen fresh-v2 test."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

from deepfake_detection.train.evaluation import report as domain_report
from deepfake_detection.train.evaluation import report_videos, video_rows


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _key(row: dict) -> str:
    return str(row["sample_id"])


def ensemble(paths: list[Path]) -> tuple[list[dict], list[list[dict]]]:
    """Average seed logits after proving prediction alignment."""
    if not paths:
        raise ValueError("at least one seed prediction is required")
    seeds = [_read_jsonl(path) for path in paths]
    maps = [{_key(row): row for row in rows} for rows in seeds]
    if any(len(mapping) != len(rows) for mapping, rows in zip(maps, seeds, strict=True)):
        raise ValueError("a prediction file contains duplicate sample IDs")
    keys = set(maps[0])
    if any(set(mapping) != keys for mapping in maps[1:]):
        raise ValueError("seed predictions are not sample aligned")
    records = []
    compared = (
        "source_id", "split", "video_id", "source_domain", "fake_method",
        "manipulation_family", "label", "lineage_video_ids",
    )
    for sample_id in sorted(keys):
        items = [mapping[sample_id] for mapping in maps]
        for column in compared:
            if any(item[column] != items[0][column] for item in items[1:]):
                raise ValueError(f"seed metadata differs for {sample_id}/{column}")
        logit = statistics.fmean(float(item["logit"]) for item in items)
        records.append({
            **{column: items[0][column] for column in ("sample_id", *compared)},
            "logit": logit,
            "score": 1.0 / (1.0 + math.exp(-logit)),
        })
    return records, seeds


def _report(records: list[dict]) -> dict:
    domains = {
        domain: domain_report(records, domain=domain)
        for domain in sorted({row["source_domain"] for row in records})
    }
    return {
        "score": statistics.fmean(result["macro"]["auroc"] for result in domains.values()),
        "domain_macro_auroc": {
            domain: result["macro"]["auroc"] for domain, result in domains.items()
        },
        "domains": domains,
    }


def paired_domain_bootstrap(
    baseline: list[dict], candidate: list[dict], *, repeats: int = 2000, seed: int = 20261008
) -> dict:
    """Resample paired videos within every domain, label, and fake-method stratum."""
    base_videos, candidate_videos = video_rows(baseline), video_rows(candidate)

    def key(row: dict) -> tuple:
        return (
            row["source_domain"], row["video_id"], row["fake_method"], row["label"]
        )

    base_map = {key(row): row for row in base_videos}
    candidate_map = {key(row): row for row in candidate_videos}
    if base_map.keys() != candidate_map.keys():
        raise ValueError("paired bootstrap requires identical video keys")
    domains = sorted({key_value[0] for key_value in base_map})
    strata: dict[tuple, list[tuple]] = {}
    for domain in domains:
        domain_keys = [value for value in base_map if value[0] == domain]
        real = [value for value in domain_keys if value[3] == 0]
        if not real:
            raise ValueError(f"domain has no real videos: {domain}")
        strata[(domain, "real")] = real
        methods = sorted({value[2] for value in domain_keys if value[3] == 1})
        if not methods:
            raise ValueError(f"domain has no fake methods: {domain}")
        for method in methods:
            strata[(domain, method)] = [
                value for value in domain_keys if value[3] == 1 and value[2] == method
            ]
    observed = _report(candidate)["score"] - _report(baseline)["score"]
    rng = random.Random(seed)
    differences = []
    for _ in range(repeats):
        sampled: dict[str, list[tuple]] = {domain: [] for domain in domains}
        for (domain, _), values in strata.items():
            sampled[domain].extend(rng.choices(values, k=len(values)))
        base_score = statistics.fmean(
            report_videos([base_map[value] for value in sampled[domain]], domain=domain)[
                "macro"
            ]["auroc"]
            for domain in domains
        )
        candidate_score = statistics.fmean(
            report_videos([candidate_map[value] for value in sampled[domain]], domain=domain)[
                "macro"
            ]["auroc"]
            for domain in domains
        )
        differences.append(candidate_score - base_score)
    differences.sort()
    low = differences[math.floor(0.025 * (repeats - 1))]
    high = differences[math.ceil(0.975 * (repeats - 1))]
    return {
        "difference": observed,
        "ci95": [low, high],
        "replicates": repeats,
        "unit": "paired videos stratified by domain, label, and fake method",
        "clear_improvement": observed >= 0.05 and low > 0,
    }


def compare(baseline_paths: list[Path], candidate_paths: list[Path]) -> dict:
    baseline, baseline_seeds = ensemble(baseline_paths)
    candidate, candidate_seeds = ensemble(candidate_paths)
    if {_key(row) for row in baseline} != {_key(row) for row in candidate}:
        raise ValueError("E7 and E8 predictions are not sample aligned")
    base_map, candidate_map = {_key(row): row for row in baseline}, {
        _key(row): row for row in candidate
    }
    for sample_id in base_map:
        for column in (
            "source_id", "video_id", "source_domain", "fake_method", "label",
            "lineage_video_ids",
        ):
            if base_map[sample_id][column] != candidate_map[sample_id][column]:
                raise ValueError(f"E7 and E8 metadata differs for {sample_id}/{column}")
    baseline_report, candidate_report = _report(baseline), _report(candidate)
    baseline_seed_scores = [_report(records)["score"] for records in baseline_seeds]
    candidate_seed_scores = [_report(records)["score"] for records in candidate_seeds]
    return {
        "baseline_e7": baseline_report,
        "candidate_e8": candidate_report,
        "seed_spread": {
            "baseline_e7": baseline_seed_scores,
            "candidate_e8": candidate_seed_scores,
        },
        "bootstrap": paired_domain_bootstrap(baseline, candidate),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="append", type=Path, required=True)
    parser.add_argument("--candidate", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = compare(args.baseline, args.candidate)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output.exists() and args.output.read_text(encoding="utf-8") != payload:
        raise RuntimeError(f"refusing to overwrite changed comparison: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
