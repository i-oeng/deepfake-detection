"""Compare open-world predictions of candidate seeds against a baseline on one split.

Inputs are the JSONL predictions written by ``deepfake_detection.train.openworld``.
For every domain the script reports each model's macro video AUROC, the candidate
seed mean and spread, and a source-lineage grouped paired bootstrap of the
candidate-minus-baseline difference for every seed.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from deepfake_detection.train.evaluation import grouped_bootstrap_difference
from deepfake_detection.train.openworld import open_world_report


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _named(values: list[str]) -> dict[str, Path]:
    pairs = [value.split("=", 1) for value in values]
    if any(len(pair) != 2 for pair in pairs):
        raise ValueError("expected NAME=PATH")
    return {name: Path(path) for name, path in pairs}


def _spread(values: list[float]) -> dict[str, float]:
    return {"mean": statistics.fmean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values), "max": max(values)}


def summarize(baseline: tuple[str, Path], candidates: dict[str, Path], repeats: int) -> dict:
    base_name, base_path = baseline
    base = _read(base_path)
    base_report = open_world_report(base)
    reports = {name: open_world_report(_read(path)) for name, path in candidates.items()}
    domains = sorted(base_report["domain_macro_auroc"])
    result: dict = {"baseline": base_name, "candidates": sorted(candidates), "domains": {}}
    for domain in domains:
        entry = {
            "baseline_auroc": base_report["domain_macro_auroc"][domain],
            "candidate_auroc": {name: report["domain_macro_auroc"][domain]
                                for name, report in reports.items()},
            "by_method": {
                method: {"baseline": values["auroc"],
                         **{name: report["domains"][domain]["by_method"][method]["auroc"]
                            for name, report in reports.items()}}
                for method, values in base_report["domains"][domain]["by_method"].items()
            },
            "bootstrap": {},
        }
        entry["candidate_spread"] = _spread(list(entry["candidate_auroc"].values()))
        for name, path in candidates.items():
            entry["bootstrap"][name] = grouped_bootstrap_difference(
                base, _read(path), domain=domain, repeats=repeats)
        result["domains"][domain] = entry
    result["score"] = {
        "baseline": base_report["score"],
        "candidates": {name: report["score"] for name, report in reports.items()},
        "candidate_spread": _spread([report["score"] for report in reports.values()]),
    }
    result["family_mean_auroc"] = {
        "baseline": base_report["family_mean_auroc"],
        **{name: report["family_mean_auroc"] for name, report in reports.items()},
    }
    return result


def markdown(summary: dict) -> str:
    names = summary["candidates"]
    lines = [
        "| Domain | Baseline | " + " | ".join(names) + " | Mean ± SD | Seeds with CI > 0 |",
        "|---|---:|" + "---:|" * len(names) + "---:|---:|",
    ]
    for domain, entry in summary["domains"].items():
        positive = sum(result["ci95"][0] > 0 for result in entry["bootstrap"].values())
        spread = entry["candidate_spread"]
        lines.append(
            f"| {domain} | {entry['baseline_auroc']:.3f} | "
            + " | ".join(f"{entry['candidate_auroc'][name]:.3f}" for name in names)
            + f" | {spread['mean']:.3f} ± {spread['sd']:.3f} | {positive}/{len(names)} |"
        )
    score = summary["score"]
    lines.append(
        f"| **Mean of domains** | {score['baseline']:.3f} | "
        + " | ".join(f"{score['candidates'][name]:.3f}" for name in names)
        + f" | {score['candidate_spread']['mean']:.3f} ± {score['candidate_spread']['sd']:.3f}"
        + " | |"
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="NAME=PATH to baseline JSONL")
    parser.add_argument("--candidate", action="append", required=True,
                        help="NAME=PATH to a candidate seed's JSONL; repeat per seed")
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    baseline = next(iter(_named([args.baseline]).items()))
    summary = summarize(baseline, _named(args.candidate), args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    table = markdown(summary)
    args.output.with_suffix(".md").write_text(table, encoding="utf-8")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
