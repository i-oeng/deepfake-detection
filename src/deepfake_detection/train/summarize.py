"""Summarize three frozen-protocol seeds without changing model selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from .evaluation import report
from .fusion import align, load_export

METRICS = ("auroc", "average_precision", "tpr_at_1pct_fpr", "brier", "ece_10")


def _spread(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sample_stddev": statistics.stdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def summarize(run_dirs: list[Path], split: str, output: Path) -> dict:
    if len(run_dirs) != 3:
        raise ValueError("a finalist summary requires exactly three run directories")
    exports = [load_export(directory, split) for directory in run_dirs]
    rows = [item[0] for item in exports]
    metadata = [item[1] for item in exports]
    for candidate in rows[1:]:
        align(rows[0], candidate)
    provenance_keys = (
        "manifest_id",
        "manifest_sha256",
        "sample_preprocessing_id",
        "preprocessing_id",
        "model",
    )
    reference = metadata[0]
    if any(
        any(candidate.get(key) != reference.get(key) for key in provenance_keys)
        for candidate in metadata[1:]
    ):
        raise ValueError("seed runs do not share model, manifest, and preprocessing provenance")
    seeds = [int(item["seed"]) for item in metadata]
    if len(set(seeds)) != 3:
        raise ValueError("seed runs must have three distinct seeds")

    reports = [
        {domain: report(predictions, domain=domain) for domain in ("ff", "cdf")}
        for predictions in rows
    ]
    spread: dict[str, dict] = {}
    for domain in ("ff", "cdf"):
        methods = sorted(reports[0][domain]["by_method"])
        spread[domain] = {
            "macro": {
                metric: _spread([result[domain]["macro"][metric] for result in reports])
                for metric in METRICS
            },
            "by_method": {
                method: {
                    metric: _spread(
                        [result[domain]["by_method"][method][metric] for result in reports]
                    )
                    for metric in METRICS
                }
                for method in methods
            },
        }
    result = {
        "split": split,
        "model": reference["model"],
        "manifest_id": reference["manifest_id"],
        "manifest_sha256": reference["manifest_sha256"],
        "seeds": seeds,
        "runs": [
            {
                "directory": str(directory),
                "seed": seed,
                "checkpoint_sha256": item["checkpoint_sha256"],
                "metrics": metrics,
            }
            for directory, seed, item, metrics in zip(
                run_dirs, seeds, metadata, reports, strict=True
            )
        ],
        "spread": spread,
    }
    content = json.dumps(result, indent=2, sort_keys=True) + "\n"
    digest = hashlib.sha256(content.encode()).hexdigest()[:20]
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{reference['model']}-{split}-three-seed-{digest}.json"
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise RuntimeError(f"refusing to overwrite changed seed summary: {path}")
    path.write_text(content, encoding="utf-8")
    return {"report": str(path), "model": reference["model"], "seeds": seeds}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output", type=Path, default=Path("reports/seed_spread"))
    args = parser.parse_args(argv)
    print(json.dumps(summarize(args.run, args.split, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
