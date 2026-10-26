"""Apply the frozen round 3 validation rule to E8 and E9 seed runs.

E9 advances to the fresh v3 test only when, averaged over seeds, it

- lowers the validation real-source bias (mean video AUROC between FF++ reals and
  each other validation real source; 0.5 is unbiased) by at least 0.03,
- keeps the mean domain macro video AUROC no more than 0.01 below E8, and
- keeps FF++ validation AUROC no more than 0.01 below E8.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from deepfake_detection.train import source_bias

try:
    from select_round2 import summarize
except ModuleNotFoundError:  # imported by the tests as scripts.select_round3
    from scripts.select_round2 import summarize

MINIMUM_BIAS_DROP = 0.03
SCORE_GUARDRAIL = 0.01
FF_GUARDRAIL = 0.01


def _run(directory: Path) -> tuple[Path, Path]:
    report, predictions = directory / "report.json", directory / "validation.jsonl"
    if not report.is_file() or not predictions.is_file():
        raise FileNotFoundError(f"run lacks a report or validation predictions: {directory}")
    return report, predictions


def summarize_runs(directories: list[Path]) -> dict:
    runs = [_run(directory) for directory in directories]
    result = summarize([report for report, _ in runs])
    seeds = [source_bias.summarize(source_bias._read_jsonl(predictions))
             for _, predictions in runs]
    result["source_bias"] = {
        "mean_auroc": statistics.fmean(seed["validation_mean_auroc"] for seed in seeds),
        "by_source": {
            name: statistics.fmean(seed["validation"][name]["auroc_vs_reference"]
                                   for seed in seeds)
            for name in sorted(seeds[0]["validation"])
        },
        "seed_mean_auroc": [seed["validation_mean_auroc"] for seed in seeds],
    }
    return result


def select(baseline_runs: list[Path], candidate_runs: list[Path]) -> dict:
    baseline, candidate = summarize_runs(baseline_runs), summarize_runs(candidate_runs)
    if baseline["domain_mean_auroc"].keys() != candidate["domain_mean_auroc"].keys():
        raise ValueError("baseline and candidate use different validation domains")
    bias_drop = (baseline["source_bias"]["mean_auroc"]
                 - candidate["source_bias"]["mean_auroc"])
    score_change = candidate["score_mean"] - baseline["score_mean"]
    ff_change = candidate["domain_mean_auroc"]["ff"] - baseline["domain_mean_auroc"]["ff"]
    advance = (bias_drop >= MINIMUM_BIAS_DROP and score_change >= -SCORE_GUARDRAIL
               and ff_change >= -FF_GUARDRAIL)
    return {
        "advance_e9": advance,
        "rule": {"minimum_bias_drop": MINIMUM_BIAS_DROP,
                 "maximum_score_drop": SCORE_GUARDRAIL, "maximum_ff_auroc_drop": FF_GUARDRAIL},
        "bias_drop": bias_drop,
        "score_change": score_change,
        "ff_change": ff_change,
        "baseline_e8": baseline,
        "candidate_e9": candidate,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", action="append", type=Path, required=True)
    parser.add_argument("--candidate-run", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = select(args.baseline_run, args.candidate_run)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output.exists() and args.output.read_text(encoding="utf-8") != payload:
        raise RuntimeError(f"refusing to overwrite changed selection: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result["advance_e9"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
