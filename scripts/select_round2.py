"""Apply the frozen Round 2 validation rule to E7 and E8 seed reports."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

STABLE_DOMAIN = "stable_cascade_lite"
FF_GUARDRAIL = 0.01


def _read(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    validation = report.get("validation")
    if not isinstance(validation, dict) or not isinstance(
        validation.get("domain_macro_auroc"), dict
    ):
        raise ValueError(f"report has no validation metrics: {path}")
    return report


def summarize(paths: list[Path]) -> dict:
    if not paths:
        raise ValueError("at least one seed report is required")
    reports = [_read(path) for path in paths]
    domains = [set(report["validation"]["domain_macro_auroc"]) for report in reports]
    if any(value != domains[0] for value in domains[1:]):
        raise ValueError("seed reports have different validation domains")
    manifest_ids = {report["manifest_id"] for report in reports}
    if len(manifest_ids) != 1:
        raise ValueError("seed reports use different manifests")
    scores = [float(report["validation"]["score"]) for report in reports]
    by_domain = {
        domain: statistics.fmean(
            float(report["validation"]["domain_macro_auroc"][domain]) for report in reports
        )
        for domain in sorted(domains[0])
    }
    original = [value for domain, value in by_domain.items() if domain != STABLE_DOMAIN]
    if STABLE_DOMAIN not in by_domain or "ff" not in by_domain or not original:
        raise ValueError("Round 2 validation domains are incomplete")
    return {
        "manifest_id": manifest_ids.pop(),
        "seeds": len(reports),
        "score_mean": statistics.fmean(scores),
        "score_min": min(scores),
        "score_max": max(scores),
        "score_stdev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        "domain_mean_auroc": by_domain,
        "original_v1_domain_mean": statistics.fmean(original),
        "reports": [str(path) for path in paths],
    }


def select(baseline_paths: list[Path], candidate_paths: list[Path]) -> dict:
    baseline, candidate = summarize(baseline_paths), summarize(candidate_paths)
    if baseline["manifest_id"] != candidate["manifest_id"]:
        raise ValueError("baseline and candidate use different manifests")
    if baseline["domain_mean_auroc"].keys() != candidate["domain_mean_auroc"].keys():
        raise ValueError("baseline and candidate use different validation domains")
    score_gain = candidate["score_mean"] - baseline["score_mean"]
    ff_change = candidate["domain_mean_auroc"]["ff"] - baseline["domain_mean_auroc"]["ff"]
    advance = score_gain > 0 and ff_change >= -FF_GUARDRAIL
    return {
        "advance_e8": advance,
        "rule": {
            "mean_score_must_exceed_e7": True,
            "maximum_ff_auroc_drop": FF_GUARDRAIL,
            "tie_breakers": [STABLE_DOMAIN, "original_v1_domain_mean"],
        },
        "score_gain": score_gain,
        "ff_change": ff_change,
        "baseline_e7": baseline,
        "candidate_e8": candidate,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", action="append", type=Path, required=True)
    parser.add_argument("--candidate-report", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = select(args.baseline_report, args.candidate_report)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output.exists() and args.output.read_text(encoding="utf-8") != payload:
        raise RuntimeError(f"refusing to overwrite changed selection: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result["advance_e8"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
