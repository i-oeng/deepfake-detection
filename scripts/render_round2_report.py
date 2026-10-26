"""Render the frozen Round 2 result as Markdown from immutable JSON reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(value: float) -> str:
    return f"{float(value):.3f}"


def render(selection: dict, comparison: dict, release: dict, audit: dict) -> str:
    e7, e8 = comparison["baseline_e7"], comparison["candidate_e8"]
    bootstrap = comparison["bootstrap"]
    release_test = release["test"]
    lines = [
        "# Open-world training round 2 result",
        "",
        f"- Round 2 manifest: `{audit['round2_manifest_id']}`",
        f"- Fresh v2 manifest: `{audit['fresh_manifest_id']}`",
        f"- Added, audited rows: {audit['addition_rows']:,}",
        f"- Development decision: **{'advance E8' if selection['advance_e8'] else 'retain E7'}**",
        "",
        "## Development selection",
        "",
        "| Recipe | Seeds | Mean domain macro video AUROC | FF++ AUROC | Stable Cascade AUROC |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, key in (("E7", "baseline_e7"), ("E8", "candidate_e8")):
        result = selection[key]
        lines.append(
            f"| {label} | {result['seeds']} | {_metric(result['score_mean'])} | "
            f"{_metric(result['domain_mean_auroc']['ff'])} | "
            f"{_metric(result['domain_mean_auroc']['stable_cascade_lite'])} |"
        )
    lines += [
        "",
        f"E8 changed the development score by {_metric(selection['score_gain'])} and FF++ by "
        f"{_metric(selection['ff_change'])}. The allowed FF++ drop was 0.010.",
        "",
        "## Fresh v2",
        "",
        "The primary score is the unweighted mean of the three domain macro video AUROCs.",
        "",
        "| Model | Primary AUROC | Seed scores |",
        "|---|---:|---|",
        f"| Release CLIP B/16 | {_metric(release_test['score'])} | one checkpoint |",
        f"| E7 | {_metric(e7['score'])} | "
        f"{', '.join(_metric(value) for value in comparison['seed_spread']['baseline_e7'])} |",
        f"| E8 | {_metric(e8['score'])} | "
        f"{', '.join(_metric(value) for value in comparison['seed_spread']['candidate_e8'])} |",
        "",
        "| Domain | Model | AUROC | Average precision | TPR at 1% FPR | Brier | ECE-10 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for domain in sorted(e8["domains"]):
        for label, report in (("E7", e7), ("E8", e8)):
            macro = report["domains"][domain]["macro"]
            lines.append(
                f"| `{domain}` | {label} | {_metric(macro['auroc'])} | "
                f"{_metric(macro['average_precision'])} | {_metric(macro['tpr_at_1pct_fpr'])} | "
                f"{_metric(macro['brier'])} | {_metric(macro['ece_10'])} |"
            )
    lines += [
        "",
        "## Paired uncertainty and decision",
        "",
        f"E8 minus E7: **{_metric(bootstrap['difference'])}** AUROC; paired 95% CI "
        f"[{_metric(bootstrap['ci95'][0])}, {_metric(bootstrap['ci95'][1])}] from "
        f"{bootstrap['replicates']:,} bootstrap replicates ({bootstrap['unit']}).",
        "",
        "The preregistered clear-improvement gate (gain at least 0.05 and CI lower bound above "
        f"zero) was **{'met' if bootstrap['clear_improvement'] else 'not met'}**.",
        "",
        "## Per-method fresh-v2 AUROC",
        "",
        "| Domain | Method | E7 | E8 |",
        "|---|---|---:|---:|",
    ]
    for domain in sorted(e8["domains"]):
        base_methods = e7["domains"][domain]["by_method"]
        candidate_methods = e8["domains"][domain]["by_method"]
        for method in sorted(candidate_methods):
            lines.append(
                f"| `{domain}` | `{method}` | {_metric(base_methods[method]['auroc'])} | "
                f"{_metric(candidate_methods[method]['auroc'])} |"
            )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--release-report", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = render(
        _load(args.selection), _load(args.comparison), _load(args.release_report),
        _load(args.audit),
    )
    if args.output.exists() and args.output.read_text(encoding="utf-8") != payload:
        raise RuntimeError(f"refusing to overwrite changed report: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
