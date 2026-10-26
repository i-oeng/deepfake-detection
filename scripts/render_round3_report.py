"""Render the round 3 result from its frozen selection, comparison, and audit files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

NAMES = {"release": "Release CLIP B/16", "e7": "E7", "baseline_e8": "E8",
         "candidate_e9": "E9"}


def _read(path: Path | None) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path else None


def _ci(values: list[float]) -> str:
    return f"[{values[0]:.3f}, {values[1]:.3f}]"


def selection_section(selection: dict) -> list[str]:
    base, cand = selection["baseline_e8"], selection["candidate_e9"]
    rule = selection["rule"]
    lines = [
        "## Development selection", "",
        "| Recipe | Seeds | Mean domain AUROC | FF++ AUROC | Stable Cascade AUROC "
        "| Real-source bias |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, summary in (("E8", base), ("E9", cand)):
        domains = summary["domain_mean_auroc"]
        lines.append(f"| {name} | {summary['seeds']} | {summary['score_mean']:.3f} "
                     f"| {domains['ff']:.3f} | {domains['stable_cascade_lite']:.3f} "
                     f"| {summary['source_bias']['mean_auroc']:.3f} |")
    lines += [
        "",
        "Real-source bias is the mean video AUROC between FF++ validation reals and each "
        "other validation real source (0.5 = the score carries no source information).",
        "",
        "| Validation real source | E8 | E9 |", "|---|---:|---:|",
    ]
    for source, value in base["source_bias"]["by_source"].items():
        lines.append(f"| `{source}` | {value:.3f} "
                     f"| {cand['source_bias']['by_source'][source]:.3f} |")
    verdict = "advance E9" if selection["advance_e9"] else "do not advance E9"
    lines += [
        "",
        f"E9 changed the bias by {-selection['bias_drop']:+.3f} (required: at most "
        f"-{rule['minimum_bias_drop']:.2f}), the mean domain AUROC by "
        f"{selection['score_change']:+.3f}, and FF++ AUROC by {selection['ff_change']:+.3f} "
        f"(each allowed to fall by at most {rule['maximum_score_drop']:.2f}). "
        f"Decision: **{verdict}**.", "",
    ]
    return lines


def comparison_section(comparison: dict) -> list[str]:
    primary, decision = comparison["primary"], comparison["decision"]
    detection, bias = primary["detection"], primary["source_bias"]
    models = comparison["models"]
    e8, e9 = models["baseline_e8"], models["candidate_e9"]
    lines = [
        "## Fresh v3", "",
        "Seed logits are averaged per model. Both primary outcomes and the decision rule "
        "were fixed in the [protocol](round3_protocol.md) before scoring.", "",
        "| Primary outcome | E8 | E9 | Change | 95% CI |", "|---|---:|---:|---:|---:|",
        f"| Detection: `openfake` AUROC | {e8['domains']['openfake']['auroc']:.3f} "
        f"| {e9['domains']['openfake']['auroc']:.3f} | {detection['difference']:+.3f} "
        f"| {_ci(detection['ci95'])} |",
        f"| Real-source bias (`openfake`, `lfw` reals vs FF++) | {bias['baseline']:.3f} "
        f"| {bias['candidate']:.3f} | {-bias['drop']:+.3f} "
        f"| {_ci([-bias['ci95'][1], -bias['ci95'][0]])} |",
        "",
        f"- Bias reduced by at least {decision['rule']['minimum_bias_drop']:.2f} with the "
        f"interval below zero: **{'yes' if decision['bias_reduced'] else 'no'}**.",
        f"- Detection non-inferior (lower bound above {decision['rule']['detection_margin']:+.2f}):"
        f" **{'yes' if decision['detection_non_inferior'] else 'no'}**; improved (lower bound "
        f"above zero): **{'yes' if decision['detection_improved'] else 'no'}**.",
        f"- Round 3 succeeds: **{'yes' if decision['round3_succeeds'] else 'no'}**.", "",
        "### All models", "",
        "FPR is the share of real videos called fake at each model's threshold locked on "
        "validation reals (10% false positives, real sources weighted equally).", "",
        "| Model | `openfake` AUROC | `openfake_reddit` AUROC | Bias | `openfake` real FPR "
        "| `openfake_reddit` real FPR | `lfw` FPR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("release", "e7", "baseline_e8", "candidate_e9"):
        if key not in models:
            continue
        model = models[key]
        tested = model["source_bias"]["test"]
        lines.append(
            f"| {NAMES[key]} | {model['domains']['openfake']['auroc']:.3f} "
            f"| {model['domains']['openfake_reddit']['auroc']:.3f} "
            f"| {model['bias_primary']:.3f} "
            + " | ".join(f"{tested[name]['fpr_at_threshold']:.0%}"
                         for name in ("openfake", "openfake_reddit", "lfw")) + " |")
    secondary = comparison["secondary_detection"].get("openfake_reddit")
    if secondary:
        lines += ["", f"`openfake_reddit` (40 real videos, low-level shortcut 0.72, not gated): "
                      f"E9 minus E8 {secondary['difference']:+.3f}, 95% CI "
                      f"{_ci(secondary['ci95'])}."]
    lines += ["", "### `openfake` generators with at least 10 videos", "",
              "| Generator | Videos | E8 | E9 |", "|---|---:|---:|---:|"]
    for name, value in e8["generators"].items():
        lines.append(f"| `{name}` | {value['videos']} | {value['auroc']:.3f} "
                     f"| {e9['generators'][name]['auroc']:.3f} |")
    lines.append("")
    return lines


def fresh_v2_section(comparison: dict) -> list[str]:
    base, cand = comparison["baseline_e7"], comparison["candidate_e8"]
    lines = ["## Fresh v2 (secondary, second use)", "",
             "Fresh v2 was scored at the end of round 2 and then inspected, so it is no longer "
             "untouched. It is reported for continuity only.", "",
             "| Domain | E8 | E9 |", "|---|---:|---:|"]
    for domain, value in base["domain_macro_auroc"].items():
        lines.append(f"| `{domain}` | {value:.3f} | {cand['domain_macro_auroc'][domain]:.3f} |")
    lines.append(f"| **Mean** | **{base['score']:.3f}** | **{cand['score']:.3f}** |")
    lines.append("")
    return lines


def render(selection: dict, audit: dict, comparison: dict | None,
           fresh_v2: dict | None) -> str:
    lines = [
        "# Open-world training round 3 result", "",
        f"- Round 3 manifest: `{audit['round3_manifest_id']}`; audit "
        f"{'passed' if audit['passed'] else 'FAILED'}",
        "- Training real namespaces (images): " + ", ".join(
            f"`{name}` {count:,}" for name, count in audit["training_real_namespaces"].items()),
        f"- Development decision: **{'advance E9' if selection['advance_e9'] else 'stop'}**",
        "",
    ]
    lines += selection_section(selection)
    if comparison:
        lines += comparison_section(comparison)
    else:
        lines += ["Fresh v3 was not scored, because E9 did not advance.", ""]
    if fresh_v2:
        lines += fresh_v2_section(fresh_v2)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--fresh-v2-comparison", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    text = render(_read(args.selection), _read(args.audit), _read(args.comparison),
                  _read(args.fresh_v2_comparison))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
