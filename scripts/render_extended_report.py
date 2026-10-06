#!/usr/bin/env python3
"""Render the extended evaluation (P1-P5, profile, crop agreement) into one report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from deepfake_detection.train import corruptions

METRICS = (
    ("auroc", "AUROC"),
    ("average_precision", "AP"),
    ("tpr_at_1pct_fpr", "TPR@1% FPR"),
    ("brier", "Brier"),
    ("ece_10", "ECE-10"),
)
FAMILIES = ("ResNet18", "ConvNeXt-Tiny", "CLIP ViT-B/16", "Frequency")
PROTOCOLS = (
    ("ff", "P1. Seen methods, FF++ (20 videos per method)"),
    ("cdf", "P2. Seen methods, Celeb-DF domain (10 videos per method)"),
    ("celeba", "P3. StarGANv2 face editing on CelebA (300 + 300 images)"),
)
ROBUSTNESS = (
    ("frozen/ff", "Frozen unseen-method test, FF++"),
    ("heldout/ff", "P1 seen methods, FF++"),
    ("frozen/cdf", "Frozen unseen-method test, Celeb-DF"),
    ("heldout/cdf", "P2 seen methods, Celeb-DF"),
)


def spread(summary: dict) -> str:
    if summary["n"] == 1:
        return f"{summary['mean']:.4f}"
    return (f"{summary['mean']:.4f} ± {summary['sd']:.4f} "
            f"[{summary['min']:.4f}, {summary['max']:.4f}]")


def family_tables(report: dict) -> list[str]:
    lines: list[str] = []
    for domain, title in PROTOCOLS:
        lines += [f"### {title}", "",
                  "Mean ± sample SD [min, max] over seeds; one value when a family has one run.",
                  "", "| Model | Seeds | " + " | ".join(label for _, label in METRICS) + " |",
                  "|---|---:|" + "---:|" * len(METRICS)]
        for family in FAMILIES:
            if family not in report["families"]:
                continue
            values = report["families"][family][domain]
            seeds = values["auroc"]["n"]
            cells = " | ".join(spread(values[key]) for key, _ in METRICS)
            lines.append(f"| {family} | {seeds} | {cells} |")
        lines.append("")
    return lines


def method_table(report: dict, run: str) -> list[str]:
    lines = [f"### Per-method AUROC, {run}", "",
             "| Protocol | Method | Videos (incl. real) | AUROC | AP |", "|---|---|---:|---:|---:|"]
    for domain, title in PROTOCOLS:
        result = report["runs"][run]["heldout"][domain]
        for method, values in sorted(result["by_method"].items()):
            lines.append(f"| {title.split('.')[0]} | {method} | {values['videos']} | "
                         f"{values['auroc']:.4f} | {values['average_precision']:.4f} |")
    return [*lines, ""]


def comparisons(report: dict) -> list[str]:
    lines = ["| Comparison | Difference | 95% CI | Source groups | Clear improvement |",
             "|---|---:|---:|---:|---|"]
    for name, result in report["comparisons"].items():
        low, high = result["ci95"]
        lines.append(f"| {name} | {result['difference']:+.4f} | [{low:+.4f}, {high:+.4f}] | "
                     f"{result['source_groups']} | "
                     f"{'yes' if result['clear_improvement'] else 'no'} |")
    return [*lines, ""]


def robustness(report: dict, runs: dict[str, str]) -> list[str]:
    lines: list[str] = []
    for key, title in ROBUSTNESS:
        settings = None
        rows = []
        for label, run in runs.items():
            table = report["robustness"].get(run, {}).get(key)
            if not table:
                continue
            # Pre-registered order: corruption type, then increasing severity.
            settings = settings or [corruptions.label(*s) for s in corruptions.settings()
                                    if corruptions.label(*s) in table]
            cells = [f"{table['clean']['auroc']:.3f}"] + [
                f"{table[name]['auroc']:.3f}" for name in settings]
            rows.append(f"| {label} | " + " | ".join(cells) + " |")
        if not rows:
            continue
        lines += [f"### {title}", "", "Macro video AUROC per corruption.", "",
                  "| Model | clean | " + " | ".join(settings) + " |",
                  "|---|" + "---:|" * (len(settings) + 1), *rows, ""]
    return lines


def p5(result: dict) -> list[str]:
    lines = [
        f"Videos: {result['videos']}; identity-disjoint from DF40 Celeb-DF development: "
        f"{result['identity_disjoint_videos']}.", "",
        "| Bundle | FAKE / REAL / ABSTAIN | Accuracy at threshold | AUROC (all) | "
        "AUROC (identity-disjoint) | AUROC, abstain = 0.5 | AP | TPR@1% FPR | Brier | ECE-10 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, entry in result["bundles"].items():
        primary = entry["excluding_abstentions"]
        decisions = entry["decisions"]
        lines.append(
            f"| {name} | {decisions['FAKE']} / {decisions['REAL']} / {decisions['ABSTAIN']} | "
            f"{entry['accuracy_at_threshold']:.4f} | {primary['all']['auroc']:.4f} | "
            f"{primary['identity_disjoint']['auroc']:.4f} | "
            f"{entry['abstain_as_0.5']['all']['auroc']:.4f} | "
            f"{primary['all']['average_precision']:.4f} | "
            f"{primary['all']['tpr_at_1pct_fpr']:.4f} | "
            f"{primary['all']['brier']:.4f} | {primary['all']['ece_10']:.4f} |")
    h2 = result["h2"]
    low, high = h2["ci95"]
    lines += ["", f"**H2** ({h2['candidate']} minus {h2['baseline']}, video AUROC): "
              f"{h2['difference']:+.4f}, 95% CI [{low:+.4f}, {high:+.4f}], "
              f"{h2['source_groups']} identity groups, {h2['valid_replicates']} valid replicates.",
              ""]
    return lines


def profile(result: dict) -> list[str]:
    lines = [f"Measured on {result['gpu']} with PyTorch {result['torch']}, mixed precision.", "",
             "| Model | Parameters (M) | Latency, batch 1 (ms) | Throughput, batch 32 (img/s) | "
             "Peak memory, batch 32 (MiB) |", "|---|---:|---:|---:|---:|"]
    for name, entry in result["models"].items():
        one, many = entry["latency"]
        lines.append(f"| {name} | {entry['parameters'] / 1e6:.1f} | {one['median_ms']:.1f} | "
                     f"{many['images_per_second']:.0f} | {entry['peak_memory_mib_batch32']:.0f} |")
    models = result["models"]
    if {"CLIP ViT-B/16", "Frequency"} <= models.keys():
        rgb, frequency = models["CLIP ViT-B/16"], models["Frequency"]
        lines += ["", "Late fusion runs both branches, so CLIP + frequency costs "
                  f"{(rgb['parameters'] + frequency['parameters']) / 1e6:.1f} M parameters and "
                  f"{rgb['latency'][0]['median_ms'] + frequency['latency'][0]['median_ms']:.1f} ms "
                  "per image at batch 1, for no accuracy gain on the frozen test."]
    return [*lines, ""]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extended", type=Path, required=True)
    parser.add_argument("--p5", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--crop-agreement", type=Path, required=True)
    parser.add_argument("--outcome", type=Path, required=True, help="Markdown outcome section")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    extended = json.loads(args.extended.read_text(encoding="utf-8"))
    crop = json.loads(args.crop_agreement.read_text(encoding="utf-8"))
    robust_runs = {"ResNet18": "resnet18-20261006", "ConvNeXt-Tiny 20261006": "convnext-20261006",
                   "CLIP ViT-B/16 20261006": "clip-20261006",
                   "Frequency 20261006": "frequency-20261006"}
    lines = [
        "# DF40 extended evaluation and release result", "",
        args.outcome.read_text(encoding="utf-8").strip(), "",
        "## Frozen inputs", "",
        "- Protocol: [`df40_extended_evaluation_protocol.md`]"
        "(df40_extended_evaluation_protocol.md)",
        f"- Held-out manifest: `{extended['heldout_manifest']}`, gated by "
        f"[`df40_heldout_v1_{extended['heldout_manifest']}.json`]"
        f"(../identity_audit/df40_heldout_v1_{extended['heldout_manifest']}.json)",
        f"- Frozen unseen-method manifest: `{extended['frozen_manifest']}`",
        f"- Runs without readable checkpoints: {', '.join(extended['missing_runs']) or 'none'}",
        "",
        "## Held-out protocols P1-P3", "", *family_tables(extended),
        *method_table(extended, "clip-20261006"),
        "## Pre-registered comparisons", "",
        "Grouped paired bootstrap, 2,000 replicates, source-video lineage groups. "
        "Clear improvement requires a difference of at least 0.05 and a 95% interval above zero.",
        "", *comparisons(extended),
        "## P4. Robustness", "", *robustness(extended, robust_runs),
        "## P5. Raw Celeb-DF v2 video through the inference pipeline", "",
        *p5(json.loads(args.p5.read_text(encoding="utf-8"))),
        "## Face-crop agreement with DF40", "",
        f"On {crop['frames']} real Celeb-DF frames that DF40 also cropped, landmarks in our "
        f"1.3-margin crops lie a median {crop['median_landmark_px']:.1f} px "
        f"(mean {crop['mean_landmark_px']:.1f} px) from DF40's on a 256 px crop; mean absolute "
        f"pixel difference is {crop['pixel_mae']:.1f} of 255. Other margins were "
        f"{', '.join(f'{k}: {v:.1f} px' for k, v in crop['mean_by_scale'].items())} (mean). "
        "The check gated nothing; the margin stayed at DeepfakeBench's 1.3.", "",
        "## Cost", "", *profile(json.loads(args.profile.read_text(encoding="utf-8"))),
    ]
    args.output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
