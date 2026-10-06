#!/usr/bin/env python3
"""Render the frozen RGB/frequency final evaluation into a reviewable report."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

METRICS = (
    ("auroc", "AUROC"),
    ("average_precision", "AP"),
    ("tpr_at_1pct_fpr", "TPR@1% FPR"),
    ("brier", "Brier"),
    ("ece_10", "ECE-10"),
)
VARIANTS = (
    ("rgb", "RGB"),
    ("frequency", "Frequency"),
    ("equal_logits", "Equal logits"),
    ("tuned_logits", "Tuned logits"),
    ("gated_features", "Gated features†"),
)
BACKBONES = (
    ("convnext_tiny", "ConvNeXt-Tiny"),
    ("clip_vit_b16", "CLIP ViT-B/16"),
)
DOMAINS = (("ff", "FF++ unseen methods (primary)"), ("cdf", "Celeb-DF transfer"))


def load_one(directory: Path) -> tuple[Path, dict]:
    files = sorted(directory.glob("fusion-*.json"))
    if len(files) != 1:
        raise ValueError(f"expected one fusion JSON in {directory}, found {len(files)}")
    return files[0], json.loads(files[0].read_text(encoding="utf-8"))


def seed_from(name: str) -> int:
    match = re.search(r"(?:-freq-|frequency_resnet18-)(20\d{6})(?:-|$)", name)
    if not match:
        raise ValueError(f"cannot recover frequency seed from {name}")
    return int(match.group(1))


def aggregate(values: list[float]) -> str:
    mean = statistics.fmean(values)
    spread = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.4f} ± {spread:.4f} [{min(values):.4f}, {max(values):.4f}]"


def metric_rows(runs: list[dict], domain: str, variants: tuple[str, ...]) -> list[str]:
    rows: list[str] = []
    methods = ["macro", *sorted(runs[0]["final_test"]["rgb"][domain]["by_method"])]
    for method in methods:
        for variant in variants:
            reports = [run["final_test"][variant][domain] for run in runs]
            points = [report["macro"] if method == "macro" else report["by_method"][method]
                      for report in reports]
            cells = [aggregate([float(point[key]) for point in points]) for key, _ in METRICS]
            rows.append(
                f"| {method} | {dict(VARIANTS)[variant]} | " + " | ".join(cells) + " |"
            )
    return rows


def macro_rows(runs: list[dict], domain: str) -> list[str]:
    rows: list[str] = []
    for variant, label in VARIANTS:
        points = [run["final_test"][variant][domain]["macro"] for run in runs]
        cells = [aggregate([float(point[key]) for point in points]) for key, _ in METRICS]
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    return rows


def comparison_path(root: Path, label: str) -> Path:
    files = sorted((root / "comparison").glob(f"{label}-test-*.json"))
    if len(files) != 1:
        raise ValueError(f"expected one {label} comparison, found {len(files)}")
    return files[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=Path("reports/fusion"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/experiments/df40_unseen_frequency_final.md"),
    )
    args = parser.parse_args()

    all_runs: dict[str, list[tuple[str, Path, dict]]] = {}
    for prefix, _ in BACKBONES:
        entries = []
        for directory in sorted((args.reports_root / "final").glob(f"{prefix}-*-freq-*")):
            report_path, report = load_one(directory)
            entries.append((directory.name, report_path, report))
        if len(entries) != 3:
            raise ValueError(f"expected three final reports for {prefix}, found {len(entries)}")
        all_runs[prefix] = entries

    comparisons = {
        "convnext_tiny": (
            comparison_path(args.reports_root, "convnext_plus_frequency"),
            json.loads(
                comparison_path(args.reports_root, "convnext_plus_frequency").read_text(
                    encoding="utf-8"
                )
            ),
        ),
        "clip_vit_b16": (
            comparison_path(args.reports_root, "clip_plus_frequency"),
            json.loads(
                comparison_path(args.reports_root, "clip_plus_frequency").read_text(
                    encoding="utf-8"
                )
            ),
        ),
    }

    first = all_runs["convnext_tiny"][0][2]
    manifest_id = first["manifest_id"]
    manifest_sha = first["manifest_sha256"]
    for entries in all_runs.values():
        for _, _, run in entries:
            if (run["manifest_id"], run["manifest_sha256"]) != (manifest_id, manifest_sha):
                raise ValueError("final reports do not share one frozen manifest")

    gate_mismatches: list[str] = []
    for entries in all_runs.values():
        for pair_name, _, final in entries:
            _, dev = load_one(args.reports_root / "development" / pair_name)
            if final["selected_rgb_weight"] != dev["selected_rgb_weight"]:
                raise ValueError(f"final scalar weight changed for {pair_name}")
            for branch in ("rgb", "frequency"):
                if final[branch]["checkpoint_sha256"] != dev[branch]["checkpoint_sha256"]:
                    raise ValueError(f"{branch} checkpoint changed for {pair_name}")
            if final["gated_features"]["checkpoint_sha256"] != dev["gated_features"][
                "checkpoint_sha256"
            ]:
                gate_mismatches.append(pair_name)

    lines = [
        "# DF40 unseen-method frequency and fusion final evaluation",
        "",
        "## Outcome",
        "",
        "The frequency branch did **not** improve either frozen RGB finalist under the "
        "pre-registered rule. ConvNeXt-Tiny + frequency passed in 0/3 seeds; CLIP "
        "ViT-B/16 + frequency passed in 0/3 seeds. The tuned RGB/frequency blends still "
        "meet the original clear-improvement gate versus the rerun ResNet18 baseline on "
        "FF++ in every seed, because the RGB branches already provide that improvement.",
        "",
        "The untouched final test was read only after development fusion weights, checkpoint "
        "hashes, and the decision rule were committed. FF++ is the primary result; Celeb-DF "
        "is reported separately as cross-dataset transfer.",
        "",
        "## Frozen provenance",
        "",
        f"- Manifest ID: `{manifest_id}`",
        f"- Manifest SHA-256: `{manifest_sha}`",
        "- Bootstrap: 2,000 source-lineage-grouped paired replicates, seed 20261006",
        "- Scalar selection: FF++ development macro video AUROC on weights 0.00–1.00 in 0.05 steps",
        "",
        "| Pair | RGB weight | RGB checkpoint SHA-256 | Frequency checkpoint SHA-256 |",
        "|---|---:|---|---|",
    ]
    for prefix, _ in BACKBONES:
        for pair_name, _, run in all_runs[prefix]:
            lines.append(
                f"| {pair_name} | {run['selected_rgb_weight']:.2f} | "
                f"`{run['rgb']['checkpoint_sha256']}` | "
                f"`{run['frequency']['checkpoint_sha256']}` |"
            )

    lines.extend(
        [
            "",
            "† The gated head is descriptive. Its development checkpoint file was not retained; "
            f"the final run recreated the fixed-seed head and did not reproduce the stored byte "
            f"hash in {len(gate_mismatches)}/6 pairs. Gated results are excluded from both "
            "verdicts.",
            "",
            "## Macro video metrics across three frequency seeds",
            "",
            "Each cell is mean ± sample standard deviation [minimum, maximum]. CLIP uses one "
            "frozen RGB checkpoint paired with the three frequency seeds.",
        ]
    )
    for domain, domain_label in DOMAINS:
        lines.extend(["", f"### {domain_label}", ""])
        for prefix, backbone_label in BACKBONES:
            runs = [item[2] for item in all_runs[prefix]]
            lines.extend(
                [
                    f"#### {backbone_label}",
                    "",
                    "| Variant | AUROC | AP | TPR@1% FPR | Brier | ECE-10 |",
                    "|---|---:|---:|---:|---:|---:|",
                    *macro_rows(runs, domain),
                    "",
                ]
            )

    lines.extend(
        [
            "## Per-method video metrics",
            "",
            "These tables cover the independent RGB and frequency models and the confirmatory "
            "validation-tuned scalar blend. Each cell uses the same three-seed summary format.",
        ]
    )
    selected_variants = ("rgb", "frequency", "tuned_logits")
    for domain, domain_label in DOMAINS:
        lines.extend(["", f"### {domain_label}", ""])
        for prefix, backbone_label in BACKBONES:
            runs = [item[2] for item in all_runs[prefix]]
            lines.extend(
                [
                    f"#### {backbone_label}",
                    "",
                    "| Method | Model | AUROC | AP | TPR@1% FPR | Brier | ECE-10 |",
                    "|---|---|---:|---:|---:|---:|---:|",
                    *metric_rows(runs, domain, selected_variants),
                    "",
                ]
            )

    lines.extend(["## Confirmatory tuned-blend difference from the same RGB model", ""])
    for prefix, backbone_label in BACKBONES:
        comparison_file, comparison = comparisons[prefix]
        verdict_value = str(comparison["verdict"]["frequency_helps"]).lower()
        passing_seeds = comparison["verdict"]["passing_seeds"]
        seed_count = comparison["verdict"]["seeds"]
        lines.extend(
            [
                f"### {backbone_label}",
                "",
                f"Verdict: **frequency helps = {verdict_value}** "
                f"({passing_seeds}/{seed_count} seeds).",
                "",
                "| Seed | Domain | Difference | 95% CI | Source groups | Pass |",
                "|---:|---|---:|---:|---:|---|",
            ]
        )
        for pair in comparison["pairs"]:
            seed = seed_from(pair["frequency_run"])
            ff_diff = pair["domains"]["ff"]["tuned_minus_rgb"]["difference"]
            passing = False
            for domain, domain_label in DOMAINS:
                result = pair["domains"][domain]["tuned_minus_rgb"]
                low, high = result["ci95"]
                if low > 0 and ff_diff >= -0.01:
                    passing = True
                lines.append(
                    f"| {seed} | {domain_label} | {result['difference']:+.4f} | "
                    f"[{low:+.4f}, {high:+.4f}] | {result['source_groups']} | "
                    f"{('yes' if passing else 'no') if domain == 'cdf' else '—'} |"
                )
        lines.extend(["", f"Machine-readable source: `{comparison_file.as_posix()}`", ""])

    lines.extend(
        [
            "## Original clear-improvement gate versus rerun ResNet18",
            "",
            "The gate requires an FF++ macro video AUROC difference of at least +0.05 with a "
            "95% source-grouped bootstrap interval excluding zero.",
            "",
            "| Backbone | Seed | Tuned-minus-ResNet18 | 95% CI | Source groups | "
            "Clear improvement |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    clear_flags = []
    for prefix, backbone_label in BACKBONES:
        for pair_name, _, run in all_runs[prefix]:
            seed = seed_from(pair_name)
            result = run["comparison_with_resnet18"]["tuned_logits"]
            clear_flags.append(bool(result["clear_improvement"]))
            low, high = result["ci95"]
            lines.append(
                f"| {backbone_label} | {seed} | {result['difference']:+.4f} | "
                f"[{low:+.4f}, {high:+.4f}] | {result['source_groups']} | "
                f"{'yes' if result['clear_improvement'] else 'no'} |"
            )
    lines.extend(
        [
            "",
            f"Result: **clear improvement = {str(all(clear_flags)).lower()}** for all six frozen "
            "RGB/frequency pairings versus ResNet18. This does not imply an incremental benefit "
            "from frequency; the paired RGB comparisons above answer that question and are null.",
            "",
            "## Review artifacts",
            "",
        ]
    )
    for prefix, backbone_label in BACKBONES:
        lines.append(f"- {backbone_label} final fusion reports:")
        for _, report_path, _ in all_runs[prefix]:
            lines.append(f"  - `{report_path.as_posix()}`")
    lines.append("")

    content = "\n".join(lines)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
