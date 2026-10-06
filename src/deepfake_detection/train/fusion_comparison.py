"""Pre-registered test comparison: tuned RGB+frequency blend versus the same RGB model alone.

Each pair supplies an RGB run, a frequency run, and the development fusion report that fixed the
blend weight. The weight is read from that report, never from the command line, and both
branches' checkpoint hashes must match it, so the test split cannot influence any choice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from deepfake_detection.data.unseen import audit_unseen_manifest

from .evaluation import grouped_bootstrap_difference, report
from .fusion_contract import align, blend, load_export

DOMAINS = ("ff", "cdf")
REPEATS = 2000
SEED = 20261006
# Pre-registered decision rule (reports/experiments/df40_unseen_frequency_development_selection.md).
MAX_FF_DROP = 0.01


def frozen_weight(dev_report: dict, rgb_meta: dict, frequency_meta: dict) -> float:
    """Return the development-selected RGB weight after checking both branches' provenance."""
    # fusion.run only writes a "final_test" section after a test export was scored.
    if "final_test" in dev_report:
        raise ValueError("blend weight must come from a development-only fusion report")
    if dev_report.get("selection_split") != "validation":
        raise ValueError("blend weight was not selected on the validation split")
    for name, meta in (("rgb", rgb_meta), ("frequency", frequency_meta)):
        if dev_report[name]["checkpoint_sha256"] != meta["checkpoint_sha256"]:
            raise ValueError(f"{name} checkpoint differs from the development fusion report")
    if rgb_meta["sample_preprocessing_id"] != frequency_meta["sample_preprocessing_id"]:
        raise ValueError("RGB and frequency exports use different sample preprocessing")
    weight = float(dev_report["selected_rgb_weight"])
    if not 0 <= weight <= 1:
        raise ValueError("development blend weight is outside [0, 1]")
    return weight


def compare_pair(
    rgb_dir: Path,
    frequency_dir: Path,
    dev_report_path: Path,
    *,
    split: str = "test",
    domains: tuple[str, ...] = DOMAINS,
    repeats: int = REPEATS,
) -> dict:
    """Score one pair on ``split`` with the frozen weight and bootstrap fused minus RGB."""
    dev_report = json.loads(dev_report_path.read_text(encoding="utf-8"))
    rgb, rgb_meta = load_export(rgb_dir, split)
    frequency, frequency_meta = load_export(frequency_dir, split)
    for name, meta in (("rgb", rgb_meta), ("frequency", frequency_meta)):
        if meta["manifest_id"] != dev_report["manifest_id"]:
            raise ValueError(f"{name} export uses a different manifest than development")
    weight = frozen_weight(dev_report, rgb_meta, frequency_meta)
    rgb, frequency = align(rgb, frequency)
    fused = blend(rgb, frequency, weight)
    result = {
        "rgb_run": rgb_dir.name,
        "frequency_run": frequency_dir.name,
        "rgb_checkpoint_sha256": rgb_meta["checkpoint_sha256"],
        "frequency_checkpoint_sha256": frequency_meta["checkpoint_sha256"],
        "development_report": dev_report_path.name,
        "selected_rgb_weight": weight,
        "split": split,
        "domains": {},
    }
    for domain in domains:
        result["domains"][domain] = {
            "rgb": report(rgb, domain=domain),
            "frequency": report(frequency, domain=domain),
            "tuned_blend": report(fused, domain=domain),
            "tuned_minus_rgb": grouped_bootstrap_difference(
                rgb, fused, domain=domain, repeats=repeats, seed=SEED
            ),
        }
    return result


def pair_passes(pair: dict) -> bool:
    """Pre-registered per-seed rule: a CI above zero somewhere, and no material FF++ drop."""
    differences = {domain: pair["domains"][domain]["tuned_minus_rgb"] for domain in pair["domains"]}
    improved = any(value["ci95"][0] > 0 for value in differences.values())
    ff_ok = "ff" not in differences or differences["ff"]["difference"] >= -MAX_FF_DROP
    return improved and ff_ok


def verdict(pairs: list[dict]) -> dict:
    passing = [pair_passes(pair) for pair in pairs]
    return {
        "rule": (
            "frequency helps if, in a majority of seeds, the tuned blend's 95% grouped-bootstrap "
            "CI for macro AUROC minus RGB excludes zero on FF++ or Celeb-DF, with an FF++ "
            f"difference of at least -{MAX_FF_DROP}"
        ),
        "passing_seeds": sum(passing),
        "seeds": len(passing),
        "frequency_helps": sum(passing) * 2 > len(passing),
    }


def run(
    label: str,
    pairs: list[tuple[Path, Path, Path]],
    manifest_dir: Path,
    output: Path,
    *,
    split: str = "test",
) -> dict:
    audit = audit_unseen_manifest(manifest_dir)
    if split == "test" and not audit["passed"]:
        raise ValueError("held-out test blocked by protocol audit: " + "; ".join(audit["failures"]))
    if any(
        json.loads(dev.read_text(encoding="utf-8"))["manifest_id"] != audit["manifest_id"]
        for _, _, dev in pairs
    ):
        raise ValueError("development fusion report differs from the frozen manifest")
    results = [compare_pair(rgb, frequency, dev, split=split) for rgb, frequency, dev in pairs]
    summary = {
        "comparison": label,
        "manifest_id": audit["manifest_id"],
        "manifest_sha256": audit["manifest_sha256"],
        "split": split,
        "pairs": results,
        "verdict": verdict(results),
    }
    content = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    digest = hashlib.sha256(content.encode()).hexdigest()[:20]
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{label}-{split}-{digest}.json"
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise RuntimeError(f"refusing to overwrite changed comparison: {path}")
    path.write_text(content, encoding="utf-8")
    return {"report": str(path), **summary["verdict"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="e.g. convnext_plus_frequency")
    parser.add_argument(
        "--pair",
        nargs=3,
        action="append",
        type=Path,
        required=True,
        metavar=("RGB_RUN", "FREQUENCY_RUN", "DEV_FUSION_REPORT"),
    )
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output", type=Path, default=Path("reports/fusion/comparison"))
    args = parser.parse_args(argv)
    result = run(args.label, [tuple(pair) for pair in args.pair], args.manifest_dir, args.output,
                 split=args.split)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
