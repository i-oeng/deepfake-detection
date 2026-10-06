"""Score frozen checkpoints on the extended protocols (P1-P4) and build their reports.

Predictions are written once per (run, manifest, split, corruption) under the
evaluation output root and are never overwritten. Every score re-verifies the
manifest and the checkpoint hash recorded at training time.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import yaml

from deepfake_detection.data.config import find_project_root

from . import corruptions
from .common import read_manifest_rows, sha256_file
from .evaluation import grouped_bootstrap_difference, report

DOMAINS = {"heldout": ("ff", "cdf", "celeba"), "frozen": ("ff", "cdf")}
METRICS = ("auroc", "average_precision", "tpr_at_1pct_fpr", "brier", "ece_10")


def load_config(path: Path) -> tuple[dict, Path]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))["evaluation"]
    return config, find_project_root(path.resolve().parent)


def _prediction_path(output_root: Path, run: Path, manifest: Path, split: str,
                     corruption: tuple[str, float] | None) -> Path:
    name = "clean" if corruption is None else corruptions.label(*corruption)
    return output_root / run.name / manifest.name / f"{split}-{name}.jsonl"


def score(run_dir: Path, manifest_dir: Path, data_root: Path, output_root: Path, *,
          split: str = "test", corruption: tuple[str, float] | None = None) -> Path:
    """Predict one manifest split with a frozen checkpoint; skip work that already exists."""
    import torch

    from .benchmark import BinaryEncoder, predict

    path = _prediction_path(output_root, run_dir, manifest_dir, split, corruption)
    if path.exists():
        return path
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    training = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    checkpoint = run_dir / "best.pt"
    if sha256_file(checkpoint) != training["checkpoint_sha256"]:
        raise ValueError(f"checkpoint hash differs from the training report: {run_dir}")
    rows, manifest_path = read_manifest_rows(manifest_dir, data_root)
    rows = [row for row in rows if row["split"] == split]
    if not rows:
        raise ValueError(f"no {split} rows in {manifest_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.get("require_cuda", True) and device.type != "cuda":
        raise RuntimeError("CUDA required by benchmark config")
    model = BinaryEncoder(config["model"], config).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    records, _ = predict(model, rows, config, device, corruption=corruption)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    metadata = {
        "run": run_dir.name,
        "checkpoint_sha256": training["checkpoint_sha256"],
        "manifest_id": manifest_dir.name,
        "manifest_sha256": sha256_file(manifest_path),
        "split": split,
        "corruption": None if corruption is None else list(corruption),
        "rows": len(records),
    }
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return path


def _runs(config: dict, root: Path) -> list[dict]:
    runs = []
    for entry in config["runs"]:
        run = dict(entry)
        run["path"] = root / entry["run"]
        runs.append(run)
    return runs


def score_all(config: dict, root: Path, *, robustness: bool = True) -> list[str]:
    """Score every configured run on P1-P3 and, optionally, the P4 corruptions."""
    output_root = root / config["output_root"]
    data_root = root / config["data_root"]
    manifests = {name: root / config[f"{name}_manifest"] for name in DOMAINS}
    settings: list[tuple[str, float] | None] = [None]
    if robustness:
        settings += corruptions.settings()
    missing = []
    for run in _runs(config, root):
        if not (run["path"] / "best.pt").is_file():
            missing.append(run["name"])
            continue
        for corruption in settings:
            # P3 (CelebA) is clean-only, but it shares the held-out manifest with
            # P1-P2, so corrupted held-out scores include it and reports skip it.
            for manifest in manifests.values():
                score(run["path"], manifest, data_root, output_root, corruption=corruption)
    return missing


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _summary(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def build_report(config: dict, root: Path) -> dict:
    """Assemble per-run metrics, seed spread, pre-registered tests, and robustness."""
    output_root = root / config["output_root"]
    manifests = {name: root / config[f"{name}_manifest"] for name in DOMAINS}
    runs = [run for run in _runs(config, root)
            if _prediction_path(output_root, run["path"], manifests["heldout"], "test",
                                None).exists()]
    predictions: dict[tuple[str, str, str | None], list[dict]] = {}

    def frames(run: dict, protocol: str, corruption: tuple[str, float] | None = None):
        key = (run["name"], protocol, None if corruption is None else corruptions.label(
            *corruption))
        if key not in predictions:
            predictions[key] = _load(_prediction_path(
                output_root, run["path"], manifests[protocol], "test", corruption))
        return predictions[key]

    result: dict = {
        "heldout_manifest": manifests["heldout"].name,
        "frozen_manifest": manifests["frozen"].name,
        "runs": {},
        "families": {},
        "comparisons": {},
        "robustness": {},
        "missing_runs": [run["name"] for run in _runs(config, root) if run not in runs],
    }
    for run in runs:
        result["runs"][run["name"]] = {
            "family": run["family"],
            "seed": run["seed"],
            "heldout": {d: report(frames(run, "heldout"), domain=d)
                        for d in DOMAINS["heldout"]},
        }
    families: dict[str, list[str]] = {}
    for run in runs:
        families.setdefault(run["family"], []).append(run["name"])
    for family, names in families.items():
        result["families"][family] = {
            domain: {
                metric: _summary([result["runs"][n]["heldout"][domain]["macro"][metric]
                                  for n in names])
                for metric in METRICS
            }
            for domain in DOMAINS["heldout"]
        }
    by_name = {run["name"]: run for run in runs}
    for comparison in config["comparisons"]:
        baseline, candidate = by_name.get(comparison["baseline"]), by_name.get(
            comparison["candidate"])
        if baseline is None or candidate is None:
            continue
        protocol = comparison.get("protocol", "heldout")
        result["comparisons"][comparison["name"]] = {
            **comparison,
            **grouped_bootstrap_difference(
                frames(baseline, protocol), frames(candidate, protocol),
                domain=comparison["domain"],
            ),
        }
    for run in runs:
        rows = {}
        for protocol in DOMAINS:
            clean = {d: report(frames(run, protocol), domain=d)["macro"]["auroc"]
                     for d in DOMAINS["frozen"]}
            for corruption in corruptions.settings():
                path = _prediction_path(output_root, run["path"], manifests[protocol], "test",
                                        corruption)
                if not path.exists():
                    continue
                for domain in DOMAINS["frozen"]:
                    value = report(frames(run, protocol, corruption), domain=domain)["macro"][
                        "auroc"]
                    rows.setdefault(f"{protocol}/{domain}", {})[corruptions.label(
                        *corruption)] = {"auroc": value, "drop": clean[domain] - value}
            for domain in DOMAINS["frozen"]:
                rows.setdefault(f"{protocol}/{domain}", {})["clean"] = {
                    "auroc": clean[domain], "drop": 0.0}
        result["robustness"][run["name"]] = rows
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("score", help="Score one run on one manifest split")
    one.add_argument("--run", type=Path, required=True)
    one.add_argument("--manifest-dir", type=Path, required=True)
    one.add_argument("--data-root", type=Path, required=True)
    one.add_argument("--output-root", type=Path, required=True)
    one.add_argument("--split", default="test")
    one.add_argument("--corruption", choices=sorted(corruptions.SEVERITIES))
    one.add_argument("--level", type=float)
    every = commands.add_parser("score-all", help="Score every configured run")
    every.add_argument("--config", type=Path, required=True)
    every.add_argument("--run-name", action="append", help="Limit to these configured runs")
    every.add_argument("--clean-only", action="store_true")
    summary = commands.add_parser("report", help="Write the extended evaluation JSON report")
    summary.add_argument("--config", type=Path, required=True)
    summary.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "score":
        corruption = None
        if args.corruption is not None:
            if args.level is None:
                parser.error("--corruption requires --level")
            corruption = (args.corruption, args.level)
        print(score(args.run.resolve(), args.manifest_dir.resolve(), args.data_root.resolve(),
                    args.output_root.resolve(), split=args.split, corruption=corruption))
        return 0
    config, root = load_config(args.config)
    if args.command == "score-all":
        if args.run_name:
            config["runs"] = [run for run in config["runs"] if run["name"] in args.run_name]
        missing = score_all(config, root, robustness=not args.clean_only)
        print(json.dumps({"missing_runs": missing}))
        return 0
    result = build_report(config, root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
