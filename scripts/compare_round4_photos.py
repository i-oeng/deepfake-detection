"""Development comparison for round 4's photo data: E10-photos against E9.

E10-photos trains E9's recipe on round 3's data plus the AI-photo additions
(``data/round4.py``). Fresh v4 stays unscored: the two models are compared on
retired sets only (``reports/openworld/round4_protocol.md``, "Development
selection" and "Threshold"):

- target domains, macro video AUROC from seed-averaged logits: fresh v2's ``dff``,
  ``celebdf_video`` and ``sfhq_t2i``, and fresh v3's ``openfake`` restricted to
  generators of OpenFake's train split (its test-only generators supply fresh v4);
- the round 4 threshold (``train.threshold.lock``) over those domains, with every
  validation real source and ``lfw`` at most 10% false positives, and balanced
  accuracy at it per domain, next to balanced accuracy at round 3's threshold;
- round 3's validation guardrails, averaged over seeds: FF++ AUROC may fall by at
  most 0.01 and the real-source bias may rise by at most 0.03.

E10-photos advances when its mean target AUROC is higher and both guardrails hold.
``sfhq_t2i`` and ``openfake`` come from the same pipelines as the new training data,
and the threshold is locked on the domains it is scored on, so these balanced
accuracies are upper estimates; fresh v4 gives the honest number.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from deepfake_detection.data.fresh_v3 import generator_of
from deepfake_detection.train import source_bias, threshold
from deepfake_detection.train.evaluation import report as domain_report
from deepfake_detection.train.evaluation import video_rows
from deepfake_detection.train.metrics import auroc as _auroc

try:
    from compare_round2 import ensemble
    from select_round3 import summarize_runs
except ModuleNotFoundError:  # imported by the tests as scripts.compare_round4_photos
    from scripts.compare_round2 import ensemble
    from scripts.select_round3 import summarize_runs

FRESH_V2_DOMAINS = ("dff", "celebdf_video", "sfhq_t2i")
OPENFAKE = "openfake"
TARGETS = (*FRESH_V2_DOMAINS, OPENFAKE)
# Real-only or shortcut-prone sets: reported, not used to lock the threshold.
REPORTED_ONLY = ("openfake_reddit",)
LFW = "lfw"
FF_GUARDRAIL = 0.01
BIAS_GUARDRAIL = 0.03


def train_generators(metadata: Path) -> set[str]:
    """OpenFake's train-split fake generators, from ``openfake_metadata.json``."""
    return set(json.loads(metadata.read_text(encoding="utf-8"))["fake_generators"]["train_only"])


def targets(fresh_v2: list[dict], fresh_v3: list[dict], generators: set[str]) -> list[dict]:
    """Target-domain records, ``lfw`` and the reported-only sets; ``openfake`` fakes
    only from ``generators``."""
    kept = [row for row in fresh_v2 if row["source_domain"] in FRESH_V2_DOMAINS]
    for row in fresh_v3:
        domain = row["source_domain"]
        if domain == OPENFAKE and int(row["label"]) == 1 \
                and generator_of(row["video_id"]) not in generators:
            continue
        if domain in (OPENFAKE, LFW, *REPORTED_ONLY):
            kept.append(row)
    return kept


def _real_sources(validation: list[dict], target: list[dict]) -> dict[str, list[float]]:
    sources = dict(source_bias.real_videos(validation))
    for name, scores in source_bias.real_videos(target).items():
        if name in sources:
            raise ValueError(f"real source {name} is in validation and a target set")
        if name not in REPORTED_ONLY:
            sources[name] = scores
    return sources


def _accuracy(videos: list[dict], cut: float, domains) -> dict:
    result = {}
    for domain in domains:
        rows = [video for video in videos if video["source_domain"] == domain]
        result[domain] = {key: value for key, value in threshold.balanced_accuracy(
            rows, cut).items() if key in {"balanced_accuracy", "interval", "tpr", "tnr",
                                          "videos"}}
    return result


def evaluate(validation: list[dict], target: list[dict]) -> dict:
    """Target AUROCs, both thresholds and balanced accuracy at each for one model."""
    videos = video_rows(target)
    auroc = {domain: domain_report(target, domain=domain)["macro"]["auroc"]
             for domain in TARGETS}
    domains = {name: labels for name, labels in threshold.by_domain(target).items()
               if name in TARGETS}
    sources = _real_sources(validation, target)
    locked = threshold.lock(domains, sources)
    # Round 3 calls a score above its threshold fake; balanced_accuracy uses >=.
    old = math.nextafter(source_bias.summarize(validation)["threshold"], math.inf)
    reals = {**source_bias.real_videos(validation), **source_bias.real_videos(target)}
    fprs = {name: sum(score >= locked["threshold"] for score in scores) / len(scores)
            for name, scores in sorted(reals.items())}
    return {
        "target_auroc": auroc,
        "target_mean_auroc": statistics.fmean(auroc.values()),
        "threshold": locked,
        "balanced_accuracy": _accuracy(videos, locked["threshold"], TARGETS),
        "real_fpr": fprs,
        "round3_threshold": old,
        "round3_threshold_balanced_accuracy": _accuracy(videos, old, TARGETS),
        "openfake_generators": _generators(videos),
    }


def _generators(videos: list[dict]) -> dict[str, dict]:
    rows = [video for video in videos if video["source_domain"] == OPENFAKE]
    reals = [video for video in rows if video["label"] == 0]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for video in rows:
        if video["label"] == 1:
            grouped[generator_of(video["video_id"])].append(video)
    return {name: {"videos": len(fakes),
                   "auroc": _auroc([video["label"] for video in reals + fakes],
                                   [video["score"] for video in reals + fakes])}
            for name, fakes in sorted(grouped.items()) if len(fakes) >= 10}


def decide(baseline: dict, candidate: dict, guard_baseline: dict, guard_candidate: dict) -> dict:
    target_change = candidate["target_mean_auroc"] - baseline["target_mean_auroc"]
    ff_change = (guard_candidate["domain_mean_auroc"]["ff"]
                 - guard_baseline["domain_mean_auroc"]["ff"])
    bias_change = (guard_candidate["source_bias"]["mean_auroc"]
                   - guard_baseline["source_bias"]["mean_auroc"])
    return {"advances": target_change > 0 and ff_change >= -FF_GUARDRAIL
            and bias_change <= BIAS_GUARDRAIL,
            "target_mean_auroc_change": target_change, "ff_auroc_change": ff_change,
            "real_source_bias_change": bias_change,
            "rule": {"maximum_ff_auroc_drop": FF_GUARDRAIL,
                     "maximum_bias_rise": BIAS_GUARDRAIL}}


def compare(models: dict[str, dict[str, list[Path]]], baseline: str, candidate: str,
            metadata: Path) -> dict:
    generators = train_generators(metadata)
    results, guards = {}, {}
    for name, paths in models.items():
        validation = ensemble([run / "validation.jsonl" for run in paths["run"]])[0]
        target = targets(ensemble(paths["fresh_v2"])[0], ensemble(paths["fresh_v3"])[0],
                         generators)
        results[name] = evaluate(validation, target)
        guards[name] = summarize_runs(paths["run"])
        results[name]["validation"] = {
            "ff_auroc": guards[name]["domain_mean_auroc"]["ff"],
            "domain_mean_auroc": guards[name]["domain_mean_auroc"],
            "real_source_bias": guards[name]["source_bias"]["mean_auroc"],
            "seeds": guards[name]["seeds"]}
    return {"models": results,
            "decision": decide(results[baseline], results[candidate], guards[baseline],
                               guards[candidate]),
            "baseline": baseline, "candidate": candidate}


def markdown(result: dict) -> str:
    names = [result["baseline"], result["candidate"]]
    models = result["models"]
    lines = ["| Domain | " + " | ".join(f"{name} AUROC" for name in names) + " | "
             + " | ".join(f"{name} balanced accuracy" for name in names) + " |",
             "|---|" + "---:|" * 4]
    for domain in TARGETS:
        cells = [f"{models[name]['target_auroc'][domain]:.3f}" for name in names]
        cells += [f"{models[name]['balanced_accuracy'][domain]['balanced_accuracy']:.3f}"
                  for name in names]
        lines.append(f"| `{domain}` | " + " | ".join(cells) + " |")
    decision = result["decision"]
    lines += ["", f"Target mean AUROC change {decision['target_mean_auroc_change']:+.3f}, "
              f"FF++ AUROC change {decision['ff_auroc_change']:+.3f}, real-source bias "
              f"change {decision['real_source_bias_change']:+.3f}: "
              f"**{'advances' if decision['advances'] else 'does not advance'}**."]
    return "\n".join(lines) + "\n"


def _grouped(values: list[str], option: str) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for value in values:
        name, separator, path = value.partition("=")
        if not separator:
            raise SystemExit(f"{option} expects NAME=PATH, got {value!r}")
        grouped[name].append(Path(path))
    return grouped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("--run", "--fresh-v2", "--fresh-v3"):
        parser.add_argument(option, action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--baseline", default="e9")
    parser.add_argument("--candidate", default="e10_photos")
    parser.add_argument("--openfake-metadata", type=Path,
                        default=Path("reports/openworld/openfake_metadata.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args(argv)
    grouped = {key: _grouped(getattr(args, key), f"--{key.replace('_', '-')}")
               for key in ("run", "fresh_v2", "fresh_v3")}
    names = set(grouped["run"])
    if any(set(paths) != names for paths in grouped.values()) \
            or {args.baseline, args.candidate} != names:
        raise SystemExit("every model needs runs and fresh v2 and v3 predictions")
    models = {name: {key: grouped[key][name] for key in grouped} for name in sorted(names)}
    result = compare(models, args.baseline, args.candidate, args.openfake_metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.write_text(markdown(result), encoding="utf-8")
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
