"""Real-source bias: whether a detector's score tells real images apart by origin.

Every training real up to round 2 is an FF++ video frame. A detector that learned
manipulation evidence scores real images from any source alike; one that learned
"FF++ frame versus anything else" separates other real sources from FF++ reals as
well as it separates fakes, and calls most real photos fake. Domain AUROC cannot
show this, because it compares each domain's fakes only with that domain's reals.

For each real source this module reports the video-level AUROC of the score with
FF++ validation reals as negatives and that source's reals as positives: 0.5 means
the score carries no source information, 1.0 means it identifies the source. It
also fixes a threshold on validation reals, every source weighted equally, and
reports each source's false-positive rate at that threshold.
"""

from __future__ import annotations

import argparse
import json
import statistics
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

from .evaluation import video_rows
from .metrics import auroc

REFERENCE = "ff"
TARGET_FPR = 0.10


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def real_videos(records: list[dict]) -> dict[str, list[float]]:
    """Mean frame score of every real video, grouped by source domain."""
    sources: dict[str, list[float]] = defaultdict(list)
    for video in video_rows(records):
        if int(video["label"]) == 0:
            sources[video["source_domain"]].append(float(video["score"]))
    return {source: sorted(scores) for source, scores in sorted(sources.items())}


def source_auroc(reference: list[float], scores: list[float]) -> float:
    """AUROC with reference reals as negatives; 0.5 means no source information."""
    return auroc([0] * len(reference) + [1] * len(scores), [*reference, *scores])


def false_positive_rate(scores: list[float], threshold: float) -> float:
    ordered = sorted(scores)
    return (len(ordered) - bisect_right(ordered, threshold)) / len(ordered)


def locked_threshold(sources: dict[str, list[float]], target_fpr: float = TARGET_FPR) -> float:
    """The lowest score threshold whose source-balanced real FPR is at most the target.

    A score above the threshold is called fake. Weighting sources equally keeps the
    largest source from setting the threshold for every other one.
    """
    if not sources or any(not scores for scores in sources.values()):
        raise ValueError("every real source needs at least one video")
    ordered = {name: sorted(scores) for name, scores in sources.items()}
    for candidate in sorted({score for scores in ordered.values() for score in scores}):
        rate = statistics.fmean(false_positive_rate(scores, candidate)
                                for scores in ordered.values())
        if rate <= target_fpr:
            return candidate
    raise AssertionError("the highest score always gives a zero false-positive rate")


def _describe(reference: list[float], scores: list[float], threshold: float) -> dict:
    return {
        "videos": len(scores),
        "median_score": statistics.median(scores),
        "auroc_vs_reference": source_auroc(reference, scores),
        "fpr_at_threshold": false_positive_rate(scores, threshold),
    }


def summarize(validation: list[dict], test: list[dict] | None = None, *,
              reference: str = REFERENCE, target_fpr: float = TARGET_FPR) -> dict:
    """Source bias on validation reals and, optionally, on test-only real sources."""
    sources = real_videos(validation)
    if reference not in sources:
        raise ValueError(f"validation predictions have no {reference} reals")
    base = sources[reference]
    threshold = locked_threshold(sources, target_fpr)
    others = {name: _describe(base, scores, threshold)
              for name, scores in sources.items() if name != reference}
    if not others:
        raise ValueError("validation predictions have no non-reference real source")
    result = {
        "reference": reference,
        "reference_videos": len(base),
        "reference_fpr_at_threshold": false_positive_rate(base, threshold),
        "target_fpr": target_fpr,
        "threshold": threshold,
        "validation": others,
        "validation_mean_auroc": statistics.fmean(
            value["auroc_vs_reference"] for value in others.values()),
    }
    if test:
        tested = {name: _describe(base, scores, threshold)
                  for name, scores in real_videos(test).items()}
        result["test"] = tested
        result["test_mean_auroc"] = statistics.fmean(
            value["auroc_vs_reference"] for value in tested.values())
        result["test_mean_fpr"] = statistics.fmean(
            value["fpr_at_threshold"] for value in tested.values())
    return result


def summarize_seeds(validation_paths: list[Path], test_paths: list[Path]) -> dict:
    """Per-seed summaries, paired by position, and their means."""
    if not validation_paths:
        raise ValueError("at least one validation prediction file is required")
    if test_paths and len(test_paths) != len(validation_paths):
        raise ValueError("test predictions must pair with validation predictions")
    seeds = [
        summarize(_read_jsonl(path), _read_jsonl(test_paths[index]) if test_paths else None)
        for index, path in enumerate(validation_paths)
    ]
    keys = [key for key in ("validation_mean_auroc", "test_mean_auroc", "test_mean_fpr")
            if key in seeds[0]]
    return {
        "seeds": seeds,
        "mean": {key: statistics.fmean(seed[key] for seed in seeds) for key in keys},
        "validation_inputs": [str(path) for path in validation_paths],
        "test_inputs": [str(path) for path in test_paths],
    }


def _pairs(values: list[str], option: str) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise SystemExit(f"{option} expects NAME=PATH, got {value!r}")
        grouped[name].append(Path(path))
    return grouped


def markdown(models: dict[str, dict]) -> str:
    """One table per split: AUROC against FF++ reals and FPR at the locked threshold."""
    lines = []
    for split in ("validation", "test"):
        sources = sorted({name for model in models.values() for seed in model["seeds"]
                          for name in seed.get(split, {})})
        if not sources:
            continue
        lines += [f"### {split.capitalize()} real sources", "",
                  "AUROC of the score, FF++ validation reals against each source's reals "
                  "(0.5 = no source information); FPR at the validation-locked threshold "
                  "in parentheses. Means over seeds.", "",
                  "| Model | " + " | ".join(f"`{name}`" for name in sources) + " |",
                  "|---|" + "---:|" * len(sources)]
        for model, summary in models.items():
            cells = []
            for name in sources:
                values = [seed[split][name] for seed in summary["seeds"]
                          if name in seed.get(split, {})]
                if not values:
                    cells.append("")
                    continue
                area = statistics.fmean(value["auroc_vs_reference"] for value in values)
                rate = statistics.fmean(value["fpr_at_threshold"] for value in values)
                cells.append(f"{area:.3f} ({rate:.0%})")
            lines.append(f"| {model} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", action="append", required=True, metavar="NAME=PATH",
                        help="validation predictions; repeat a name for each seed")
    parser.add_argument("--test", action="append", default=[], metavar="NAME=PATH",
                        help="test predictions, in the same seed order as --validation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args(argv)
    validation, test = _pairs(args.validation, "--validation"), _pairs(args.test, "--test")
    if set(test) - set(validation):
        raise SystemExit("every --test model needs --validation predictions")
    models = {name: summarize_seeds(paths, test.get(name, []))
              for name, paths in validation.items()}
    payload = json.dumps(models, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown(models), encoding="utf-8")
    print(json.dumps({name: model["mean"] for name, model in models.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
