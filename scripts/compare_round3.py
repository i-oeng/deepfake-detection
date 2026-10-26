"""Compare E9 with E8 on fresh v3: generated-face detection and real-source bias.

Seed logits are averaged per model, as in round 2. The two preregistered primary
outcomes are

- detection: macro video AUROC on ``openfake`` (real photos against current
  generators), E9 minus E8, with a paired bootstrap stratified by label;
- real-source bias: the mean, over the ``openfake`` and ``lfw`` real sources, of
  the video AUROC between FF++ validation reals and that source's reals
  (0.5 = unbiased), E8 minus E9, with a paired bootstrap over real videos.

Round 3 succeeds when the bias drops by at least 0.05 with a 95% interval above
zero and the detection difference has a 95% lower bound above -0.02. The Reddit
split, with 40 reals and a 0.72 low-level shortcut, is reported but not gated.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

from deepfake_detection.data.fresh_v3 import generator_of
from deepfake_detection.train import source_bias
from deepfake_detection.train.evaluation import report as domain_report
from deepfake_detection.train.evaluation import video_rows
from deepfake_detection.train.metrics import auroc

try:
    from compare_round2 import ensemble
except ModuleNotFoundError:  # imported by the tests as scripts.compare_round3
    from scripts.compare_round2 import ensemble

PRIMARY_DOMAIN = "openfake"
BIAS_SOURCES = ("openfake", "lfw")
MINIMUM_BIAS_DROP = 0.05
DETECTION_MARGIN = -0.02
REPEATS = 2000


def _paired(records: list[dict]) -> list[str]:
    labels: dict[str, set[int]] = defaultdict(set)
    for row in records:
        labels[row["source_domain"]].add(int(row["label"]))
    return sorted(domain for domain, values in labels.items() if values == {0, 1})


def _videos(records: list[dict], domain: str) -> dict[tuple, dict]:
    return {(row["video_id"], row["fake_method"], int(row["label"])): row
            for row in video_rows(records) if row["source_domain"] == domain}


def _interval(values: list[float]) -> list[float]:
    values = sorted(values)
    return [values[math.floor(0.025 * (len(values) - 1))],
            values[math.ceil(0.975 * (len(values) - 1))]]


def detection_bootstrap(baseline: list[dict], candidate: list[dict], domain: str, *,
                        repeats: int = REPEATS, seed: int = 20261027) -> dict:
    """Paired bootstrap of one domain's AUROC difference, resampling within each label."""
    base, cand = _videos(baseline, domain), _videos(candidate, domain)
    if base.keys() != cand.keys():
        raise ValueError(f"{domain} videos are not paired")
    strata = [[key for key in base if key[2] == label] for label in (0, 1)]

    def area(table: dict, keys: list[tuple]) -> float:
        return auroc([key[2] for key in keys], [float(table[key]["score"]) for key in keys])

    observed = area(cand, list(cand)) - area(base, list(base))
    rng = random.Random(seed)
    differences = []
    for _ in range(repeats):
        keys = [key for stratum in strata for key in rng.choices(stratum, k=len(stratum))]
        differences.append(area(cand, keys) - area(base, keys))
    return {"difference": observed, "ci95": _interval(differences), "replicates": repeats}


def _bias(reference: list[float], sources: dict[str, list[float]]) -> float:
    return statistics.fmean(source_bias.source_auroc(reference, sources[name])
                            for name in BIAS_SOURCES)


def bias_bootstrap(baseline: tuple[list[dict], list[dict]],
                   candidate: tuple[list[dict], list[dict]], *,
                   repeats: int = REPEATS, seed: int = 20261028) -> dict:
    """Paired bootstrap of the bias drop, resampling FF++ and each source's real videos."""
    def real_tables(validation: list[dict], test: list[dict]) -> dict[str, dict]:
        tables = {"ff": {key: row for key, row in _videos(validation, "ff").items()
                         if key[2] == 0}}
        for name in BIAS_SOURCES:
            tables[name] = {key: row for key, row in _videos(test, name).items()
                            if key[2] == 0}
        return tables

    base, cand = real_tables(*baseline), real_tables(*candidate)
    for name in base:
        if base[name].keys() != cand[name].keys() or not base[name]:
            raise ValueError(f"{name} real videos are not paired")
    keys = {name: sorted(table) for name, table in base.items()}

    def bias(tables: dict, chosen: dict[str, list[tuple]]) -> float:
        scores = {name: [float(tables[name][key]["score"]) for key in values]
                  for name, values in chosen.items()}
        return _bias(scores["ff"], scores)

    observed = bias(base, keys) - bias(cand, keys)
    rng = random.Random(seed)
    drops = []
    for _ in range(repeats):
        chosen = {name: rng.choices(values, k=len(values)) for name, values in keys.items()}
        drops.append(bias(base, chosen) - bias(cand, chosen))
    return {"drop": observed, "ci95": _interval(drops), "replicates": repeats,
            "baseline": bias(base, keys), "candidate": bias(cand, keys)}


def per_generator(records: list[dict], domain: str = PRIMARY_DOMAIN,
                  minimum: int = 10) -> dict[str, dict]:
    """AUROC of each generator's fakes against all of the domain's real videos."""
    videos = list(_videos(records, domain).values())
    reals = [row for row in videos if row["label"] == 0]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in videos:
        if row["label"] == 1:
            grouped[generator_of(row["video_id"])].append(row)
    return {
        name: {"videos": len(rows), "auroc": auroc(
            [0] * len(reals) + [1] * len(rows),
            [float(row["score"]) for row in reals + rows])}
        for name, rows in sorted(grouped.items()) if len(rows) >= minimum
    }


def describe(validation: list[dict], test: list[dict]) -> dict:
    domains = {domain: domain_report(test, domain=domain)["macro"] for domain in _paired(test)}
    return {
        "domains": domains,
        "source_bias": source_bias.summarize(validation, test),
        "bias_primary": _bias(source_bias.real_videos(validation)["ff"],
                              source_bias.real_videos(test)),
        "generators": per_generator(test),
    }


def compare(baseline: tuple[list[Path], list[Path]], candidate: tuple[list[Path], list[Path]],
            context: dict[str, tuple[list[Path], list[Path]]]) -> dict:
    loaded = {}
    for name, (validation, test) in {"baseline_e8": baseline, "candidate_e9": candidate,
                                     **context}.items():
        loaded[name] = (ensemble(validation)[0], ensemble(test)[0])
    detection = detection_bootstrap(loaded["baseline_e8"][1], loaded["candidate_e9"][1],
                                    PRIMARY_DOMAIN)
    secondary = {domain: detection_bootstrap(loaded["baseline_e8"][1],
                                             loaded["candidate_e9"][1], domain)
                 for domain in _paired(loaded["baseline_e8"][1]) if domain != PRIMARY_DOMAIN}
    bias = bias_bootstrap(loaded["baseline_e8"], loaded["candidate_e9"])
    bias_reduced = bias["drop"] >= MINIMUM_BIAS_DROP and bias["ci95"][0] > 0
    non_inferior = detection["ci95"][0] > DETECTION_MARGIN
    return {
        "models": {name: describe(*records) for name, records in loaded.items()},
        "primary": {
            "detection": {"domain": PRIMARY_DOMAIN, **detection},
            "source_bias": {"sources": list(BIAS_SOURCES), **bias},
        },
        "secondary_detection": secondary,
        "decision": {
            "bias_reduced": bias_reduced,
            "detection_non_inferior": non_inferior,
            "detection_improved": detection["ci95"][0] > 0,
            "round3_succeeds": bias_reduced and non_inferior,
            "rule": {"minimum_bias_drop": MINIMUM_BIAS_DROP,
                     "detection_margin": DETECTION_MARGIN},
        },
    }


def _model_paths(values: list[str], option: str) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for value in values:
        name, separator, path = value.partition("=")
        if not separator:
            raise SystemExit(f"{option} expects NAME=PATH, got {value!r}")
        grouped[name].append(Path(path))
    return grouped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ("baseline", "candidate"):
        parser.add_argument(f"--{role}-validation", action="append", type=Path, required=True)
        parser.add_argument(f"--{role}-test", action="append", type=Path, required=True)
    parser.add_argument("--context-validation", action="append", default=[],
                        metavar="NAME=PATH")
    parser.add_argument("--context-test", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    validation = _model_paths(args.context_validation, "--context-validation")
    test = _model_paths(args.context_test, "--context-test")
    if validation.keys() != test.keys():
        raise SystemExit("every context model needs validation and test predictions")
    result = compare((args.baseline_validation, args.baseline_test),
                     (args.candidate_validation, args.candidate_test),
                     {name: (validation[name], test[name]) for name in validation})
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output.exists() and args.output.read_text(encoding="utf-8") != payload:
        raise RuntimeError(f"refusing to overwrite changed comparison: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(json.dumps(result["decision"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
