"""Attribute E9's round 3 gains to its two changes: FairFace data and padding.

Four recipes share the E8 recipe, budget, and seeds:

- ``e8``: round 2 manifest;
- ``e9_data``: round 3 manifest (FairFace reals and reconstructions) with
  real-source balancing, no padding augmentation;
- ``e9_padding``: round 2 manifest with padding augmentation;
- ``e9``: both changes.

Every metric is reported as a change from E8. What the two single changes leave of
E9's change is their interaction. Fresh v3 is used a second time here, so its
numbers attribute effects but do not test them. Round 4 keeps both changes unless
exactly one arm alone is no worse than E9 by more than 0.01 on every metric; then
it keeps only that change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from compare_round3 import _model_paths, compare
    from select_round3 import summarize_runs
except ModuleNotFoundError:  # imported by the tests as scripts.ablate_round3
    from scripts.compare_round3 import _model_paths, compare
    from scripts.select_round3 import summarize_runs

ARMS = ("e8", "e9_data", "e9_padding", "e9")
SINGLE = ("e9_data", "e9_padding")
NAMES = {"e8": "E8", "e9_data": "E9-data", "e9_padding": "E9-padding", "e9": "E9"}
# Arms that must share a manifest: the data change is the manifest.
MANIFEST_PAIRS = (("e8", "e9_padding"), ("e9_data", "e9"))
TOLERANCE = 0.01
METRICS = {  # name: (label, higher is better)
    "validation_bias": ("Validation real-source bias", False),
    "validation_mean": ("Validation mean domain AUROC", True),
    "validation_ff": ("Validation FF++ AUROC", True),
    "validation_stable_cascade": ("Validation Stable Cascade AUROC", True),
    "fresh_v3_openfake": ("Fresh v3 `openfake` AUROC", True),
    "fresh_v3_bias": ("Fresh v3 real-source bias", False),
}


def _validation(summary: dict) -> dict[str, float]:
    return {"validation_bias": summary["source_bias"]["mean_auroc"],
            "validation_mean": summary["score_mean"],
            "validation_ff": summary["domain_mean_auroc"]["ff"],
            "validation_stable_cascade": summary["domain_mean_auroc"]["stable_cascade_lite"]}


def _no_worse(arm: dict[str, float], full: dict[str, float]) -> bool:
    return all(arm[name] >= full[name] - TOLERANCE if higher
               else arm[name] <= full[name] + TOLERANCE
               for name, (_, higher) in METRICS.items())


def ablate(runs: dict[str, list[Path]], tests: dict[str, list[Path]]) -> dict:
    if set(runs) != set(ARMS) or set(tests) != set(ARMS):
        raise ValueError(f"need runs and fresh v3 predictions for each of {ARMS}")
    if len({len(runs[arm]) for arm in ARMS} | {len(tests[arm]) for arm in ARMS}) != 1:
        raise ValueError("every arm needs the same number of seeds")
    summaries = {arm: summarize_runs(runs[arm]) for arm in ARMS}
    for first, second in MANIFEST_PAIRS:
        if summaries[first]["manifest_id"] != summaries[second]["manifest_id"]:
            raise ValueError(f"{first} and {second} must share a manifest")
    if summaries["e8"]["manifest_id"] == summaries["e9"]["manifest_id"]:
        raise ValueError("e8 and e9 must use different manifests")
    metrics = {arm: _validation(summary) for arm, summary in summaries.items()}
    validation = {arm: [run / "validation.jsonl" for run in runs[arm]] for arm in ARMS}
    fresh_v3 = {}
    for arm in ARMS[1:]:
        result = compare((validation["e8"], tests["e8"]), (validation[arm], tests[arm]), {})
        fresh_v3[arm] = {"detection": result["primary"]["detection"],
                         "source_bias": result["primary"]["source_bias"]}
        for name, key in (("baseline_e8", "e8"), ("candidate_e9", arm)):
            model = result["models"][name]
            metrics[key]["fresh_v3_openfake"] = model["domains"]["openfake"]["auroc"]
            metrics[key]["fresh_v3_bias"] = model["bias_primary"]
    attribution = {}
    for name in METRICS:
        change = {arm: metrics[arm][name] - metrics["e8"][name] for arm in ARMS[1:]}
        attribution[name] = {
            **{f"{arm}_change": value for arm, value in change.items()},
            "interaction": change["e9"] - change["e9_data"] - change["e9_padding"],
        }
    sufficient = [arm for arm in SINGLE if _no_worse(metrics[arm], metrics["e9"])]
    return {
        "metrics": metrics,
        "attribution": attribution,
        "fresh_v3_against_e8": fresh_v3,
        "sufficient_alone": sufficient,
        "round4_recipe": sufficient[0] if len(sufficient) == 1 else "e9",
        "rule": {"tolerance": TOLERANCE},
        "seeds": len(runs["e8"]),
        "manifests": {arm: summary["manifest_id"] for arm, summary in summaries.items()},
    }


def _ci(values: list[float]) -> str:
    return f"[{values[0]:+.3f}, {values[1]:+.3f}]"


def render(result: dict) -> str:
    metrics, attribution = result["metrics"], result["attribution"]
    lines = [
        "# Open-world round 3 ablation result", "",
        "The [protocol](round3_ablation_protocol.md) was fixed before training. "
        f"Seeds per arm: {result['seeds']}. Fresh v3 is used a second time, so its rows "
        "attribute effects but do not test them.", "",
        "## Metrics", "",
        "| Metric | E8 | E9-data | E9-padding | E9 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, (label, _) in METRICS.items():
        lines.append(f"| {label} | " + " | ".join(f"{metrics[arm][name]:.3f}" for arm in ARMS)
                     + " |")
    lines += [
        "", "## Change from E8", "",
        "Real-source bias is better when lower (0.5 = no source information); AUROC is "
        "better when higher. The interaction is E9's change minus both single changes.", "",
        "| Metric | Data alone | Padding alone | Both (E9) | Interaction |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, (label, _) in METRICS.items():
        row = attribution[name]
        lines.append(f"| {label} | {row['e9_data_change']:+.3f} | "
                     f"{row['e9_padding_change']:+.3f} | {row['e9_change']:+.3f} | "
                     f"{row['interaction']:+.3f} |")
    lines += [
        "", "## Fresh v3 against E8 (paired bootstrap, 95% CI)", "",
        "| Arm | `openfake` AUROC change | CI | Bias change | CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm, values in result["fresh_v3_against_e8"].items():
        detection, bias = values["detection"], values["source_bias"]
        lines.append(f"| {NAMES[arm]} | {detection['difference']:+.3f} "
                     f"| {_ci(detection['ci95'])} | {-bias['drop']:+.3f} "
                     f"| {_ci([-bias['ci95'][1], -bias['ci95'][0]])} |")
    sufficient = [NAMES[arm] for arm in result["sufficient_alone"]] or ["neither"]
    lines += [
        "", "## Round 4 recipe", "",
        f"Arms no worse than E9 by more than {result['rule']['tolerance']:.2f} on every "
        f"metric: {', '.join(sufficient)}. Round 4 starts from "
        f"**{NAMES[result['round4_recipe']]}**.", "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="ARM=RUN_DIR")
    parser.add_argument("--test", action="append", required=True, metavar="ARM=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args(argv)
    result = ablate(_model_paths(args.run, "--run"), _model_paths(args.test, "--test"))
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output.exists() and args.output.read_text(encoding="utf-8") != payload:
        raise RuntimeError(f"refusing to overwrite changed ablation: {args.output}")
    for path, text in ((args.output, payload), (args.markdown, render(result))):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(json.dumps({"round4_recipe": result["round4_recipe"],
                      "sufficient_alone": result["sufficient_alone"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
