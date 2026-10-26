import json

from scripts.select_round2 import select


def _report(path, manifest, scores):
    path.write_text(json.dumps({
        "manifest_id": manifest,
        "validation": {
            "score": sum(scores.values()) / len(scores),
            "domain_macro_auroc": scores,
        },
    }))
    return path


def test_selection_requires_score_gain_and_ff_guardrail(tmp_path):
    baseline = _report(tmp_path / "baseline.json", "manifest", {
        "ff": 0.80, "cdf": 0.75, "stable_cascade_lite": 0.50,
    })
    candidate = _report(tmp_path / "candidate.json", "manifest", {
        "ff": 0.795, "cdf": 0.77, "stable_cascade_lite": 0.60,
    })
    result = select([baseline], [candidate])
    assert result["advance_e8"]
    assert result["score_gain"] > 0

    failed = _report(tmp_path / "failed.json", "manifest", {
        "ff": 0.78, "cdf": 0.79, "stable_cascade_lite": 0.61,
    })
    result = select([baseline], [failed])
    assert not result["advance_e8"]
    assert result["ff_change"] < -0.01


def test_selection_aggregates_seed_reports(tmp_path):
    baseline = [
        _report(tmp_path / f"baseline-{index}.json", "manifest", {
            "ff": 0.80 + index * 0.01,
            "cdf": 0.70,
            "stable_cascade_lite": 0.50,
        })
        for index in range(2)
    ]
    candidate = [
        _report(tmp_path / f"candidate-{index}.json", "manifest", {
            "ff": 0.80,
            "cdf": 0.75,
            "stable_cascade_lite": 0.60,
        })
        for index in range(2)
    ]
    result = select(baseline, candidate)
    assert result["baseline_e7"]["seeds"] == 2
    assert result["candidate_e8"]["score_stdev"] == 0.0
