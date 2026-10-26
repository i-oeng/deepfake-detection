from scripts.render_round2_report import render


def test_markdown_report_includes_gate_and_calibration():
    validation = {
        "manifest_id": "round2", "seeds": 3, "score_mean": 0.7,
        "domain_mean_auroc": {"ff": 0.8, "stable_cascade_lite": 0.6},
    }
    method = {
        "auroc": 0.7, "average_precision": 0.8, "tpr_at_1pct_fpr": 0.1,
        "brier": 0.2, "ece_10": 0.05,
    }
    model = {
        "score": 0.7,
        "domains": {"dff": {"macro": method, "by_method": {"sd15": method}}},
    }
    markdown = render(
        {
            "advance_e8": True, "score_gain": 0.02, "ff_change": -0.005,
            "baseline_e7": validation, "candidate_e8": validation,
        },
        {
            "baseline_e7": model, "candidate_e8": model,
            "seed_spread": {"baseline_e7": [0.6, 0.7, 0.8],
                            "candidate_e8": [0.7, 0.7, 0.7]},
            "bootstrap": {
                "difference": 0.06, "ci95": [0.01, 0.1], "replicates": 2000,
                "unit": "paired videos", "clear_improvement": True,
            },
        },
        {"test": {"score": 0.5}},
        {"round2_manifest_id": "round2", "fresh_manifest_id": "fresh", "addition_rows": 10},
    )
    assert "clear-improvement gate" in markdown
    assert "ECE-10" in markdown
    assert "**met**" in markdown
