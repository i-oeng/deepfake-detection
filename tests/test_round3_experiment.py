import json
import math

from scripts import compare_round3, render_round3_report, select_round3

VALIDATION = ("ff", "cdf", "stargan", "styleclip", "whichfaceisreal", "stable_cascade_lite")


def _record(domain, video, label, score, method="", split="validation"):
    logit = math.log(score / (1 - score))
    return {
        "sample_id": f"{domain}-{video}-{label}", "source_id": f"{domain}-{video}",
        "split": split, "video_id": f"{domain}:{video}", "source_domain": domain,
        "fake_method": method if label else "",
        "manipulation_family": "entire_face_synthesis" if label else "real",
        "label": label, "lineage_video_ids": f"{domain}:{video}", "logit": logit,
        "score": score,
    }


def _validation(other_real_score):
    rows = []
    for domain in VALIDATION:
        real = 0.2 if domain == "ff" else other_real_score
        for video in range(6):
            rows.append(_record(domain, f"r{video}", 0, real + 0.01 * video))
            rows.append(_record(domain, f"f{video}", 1, 0.85 + 0.01 * video, method=domain))
    return rows


def _run(tmp_path, name, other_real_score):
    directory = tmp_path / name
    directory.mkdir()
    rows = _validation(other_real_score)
    (directory / "validation.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    domains = {domain: 0.9 for domain in VALIDATION}
    (directory / "report.json").write_text(json.dumps({
        "manifest_id": "round3", "validation": {"score": 0.9, "domain_macro_auroc": domains}}))
    return directory


def test_selection_requires_a_bias_drop_without_losing_detection(tmp_path):
    baseline = [_run(tmp_path, "e8", 0.6)]
    result = select_round3.select(baseline, [_run(tmp_path, "e9", 0.21)])
    assert result["advance_e9"]
    assert result["bias_drop"] > 0.3
    unchanged = select_round3.select(baseline, [_run(tmp_path, "e9-same", 0.6)])
    assert not unchanged["advance_e9"]


def _test_records(real_score, fake_score):
    rows = []
    for video in range(8):
        rows.append(_record("openfake", f"real-laion-{video:06d}", 0, real_score + 0.01 * video,
                            split="test"))
        generator = "flux.1-dev" if video % 2 else "midjourney-7"
        rows.append(_record("openfake", f"fake-{generator}-{video + 100:06d}", 1,
                            fake_score + 0.01 * video, method="openfake", split="test"))
        rows.append(_record("openfake_reddit", f"real-reddit-{video:06d}", 0,
                            real_score + 0.01 * video, split="test"))
        rows.append(_record("openfake_reddit", f"fake-reddit-{video + 100:06d}", 1,
                            fake_score, method="reddit_ai", split="test"))
        rows.append(_record("lfw", f"real-person-{video}", 0, real_score + 0.02 * video,
                            split="test"))
    return rows


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_comparison_gates_bias_and_detection_and_renders(tmp_path):
    base_validation = _write(tmp_path / "e8-validation.jsonl", _validation(0.6))
    cand_validation = _write(tmp_path / "e9-validation.jsonl", _validation(0.21))
    base_test = _write(tmp_path / "e8-test.jsonl", _test_records(0.6, 0.62))
    cand_test = _write(tmp_path / "e9-test.jsonl", _test_records(0.21, 0.7))
    result = compare_round3.compare(([base_validation], [base_test]),
                                    ([cand_validation], [cand_test]), {})
    decision = result["decision"]
    assert decision["bias_reduced"] and decision["round3_succeeds"]
    assert result["primary"]["detection"]["difference"] > 0
    assert set(result["models"]["candidate_e9"]["generators"]) == set()

    selection = select_round3.select([_run(tmp_path, "e8", 0.6)],
                                     [_run(tmp_path, "e9", 0.21)])
    audit = {"round3_manifest_id": "round3", "passed": True,
             "training_real_namespaces": {"fairface": 8000, "ff": 28000}}
    text = render_round3_report.render(selection, audit, result, None)
    assert "Round 3 succeeds: **yes**" in text
    assert "| E9 |" in text
