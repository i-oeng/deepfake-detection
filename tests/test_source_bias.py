import json

import pytest

from deepfake_detection.train import source_bias


def _record(domain, video, score, label=0, method=""):
    return {
        "sample_id": f"{domain}-{video}-{score}", "source_domain": domain,
        "video_id": f"{domain}:{video}", "fake_method": method, "label": label,
        "score": score, "lineage_video_ids": f"{domain}:{video}",
    }


def test_real_videos_average_frames_and_skip_fakes():
    records = [
        _record("ff", "a", 0.2), _record("ff", "a", 0.4), _record("ff", "b", 0.1),
        _record("ff", "c", 0.9, label=1, method="simswap"),
    ]
    assert source_bias.real_videos(records) == {"ff": pytest.approx([0.1, 0.3])}


def test_source_auroc_is_half_when_sources_score_alike_and_one_when_separated():
    assert source_bias.source_auroc([0.1, 0.2], [0.1, 0.2]) == 0.5
    assert source_bias.source_auroc([0.1, 0.2], [0.8, 0.9]) == 1.0


def test_locked_threshold_balances_sources():
    # 10 FF++ reals all low, 10 photo reals all high: an FF++-sized pool would
    # set the threshold below every photo; equal weighting keeps the FPR at 10%.
    sources = {"ff": [0.01 * i for i in range(10)],
               "photo": [0.5 + 0.01 * i for i in range(10)]}
    threshold = source_bias.locked_threshold(sources, 0.10)
    rates = [source_bias.false_positive_rate(scores, threshold)
             for scores in sources.values()]
    assert sum(rates) / 2 <= 0.10
    assert threshold == pytest.approx(0.57)


def test_summarize_reports_validation_and_test_sources():
    validation = [_record("ff", f"f{i}", 0.1 + 0.01 * i) for i in range(5)]
    validation += [_record("cdf", f"c{i}", 0.6 + 0.01 * i) for i in range(5)]
    validation += [_record("cdf", "x", 0.95, label=1, method="sit")]
    test = [_record("lfw", f"l{i}", 0.12 + 0.01 * i) for i in range(4)]
    result = source_bias.summarize(validation, test)
    assert result["validation"]["cdf"]["auroc_vs_reference"] == 1.0
    assert result["validation_mean_auroc"] == 1.0
    assert 0.0 < result["test"]["lfw"]["auroc_vs_reference"] < 1.0
    assert "ff" not in result["validation"]


def test_summarize_requires_reference_reals():
    with pytest.raises(ValueError, match="no ff reals"):
        source_bias.summarize([_record("cdf", "a", 0.5)])


def test_cli_groups_seeds_by_model(tmp_path):
    paths = []
    for seed in range(2):
        path = tmp_path / f"validation-{seed}.jsonl"
        rows = [_record("ff", "a", 0.1), _record("ff", "b", 0.2),
                _record("stargan", "c", 0.3 + seed), _record("stargan", "d", 0.05)]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        paths.append(path)
    output, table = tmp_path / "bias.json", tmp_path / "bias.md"
    assert source_bias.main([*(f"--validation=e8={path}" for path in paths),
                             "--output", str(output), "--markdown", str(table)]) == 0
    result = json.loads(output.read_text())
    assert len(result["e8"]["seeds"]) == 2
    assert "| e8 |" in table.read_text()
