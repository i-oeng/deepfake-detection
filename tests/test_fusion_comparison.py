import hashlib
import json
import random

import numpy as np
import pytest

from deepfake_detection.train.fusion_comparison import (
    compare_pair,
    frozen_weight,
    pair_passes,
    verdict,
)


def _export(directory, split, rows, checkpoint: bytes) -> dict:
    directory.mkdir()
    (directory / "best.pt").write_bytes(checkpoint)
    (directory / f"{split}.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    np.savez_compressed(
        directory / f"{split}_embeddings.npz",
        sample_ids=np.array([row["sample_id"] for row in rows]),
        embeddings=np.zeros((len(rows), 2), dtype=np.float32),
    )
    metadata = {
        "manifest_id": "manifest",
        "manifest_sha256": "a" * 64,
        "sample_preprocessing_id": "shared-crops-v1",
        "preprocessing_id": directory.name,
        "checkpoint_file": "best.pt",
        "checkpoint_sha256": hashlib.sha256(checkpoint).hexdigest(),
    }
    (directory / f"{split}_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return metadata


def _rows(split: str, rng: random.Random, signal: float) -> list[dict]:
    """Eight source groups, each with a real video and one video per fake method."""
    rows = []
    for group in range(8):
        for method, label in (("", 0), ("method-a", 1), ("method-b", 1)):
            for frame in range(2):
                logit = signal * (1 if label else -1) + rng.gauss(0, 1)
                rows.append({
                    "sample_id": f"g{group}-{method or 'real'}-{frame}",
                    "split": split,
                    "video_id": f"g{group}-{method or 'real'}",
                    "source_domain": "ff",
                    "fake_method": method,
                    "label": label,
                    "logit": logit,
                    "score": 1 / (1 + np.exp(-logit)),
                    "lineage_video_ids": f"ff:src{group}",
                })
    return rows


@pytest.fixture
def pair(tmp_path):
    rgb_meta = _export(tmp_path / "rgb", "validation", _rows("validation", random.Random(1), 0.4),
                       b"rgb")
    freq_meta = _export(tmp_path / "freq", "validation",
                        _rows("validation", random.Random(2), 1.5), b"freq")
    dev = {"manifest_id": "manifest", "selection_split": "validation",
           "selected_rgb_weight": 0.3, "rgb": rgb_meta, "frequency": freq_meta}
    dev_path = tmp_path / "fusion-dev.json"
    dev_path.write_text(json.dumps(dev), encoding="utf-8")
    return tmp_path / "rgb", tmp_path / "freq", dev_path, dev, rgb_meta, freq_meta


def test_weight_comes_from_development_report_with_matching_hashes(pair) -> None:
    *_, dev, rgb_meta, freq_meta = pair
    assert frozen_weight(dev, rgb_meta, freq_meta) == 0.3
    with pytest.raises(ValueError, match="rgb checkpoint differs"):
        frozen_weight(dev, {**rgb_meta, "checkpoint_sha256": "0" * 64}, freq_meta)
    with pytest.raises(ValueError, match="development-only"):
        frozen_weight({**dev, "final_test": {}}, rgb_meta, freq_meta)
    with pytest.raises(ValueError, match="validation split"):
        frozen_weight({**dev, "selection_split": "test"}, rgb_meta, freq_meta)


def test_compare_pair_scores_frozen_blend_against_rgb(pair) -> None:
    rgb_dir, freq_dir, dev_path, *_ = pair
    result = compare_pair(rgb_dir, freq_dir, dev_path, split="validation", domains=("ff",),
                          repeats=200)
    assert result["selected_rgb_weight"] == 0.3
    ff = result["domains"]["ff"]
    assert set(ff) == {"rgb", "frequency", "tuned_blend", "tuned_minus_rgb"}
    difference = ff["tuned_minus_rgb"]
    assert difference["difference"] == pytest.approx(
        ff["tuned_blend"]["macro"]["auroc"] - ff["rgb"]["macro"]["auroc"]
    )
    assert difference["ci95"][0] <= difference["difference"] <= difference["ci95"][1]
    assert difference["source_groups"] == 8


def test_pure_rgb_weight_cannot_pass(pair, tmp_path) -> None:
    rgb_dir, freq_dir, _, dev, *_ = pair
    dev_path = tmp_path / "fusion-rgb-only.json"
    dev_path.write_text(json.dumps({**dev, "selected_rgb_weight": 1.0}), encoding="utf-8")
    result = compare_pair(rgb_dir, freq_dir, dev_path, split="validation", domains=("ff",),
                          repeats=200)
    assert result["domains"]["ff"]["tuned_minus_rgb"]["ci95"] == [0.0, 0.0]
    assert not pair_passes(result)


def _summary(ff_low: float, ff_diff: float, cdf_low: float) -> dict:
    return {"domains": {
        "ff": {"tuned_minus_rgb": {"ci95": [ff_low, 1.0], "difference": ff_diff}},
        "cdf": {"tuned_minus_rgb": {"ci95": [cdf_low, 1.0], "difference": 0.1}},
    }}


def test_decision_rule_needs_a_ci_above_zero_without_ff_regression() -> None:
    assert pair_passes(_summary(ff_low=0.001, ff_diff=0.01, cdf_low=-0.1))
    assert pair_passes(_summary(ff_low=-0.02, ff_diff=-0.005, cdf_low=0.02))
    assert not pair_passes(_summary(ff_low=-0.02, ff_diff=-0.02, cdf_low=0.02))
    assert not pair_passes(_summary(ff_low=-0.01, ff_diff=0.0, cdf_low=0.0))


def test_verdict_requires_a_majority_of_seeds() -> None:
    passing, failing = _summary(0.01, 0.02, -0.1), _summary(-0.01, 0.0, -0.1)
    assert verdict([passing, passing, failing])["frequency_helps"]
    assert not verdict([passing, failing, failing])["frequency_helps"]
