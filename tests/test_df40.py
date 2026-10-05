from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from deepfake_detection.data.df40 import normalize_df40


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_normalizes_official_layout_and_deduplicates_shared_real_frames(tmp_path: Path) -> None:
    source = tmp_path / "json"
    real = "/old/FaceForensics++/original_sequences/youtube/c23/frames/001/000.png"
    _write(
        source / "simswap_ff.json",
        {"simswap_ff": {
            "roop_Real": {"train": {"c23": {"001": {"label": "roop_Real", "frames": [real]}}}},
            "roop_Fake": {"train": {"c23": {"001_002": {
                "label": "roop_Fake", "frames": ["/old/DF40/simswap/ff/001_002/003.png"]
            }}}},
        }},
    )
    _write(
        source / "wav2lip_ff.json",
        {"wav2lip_ff": {
            "wav2lip_Real": {"train": {"c23": {"001": {
                "label": "wav2lip_Real", "frames": [real]
            }}}},
            "wav2lip_Fake": {"val": {"c23": {"003": {
                "label": "wav2lip_Fake", "frames": ["/old/DF40/wav2lip/ff/003/004.png"]
            }}}},
        }},
    )
    output = tmp_path / "metadata.csv"
    summary = normalize_df40(source, output)
    data = pd.read_csv(output, dtype=str, keep_default_na=False)

    assert summary["rows"] == 3
    assert data.groupby(["split", "label"]).size().to_dict() == {
        ("train", "REAL"): 1,
        ("train", "FAKE"): 1,
        ("validation", "FAKE"): 1,
    }
    assert set(data["source_domain"]) == {"ff"}
    assert set(data["fake_method"]) == {"", "simswap", "wav2lip"}
    assert all(path.startswith("images/") for path in data["relative_path"])
    assert data["image_id"].is_unique
    assert normalize_df40(source, output)["rows"] == 3


def test_rejects_official_frame_reused_in_different_splits(tmp_path: Path) -> None:
    source = tmp_path / "json"
    frame = "/old/DF40/simswap/ff/video/000.png"
    _write(source / "simswap_ff.json", {"simswap_ff": {"roop_Fake": {
        "train": {"video": {"label": "roop_Fake", "frames": [frame]}},
        "test": {"video": {"label": "roop_Fake", "frames": [frame]}},
    }}})
    with pytest.raises(ValueError, match="conflicting official frame assignment"):
        normalize_df40(source, tmp_path / "metadata.csv")


def test_rejects_unrecognized_catalog_instead_of_guessing_domain(tmp_path: Path) -> None:
    source = tmp_path / "json"
    _write(source / "DF40_all.json", {})
    with pytest.raises(ValueError, match="unsupported DF40 catalog filename"):
        normalize_df40(source, tmp_path / "metadata.csv")
