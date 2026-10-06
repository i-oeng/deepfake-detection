from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from deepfake_detection.train.evaluation import report
from deepfake_detection.train.fusion_contract import align, blend, load_export


def _row(sample: str, label: int, score: float, video: str, method: str = "") -> dict:
    return {
        "sample_id": sample,
        "split": "validation",
        "video_id": video,
        "source_domain": "ff",
        "fake_method": method,
        "label": label,
        "score": score,
        "logit": score,
        "lineage_video_ids": f"ff:{video}",
    }


def test_frozen_report_is_per_method_and_macro() -> None:
    rows = [
        _row("r1", 0, 0.1, "real-1"),
        _row("r2", 0, 0.2, "real-2"),
        _row("a1", 1, 0.8, "fake-a", "method-a"),
        _row("b1", 1, 0.9, "fake-b", "method-b"),
    ]
    result = report(rows)
    assert result["macro"]["auroc"] == 1.0
    assert result["macro"]["average_precision"] == 1.0
    assert set(result["by_method"]) == {"method-a", "method-b"}


def test_prediction_alignment_uses_sample_id_and_metadata() -> None:
    left = [_row("a", 0, 0.1, "v1"), _row("b", 1, 0.9, "v2", "fake")]
    right = [dict(left[1]), dict(left[0])]
    ordered_left, ordered_right = align(left, right)
    assert [row["sample_id"] for row in ordered_left] == ["a", "b"]
    assert [row["sample_id"] for row in ordered_right] == ["a", "b"]
    assert len(blend(left, right, 0.5)) == 2
    right[0]["video_id"] = "changed"
    with pytest.raises(ValueError, match="metadata mismatch"):
        align(left, right)


def test_export_loader_checks_embedding_order_and_checkpoint_hash(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    checkpoint = run / "frequency.pt"
    checkpoint.write_bytes(b"selected frequency weights")
    rows = [_row("a", 0, 0.1, "v1"), _row("b", 1, 0.9, "v2", "fake")]
    (run / "validation.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    np.savez_compressed(
        run / "validation_embeddings.npz",
        sample_ids=np.array(["a", "b"]),
        embeddings=np.zeros((2, 4), dtype=np.float32),
    )
    metadata = {
        "manifest_id": "manifest",
        "manifest_sha256": "a" * 64,
        "sample_preprocessing_id": "shared-crops-v1",
        "preprocessing_id": "frequency-fft-v1",
        "checkpoint_file": "frequency.pt",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    (run / "validation_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    loaded, loaded_metadata = load_export(run, "validation")
    assert [row["sample_id"] for row in loaded] == ["a", "b"]
    assert loaded_metadata == metadata

    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint hash differs"):
        load_export(run, "validation")
