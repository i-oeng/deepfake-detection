from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from deepfake_detection.train import corruptions
from deepfake_detection.train.extended import _prediction_path, build_report


def _image() -> Image.Image:
    # Odd-sized texture: a smooth gradient would be invariant under blur.
    texture = np.random.default_rng(0).integers(0, 256, (259, 259, 3), dtype=np.uint8)
    return Image.fromarray(texture)


@pytest.mark.parametrize(
    "name,level",
    [setting for setting in corruptions.settings() if setting[0] != "h264"],
)
def test_corruptions_preserve_size_and_are_seeded_by_sample(name: str, level: float) -> None:
    first = corruptions.apply(_image(), name, level, "sample-a")
    again = corruptions.apply(_image(), name, level, "sample-a")
    assert first.size == (259, 259)
    assert first.tobytes() == again.tobytes()
    assert first.tobytes() != _image().tobytes()


def test_noise_differs_between_samples() -> None:
    a = corruptions.apply(_image(), "noise", 5, "sample-a")
    b = corruptions.apply(_image(), "noise", 5, "sample-b")
    assert a.tobytes() != b.tobytes()


def test_unregistered_severity_is_rejected() -> None:
    with pytest.raises(ValueError, match="unregistered corruption"):
        corruptions.apply(_image(), "jpeg", 10, "sample-a")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_h264_round_trip_handles_odd_sizes() -> None:
    first = corruptions.apply(_image(), "h264", 38, "sample-a")
    assert first.size == (259, 259)
    assert first.tobytes() == corruptions.apply(_image(), "h264", 38, "sample-a").tobytes()


def _frames(domain: str, quality: float) -> list[dict]:
    rows = []
    for method in ("simswap", "real") if domain != "celeba" else ("starganv2", "real"):
        for video in range(6):
            label = int(method != "real")
            for frame in range(2):
                rows.append({
                    "source_domain": domain, "video_id": f"{domain}:{method}{video}",
                    "fake_method": "" if method == "real" else method, "label": label,
                    "lineage_video_ids": f"{domain}:{video}",
                    "score": 0.5 + quality * (label - 0.5) + 0.01 * (video + frame),
                })
    return rows


def test_report_collects_runs_comparisons_and_robustness(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("")
    heldout, frozen = tmp_path / "m" / "heldout", tmp_path / "m" / "frozen"
    config = {
        "output_root": "out", "data_root": "data",
        "heldout_manifest": "m/heldout", "frozen_manifest": "m/frozen",
        "runs": [
            {"name": "weak", "family": "A", "seed": 1, "run": "runs/weak"},
            {"name": "strong", "family": "B", "seed": 1, "run": "runs/strong"},
            {"name": "absent", "family": "B", "seed": 2, "run": "runs/absent"},
        ],
        "comparisons": [
            {"name": "strong minus weak", "domain": "ff", "baseline": "weak",
             "candidate": "strong"},
        ],
    }
    for name, quality in (("weak", 0.1), ("strong", 0.8)):
        run = tmp_path / "runs" / name
        for manifest, domains in ((heldout, ("ff", "cdf", "celeba")), (frozen, ("ff", "cdf"))):
            for corruption in (None, ("jpeg", 30)):
                path = _prediction_path(tmp_path / "out", run, manifest, "test", corruption)
                path.parent.mkdir(parents=True, exist_ok=True)
                factor = 1.0 if corruption is None else 0.5
                rows = [row for d in domains for row in _frames(d, quality * factor)]
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = build_report(config, tmp_path)
    assert set(result["runs"]) == {"weak", "strong"}
    assert result["missing_runs"] == ["absent"]
    assert result["runs"]["strong"]["heldout"]["celeba"]["macro"]["auroc"] == 1.0
    assert result["comparisons"]["strong minus weak"]["source_groups"] == 6
    jpeg = result["robustness"]["strong"]["heldout/ff"]["jpeg-30"]
    assert jpeg["drop"] == pytest.approx(
        result["robustness"]["strong"]["heldout/ff"]["clean"]["auroc"] - jpeg["auroc"]
    )
