import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("cv2")

from deepfake_detection.train.openworld import (  # noqa: E402
    SBI_METHOD,
    alignment_uniformity,
    source_fingerprint,
    with_pseudo_fakes,
)
from deepfake_detection.train.sbi import face_mask, self_blend  # noqa: E402


def _rows():
    rows = []
    for method in ("a", "b"):
        for video in range(3):
            rows.append({"label": "FAKE", "fake_method": method, "video_id": f"{method}{video}"})
    for video in range(4):
        rows.append({"label": "REAL", "fake_method": "", "video_id": f"r{video}"})
    return rows


def test_pseudo_fakes_receive_the_configured_share_of_fake_mass():
    rows, weights = with_pseudo_fakes(_rows(), 0.4)
    assert sum(row["fake_method"] == SBI_METHOD for row in rows) == 4
    mass = {}
    for row, weight in zip(rows, weights, strict=True):
        key = row["fake_method"] or "real"
        mass[key] = mass.get(key, 0.0) + weight
    assert mass["real"] == pytest.approx(0.5)
    assert mass[SBI_METHOD] == pytest.approx(0.2)
    assert mass["a"] == pytest.approx(0.15) and mass["b"] == pytest.approx(0.15)
    plain, plain_weights = with_pseudo_fakes(_rows(), 0.0)
    assert len(plain) == 10 and sum(plain_weights) == pytest.approx(1.0)


def test_self_blend_changes_only_the_face_region():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
    for seed in range(5):
        blended = self_blend(image, np.random.default_rng(seed))
        assert blended.shape == image.shape and blended.dtype == np.uint8
        changed = np.abs(blended.astype(int) - image.astype(int)).sum(axis=2) > 0
        # Corners lie outside every face mask.
        assert not changed[:16, :16].any() and not changed[-16:, -16:].any()
    mask = face_mask(256, np.random.default_rng(1))
    assert mask.max() == pytest.approx(1.0) and mask[128, 128] > 0.5


def test_alignment_and_uniformity():
    import torch

    features = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    alignment, uniformity = alignment_uniformity(features, labels)
    assert float(alignment) == pytest.approx(0.0)
    assert float(uniformity) < 0


def test_source_fingerprint_changes_with_training_source(tmp_path, monkeypatch):
    from deepfake_detection.train import openworld

    monkeypatch.setattr(openworld, "TRAINING_SOURCES", ("train.py", "data.py"))
    (tmp_path / "train.py").write_text("first")
    (tmp_path / "data.py").write_text("data")
    before, files = source_fingerprint(tmp_path)
    assert set(files) == {"train.py", "data.py"}

    (tmp_path / "train.py").write_text("second")
    after, _ = source_fingerprint(tmp_path)
    assert after != before
