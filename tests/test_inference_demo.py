from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("cv2")

from deepfake_detection.inference import demo, face  # noqa: E402


class _Detector:
    def __init__(self, faces: list[face.Face]) -> None:
        self.faces = faces
        self.calls = 0

    def detect(self, _rgb: np.ndarray) -> list[face.Face]:
        self.calls += 1
        return self.faces


class _Scorer:
    bundle = {
        "calibration": {"a": 0.5, "b": -0.2},
        "model_version": "test-model-123",
        "threshold": 0.55,
    }
    device = "cpu"

    def frame_logits(self, crops: list[np.ndarray]) -> list[float]:
        assert len(crops) == 1 and crops[0].shape == (256, 256, 3)
        return [2.0]

    def decide(self, logits: list[float], *, video: bool) -> dict:
        assert not video
        if not logits:
            return {"label": "ABSTAIN", "probability_fake": None,
                    "aggregated_logit": None}
        probability = 1 / (1 + math.exp(-(0.5 * logits[0] - 0.2)))
        return {"label": "FAKE", "probability_fake": probability,
                "aggregated_logit": logits[0]}


def test_demo_scores_one_detection_and_renders_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "example.png"
    Image.new("RGB", (240, 200), "#263544").save(path)
    selected = face.Face((25.0, 20.0, 130.0, 130.0), 0.98, face.TEMPLATE + 45)
    extra = face.Face((180.0, 30.0, 30.0, 30.0), 0.72, face.TEMPLATE)
    detector = _Detector([selected, extra])

    analysis = demo.analyze_image(path, _Scorer(), detector)

    assert detector.calls == 1
    assert analysis.selected is selected
    assert analysis.frame_logit == 2.0
    assert analysis.calibrated_logit == pytest.approx(0.8)
    assert analysis.decision["label"] == "FAKE"
    terminal = "\n".join(text for text, _style in demo.terminal_lines(analysis))
    assert "selected bb    x=25 y=20 w=130 h=130" in terminal
    assert "P(fake)" in terminal and "> VERDICT      FAKE" in terminal

    snapshot = demo.render_snapshot(analysis, max_image_size=(300, 300))
    assert snapshot.width > 700
    assert snapshot.height >= 600


def test_demo_abstains_without_a_face(tmp_path: Path) -> None:
    path = tmp_path / "empty.png"
    Image.new("RGB", (100, 80), "black").save(path)

    analysis = demo.analyze_image(path, _Scorer(), _Detector([]))

    assert not analysis.accepted
    assert analysis.reason == "no_face"
    assert analysis.decision["label"] == "ABSTAIN"
    assert analysis.frame_logit is None
    assert "selected       none" in "\n".join(
        text for text, _style in demo.terminal_lines(analysis)
    )
