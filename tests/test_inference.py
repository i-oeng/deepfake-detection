from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("cv2")

from deepfake_detection.inference import bundle, face, media  # noqa: E402


def test_similarity_recovers_rotation_scale_and_translation() -> None:
    angle, scale, shift = 0.3, 1.7, np.array([12.0, -5.0])
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    source = face.TEMPLATE.copy()
    destination = source @ (scale * rotation).T + shift
    matrix = face.similarity(source, destination)
    np.testing.assert_allclose(matrix[:, :2], scale * rotation, atol=1e-9)
    np.testing.assert_allclose(matrix[:, 2], shift, atol=1e-9)


def test_crop_template_matches_deepfakebench_margin() -> None:
    template = face.crop_template(256, 1.3)
    # The 1.3 margin shrinks the ArcFace layout towards the crop centre.
    expected = face.TEMPLATE * (256 / 112) / 1.3 + 256 * 0.15 / 1.3
    np.testing.assert_allclose(template, expected)
    assert template[0, 0] < template[1, 0] and template[3, 0] < template[4, 0]


def test_align_places_landmarks_on_template() -> None:
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    landmarks = face.TEMPLATE * 2 + 60
    for x, y in landmarks:
        image[int(round(y)) - 1: int(round(y)) + 2, int(round(x)) - 1: int(round(x)) + 2] = 255
    crop = face.align(image, landmarks)
    assert crop.shape == (256, 256, 3)
    for x, y in face.crop_template():
        assert crop[int(round(y)), int(round(x))].max() > 0


class _Detector:
    def __init__(self, faces: list[face.Face]) -> None:
        self.faces = faces

    def detect(self, rgb: np.ndarray) -> list[face.Face]:
        return sorted(self.faces, key=lambda f: f.area, reverse=True)


def _face(size: float, score: float) -> face.Face:
    return face.Face((10.0, 10.0, size, size), score, face.TEMPLATE + 10)


def test_policy_scores_largest_confident_face_and_records_count() -> None:
    rgb = np.zeros((200, 200, 3), dtype=np.uint8)
    result = face.extract_face(rgb, _Detector([_face(150, 0.6), _face(90, 0.95)]))
    assert result.reason is None and result.faces_detected == 2
    assert result.face is not None and result.face.score == 0.95


@pytest.mark.parametrize(
    "faces,reason",
    [([], "no_face"), ([_face(120, 0.7)], "low_confidence"), ([_face(40, 0.99)], "face_too_small")],
)
def test_policy_abstains(faces: list[face.Face], reason: str) -> None:
    result = face.extract_face(np.zeros((200, 200, 3), dtype=np.uint8), _Detector(faces))
    assert result.crop is None and result.reason == reason


def test_aggregations() -> None:
    logits = [-2.0, 0.0, 1.0, 3.0, 8.0]
    assert bundle.aggregate(logits, "mean_logit") == pytest.approx(2.0)
    assert bundle.aggregate(logits, "median_logit") == 1.0
    assert bundle.aggregate(logits, "trimmed_mean_logit") == pytest.approx(4 / 3)
    assert bundle.aggregate(logits, "top25_mean_logit") == pytest.approx(5.5)
    mean_probability = sum(1 / (1 + math.exp(-v)) for v in logits) / len(logits)
    assert bundle._sigmoid(bundle.aggregate(logits, "mean_probability")) == pytest.approx(
        mean_probability)


def test_platt_recovers_known_parameters() -> None:
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 3, 20000)
    labels = (rng.random(20000) < 1 / (1 + np.exp(-(0.5 * logits - 1.0)))).astype(int)
    a, b = bundle.fit_platt(logits.tolist(), labels.tolist())
    assert a == pytest.approx(0.5, abs=0.03) and b == pytest.approx(-1.0, abs=0.06)


def _frames(spread: float) -> list[dict]:
    rows = []
    for method in ("facedancer", "real"):
        for video in range(10):
            label = int(method != "real")
            for frame in range(4):
                rows.append({"source_domain": "ff", "video_id": f"{method}{video}",
                             "fake_method": "" if method == "real" else method, "label": label,
                             "logit": (2 * label - 1) * (video % 3) + spread * (frame - 1.5)})
    return rows


def test_aggregation_ties_resolve_to_mean_probability() -> None:
    chosen, table = bundle.select_aggregation(_frames(0.1))
    assert set(table) == set(bundle.AGGREGATIONS)
    assert chosen == "mean_probability"


def test_even_positions_match_subset_sampler() -> None:
    assert media.even_positions(5, 8) == [0, 1, 2, 3, 4]
    assert media.even_positions(100, 4) == [0, 33, 66, 99]
    with pytest.raises(ValueError):
        media.media_kind(media.Path("clip.gif"))
