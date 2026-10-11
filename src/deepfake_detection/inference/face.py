"""Versioned face detection and alignment matching the DF40 crop geometry.

DF40 distributes DeepfakeBench crops: a five-landmark similarity alignment to the
ArcFace template, enlarged by a 1.3 margin and resampled to 256 x 256. This
module reproduces that geometry from raw frames with OpenCV's YuNet detector, so
models trained on DF40 crops see comparable inputs at inference time.
"""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/47534e27c9851bb1128ccc0102f1145e27f23f98/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
DETECTOR_ID = f"yunet-2023mar-{YUNET_SHA256[:12]}"
CROP_SIZE = 256
CROP_SCALE = 1.3
ALIGNMENT_ID = f"arcface5-scale{CROP_SCALE:g}-{CROP_SIZE}-bilinear-v1"
# ArcFace 112 x 112 template: eye centres, nose tip, mouth corners in image order.
TEMPLATE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float64,
)


def default_detector_path() -> Path:
    return Path.home() / ".cache" / "deepfake-detection" / f"{DETECTOR_ID}.onnx"


def ensure_detector(path: Path | None = None) -> Path:
    """Return the pinned YuNet model, downloading it once and verifying its SHA-256."""
    path = path or default_detector_path()
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".part")
        with urllib.request.urlopen(YUNET_URL, timeout=60) as response:
            temporary.write_bytes(response.read())
        temporary.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != YUNET_SHA256:
        raise ValueError(f"face detector hash mismatch: {path}")
    return path


@dataclass(frozen=True)
class Face:
    box: tuple[float, float, float, float]
    score: float
    landmarks: np.ndarray  # 5 x 2, image order: eyes, nose, mouth corners

    @property
    def short_side(self) -> float:
        return min(self.box[2], self.box[3])

    @property
    def area(self) -> float:
        return self.box[2] * self.box[3]


class FaceDetector:
    """YuNet detections sorted by box area, largest first."""

    def __init__(self, model: Path | None = None, *, score_threshold: float = 0.5) -> None:
        import cv2

        self._cv2 = cv2
        self._detector = cv2.FaceDetectorYN.create(
            str(ensure_detector(model)), "", (320, 320), score_threshold, 0.3, 5000
        )

    def detect(self, rgb: np.ndarray) -> list[Face]:
        height, width = rgb.shape[:2]
        self._detector.setInputSize((width, height))
        _, detections = self._detector.detect(self._cv2.cvtColor(rgb, self._cv2.COLOR_RGB2BGR))
        faces = []
        for row in detections if detections is not None else ():
            points = row[4:14].reshape(5, 2).astype(np.float64)
            eyes = points[:2][np.argsort(points[:2, 0])]
            mouth = points[3:5][np.argsort(points[3:5, 0])]
            faces.append(Face(tuple(float(v) for v in row[:4]), float(row[14]),
                              np.vstack([eyes, points[2:3], mouth])))
        return sorted(faces, key=lambda face: face.area, reverse=True)


def crop_template(size: int = CROP_SIZE, scale: float = CROP_SCALE) -> np.ndarray:
    """Landmark destinations in the enlarged output crop (DeepfakeBench convention)."""
    template = TEMPLATE * (size / 112.0)
    margin = size * (scale - 1) / 2.0
    return (template + margin) * (size / (size + 2 * margin))


def similarity(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (Umeyama), as a 2 x 3 affine matrix."""
    source_mean, destination_mean = source.mean(axis=0), destination.mean(axis=0)
    source_centered = source - source_mean
    destination_centered = destination - destination_mean
    covariance = destination_centered.T @ source_centered / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.ones(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[-1] = -1
    rotation = u @ np.diag(sign) @ vt
    variance = (source_centered**2).sum() / len(source)
    factor = (singular * sign).sum() / variance
    translation = destination_mean - factor * rotation @ source_mean
    return np.hstack([factor * rotation, translation[:, None]])


def align(rgb: np.ndarray, landmarks: np.ndarray, size: int = CROP_SIZE,
          scale: float = CROP_SCALE) -> np.ndarray:
    """Warp ``rgb`` so ``landmarks`` land on the enlarged template."""
    import cv2

    matrix = similarity(landmarks, crop_template(size, scale))
    return cv2.warpAffine(rgb, matrix, (size, size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


MIN_SCORE = 0.80
MIN_SHORT_SIDE = 64.0
POLICY_ID = f"largest-face-score{MIN_SCORE:g}-side{MIN_SHORT_SIDE:g}-v1"


@dataclass(frozen=True)
class FaceResult:
    """One frame's face decision: an aligned crop, or the reason there is none."""

    crop: np.ndarray | None
    face: Face | None
    faces_detected: int
    reason: str | None


def extract_face(rgb: np.ndarray, detector: FaceDetector) -> FaceResult:
    """Apply the pre-registered policy: the largest confident face, if it is large enough."""
    faces = detector.detect(rgb)
    if not faces:
        return FaceResult(None, None, 0, "no_face")
    confident = [face for face in faces if face.score >= MIN_SCORE]
    if not confident:
        return FaceResult(None, max(faces, key=lambda f: f.score), len(faces), "low_confidence")
    face = confident[0]
    if face.short_side < MIN_SHORT_SIDE:
        return FaceResult(None, face, len(faces), "face_too_small")
    return FaceResult(align(rgb, face.landmarks), face, len(faces), None)
