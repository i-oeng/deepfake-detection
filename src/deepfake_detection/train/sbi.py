"""Self-blended pseudo-fakes from real face crops (Shiohara and Yamasaki, CVPR 2022).

A real crop is blended with a slightly transformed copy of itself inside a soft,
deformed face mask. The result carries blending boundaries and statistical
mismatches that many face-manipulation pipelines share, but no fingerprint of
any particular generator.

DF40 crops are aligned to the ArcFace five-point template, enlarged by 1.3, so
the landmark positions are known from the crop geometry and need no detector.
"""

from __future__ import annotations

import cv2
import numpy as np

from deepfake_detection.inference.face import crop_template

BLEND_RATIOS = (0.25, 0.5, 0.75, 1.0, 1.0, 1.0)


def face_mask(size: int, rng: np.random.Generator) -> np.ndarray:
    """A soft elliptical inner-face mask with random extent, rotation, and warping."""
    points = crop_template(size)
    eyes, mouth = points[:2].mean(axis=0), points[3:].mean(axis=0)
    spread_x = float(np.linalg.norm(points[1] - points[0]))
    spread_y = float(np.linalg.norm(mouth - eyes))
    center = (eyes + mouth) / 2 + rng.normal(0, 0.04 * size, 2)
    axes = (spread_x * rng.uniform(0.9, 1.4), spread_y * rng.uniform(1.3, 1.9))
    mask = np.zeros((size, size), dtype=np.float32)
    cv2.ellipse(mask, (round(center[0]), round(center[1])),
                (max(1, round(axes[0])), max(1, round(axes[1]))),
                float(rng.uniform(-12, 12)), 0, 360, 1.0, -1)
    # Low-frequency displacement field makes the boundary irregular.
    grid = max(2, size // 32)
    field = rng.normal(0, size * 0.03, (2, grid, grid)).astype(np.float32)
    dx = cv2.resize(field[0], (size, size), interpolation=cv2.INTER_CUBIC)
    dy = cv2.resize(field[1], (size, size), interpolation=cv2.INTER_CUBIC)
    xs, ys = np.meshgrid(np.arange(size, dtype=np.float32), np.arange(size, dtype=np.float32))
    mask = cv2.remap(mask, xs + dx, ys + dy, cv2.INTER_LINEAR)
    if rng.random() < 0.5:
        kernel = np.ones((3, 3), np.uint8)
        steps = int(rng.integers(1, 6))
        mask = (cv2.erode if rng.random() < 0.5 else cv2.dilate)(mask, kernel, iterations=steps)
    sigma = float(rng.uniform(0.01, 0.06) * size)
    mask = cv2.GaussianBlur(mask, (0, 0), sigma)
    peak = float(mask.max())
    return mask / peak if peak > 0 else mask


def _source(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Color, sharpness, and small geometric changes applied to the blended copy."""
    source = image.astype(np.float32)
    if rng.random() < 0.8:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + rng.uniform(-6, 6)) % 180
        hsv[..., 1] *= rng.uniform(0.8, 1.2)
        hsv[..., 2] *= rng.uniform(0.85, 1.15)
        source = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8),
                              cv2.COLOR_HSV2RGB).astype(np.float32)
    if rng.random() < 0.5:
        blurred = cv2.GaussianBlur(source, (0, 0), float(rng.uniform(0.5, 1.5)))
        source = blurred if rng.random() < 0.5 else source + rng.uniform(0.3, 1.0) * (
            source - blurred)
    size = image.shape[1], image.shape[0]
    scale = rng.uniform(0.95, 1.05)
    shift = rng.uniform(-0.03, 0.03, 2) * np.array(size)
    matrix = cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), 0.0, scale)
    matrix[:, 2] += shift
    source = cv2.warpAffine(source, matrix, size, flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT)
    return np.clip(source, 0, 255)


def self_blend(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return a pseudo-fake of an aligned square RGB crop, same shape and dtype."""
    if image.ndim != 3 or image.shape[0] != image.shape[1]:
        raise ValueError("self_blend expects an aligned square RGB crop")
    mask = face_mask(image.shape[0], rng) * float(rng.choice(BLEND_RATIOS))
    mask = mask[..., None]
    blended = mask * _source(image, rng) + (1 - mask) * image.astype(np.float32)
    return np.clip(np.rint(blended), 0, 255).astype(np.uint8)
