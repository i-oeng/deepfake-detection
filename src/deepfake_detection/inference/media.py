"""Read still images and evenly sampled video frames as RGB arrays."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}


def media_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    raise ValueError(f"unsupported media type: {path.name}")


def read_image(path: Path) -> np.ndarray:
    from PIL import Image, ImageOps

    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB"))


def even_positions(length: int, limit: int) -> list[int]:
    """Evenly spaced indices, matching the DF40 subset sampler."""
    if length <= limit:
        return list(range(length))
    if limit == 1:
        return [length // 2]
    return [round(index * (length - 1) / (limit - 1)) for index in range(limit)]


def sample_video(path: Path, frames: int) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame_index, rgb)`` for evenly spaced frames, decoding sequentially.

    Sequential decoding avoids inexact container seeks. The frame count is taken
    from a first decoding pass when the container does not report it.
    """
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path.name}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        while capture.grab():
            total += 1
        capture.release()
        capture = cv2.VideoCapture(str(path))
    wanted = set(even_positions(total, frames))
    index = 0
    try:
        while wanted and index <= max(wanted):
            ok, frame = capture.read()
            if not ok:
                break
            if index in wanted:
                wanted.discard(index)
                yield index, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            index += 1
    finally:
        capture.release()
