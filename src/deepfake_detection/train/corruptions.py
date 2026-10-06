"""Seeded image corruptions for the pre-registered robustness protocol (P4).

Each corruption maps a stored face crop to a degraded crop of the same size before
any model preprocessing. Randomness is derived from the sample ID, so every model
sees identical corrupted pixels.
"""

from __future__ import annotations

import hashlib
import io
import subprocess

import numpy as np
from PIL import Image, ImageFilter

SEVERITIES: dict[str, tuple[float, ...]] = {
    "jpeg": (90, 70, 50, 30),
    "resize": (0.75, 0.5, 0.25),
    "blur": (0.5, 1.0, 2.0),
    "noise": (2, 5, 10),
    "crop": (0.9, 0.8, 0.7),
    "h264": (23, 30, 38),
}


def settings() -> list[tuple[str, float]]:
    """Every pre-registered (corruption, severity) pair, in report order."""
    return [(name, level) for name, levels in SEVERITIES.items() for level in levels]


def label(name: str, level: float) -> str:
    return f"{name}-{level:g}"


def _rng(sample_id: str, name: str, level: float) -> np.random.Generator:
    digest = hashlib.sha256(f"{sample_id}\0{name}\0{level:g}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _h264(image: Image.Image, crf: float) -> Image.Image:
    """Encode one frame with libx264 (4:2:0, padded to even size) and decode it back."""
    width, height = image.size
    padded_w, padded_h = width + width % 2, height + height % 2
    canvas = Image.new("RGB", (padded_w, padded_h))
    canvas.paste(image, (0, 0))
    encode = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{padded_w}x{padded_h}", "-i", "-", "-c:v", "libx264", "-preset", "medium",
         "-crf", f"{crf:g}", "-threads", "1", "-pix_fmt", "yuv420p", "-f", "h264", "-"],
        input=canvas.tobytes(), capture_output=True, check=True,
    ).stdout
    decoded = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "h264", "-i", "-", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"],
        input=encode, capture_output=True, check=True,
    ).stdout
    frame = Image.frombytes("RGB", (padded_w, padded_h), decoded[: padded_w * padded_h * 3])
    return frame.crop((0, 0, width, height))


def apply(image: Image.Image, name: str, level: float, sample_id: str) -> Image.Image:
    """Return ``image`` corrupted at one pre-registered severity; size is preserved."""
    if level not in SEVERITIES.get(name, ()):
        raise ValueError(f"unregistered corruption: {name} at {level}")
    image = image.convert("RGB")
    width, height = image.size
    if name == "jpeg":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=int(level))
        buffer.seek(0)
        with Image.open(buffer) as encoded:
            return encoded.convert("RGB")
    if name == "resize":
        small = (max(1, round(width * level)), max(1, round(height * level)))
        return image.resize(small, Image.Resampling.BICUBIC).resize(
            (width, height), Image.Resampling.BICUBIC
        )
    if name == "blur":
        return image.filter(ImageFilter.GaussianBlur(level))
    if name == "noise":
        pixels = np.asarray(image, dtype=np.float32)
        pixels += _rng(sample_id, name, level).normal(0.0, level, pixels.shape)
        return Image.fromarray(np.clip(np.rint(pixels), 0, 255).astype(np.uint8))
    if name == "crop":
        kept_w, kept_h = round(width * level), round(height * level)
        left, top = (width - kept_w) // 2, (height - kept_h) // 2
        return image.crop((left, top, left + kept_w, top + kept_h)).resize(
            (width, height), Image.Resampling.BICUBIC
        )
    return _h264(image, level)
