"""Content-addressed face-crop artifacts for raw images and videos (plan phase 2).

The artifact ID is derived from every input file's SHA-256, the detector,
alignment, and policy versions, and the frame budget. Rebuilding the same inputs
with the same configuration resolves to the same directory. Every sampled frame
gets one ``crops.csv`` row: a crop with its geometry, or an exclusion reason.
Labels and splits are never read.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from deepfake_detection.train.common import sha256_file

from . import face, media

FIELDS = (
    "media_id", "kind", "frame_index", "status", "reason", "crop_path", "crop_sha256",
    "faces_detected", "score", "box", "landmarks",
)


def _safe(media_id: str) -> str:
    return hashlib.sha256(media_id.encode()).hexdigest()[:16]


def _process(job: tuple[str, str, int, str]) -> list[dict[str, str]]:
    from PIL import Image

    media_id, path, frames, output = job
    detector = face.FaceDetector()
    source = Path(path)
    kind = media.media_kind(source)
    if kind == "image":
        samples = [(0, media.read_image(source))]
    else:
        samples = list(media.sample_video(source, frames))
    rows = []
    for index, rgb in samples:
        result = face.extract_face(rgb, detector)
        row = {"media_id": media_id, "kind": kind, "frame_index": str(index),
               "faces_detected": str(result.faces_detected), "reason": result.reason or "",
               "status": "ok" if result.crop is not None else "excluded",
               "crop_path": "", "crop_sha256": "", "score": "", "box": "", "landmarks": ""}
        if result.face is not None:
            row["score"] = f"{result.face.score:.4f}"
            row["box"] = json.dumps([round(v, 2) for v in result.face.box])
            row["landmarks"] = json.dumps(np.round(result.face.landmarks, 2).tolist())
        if result.crop is not None:
            relative = Path("crops") / _safe(media_id) / f"{index:06d}.png"
            target = Path(output) / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(result.crop).save(target, optimize=False)
            row["crop_path"] = relative.as_posix()
            row["crop_sha256"] = sha256_file(target)
        rows.append(row)
    if not samples:
        rows.append({**dict.fromkeys(FIELDS, ""), "media_id": media_id, "kind": kind,
                     "frame_index": "",
                     "status": "excluded", "reason": "unreadable_media", "faces_detected": "0"})
    return rows


def artifact_id(inputs: list[tuple[str, Path]], frames: int) -> tuple[str, dict]:
    identity = {
        "inputs": [[media_id, sha256_file(path)] for media_id, path in inputs],
        "frames": frames,
        "detector": face.DETECTOR_ID,
        "alignment": face.ALIGNMENT_ID,
        "policy": face.POLICY_ID,
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return digest[:20], identity


def build_crops(inputs: list[tuple[str, Path]], output_root: Path, *, frames: int = 32,
                workers: int = 8) -> Path:
    """Create (or reuse) the crop artifact for ``inputs``, a list of (media_id, path)."""
    if len({media_id for media_id, _ in inputs}) != len(inputs):
        raise ValueError("media IDs must be unique")
    face.ensure_detector()
    identifier, identity = artifact_id(inputs, frames)
    directory = output_root / identifier
    if (directory / "artifact.json").is_file():
        return directory
    staging = output_root / f".{identifier}.part"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    jobs = [(media_id, str(path), frames, str(staging)) for media_id, path in inputs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_process, jobs))
    rows = [row for result in results for row in result]
    with (staging / "crops.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        **identity,
        "artifact_id": identifier,
        "crops_csv_sha256": sha256_file(staging / "crops.csv"),
        "media": len(inputs),
        "frames": len(rows),
        "crops": sum(row["status"] == "ok" for row in rows),
        "exclusions": {reason: sum(row["reason"] == reason for row in rows)
                       for reason in sorted({row["reason"] for row in rows} - {""})},
    }
    (staging / "artifact.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(staging, directory)
    return directory


def read_crops(directory: Path) -> list[dict[str, str]]:
    """Load a crop artifact after checking its index hash."""
    summary = json.loads((directory / "artifact.json").read_text(encoding="utf-8"))
    if sha256_file(directory / "crops.csv") != summary["crops_csv_sha256"]:
        raise ValueError(f"crop index hash mismatch: {directory}")
    with (directory / "crops.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))
