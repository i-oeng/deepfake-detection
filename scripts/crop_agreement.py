#!/usr/bin/env python3
"""Compare our aligned crops of raw Celeb-DF frames with DF40's crops of the same frames.

Real Celeb-DF videos appear in DF40 as crops with a frame index. For a seeded
sample of those frames, this decodes the same frame from the published archive,
aligns it with the release face pipeline, and measures how far YuNet landmarks
in our crop lie from the landmarks in DF40's crop. Uses real frames only.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import tempfile
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
from PIL import Image

from deepfake_detection.inference import face

SCALES = (1.2, 1.25, 1.3, 1.35, 1.4)


def _frame(video: Path, index: int) -> np.ndarray | None:
    import cv2

    capture = cv2.VideoCapture(str(video))
    frame = None
    for _ in range(index + 1):
        ok, frame = capture.read()
        if not ok:
            frame = None
            break
    capture.release()
    return None if frame is None else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.catalog.open(newline="", encoding="utf-8") as stream:
        catalog = {row["video_id"]: row for row in csv.DictReader(stream)}
    with args.manifest.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["source_domain"] == "cdf"
                and row["label"] == "REAL" and row["video_id"] in catalog]
    random.Random(0).shuffle(rows)
    detector = face.FaceDetector()
    distances: dict[float, list[float]] = {scale: [] for scale in SCALES}
    errors: list[float] = []
    with ZipFile(args.archive) as archive, tempfile.TemporaryDirectory() as scratch:
        members = {PurePosixPath(name).as_posix(): name for name in archive.namelist()}
        for row in rows:
            if len(errors) >= args.frames:
                break
            wanted = catalog[row["video_id"]]["archive_path"]
            matches = [name for key, name in members.items() if key.endswith(wanted)]
            if len(matches) != 1:
                continue
            video = Path(scratch) / Path(wanted).name
            if not video.exists():
                video.write_bytes(archive.read(matches[0]))
            rgb = _frame(video, int(row["frame_index"]))
            reference = np.asarray(Image.open(args.data_root / row["relative_path"]).convert("RGB"))
            faces = detector.detect(rgb) if rgb is not None else []
            reference_faces = detector.detect(reference)
            if not faces or not reference_faces:
                continue
            for scale in SCALES:
                ours = face.align(rgb, faces[0].landmarks, reference.shape[0], scale)
                found = detector.detect(ours)
                if found:
                    distances[scale].append(float(np.linalg.norm(
                        found[0].landmarks - reference_faces[0].landmarks, axis=1).mean()))
                if scale == face.CROP_SCALE:
                    errors.append(float(np.abs(ours.astype(float) - reference).mean()))
    chosen = distances[face.CROP_SCALE]
    result = {
        "frames": len(errors),
        "median_landmark_px": statistics.median(chosen),
        "mean_landmark_px": statistics.fmean(chosen),
        "pixel_mae": statistics.fmean(errors),
        "mean_by_scale": {f"{scale:g}": statistics.fmean(values)
                          for scale, values in distances.items()},
        "detector": face.DETECTOR_ID,
        "alignment": face.ALIGNMENT_ID,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
