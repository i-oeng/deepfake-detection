#!/usr/bin/env python3
"""Exploratory: score raw Celeb-DF videos on whole frames instead of face crops.

For every video in a crop artifact's input list, the same evenly spaced frames
are scored by each release bundle three ways: the stored aligned face crop, the
whole frame stretched to the model input, and the whole frame letterboxed to a
square first. Video scores use each bundle's aggregation; AUROC is threshold-free,
so calibration does not affect the comparison. Not pre-registered.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from deepfake_detection.inference import bundle, crops, media
from deepfake_detection.inference.celebdf import _identities
from deepfake_detection.train.evaluation import grouped_bootstrap_difference, report

VARIANTS = ("face_crop", "whole_stretched", "whole_letterboxed")


def letterbox(rgb: np.ndarray) -> np.ndarray:
    height, width = rgb.shape[:2]
    side = max(height, width)
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    top, left = (side - height) // 2, (side - width) // 2
    canvas[top: top + height, left: left + width] = rgb
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crops", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--bundle", action="append", required=True, help="NAME=bundle-dir")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scorers = {name: bundle.Scorer(Path(path))
               for name, path in (item.split("=", 1) for item in args.bundle)}
    with args.catalog.open(newline="", encoding="utf-8") as stream:
        labels = {row["video_id"]: row["label"] for row in csv.DictReader(stream)}
    with args.inputs.open(newline="", encoding="utf-8") as stream:
        paths = {row["media_id"]: Path(row["path"]) for row in csv.DictReader(stream)}
    crop_rows: dict[str, dict[int, str]] = defaultdict(dict)
    for row in crops.read_crops(args.crops):
        if row["status"] == "ok":
            crop_rows[row["media_id"]][int(row["frame_index"])] = row["crop_path"]
    frames_per_video = json.loads((args.crops / "artifact.json").read_text())["frames"]
    logits: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for number, (video, path) in enumerate(sorted(paths.items()), 1):
        sampled = list(media.sample_video(path, frames_per_video))
        inputs = {
            "face_crop": [np.asarray(Image.open(args.crops / crop_rows[video][i]).convert("RGB"))
                          for i, _ in sampled if i in crop_rows[video]],
            "whole_stretched": [rgb for _, rgb in sampled],
            "whole_letterboxed": [letterbox(rgb) for _, rgb in sampled],
        }
        for name, scorer in scorers.items():
            for variant, images in inputs.items():
                if images:
                    logits[name][variant][video] = bundle.aggregate(
                        scorer.frame_logits(images), scorer.bundle["aggregation"])
        if number % 50 == 0:
            print(f"{number}/{len(paths)} videos", flush=True)
    result: dict = {"videos": len(paths), "models": {}, "paired": {}}
    shared = set.intersection(*(set(v) for per in logits.values() for v in per.values()))

    def rows(name: str, variant: str) -> list[dict]:
        return [{"source_domain": "cdf", "video_id": video, "label": int(labels[video] == "FAKE"),
                 "fake_method": "celebdf_v2" if labels[video] == "FAKE" else "",
                 "score": float(1 / (1 + np.exp(-logits[name][variant][video]))),
                 "lineage_video_ids": "|".join(sorted(_identities(video)))
                 or f"cdf:video:{video}"}
                for video in sorted(shared)]

    for name in scorers:
        result["models"][name] = {
            variant: report(rows(name, variant), domain="cdf")["macro"] for variant in VARIANTS
        }
        for variant in VARIANTS[1:]:
            result["paired"][f"{name}: {variant} minus face_crop"] = grouped_bootstrap_difference(
                rows(name, "face_crop"), rows(name, variant), domain="cdf")
    result["scored_videos"] = len(shared)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name, variants in result["models"].items():
        print(name, {variant: round(values["auroc"], 4) for variant, values in variants.items()})
    for name, value in result["paired"].items():
        print(name, round(value["difference"], 4), [round(x, 4) for x in value["ci95"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
