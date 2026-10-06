"""``deepfake-detect``: build release bundles and classify images or videos.

Results report the model version, threshold, and face decisions. No media, crop,
or embedding is written unless the caller builds a crop artifact explicitly.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from . import bundle, crops, face, media


def _decision(scorer: bundle.Scorer, logits: list[float], kind: str, reasons: list[str],
              faces: list[int], sampled: int) -> dict:
    decision = scorer.decide(logits, video=kind == "video")
    abstain = None
    if decision["label"] == "ABSTAIN":
        counts = {reason: reasons.count(reason) for reason in sorted(set(reasons))}
        abstain = "too_few_faces" if kind == "video" else (reasons[0] if reasons else "no_face")
        decision["exclusions"] = counts
    return {
        **decision,
        "kind": kind,
        "model_version": scorer.bundle["model_version"],
        "threshold": scorer.bundle["threshold"],
        "aggregation": scorer.bundle["aggregation"],
        "frames_sampled": sampled,
        "frames_scored": len(logits),
        "max_faces_in_frame": max(faces, default=0),
        "abstain_reason": abstain,
    }


def detect(scorer: bundle.Scorer, path: Path, detector: face.FaceDetector) -> dict:
    """Classify one image or video end to end, holding crops only in memory."""
    kind = media.media_kind(path)
    frames = ([(0, media.read_image(path))] if kind == "image"
              else list(media.sample_video(path, scorer.bundle["video_frames"])))
    accepted, reasons, faces = [], [], []
    for _, rgb in frames:
        result = face.extract_face(rgb, detector)
        faces.append(result.faces_detected)
        if result.crop is None:
            reasons.append(result.reason or "")
        else:
            accepted.append(result.crop)
    logits = scorer.frame_logits(accepted) if accepted else []
    return {"input": path.name, **_decision(scorer, logits, kind, reasons, faces, len(frames))}


def score_crop_artifact(scorer: bundle.Scorer, directory: Path) -> list[dict]:
    """Decide every media item in a crop artifact; used for raw-video evaluation."""
    import numpy as np
    from PIL import Image

    rows = crops.read_crops(directory)
    by_media: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_media[row["media_id"]].append(row)
    output = []
    for media_id, items in sorted(by_media.items()):
        accepted = [np.asarray(Image.open(directory / row["crop_path"]).convert("RGB"))
                    for row in items if row["status"] == "ok"]
        reasons = [row["reason"] for row in items if row["status"] != "ok"]
        faces = [int(row["faces_detected"] or 0) for row in items]
        kind = items[0]["kind"]
        logits = scorer.frame_logits(accepted) if accepted else []
        output.append({"media_id": media_id,
                       **_decision(scorer, logits, kind, reasons, faces, len(items))})
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("bundle", help="Build a release bundle from a frozen run")
    make.add_argument("--run", type=Path, required=True)
    make.add_argument("--output", type=Path, required=True)
    build = commands.add_parser("crops", help="Build a content-addressed face-crop artifact")
    build.add_argument("--inputs", type=Path, required=True, help="CSV with media_id,path")
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--frames", type=int, default=32)
    build.add_argument("--workers", type=int, default=8)
    scored = commands.add_parser("score-crops", help="Decide every item in a crop artifact")
    scored.add_argument("--bundle", type=Path, required=True)
    scored.add_argument("--crops", type=Path, required=True)
    scored.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("detect", help="Classify images or videos")
    run.add_argument("--bundle", type=Path, required=True)
    run.add_argument("--device")
    run.add_argument("paths", type=Path, nargs="+")
    args = parser.parse_args(argv)
    if args.command == "bundle":
        print(json.dumps(bundle.make_bundle(args.run, args.output), indent=2))
    elif args.command == "crops":
        with args.inputs.open(newline="", encoding="utf-8") as stream:
            inputs = [(row["media_id"], Path(row["path"]).resolve())
                      for row in csv.DictReader(stream)]
        print(crops.build_crops(inputs, args.output_root, frames=args.frames,
                                workers=args.workers))
    elif args.command == "score-crops":
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite {args.output}")
        results = score_crop_artifact(bundle.Scorer(args.bundle), args.crops)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in results))
        print(args.output)
    else:
        scorer = bundle.Scorer(args.bundle, device=args.device)
        detector = face.FaceDetector()
        status = 0
        for path in args.paths:
            try:
                print(json.dumps(detect(scorer, path, detector), sort_keys=True), flush=True)
            except (OSError, ValueError) as error:
                print(json.dumps({"input": path.name, "label": "ERROR", "error": str(error)}),
                      flush=True)
                status = 1
        return status
    return 0


if __name__ == "__main__":
    sys.exit(main())
