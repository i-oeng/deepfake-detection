"""Release bundles: a frozen checkpoint plus development-fixed aggregation and calibration.

``make_bundle`` reads only the checkpoint's FF++ development (validation) frame
predictions. It fixes, in order: the video aggregation rule, Platt scaling of the
aggregated logit, and the balanced-accuracy threshold on calibrated scores. No
test, held-out, or Celeb-DF data enters a bundle.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from deepfake_detection.train.common import sha256_file
from deepfake_detection.train.evaluation import report_videos
from deepfake_detection.train.metrics import validation_threshold

from . import face

BUNDLE_VERSION = 1
# Preference order also breaks ties within TIE_MARGIN of the best development AUROC.
AGGREGATIONS = (
    "mean_probability", "mean_logit", "median_logit", "trimmed_mean_logit", "top25_mean_logit",
)
TIE_MARGIN = 0.002


def _logit(probability: float) -> float:
    probability = min(max(probability, 1e-7), 1 - 1e-7)
    return math.log(probability / (1 - probability))


def _sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-value)) if value >= 0 else math.exp(value) / (1 + math.exp(value))


def aggregate(logits: list[float], method: str) -> float:
    """Combine frame logits into one video logit."""
    if not logits:
        raise ValueError("no frame logits to aggregate")
    ordered = sorted(logits)
    if method == "mean_probability":
        return _logit(statistics.fmean(_sigmoid(value) for value in ordered))
    if method == "mean_logit":
        return statistics.fmean(ordered)
    if method == "median_logit":
        return statistics.median(ordered)
    if method == "trimmed_mean_logit":
        cut = int(len(ordered) * 0.2)
        return statistics.fmean(ordered[cut: len(ordered) - cut] or ordered)
    if method == "top25_mean_logit":
        return statistics.fmean(ordered[-max(1, math.ceil(len(ordered) * 0.25)):])
    raise ValueError(f"unknown aggregation: {method}")


def _videos(frames: list[dict], method: str) -> list[dict]:
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for row in frames:
        key = (row["source_domain"], row["video_id"], row["fake_method"], int(row["label"]))
        grouped[key].append(float(row["logit"]))
    return [
        {"source_domain": d, "video_id": v, "fake_method": m, "label": label,
         "logit": aggregate(logits, method), "score": _sigmoid(aggregate(logits, method))}
        for (d, v, m, label), logits in sorted(grouped.items())
    ]


def select_aggregation(frames: list[dict]) -> tuple[str, dict[str, float]]:
    """Pick the aggregation by FF++ development macro video AUROC (pre-registered rule)."""
    auroc = {
        method: report_videos(_videos(frames, method), domain="ff")["macro"]["auroc"]
        for method in AGGREGATIONS
    }
    best = max(auroc.values())
    chosen = next(method for method in AGGREGATIONS if auroc[method] >= best - TIE_MARGIN)
    return chosen, auroc


def fit_platt(logits: list[float], labels: list[int], iterations: int = 100) -> tuple[float, float]:
    """Maximum-likelihood ``sigmoid(a * logit + b)`` by damped Newton's method."""
    x = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    design = np.stack([x, np.ones_like(x)], axis=1)

    def loss(weights: np.ndarray) -> float:
        z = design @ weights
        return float(np.sum(np.logaddexp(0, z) - y * z))

    weights = np.array([1.0, 0.0])
    for _ in range(iterations):
        probability = 0.5 * (1 + np.tanh(0.5 * (design @ weights)))
        gradient = design.T @ (probability - y)
        curvature = design.T @ (design * (probability * (1 - probability))[:, None])
        step = np.linalg.solve(curvature + 1e-9 * np.eye(2), gradient)
        # Backtracking keeps every update a likelihood improvement; plain Newton
        # can oscillate when the starting slope is far from the optimum.
        size, current = 1.0, loss(weights)
        while loss(weights - size * step) > current and size > 1e-8:
            size /= 2
        weights = weights - size * step
        if np.abs(size * step).max() < 1e-10:
            break
    return float(weights[0]), float(weights[1])


def make_bundle(run_dir: Path, output: Path) -> dict:
    """Write a self-contained, content-versioned release bundle for one frozen run."""
    run_dir = run_dir.resolve()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    training = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    if sha256_file(run_dir / "best.pt") != training["checkpoint_sha256"]:
        raise ValueError("checkpoint hash differs from the training report")
    frames = [json.loads(line) for line in
              (run_dir / "validation.jsonl").read_text(encoding="utf-8").splitlines() if line]
    aggregation, table = select_aggregation(frames)
    videos = [row for row in _videos(frames, aggregation) if row["source_domain"] == "ff"]
    a, b = fit_platt([row["logit"] for row in videos], [row["label"] for row in videos])
    calibrated = [_sigmoid(a * row["logit"] + b) for row in videos]
    threshold = validation_threshold([row["label"] for row in videos], calibrated)
    manifest = {
        "bundle_version": BUNDLE_VERSION,
        "run_id": run_dir.name,
        "model": config["model"],
        "model_config": config,
        "checkpoint_file": "best.pt",
        "checkpoint_sha256": training["checkpoint_sha256"],
        "training_manifest_id": training["manifest_id"],
        "sample_preprocessing_id": training["sample_preprocessing_id"],
        "preprocessing_id": training["preprocessing_id"],
        "face_detector_id": face.DETECTOR_ID,
        "face_alignment_id": face.ALIGNMENT_ID,
        "face_policy_id": face.POLICY_ID,
        "video_frames": 32,
        "min_video_faces": 4,
        "aggregation": aggregation,
        "aggregation_development_auroc": table,
        "calibration": {"method": "platt", "a": a, "b": b,
                        "fit": "FF++ development videos", "videos": len(videos)},
        "threshold": threshold,
        "threshold_source": "FF++ development calibrated video balanced accuracy",
    }
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest["model_version"] = (
        f"{config['model']}-{hashlib.sha256(payload.encode()).hexdigest()[:12]}"
    )
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(run_dir / "best.pt", output / "best.pt")
    (output / "bundle.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


class Scorer:
    """Load a bundle once and turn aligned face crops into calibrated video decisions."""

    def __init__(self, bundle_dir: Path, *, device: str | None = None) -> None:
        import torch

        from deepfake_detection.train.benchmark import BinaryEncoder, build_transform

        self.bundle = json.loads((bundle_dir / "bundle.json").read_text(encoding="utf-8"))
        checkpoint = bundle_dir / self.bundle["checkpoint_file"]
        if sha256_file(checkpoint) != self.bundle["checkpoint_sha256"]:
            raise ValueError(f"bundle checkpoint hash mismatch: {bundle_dir}")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = BinaryEncoder(self.bundle["model"], self.bundle["model_config"])
        self.model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
        )
        self.model.to(self.device).eval()
        self.transform = build_transform(self.bundle["model"], training=False)
        self._torch = torch

    def frame_logits(self, crops: list[np.ndarray], batch_size: int = 32) -> list[float]:
        from PIL import Image

        torch = self._torch
        logits: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(crops), batch_size):
                batch = torch.stack([self.transform(Image.fromarray(crop))
                                     for crop in crops[start: start + batch_size]])
                with torch.autocast(device_type=self.device.type,
                                    enabled=self.device.type == "cuda"):
                    output, _ = self.model(batch.to(self.device))
                logits.extend(output.float().cpu().tolist())
        return logits

    def decide(self, logits: list[float], *, video: bool) -> dict:
        """Aggregate, calibrate, and threshold; abstain when too few faces were scored."""
        minimum = self.bundle["min_video_faces"] if video else 1
        if len(logits) < minimum:
            return {"label": "ABSTAIN", "probability_fake": None, "aggregated_logit": None}
        value = aggregate(logits, self.bundle["aggregation"])
        calibration = self.bundle["calibration"]
        probability = _sigmoid(calibration["a"] * value + calibration["b"])
        label = "FAKE" if probability >= self.bundle["threshold"] else "REAL"
        return {"label": label, "probability_fake": probability, "aggregated_logit": value}
