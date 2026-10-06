"""Dependency-light export validation and prediction alignment for fusion."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

REQUIRED_METADATA = {
    "manifest_id",
    "manifest_sha256",
    "sample_preprocessing_id",
    "preprocessing_id",
    "checkpoint_file",
    "checkpoint_sha256",
}
ROW_IDENTITY = ("sample_id", "split", "video_id", "source_domain", "fake_method", "label")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_export(directory: Path, split: str) -> tuple[list[dict], dict]:
    path = directory / f"{split}.jsonl"
    metadata = json.loads((directory / f"{split}_metadata.json").read_text(encoding="utf-8"))
    if REQUIRED_METADATA - metadata.keys() or any(not metadata[key] for key in REQUIRED_METADATA):
        raise ValueError(f"incomplete model provenance in {directory}/{split}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows or len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError(f"empty or duplicate sample IDs in {path}")
    if any(row["split"] != split for row in rows):
        raise ValueError(f"unexpected split in {path}")
    embedding_path = directory / f"{split}_embeddings.npz"
    with np.load(embedding_path) as payload:
        embedding_ids = payload["sample_ids"].astype(str).tolist()
        shape = payload["embeddings"].shape
    if embedding_ids != [row["sample_id"] for row in rows] or shape[0] != len(rows):
        raise ValueError(f"embedding sample IDs are not aligned with {path}")
    checkpoint_name = Path(str(metadata["checkpoint_file"]))
    if checkpoint_name.is_absolute() or ".." in checkpoint_name.parts:
        raise ValueError(f"unsafe checkpoint path in {directory}/{split}")
    checkpoint = directory / checkpoint_name
    if not checkpoint.is_file():
        raise ValueError(f"checkpoint is missing from {directory}")
    if _sha256(checkpoint) != metadata["checkpoint_sha256"]:
        raise ValueError(f"checkpoint hash differs from export metadata in {directory}")
    return rows, metadata


def load_embeddings(directory: Path, split: str) -> tuple[list[dict], np.ndarray, dict]:
    rows, metadata = load_export(directory, split)
    with np.load(directory / f"{split}_embeddings.npz") as payload:
        embeddings = payload["embeddings"].astype(np.float32)
    return rows, embeddings, metadata


def align(left: list[dict], right: list[dict]) -> tuple[list[dict], list[dict]]:
    left_map = {row["sample_id"]: row for row in left}
    right_map = {row["sample_id"]: row for row in right}
    if (
        len(left_map) != len(left)
        or len(right_map) != len(right)
        or left_map.keys() != right_map.keys()
    ):
        raise ValueError("prediction sample IDs are missing, duplicated, or unaligned")
    ordered = sorted(left_map)
    a, b = [left_map[key] for key in ordered], [right_map[key] for key in ordered]
    for x, y in zip(a, b, strict=True):
        if any(x.get(key) != y.get(key) for key in ROW_IDENTITY):
            raise ValueError(f"prediction metadata mismatch for {x['sample_id']}")
    return a, b


def blend(left: list[dict], right: list[dict], weight: float) -> list[dict]:
    a, b = align(left, right)
    if not 0 <= weight <= 1:
        raise ValueError("blend weight must be in [0, 1]")
    output = []
    for x, y in zip(a, b, strict=True):
        logit = weight * float(x["logit"]) + (1 - weight) * float(y["logit"])
        probability = 1 / (1 + math.exp(-max(-80, min(80, logit))))
        output.append({**x, "logit": logit, "score": probability})
    return output
