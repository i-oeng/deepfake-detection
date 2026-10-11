"""Framework-independent helpers shared by manifest-pinned training runs."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import yaml

from deepfake_detection.data.config import find_project_root
from deepfake_detection.data.manifest import verify_manifest

REQUIRED = {"source_id", "relative_path", "split", "label", "status", "video_id",
            "source_domain", "fake_method"}
SPLITS = ("train", "validation", "test")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> tuple[dict, Path]:
    path = path.resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("experiment"), dict):
        raise ValueError("training config needs an experiment mapping")
    return config["experiment"], find_project_root(path.parent)


def load_manifest_rows(config: dict, root: Path) -> tuple[list[dict[str, str]], Path, Path]:
    manifest_dir = (root / str(config["manifest_dir"])).resolve()
    data_root = (root / str(config["data_root"])).resolve()
    rows, manifest_path = read_manifest_rows(manifest_dir, data_root)
    if manifest_dir.name != str(config["manifest_id"]):
        raise ValueError("configured manifest ID differs from its directory")
    if sha256_file(manifest_path) != str(config["manifest_sha256"]):
        raise ValueError("configured manifest SHA-256 differs from the artifact")
    counts = Counter(row["split"] for row in rows)
    if set(counts) != set(SPLITS):
        raise ValueError(f"expected train/validation/test splits: {counts}")
    return rows, manifest_path, data_root


def read_manifest_rows(manifest_dir: Path, data_root: Path) -> tuple[list[dict[str, str]], Path]:
    """Verify an immutable manifest and resolve every image safely under ``data_root``."""
    manifest_dir = manifest_dir.resolve()
    data_root = data_root.resolve()
    valid, errors = verify_manifest(manifest_dir)
    if not valid:
        raise ValueError(f"invalid immutable manifest: {errors}")
    manifest_path = manifest_dir / "manifest.csv"
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not REQUIRED.issubset(reader.fieldnames or ()):
            raise ValueError("manifest lacks required training columns")
        rows = list(reader)
    if not rows:
        raise ValueError("empty training manifest")
    for row in rows:
        if row["status"] != "ok" or row["label"] not in {"REAL", "FAKE"}:
            raise ValueError(f"invalid image status or label: {row['source_id']}")
        relative = Path(row["relative_path"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "images":
            raise ValueError(f"unsafe manifest path: {relative}")
        image = (data_root / relative).resolve()
        if not image.is_relative_to(data_root) or not image.is_file():
            raise ValueError(f"manifest image is missing or escapes data root: {relative}")
        row["_path"] = str(image)
    return rows, manifest_path


def split_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    return {split: [row for row in rows if row["split"] == split] for split in SPLITS}


def sampling_weights(rows: list[dict[str, str]]) -> list[float]:
    """Give REAL and FAKE half the mass each, split evenly by method, then by video."""
    methods = sorted({row["fake_method"] for row in rows if row["label"] == "FAKE"})
    if not methods:
        raise ValueError("training split has no fake methods")
    group_counts = Counter((row["label"], row["fake_method"], row["video_id"]) for row in rows)
    groups_per_stratum = Counter((label, method) for label, method, _ in group_counts)
    weights = []
    for row in rows:
        label, method, video = row["label"], row["fake_method"], row["video_id"]
        mass = 0.5 if label == "REAL" else 0.5 / len(methods)
        weights.append(mass / groups_per_stratum[label, method]
                       / group_counts[label, method, video])
    return weights


def git_commit(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def run_id(config: dict, commit: str, manifest_sha256: str) -> str:
    """Content-derived run ID: the same config, commit, and manifest give the same ID."""
    identity = json.dumps({"config": config, "commit": commit,
                           "manifest_sha256": manifest_sha256}, sort_keys=True).encode()
    return hashlib.sha256(identity).hexdigest()[:20]


def create_run_directory(config: dict, root: Path, run: str, *,
                         default_output_root: str) -> Path:
    directory = root / str(config.get("output_root", default_output_root)) / run
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return directory
