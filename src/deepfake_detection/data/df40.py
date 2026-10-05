"""Normalize the official DeepfakeBench DF40 JSON catalogs.

The published JSON files index frames by dataset, class, split, and sometimes
compression before the video record. They contain paths from the authors'
machines, so the normalized catalog retains those paths as provenance and
assigns a portable destination under ``images/`` for later materialization.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Any

METHODS = {
    "simswap": ("simswap", "face_swap"),
    "wav2lip": ("wav2lip", "face_reenactment"),
    "stylegan2": ("stylegan2", "entire_face_synthesis"),
    "sd2.1": ("sd21", "entire_face_synthesis"),
    "blendface": ("blendface", "face_swap"),
    "sadtalker": ("sadtalker", "face_reenactment"),
    "dit": ("dit", "entire_face_synthesis"),
    "starganv2": ("starganv2", "face_edit"),
}
SPLITS = {"train": "train", "val": "validation", "validation": "validation", "test": "test"}
COMPRESSIONS = {"c23", "c40", "raw"}
COLUMNS = (
    "image_id", "relative_path", "label", "split", "fake_method",
    "manipulation_family", "source_domain", "video_id", "identity_id",
    "frame_index", "compression", "source_json", "source_path",
    "source_video_id", "target_identity_id",
)


def _method_and_domain(path: Path) -> tuple[str, str, str]:
    stem = path.stem.casefold()
    if stem.endswith("_cdf"):
        method_key, domain = stem[:-4], "cdf"
    elif stem.endswith("_ff"):
        method_key, domain = stem[:-3], "ff"
    elif stem == "starganv2":
        method_key, domain = stem, "celeba"
    else:
        raise ValueError(f"unsupported DF40 catalog filename: {path.name}")
    if method_key not in METHODS:
        raise ValueError(f"unsupported DF40 method in {path.name}")
    method, family = METHODS[method_key]
    return method, family, domain


def _records(node: Any, keys: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], dict]]:
    if not isinstance(node, dict):
        raise ValueError(f"expected an object at {'/'.join(keys) or '<root>'}")
    if "frames" in node:
        yield keys, node
        return
    for key, value in node.items():
        yield from _records(value, (*keys, str(key)))


def _field(keys: tuple[str, ...], mapping: dict[str, str], name: str) -> str:
    matches = {mapping[key.casefold()] for key in keys if key.casefold() in mapping}
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {name} in {'/'.join(keys)}")
    return matches.pop()


def _label(record: dict, keys: tuple[str, ...]) -> str:
    candidates = [str(record.get("label", "")), *keys]
    labels = set()
    for candidate in candidates:
        lowered = candidate.casefold()
        if re.search(r"(^|[_-])real($|[_-])", lowered):
            labels.add("REAL")
        elif re.search(r"(^|[_-])fake($|[_-])", lowered):
            labels.add("FAKE")
    if len(labels) != 1:
        raise ValueError(f"missing or conflicting real/fake label in {'/'.join(keys)}")
    return labels.pop()


def _safe_part(value: str, name: str) -> str:
    if not value or value in {".", ".."} or any(char in value for char in '/\\:<>"|?*\x00'):
        raise ValueError(f"unsafe {name}: {value!r}")
    return value


def _frame_index(frame_name: str, position: int) -> str:
    match = re.search(r"(\d+)$", Path(frame_name).stem)
    return str(int(match.group(1))) if match else str(position)


def normalize_df40(input_dir: Path, output: Path) -> dict[str, Any]:
    """Write one deterministic row per unique official frame path.

    Repeated real references across method JSONs are collapsed. Conflicting
    split or label claims fail so they cannot silently contaminate a pilot.
    """
    sources = sorted(input_dir.glob("*.json"), key=lambda path: path.name.casefold())
    if not sources:
        raise FileNotFoundError(f"no JSON catalogs found in {input_dir}")

    rows: dict[str, dict[str, str]] = {}
    for source in sources:
        method, family, domain = _method_and_domain(source)
        with source.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        for keys, record in _records(payload):
            split = _field(keys, SPLITS, "split")
            compression = next((key for key in keys if key.casefold() in COMPRESSIONS), "")
            label = _label(record, keys)
            frames = record["frames"]
            if not isinstance(frames, list) or not all(isinstance(p, str) for p in frames):
                raise ValueError(f"invalid frames list in {source.name}: {'/'.join(keys)}")
            video_name = _safe_part(keys[-1], "video name")
            video_id = f"{domain}:{video_name}"
            for position, original in enumerate(sorted(frames)):
                normalized = original.replace("\\", "/")
                frame_name = _safe_part(PurePosixPath(normalized).name, "frame name")
                if not normalized or not frame_name:
                    raise ValueError(f"empty frame path in {source.name}")
                # An absolute source path is provenance, never a local read target.
                path_token = hashlib.sha256(normalized.encode()).hexdigest()[:12]
                relative = PurePosixPath(
                    "images", domain, method if label == "FAKE" else "real",
                    split, compression or "none", video_name,
                    f"{path_token}_{frame_name}",
                ).as_posix()
                image_id = hashlib.sha256(relative.encode()).hexdigest()[:24]
                row = {
                    "image_id": image_id,
                    "relative_path": relative,
                    "label": label,
                    "split": split,
                    "fake_method": method if label == "FAKE" else "",
                    "manipulation_family": family if label == "FAKE" else "real",
                    "source_domain": domain,
                    "video_id": video_id,
                    # Official JSON does not consistently provide person identities.
                    "identity_id": str(record.get("identity_id", "")),
                    "frame_index": _frame_index(frame_name, position),
                    "compression": compression,
                    "source_json": source.name,
                    "source_path": normalized,
                    "source_video_id": str(record.get("source_video_id", "")),
                    "target_identity_id": str(record.get("target_identity_id", "")),
                }
                existing = rows.get(normalized)
                if existing is not None:
                    if (
                        existing["label"], existing["split"],
                        existing["source_domain"], existing["fake_method"],
                    ) != (
                        label, split, domain, row["fake_method"],
                    ):
                        raise ValueError(f"conflicting official frame assignment: {normalized}")
                    continue
                rows[normalized] = row

    if not rows:
        raise ValueError("official catalogs contain no frame paths")
    # Sort by the portable path to make the result independent of JSON key order.
    ordered = sorted(rows.values(), key=lambda row: row["relative_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(ordered)
    return {"catalog": str(output), "sources": len(sources), "rows": len(ordered)}
