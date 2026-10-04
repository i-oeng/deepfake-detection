#!/usr/bin/env python3
"""Normalize selected official DF40 protocol JSONs for an image-only pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath

METHODS = {
    "simswap": ("simswap_ff.json", "fs", "ff", {"train", "validation"}),
    "wav2lip": ("wav2lip_ff.json", "fr", "ff", {"train", "validation"}),
    "stylegan2": ("StyleGAN2_ff.json", "efs", "ff", {"train", "validation"}),
    "sd21": ("sd2.1_ff.json", "efs", "ff", {"train", "validation"}),
    "blendface": ("blendface_ff.json", "fs", "ff", {"test"}),
    "sadtalker": ("sadtalker_ff.json", "fr", "ff", {"test"}),
    "dit": ("DiT_ff.json", "efs", "ff", {"test"}),
    "starganv2": ("starganv2.json", "fe", "celeba", {"test"}),
}

FIELDNAMES = [
    "image_id",
    "relative_path",
    "label",
    "split",
    "fake_method",
    "manipulation_family",
    "source_domain",
    "video_id",
    "identity_id",
    "frame_index",
    "compression",
]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json-root",
        type=Path,
        default=project_root / "data/raw/df40/official_json/dataset_json",
        help="directory containing the official DF40 protocol JSON files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "data/raw/df40/metadata.csv",
        help="normalized CSV destination",
    )
    parser.add_argument(
        "--exclude-paths",
        type=Path,
        help="optional UTF-8 file containing one unavailable relative image path per line",
    )
    return parser.parse_args()


def canonical_split(value: str) -> str:
    return "validation" if value == "val" else value


def frame_number(path: str) -> str:
    stem = PurePosixPath(path).stem
    digits = "".join(char for char in stem if char.isdigit())
    return digits or stem


def stable_id(method: str, split: str, label: str, path: str) -> str:
    digest = hashlib.sha256(path.encode("utf-8")).hexdigest()[:20]
    return f"df40-{method}-{split}-{label.lower()}-{digest}"


def rows_for_method(
    json_root: Path,
    method: str,
    filename: str,
    family: str,
    domain: str,
    allowed_splits: set[str],
) -> list[dict[str, str]]:
    source = json_root / filename
    data = json.loads(source.read_text(encoding="utf-8"))
    root = next(iter(data.values()))
    rows: list[dict[str, str]] = []
    for label_bucket, splits in root.items():
        label = "FAKE" if label_bucket.lower().endswith("_fake") else "REAL"
        for source_split, groups in splits.items():
            split = canonical_split(source_split)
            if label == "FAKE" and split not in allowed_splits:
                continue
            for group_id, info in groups.items():
                for frame_path in info.get("frames", []):
                    rows.append(
                        {
                            "image_id": stable_id(method, split, label, frame_path),
                            "relative_path": frame_path,
                            "label": label,
                            "split": split,
                            "fake_method": method if label == "FAKE" else "",
                            "manipulation_family": family if label == "FAKE" else "real",
                            "source_domain": domain,
                            "video_id": (
                                f"{domain}:{method}:{group_id}"
                                if label == "FAKE"
                                else f"{domain}:{group_id}"
                            ),
                            "identity_id": (
                                f"{domain}:{method}:{group_id}"
                                if label == "FAKE"
                                else f"{domain}:{group_id}"
                            ),
                            "frame_index": frame_number(frame_path),
                            "compression": "c23" if "/c23/" in frame_path else "",
                        }
                    )
    return rows


def load_exclusions(path: Path | None) -> set[str]:
    if path is None:
        return set()
    return {
        line.strip().replace("\\", "/")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def normalize(json_root: Path, excluded_paths: set[str]) -> list[dict[str, str]]:
    fake_rows: list[dict[str, str]] = []
    ff_real_by_path: dict[str, dict[str, str]] = {}
    celeba_real_by_path: dict[str, dict[str, str]] = {}
    for method, (filename, family, domain, splits) in METHODS.items():
        for row in rows_for_method(json_root, method, filename, family, domain, splits):
            if row["label"] == "FAKE":
                fake_rows.append(row)
            elif domain == "ff" and method == "simswap":
                ff_real_by_path.setdefault(row["relative_path"], row)
            elif domain == "celeba" and method == "starganv2" and row["split"] == "test":
                celeba_real_by_path.setdefault(row["relative_path"], row)

    # Official method JSONs reuse some FF++ real videos across validation and
    # test. Repartition each unique real-video group once so it cannot leak.
    ff_groups = sorted(
        {row["video_id"] for row in ff_real_by_path.values()},
        key=lambda value: hashlib.sha256(f"df40-real-split:{value}".encode()).hexdigest(),
    )
    train_end = round(len(ff_groups) * 0.60)
    validation_end = train_end + round(len(ff_groups) * 0.15)
    group_split = {
        group: "train" if index < train_end else "validation" if index < validation_end else "test"
        for index, group in enumerate(ff_groups)
    }
    for row in ff_real_by_path.values():
        row["split"] = group_split[row["video_id"]]
        row["image_id"] = stable_id("real", row["split"], "REAL", row["relative_path"])
    for row in celeba_real_by_path.values():
        row["image_id"] = stable_id("real", row["split"], "REAL", row["relative_path"])

    rows = fake_rows + list(ff_real_by_path.values()) + list(celeba_real_by_path.values())
    rows = [row for row in rows if row["relative_path"] not in excluded_paths]
    rows.sort(
        key=lambda row: (
            row["split"],
            row["label"],
            row["fake_method"],
            row["relative_path"],
        )
    )

    image_ids = {row["image_id"] for row in rows}
    paths = {row["relative_path"] for row in rows}
    if len(image_ids) != len(rows):
        raise RuntimeError("normalization produced duplicate image IDs")
    if len(paths) != len(rows):
        raise RuntimeError("normalization produced duplicate relative paths")
    return rows


def main() -> None:
    args = parse_args()
    excluded_paths = load_exclusions(args.exclude_paths)
    rows = normalize(args.json_root, excluded_paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "metadata": str(args.output),
                "rows": len(rows),
                "fake": sum(row["label"] == "FAKE" for row in rows),
                "real": sum(row["label"] == "REAL" for row in rows),
                "excluded": len(excluded_paths),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
