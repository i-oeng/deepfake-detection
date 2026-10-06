"""Create an immutable pilot subset without cross-split image collisions."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from deepfake_detection import __version__

from .manifest import verify_manifest
from .subset import verify_subset

FINGERPRINTS = ("file_sha256", "pixel_sha256", "dhash64")
SPLIT_PRIORITY = {"train": 0, "validation": 1, "test": 2}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"empty CSV: {path}")
        return list(reader.fieldnames), list(reader)


def prune_cross_split(
    subset_dir: str | Path, manifest_dir: str | Path, output_root: str | Path
) -> dict[str, object]:
    """Keep the highest-priority split for each colliding fingerprint.

    Whole video groups are removed so nearby frames cannot remain across splits.
    The input manifest must describe exactly the verified input subset.
    """
    subset_dir, manifest_dir = Path(subset_dir), Path(manifest_dir)
    valid, errors = verify_subset(subset_dir)
    if not valid:
        raise ValueError(f"invalid subset: {errors}")
    valid, errors = verify_manifest(manifest_dir)
    if not valid:
        raise ValueError(f"invalid manifest: {errors}")
    subset_path, manifest_path = subset_dir / "subset.csv", manifest_dir / "manifest.csv"
    columns, subset = _read_csv(subset_path)
    _, manifest = _read_csv(manifest_path)
    required_subset = {"image_id", "split", "video_id", "label"}
    required_manifest = {"source_id", "split", "video_id", "status", *FINGERPRINTS}
    if not required_subset.issubset(columns) or not manifest or not required_manifest.issubset(
        manifest[0]
    ):
        raise ValueError("subset or manifest is missing required columns")
    subset_by_id = {row["image_id"]: row for row in subset}
    manifest_by_id = {row["source_id"]: row for row in manifest}
    if len(subset_by_id) != len(subset) or set(subset_by_id) != set(manifest_by_id):
        raise ValueError("manifest source IDs must match the subset exactly")
    if len(manifest_by_id) != len(manifest):
        raise ValueError("manifest has duplicate source IDs")
    for source_id, row in manifest_by_id.items():
        original = subset_by_id[source_id]
        if row["status"] != "ok" or any(not row[column] for column in FINGERPRINTS):
            raise ValueError(f"image has no complete fingerprints: {source_id}")
        if row["split"] != original["split"] or row["video_id"] != original["video_id"]:
            raise ValueError(f"manifest metadata differs from subset: {source_id}")
        if not row["video_id"] or row["split"] not in SPLIT_PRIORITY:
            raise ValueError(f"missing video ID or unsupported split: {source_id}")

    remove: set[tuple[str, str]] = set()
    collisions = Counter()
    for column in FINGERPRINTS:
        groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in manifest:
            groups[row[column]].append(row)
        for group in groups.values():
            splits = {row["split"] for row in group}
            if len(splits) < 2:
                continue
            collisions[column] += 1
            keep_split = max(splits, key=SPLIT_PRIORITY.__getitem__)
            remove.update(
                (row["split"], row["video_id"])
                for row in group if row["split"] != keep_split
            )

    kept = [row for row in subset if (row["split"], row["video_id"]) not in remove]
    if not kept:
        raise ValueError("pruning would empty the subset")
    for column in FINGERPRINTS:
        groups: dict[str, set[str]] = defaultdict(set)
        for row in kept:
            matched = manifest_by_id[row["image_id"]]
            groups[matched[column]].add(matched["split"])
        if any(len(splits) > 1 for splits in groups.values()):
            raise RuntimeError(f"cross-split {column} collisions remain")

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="prune-", dir=root) as temporary:
        staging = Path(temporary)
        output = staging / "subset.csv"
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(kept)
        digest = _sha256(output)
        subset_id = digest[:20]
        directory = root / subset_id
        if directory.exists():
            valid, errors = verify_subset(directory)
            if not valid or _sha256(directory / "subset.csv") != digest:
                raise RuntimeError(f"existing immutable subset differs: {directory}: {errors}")
            reused = True
        else:
            metadata = {
                "schema_version": 1,
                "builder_version": __version__,
                "purpose": "cross-split collision-pruned pilot",
                "subset_id": subset_id,
                "subset_sha256": digest,
                "row_count": len(kept),
                "source_subset_sha256": _sha256(subset_path),
                "source_manifest_sha256": _sha256(manifest_path),
                "split_priority": SPLIT_PRIORITY,
                "collision_groups": dict(collisions),
                "removed_video_groups": [list(group) for group in sorted(remove)],
                "removed_rows": len(subset) - len(kept),
                "selection_counts": [
                    {"split": split, "label": label, "images": count}
                    for (split, label), count in sorted(
                        Counter((row["split"], row["label"]) for row in kept).items()
                    )
                ],
            }
            metadata_path = staging / "subset.json"
            metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
            (staging / "SHA256SUMS").write_text(
                f"{digest}  subset.csv\n{_sha256(metadata_path)}  subset.json\n"
            )
            staging.rename(directory)
            reused = False
    return {
        "subset_id": subset_id,
        "subset": str(directory / "subset.csv"),
        "rows": len(kept),
        "removed_rows": len(subset) - len(kept),
        "removed_video_groups": len(remove),
        "collision_groups": dict(collisions),
        "reused": reused,
    }
