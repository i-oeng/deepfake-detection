from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from deepfake_detection.data.prune import prune_cross_split
from deepfake_detection.data.subset import verify_subset


def _artifact(root: Path, kind: str, rows: list[dict[str, str]]) -> Path:
    root.mkdir(parents=True)
    csv_name = f"{kind}.csv"
    with (root / csv_name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    digest = hashlib.sha256((root / csv_name).read_bytes()).hexdigest()
    directory = root.parent / digest[:20]
    root.rename(directory)
    metadata = {
        f"{kind}_id": digest[:20], f"{kind}_sha256": digest, "row_count": len(rows)
    }
    data = (json.dumps(metadata) + "\n").encode()
    (directory / f"{kind}.json").write_bytes(data)
    (directory / "SHA256SUMS").write_text(
        f"{digest}  {csv_name}\n"
        f"{hashlib.sha256(data).hexdigest()}  {kind}.json\n"
    )
    return directory


def test_pruning_removes_whole_lower_priority_video_groups(tmp_path: Path) -> None:
    subset_rows = [
        {"image_id": sample, "split": split, "video_id": video, "label": label}
        for sample, split, video, label in [
            ("train-1", "train", "train-video", "FAKE"),
            ("train-2", "train", "train-video", "FAKE"),
            ("val-1", "validation", "val-video", "FAKE"),
            ("test-1", "test", "test-video", "FAKE"),
        ]
    ]
    manifest_rows = [
        {
            "source_id": row["image_id"],
            "split": row["split"],
            "video_id": row["video_id"],
            "status": "ok",
            "file_sha256": file_hash,
            "pixel_sha256": file_hash,
            "dhash64": visual_hash,
        }
        for row, file_hash, visual_hash in zip(
            subset_rows, ["same", "unique", "same", "other"],
            ["a", "b", "a", "a"], strict=True,
        )
    ]
    subset_dir = _artifact(tmp_path / "subset-staging", "subset", subset_rows)
    manifest_dir = _artifact(tmp_path / "manifest-staging", "manifest", manifest_rows)
    result = prune_cross_split(subset_dir, manifest_dir, tmp_path / "clean")
    assert result["rows"] == 1
    assert result["removed_rows"] == 3
    assert result["removed_video_groups"] == 2
    clean_dir = Path(str(result["subset"])).parent
    assert verify_subset(clean_dir) == (True, [])
    with (clean_dir / "subset.csv").open(newline="") as stream:
        assert [row["image_id"] for row in csv.DictReader(stream)] == ["test-1"]
    assert prune_cross_split(subset_dir, manifest_dir, tmp_path / "clean")["reused"]

    manifest_rows[0]["source_id"] = "unexpected"
    wrong_manifest = _artifact(tmp_path / "wrong-manifest-staging", "manifest", manifest_rows)
    with pytest.raises(ValueError, match="source IDs must match"):
        prune_cross_split(subset_dir, wrong_manifest, tmp_path / "clean")
