from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from deepfake_detection.data.materialize import materialize_df40

FIELDNAMES = [
    "image_id", "relative_path", "source_path", "split", "label", "fake_method",
    "source_domain",
]


def _subset(tmp_path: Path, rows: list[dict[str, str]]) -> Path:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    content = output.getvalue().encode()
    digest = hashlib.sha256(content).hexdigest()
    directory = tmp_path / digest[:20]
    directory.mkdir()
    (directory / "subset.csv").write_bytes(content)
    metadata = {
        "subset_id": digest[:20], "subset_sha256": digest, "row_count": len(rows)
    }
    metadata_bytes = (json.dumps(metadata) + "\n").encode()
    (directory / "subset.json").write_bytes(metadata_bytes)
    (directory / "SHA256SUMS").write_text(
        f"{digest}  subset.csv\n"
        f"{hashlib.sha256(metadata_bytes).hexdigest()}  subset.json\n",
        encoding="utf-8",
    )
    return directory


def _row(
    name: str, source: str, *, label: str, domain: str, method: str = "", split: str = "test"
) -> dict[str, str]:
    return {
        "image_id": name,
        "relative_path": f"images/{name}.png",
        "source_path": f"deepfakes_detection_datasets/{source}",
        "split": split,
        "label": label,
        "fake_method": method,
        "source_domain": domain,
    }


def _archive(root: Path, name: str, member: str, data: bytes) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w") as archive:
        archive.writestr(member, data)


def test_materializes_only_selected_members_and_reuses_verified_files(tmp_path: Path) -> None:
    rows = [
        _row(
            "train-fake", "DF40_train/simswap/frames/001/000.png",
            label="FAKE", domain="ff", method="simswap", split="train",
        ),
        _row(
            "test-fake", "DF40/DiT/cdf/YouTube-real/00043/458.png",
            label="FAKE", domain="cdf", method="dit",
        ),
        _row(
            "ff-real", "FaceForensics++/original_sequences/youtube/c23/frames/001/000.png",
            label="REAL", domain="ff",
        ),
        _row(
            "cdf-real", "Celeb-DF-v2/YouTube-real/frames/00011/000.png",
            label="REAL", domain="cdf",
        ),
        _row(
            "celeba-real", "DF40/starganv2/real/1007.jpg",
            label="REAL", domain="celeba",
        ),
    ]
    subset_dir = _subset(tmp_path, rows)
    downloads = tmp_path / "downloads"
    data_root = tmp_path / "data" / "raw" / "df40"
    _archive(downloads, "train/simswap.zip", "simswap/frames/001/000.png", b"train")
    _archive(
        downloads, "test/DiT.zip", "DiT/cdf/Fake_from_Youtube-real/00043/458.png", b"dit"
    )
    _archive(
        downloads, "real/FaceForensics++_real_data_for_DF40.zip",
        "FaceForensics++/original_sequences/youtube/c23/frames/001/000.png", b"ff",
    )
    _archive(
        downloads, "real/Celeb-DF-v2_real_data_for_DF40.zip",
        "Celeb-DF-v2/YouTube-real/frames/00011/000.png", b"cdf",
    )
    _archive(downloads, "test/starganv2.zip", "starganv2/real/1007.jpg", b"celeba")

    dry_run = materialize_df40(subset_dir, downloads, data_root, dry_run=True)
    assert dry_run["selected"] == 5
    assert dry_run["written"] == 0
    assert not (data_root / "images").exists()

    result = materialize_df40(subset_dir, downloads, data_root)
    assert result["written"] == 5
    assert (data_root / "images/train-fake.png").read_bytes() == b"train"
    assert (data_root / "images/test-fake.png").read_bytes() == b"dit"
    assert materialize_df40(subset_dir, downloads, data_root)["reused"] == 5

    (data_root / "images/train-fake.png").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="refusing to overwrite changed image"):
        materialize_df40(subset_dir, downloads, data_root)


def test_rejects_unsafe_destination_before_extraction(tmp_path: Path) -> None:
    row = _row(
        "unsafe", "DF40_train/simswap/frames/001/000.png",
        label="FAKE", domain="ff", method="simswap", split="train",
    )
    row["relative_path"] = "images/../../escape.png"
    subset_dir = _subset(tmp_path, [row])
    with pytest.raises(ValueError, match="unsafe destination path"):
        materialize_df40(subset_dir, tmp_path / "downloads", tmp_path / "data")
