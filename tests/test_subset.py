from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest
import yaml

from deepfake_detection.data import build_subset, load_subset_spec, verify_subset

COLUMNS = [
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
]


def _fake_rows(
    split: str,
    method: str,
    family: str,
    domain: str,
    groups: int,
    frames: int,
) -> list[dict[str, str]]:
    rows = []
    for group in range(groups):
        for frame in range(frames):
            sample_id = f"{split}-{method}-{group}-{frame}"
            rows.append(
                {
                    "image_id": sample_id,
                    "relative_path": f"images/{split}/{method}/{group}/{frame}.png",
                    "label": "FAKE",
                    "split": split,
                    "fake_method": method,
                    "manipulation_family": family,
                    "source_domain": domain,
                    "video_id": f"{split}-{method}-video-{group}",
                    "identity_id": f"{split}-{method}-identity-{group}",
                    "frame_index": str(frame),
                }
            )
    return rows


def _real_rows(
    split: str,
    domain: str,
    groups: int,
    frames: int,
) -> list[dict[str, str]]:
    rows = []
    for group in range(groups):
        for frame in range(frames):
            sample_id = f"{split}-real-{group}-{frame}"
            rows.append(
                {
                    "image_id": sample_id,
                    "relative_path": f"images/{split}/real/{group}/{frame}.png",
                    "label": "REAL",
                    "split": split,
                    "fake_method": "",
                    "manipulation_family": "real",
                    "source_domain": domain,
                    "video_id": f"{split}-real-video-{group}",
                    "identity_id": f"{split}-real-identity-{group}",
                    "frame_index": str(frame),
                }
            )
    return rows


def _write_catalog(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _fixture_project(tmp_path: Path) -> tuple[Path, Path]:
    rows = [
        *_fake_rows("train", "simswap", "face_swap", "ff", 3, 5),
        *_fake_rows("train", "wav2lip", "face_reenactment", "ff", 3, 5),
        *_real_rows("train", "ff", 4, 3),
        *_fake_rows("validation", "simswap", "face_swap", "ff", 2, 4),
        *_fake_rows("validation", "wav2lip", "face_reenactment", "ff", 2, 4),
        *_real_rows("validation", "ff", 2, 2),
        *_fake_rows("test", "blendface", "face_swap", "cdf", 2, 5),
        *_fake_rows("test", "sadtalker", "face_reenactment", "cdf", 2, 5),
        *_real_rows("test", "cdf", 4, 2),
    ]
    metadata_path = tmp_path / "data" / "raw" / "df40" / "metadata.csv"
    _write_catalog(metadata_path, rows)

    config = {
        "subset": {
            "name": "fixture_pilot",
            "source_dataset": "df40",
            "source_metadata": "data/raw/df40/metadata.csv",
            "output_root": "data/subsets",
            "seed": "7",
            "id_column": "image_id",
            "path_column": "relative_path",
            "label_column": "label",
            "split_column": "split",
            "method_column": "fake_method",
            "family_column": "manipulation_family",
            "domain_column": "source_domain",
            "frame_order_column": "frame_index",
            "group_columns": ["video_id", "identity_id", "image_id"],
            "fake_label": "FAKE",
            "real_label": "REAL",
            "real_to_fake_ratio": 1.0,
            "require_disjoint_train_test_methods": True,
            "strict_quotas": True,
            "splits": {
                "train": {
                    "fake_methods": ["simswap", "wav2lip"],
                    "max_images_per_method": 6,
                    "max_groups_per_method": 2,
                    "max_frames_per_group": 3,
                },
                "validation": {
                    "fake_methods": ["simswap", "wav2lip"],
                    "max_images_per_method": 2,
                    "max_groups_per_method": 1,
                    "max_frames_per_group": 2,
                },
                "test": {
                    "fake_methods": ["blendface", "sadtalker"],
                    "max_images_per_method": 4,
                    "max_groups_per_method": 2,
                    "max_frames_per_group": 2,
                },
            },
        }
    }
    config_path = tmp_path / "df40_subset.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path, metadata_path


def test_subset_is_deterministic_balanced_and_verifiable(tmp_path: Path) -> None:
    config_path, metadata_path = _fixture_project(tmp_path)
    spec = load_subset_spec(config_path, project_root=tmp_path)

    first = build_subset(spec)
    selected = pd.read_csv(first.subset_path, dtype=str, keep_default_na=False)

    assert first.row_count == 48
    by_split_label = selected.groupby(["split", "label"]).size().to_dict()
    assert by_split_label == {
        ("test", "FAKE"): 8,
        ("test", "REAL"): 8,
        ("train", "FAKE"): 12,
        ("train", "REAL"): 12,
        ("validation", "FAKE"): 4,
        ("validation", "REAL"): 4,
    }
    for _, group in selected.loc[
        (selected["split"] == "train") & (selected["label"] == "FAKE")
    ].groupby("selection_group"):
        assert group["frame_index"].astype(int).tolist() == [0, 2, 4]

    source = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
    source.iloc[::-1].to_csv(metadata_path, index=False, lineterminator="\n")
    second = build_subset(spec)
    assert second.subset_id == first.subset_id
    assert second.reused_existing

    valid, errors = verify_subset(first.directory)
    assert valid
    assert errors == []


def test_subset_rejects_group_leakage_across_splits(tmp_path: Path) -> None:
    config_path, metadata_path = _fixture_project(tmp_path)
    source = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
    train_group = source.loc[source["split"] == "train", "video_id"].iloc[0]
    test_index = source.index[source["split"] == "test"][0]
    source.loc[test_index, "video_id"] = train_group
    source.to_csv(metadata_path, index=False, lineterminator="\n")

    spec = load_subset_spec(config_path, project_root=tmp_path)
    with pytest.raises(ValueError, match="cross source splits"):
        build_subset(spec)


def test_subset_checks_identity_even_when_video_ids_differ(tmp_path: Path) -> None:
    config_path, metadata_path = _fixture_project(tmp_path)
    source = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
    train_identity = source.loc[source["split"] == "train", "identity_id"].iloc[0]
    test_index = source.index[source["split"] == "test"][0]
    source.loc[test_index, "identity_id"] = train_identity
    source.to_csv(metadata_path, index=False, lineterminator="\n")

    spec = load_subset_spec(config_path, project_root=tmp_path)
    with pytest.raises(ValueError, match="identity_id values cross source splits"):
        build_subset(spec)


def test_subset_rejects_seen_method_in_unseen_test(tmp_path: Path) -> None:
    config_path, _ = _fixture_project(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["subset"]["splits"]["test"]["fake_methods"] = ["simswap"]
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    spec = load_subset_spec(config_path, project_root=tmp_path)
    with pytest.raises(ValueError, match="train and test fake methods overlap"):
        build_subset(spec)
