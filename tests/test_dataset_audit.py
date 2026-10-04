from __future__ import annotations

import csv
import random
import shutil
from pathlib import Path

import pytest
import yaml
from PIL import Image

from deepfake_detection.data import (
    audit_manifest,
    build_manifest,
    load_dataset_spec,
    verify_manifest,
    write_audit_report,
)


def _pattern_image(path: Path, seed: int) -> None:
    rng = random.Random(seed)
    image = Image.new("RGB", (32, 32))
    image.putdata(
        [
            (
                (x * 17 + rng.randrange(32)) % 256,
                (y * 29 + rng.randrange(32)) % 256,
                ((x + y) * 11 + rng.randrange(32)) % 256,
            )
            for y in range(32)
            for x in range(32)
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _fixture_project(tmp_path: Path) -> Path:
    data_root = tmp_path / "data" / "raw" / "fixture"
    image_root = data_root / "images"
    _pattern_image(image_root / "train" / "real.png", seed=1)
    _pattern_image(image_root / "validation" / "fake.png", seed=2)
    duplicate = image_root / "test" / "fake_duplicate.png"
    duplicate.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(image_root / "validation" / "fake.png", duplicate)
    corrupt = image_root / "train" / "corrupt.jpg"
    corrupt.write_bytes(b"not an image")

    rows = [
        {
            "image_id": "real-1",
            "relative_path": "images/train/real.png",
            "label": "REAL",
            "split": "train",
            "identity_id": "person-real",
            "fake_method": "",
        },
        {
            "image_id": "fake-1",
            "relative_path": "images/validation/fake.png",
            "label": "FAKE",
            "split": "validation",
            "identity_id": "person-duplicate",
            "fake_method": "generator-a",
        },
        {
            "image_id": "fake-2",
            "relative_path": "images/test/fake_duplicate.png",
            "label": "FAKE",
            "split": "test",
            "identity_id": "person-duplicate",
            "fake_method": "generator-a",
        },
        {
            "image_id": "fake-corrupt",
            "relative_path": "images/train/corrupt.jpg",
            "label": "FAKE",
            "split": "train",
            "identity_id": "person-corrupt",
            "fake_method": "generator-b",
        },
        {
            "image_id": "real-missing",
            "relative_path": "images/test/missing.png",
            "label": "REAL",
            "split": "test",
            "identity_id": "person-missing",
            "fake_method": "",
        },
    ]
    metadata_path = data_root / "metadata.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    config = {
        "dataset": {
            "name": "fixture",
            "version": "1",
            "data_root": "data/raw/fixture",
            "metadata_path": "metadata.csv",
            "output_root": "data/manifests",
            "report_root": "reports/data_audit",
            "id_column": "image_id",
            "label_column": "label",
            "split_column": "split",
            "path_column": "relative_path",
            "metadata_columns": ["fake_method"],
            "group_columns": ["identity_id"],
            "leakage_check_columns": ["fake_method"],
            "expected_labels": ["REAL", "FAKE"],
            "expected_splits": ["train", "validation", "test"],
            "workers": 2,
            "quality_gates": {
                "max_missing_images": 0,
                "max_decode_errors": 0,
                "max_cross_split_exact_groups": 0,
                "max_cross_split_pixel_groups": 0,
                "max_cross_split_perceptual_groups": 0,
                "max_group_overlap_values": 0,
                "leakage_purity_threshold": 0.95,
            },
        }
    }
    config_path = tmp_path / "fixture.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def test_manifest_is_content_addressed_and_reusable(tmp_path: Path) -> None:
    config_path = _fixture_project(tmp_path)
    spec = load_dataset_spec(config_path, project_root=tmp_path)

    first = build_manifest(spec)
    second = build_manifest(spec)

    assert first.manifest_id == second.manifest_id
    assert first.manifest_sha256 == second.manifest_sha256
    assert not first.reused_existing
    assert second.reused_existing
    assert first.row_count == 5
    valid, errors = verify_manifest(first.directory)
    assert valid
    assert errors == []


def test_audit_detects_data_and_split_failures(tmp_path: Path) -> None:
    config_path = _fixture_project(tmp_path)
    spec = load_dataset_spec(config_path, project_root=tmp_path)
    manifest = build_manifest(spec)

    report = audit_manifest(manifest.manifest_path, spec)
    artifact = write_audit_report(report, spec)

    assert not artifact.passed
    assert report["summary"]["status_counts"] == {
        "decode_error": 1,
        "missing_image": 1,
        "ok": 3,
    }
    assert report["duplicates"]["file_sha256"]["cross_split_groups"] == 1
    assert report["duplicates"]["pixel_sha256"]["cross_split_groups"] == 1
    assert report["group_overlap"]["identity_id"]["overlapping_values"] == 1
    assert report["metadata_leakage_indicators"]["fake_method"]["label_purity"] == 1.0
    assert report["quality_gates"]["failures"]
    assert report["quality_gates"]["warnings"]


def test_modified_manifest_is_not_overwritten(tmp_path: Path) -> None:
    config_path = _fixture_project(tmp_path)
    spec = load_dataset_spec(config_path, project_root=tmp_path)
    artifact = build_manifest(spec)
    artifact.manifest_path.write_text("tampered\n", encoding="utf-8")

    valid, errors = verify_manifest(artifact.directory)
    assert not valid
    assert any("checksum mismatch" in error for error in errors)
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        build_manifest(spec)


def test_path_outside_dataset_root_is_rejected(tmp_path: Path) -> None:
    config_path = _fixture_project(tmp_path)
    data_root = tmp_path / "data" / "raw" / "fixture"
    metadata_path = data_root / "metadata.csv"
    rows = list(csv.DictReader(metadata_path.open(encoding="utf-8")))
    rows[0]["relative_path"] = "../../outside.png"
    with metadata_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    spec = load_dataset_spec(config_path, project_root=tmp_path)
    artifact = build_manifest(spec)
    report = audit_manifest(artifact.manifest_path, spec)
    assert report["summary"]["status_counts"]["outside_data_root"] == 1
