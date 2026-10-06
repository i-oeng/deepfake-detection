from __future__ import annotations

import csv
from pathlib import Path

from deepfake_detection.data.df40 import COLUMNS, partition_df40_heldout
from deepfake_detection.data.unseen import HELDOUT_TARGETS, audit_heldout_rows


def _catalog_row(**values: str) -> dict[str, str]:
    row = dict.fromkeys(COLUMNS, "")
    row.update(values)
    return row


def _write(path: Path, rows: list[dict[str, str]], fields: tuple[str, ...]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_heldout_partition_excludes_frozen_training_sources(tmp_path: Path) -> None:
    reference = _write(
        tmp_path / "reference.csv",
        [
            {"split": "train", "source_domain": "ff", "video_id": "ff:001_002",
             "lineage_video_ids": "ff:001|ff:002"},
            {"split": "test", "source_domain": "ff", "video_id": "ff:003",
             "lineage_video_ids": "ff:003"},
        ],
        ("split", "source_domain", "video_id", "lineage_video_ids"),
    )
    shared = {"source_json": "x.json", "manipulation_family": "face_swap"}
    catalog = _write(
        tmp_path / "catalog.csv",
        [
            # Shares source clip 002 with frozen training: excluded.
            _catalog_row(label="FAKE", split="test", fake_method="simswap", source_domain="ff",
                         video_id="ff:002_009", source_path="a/1.png", **shared),
            # Disjoint from training; frozen test sources stay eligible.
            _catalog_row(label="FAKE", split="test", fake_method="simswap", source_domain="ff",
                         video_id="ff:003_010", source_path="a/2.png", **shared),
            # The authors' training pool is never used.
            _catalog_row(label="FAKE", split="train", fake_method="simswap", source_domain="ff",
                         video_id="ff:011_012", source_path="a/3.png", **shared),
            # FOMM and unseen methods have no held-out role.
            _catalog_row(label="FAKE", split="test", fake_method="uniface", source_domain="ff",
                         video_id="ff:013_014", source_path="a/4.png", **shared),
            # The same real frame listed by two method catalogs is kept once.
            _catalog_row(label="REAL", split="test", source_domain="ff", video_id="ff:020",
                         source_path="r/1.png", source_json="a.json"),
            _catalog_row(label="REAL", split="test", source_domain="ff", video_id="ff:020",
                         source_path="r/1.png", source_json="b.json"),
            _catalog_row(label="REAL", split="test", source_domain="celeba",
                         video_id="celeba:7", source_path="c/7.jpg", source_json="c.json"),
        ],
        COLUMNS,
    )
    result = partition_df40_heldout(catalog, reference, tmp_path / "out.csv")
    with (tmp_path / "out.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert sorted(row["video_id"] for row in rows) == ["celeba:7", "ff:003_010", "ff:020"]
    assert {row["split"] for row in rows} == {"test"}
    assert {row["lineage_video_ids"] for row in rows if row["source_domain"] == "celeba"} == {
        "celeba:7"
    }
    assert result["excluded_rows_by_domain"] == {"ff": 1}


def _audit_row(index: int, domain: str, method: str, **values: str) -> dict[str, str]:
    row = {
        "sample_id": f"s{index}", "split": "test", "status": "ok",
        "label": "REAL" if method == "real" else "FAKE",
        "fake_method": "" if method == "real" else method,
        "source_domain": domain, "video_id": f"{domain}:{index}",
        "lineage_video_ids": f"{domain}:{index}", "identity_id": "",
        "target_identity_id": "", "pixel_sha256": f"p{index}", "dhash64": f"d{index}",
    }
    row.update(values)
    return row


def _complete_heldout() -> list[dict[str, str]]:
    rows, index = [], 0
    for (domain, method), count in HELDOUT_TARGETS.items():
        for _ in range(count):
            index += 1
            # CDF identities come from idNN tokens in published video names.
            video = f"cdf:id{index}_0000" if domain == "cdf" else f"{domain}:{index}"
            rows.append(_audit_row(index, domain, method, video_id=video))
    return rows


def test_heldout_gate_accepts_disjoint_complete_manifest() -> None:
    reference = [_audit_row(10**6, "ff", "simswap", split="train")]
    result = audit_heldout_rows(_complete_heldout(), reference)
    assert result["passed"] is True, result["failures"]


def test_heldout_gate_rejects_shared_training_lineage_and_short_cells() -> None:
    rows = _complete_heldout()
    reference = [_audit_row(10**6, "ff", "simswap", split="train", dhash64=rows[0]["dhash64"])]
    rows = [row for row in rows if row["fake_method"] != "starganv2"]
    result = audit_heldout_rows(rows, reference)
    assert result["passed"] is False
    assert "1 dhash keys shared with frozen train or validation" in result["failures"]
    assert "celeba/starganv2: 0 of 300 videos" in result["failures"]


def test_heldout_gate_allows_overlap_with_frozen_test() -> None:
    rows = _complete_heldout()
    reference = [_audit_row(1, "ff", "simswap", split="test")]
    assert audit_heldout_rows(rows, reference)["passed"] is True
