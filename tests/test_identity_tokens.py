from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from deepfake_detection.data.identity import (
    audit_candidate_identity_tokens,
    candidate_ids,
    write_candidate_identity_report,
)


def _manifest(tmp_path: Path) -> Path:
    rows = [
        {"source_domain": "cdf", "video_id": "cdf:id1_0001", "split": "train",
         "label": "REAL", "status": "ok"},
        {"source_domain": "cdf", "video_id": "cdf:id2_0001", "split": "validation",
         "label": "REAL", "status": "ok"},
        {"source_domain": "cdf", "video_id": "cdf:id2_id3_0002", "split": "test",
         "label": "FAKE", "status": "ok"},
        {"source_domain": "cdf", "video_id": "cdf:00043", "split": "test",
         "label": "REAL", "status": "ok"},
        {"source_domain": "ff", "video_id": "ff:id2_0004", "split": "train",
         "label": "FAKE", "status": "ok"},
    ]
    directory = tmp_path / "manifest"
    directory.mkdir()
    path = directory / "manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    final = tmp_path / digest[:20]
    directory.rename(final)
    metadata = {"manifest_id": digest[:20], "manifest_sha256": digest}
    data = (json.dumps(metadata) + "\n").encode()
    (final / "manifest.json").write_bytes(data)
    (final / "SHA256SUMS").write_text(
        f"{digest}  manifest.csv\n"
        f"{hashlib.sha256(data).hexdigest()}  manifest.json\n"
    )
    return final


def test_candidate_identity_audit_detects_source_target_overlap(tmp_path: Path) -> None:
    assert candidate_ids("cdf:id2_id3_0002") == {"id2", "id3"}
    report = audit_candidate_identity_tokens(_manifest(tmp_path))
    assert report["split_pairs"]["validation_vs_test"]["shared_tokens"] == ["id2"]
    assert report["split_pairs"]["validation_vs_test"]["affected_rows"] == 2
    assert report["split_pairs"]["train_vs_validation"]["shared_token_count"] == 0
    assert report["row_coverage"]["test"]["without_token"] == 1
    output = tmp_path / "report.json"
    assert write_candidate_identity_report(report, output) == output
    assert write_candidate_identity_report(report, output) == output
    output.write_text("changed")
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        write_candidate_identity_report(report, output)
