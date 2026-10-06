from __future__ import annotations

from deepfake_detection.data.df40 import _source_entities
from deepfake_detection.data.unseen import METHODS, audit_unseen_rows


def _rows() -> list[dict[str, str]]:
    rows = []
    index = 0
    for split, methods in METHODS.items():
        for method in sorted(methods):
            index += 1
            rows.append(
                {
                    "sample_id": f"fake-{index}",
                    "split": split,
                    "label": "FAKE",
                    "fake_method": method,
                    "video_id": f"ff:{index}",
                    "source_domain": "ff",
                    "lineage_video_ids": f"ff:{index}",
                    "identity_id": "",
                    "target_identity_id": "",
                    "pixel_sha256": f"pixel-{index}",
                    "dhash64": f"dhash-{index}",
                    "status": "ok",
                }
            )
        index += 1
        rows.append(
            {
                "sample_id": f"real-{index}",
                "split": split,
                "label": "REAL",
                "fake_method": "",
                "video_id": f"ff:{index}",
                "source_domain": "ff",
                "lineage_video_ids": f"ff:{index}",
                "identity_id": "",
                "target_identity_id": "",
                "pixel_sha256": f"pixel-{index}",
                "dhash64": f"dhash-{index}",
                "status": "ok",
            }
        )
    return rows


def test_unseen_gate_accepts_complete_disjoint_protocol() -> None:
    result = audit_unseen_rows(_rows())
    assert result["passed"] is True
    assert not result["failures"]


def test_unseen_gate_rejects_cross_split_source_identity() -> None:
    rows = _rows()
    test_row = next(row for row in rows if row["split"] == "test")
    test_row["lineage_video_ids"] = rows[0]["lineage_video_ids"]
    result = audit_unseen_rows(rows)
    assert result["passed"] is False
    assert "1 cross-split parent groups" in result["failures"]


def test_source_entity_parser_keeps_both_swap_participants() -> None:
    assert _source_entities("ff", "ff:137_165") == ("ff:137", "ff:165")
    assert _source_entities("ff", "ff:652_video_method") == ("ff:652",)
    assert _source_entities("cdf", "cdf:id9_id0_0006") == ("cdf:id0", "cdf:id9")
    assert _source_entities("cdf", "cdf:opaque") == ()
