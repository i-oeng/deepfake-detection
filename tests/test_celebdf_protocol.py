from __future__ import annotations

import csv
import json
from pathlib import Path

from deepfake_detection.inference.celebdf import build_report


def _write_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_report_handles_abstentions_identity_subset_and_comparison(tmp_path: Path) -> None:
    videos = [(f"cdf:id{i}_000{i}", "REAL") for i in range(6)]
    # Swap pairs (0,1), (2,3), (4,5) form three identity components.
    videos += [(f"cdf:id{i}_id{i ^ 1}_0000", "FAKE") for i in range(6)]
    videos += [("cdf:00001", "REAL")]
    catalog = _write_csv(tmp_path / "catalog.csv", [
        {"video_id": v, "label": label, "split": "test", "archive_path": f"{v}.mp4"}
        for v, label in videos
    ])
    reference = _write_csv(tmp_path / "manifest.csv", [
        {"split": "validation", "source_domain": "cdf", "video_id": "cdf:id0_0003"},
        {"split": "train", "source_domain": "ff", "video_id": "ff:001"},
    ])
    results = {}
    for name, strength in (("weak", 0.1), ("strong", 0.4)):
        lines = []
        for index, (video, label) in enumerate(videos):
            probability = 0.5 + strength * (1 if label == "FAKE" else -1) + 0.01 * index
            if video == "cdf:00001":
                lines.append({"media_id": video, "label": "ABSTAIN", "probability_fake": None})
                continue
            lines.append({"media_id": video, "probability_fake": probability,
                          "label": "FAKE" if probability >= 0.5 else "REAL"})
        results[name] = tmp_path / f"{name}.jsonl"
        results[name].write_text("".join(json.dumps(line) + "\n" for line in lines))
    output = build_report(results, catalog, reference, ("weak", "strong"))
    assert output["videos"] == 13
    # id0 is a development identity: its real video and both id0/id1 swaps drop out.
    assert output["identity_disjoint_videos"] == 10
    strong = output["bundles"]["strong"]
    assert strong["abstentions"] == 1
    assert strong["excluding_abstentions"]["scored_videos"] == 12
    assert strong["abstain_as_0.5"]["scored_videos"] == 13
    assert strong["excluding_abstentions"]["all"]["auroc"] == 1.0
    assert output["h2"]["difference"] == 0.0
    assert output["h2"]["source_groups"] == 3
