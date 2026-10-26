import csv
import json

from deepfake_detection.data import fresh_v3


def _metadata(folder, records):
    folder.mkdir(parents=True)
    (folder / "metadata.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))


def test_catalog_excludes_training_families_and_keeps_one_lfw_photo_per_person(
    tmp_path, monkeypatch
):
    root = tmp_path / "raw"
    _metadata(root / "round3_sources" / "openfake", [
        {"index": 0, "split": "core-train", "label": "real", "model": "laion",
         "file": "core-train/000000.jpg"},
        {"index": 1, "split": "core-test", "label": "fake", "model": "flux.1-dev",
         "file": "core-test/000001.png"},
        {"index": 2, "split": "core-test", "label": "fake", "model": "sd-2.1",
         "file": "core-test/000002.png"},
        {"index": 3, "split": "reddit-test", "label": "fake", "model": "reddit",
         "file": "reddit-test/000003.jpg"},
        {"index": 4, "split": "reddit-test", "label": "real", "model": "reddit",
         "file": "reddit-test/000004.jpg"},
    ])
    _metadata(root / "round3_sources" / "lfw", [
        {"index": 0, "filename": "Ada_Lovelace_0002.jpg", "file": "train/000000.jpg"},
        {"index": 1, "filename": "Ada_Lovelace_0001.jpg", "file": "train/000001.jpg"},
        {"index": 2, "filename": "Alan_Turing_0001.jpg", "file": "train/000002.jpg"},
    ])
    monkeypatch.setattr(fresh_v3, "LFW_PEOPLE", 5)
    output = tmp_path / "catalog.csv"
    result = fresh_v3.build_catalog(root, output, seed="test")
    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    assert result["excluded_generators"] == {"sd-2.1": 1}
    assert {row["split"] for row in rows} == {"test"}
    by_domain = {}
    for row in rows:
        by_domain.setdefault(row["source_domain"], set()).add(row["fake_method"] or "real")
    assert by_domain == {"openfake": {"openfake", "real"},
                         "openfake_reddit": {"reddit_ai", "real"}, "lfw": {"real"}}
    lfw = [row for row in rows if row["source_domain"] == "lfw"]
    assert sorted(row["member"] for row in lfw) == ["train/000001.jpg", "train/000002.jpg"]
    fake = next(row for row in rows if row["fake_method"] == "openfake")
    assert fresh_v3.generator_of(fake["video_id"]) == "flux.1-dev"
