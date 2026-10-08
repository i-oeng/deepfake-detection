import csv
import zipfile

from deepfake_detection.data import fresh


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_validation_identities_come_from_validation_cdf_lineage(tmp_path):
    path = tmp_path / "manifest.csv"
    _write_csv(path, [
        {"split": "validation", "source_domain": "cdf", "lineage_video_ids": "cdf:id1|cdf:id2"},
        {"split": "validation", "source_domain": "cdf",
         "lineage_video_ids": "synthetic:sit:cdf:id9_0001"},
        {"split": "test", "source_domain": "cdf", "lineage_video_ids": "cdf:id3"},
        {"split": "validation", "source_domain": "ff", "lineage_video_ids": "ff:001"},
    ])
    assert fresh.validation_identities(path) == {"cdf:id1", "cdf:id2"}


def test_celebdf_selection_excludes_validation_identities(tmp_path):
    path = tmp_path / "video_catalog.csv"
    _write_csv(path, [
        {"video_id": "cdf:id1_0000", "archive_path": "Celeb-real/id1_0000.mp4", "label": "REAL"},
        {"video_id": "cdf:id3_0000", "archive_path": "Celeb-real/id3_0000.mp4", "label": "REAL"},
        {"video_id": "cdf:00000", "archive_path": "YouTube-real/00000.mp4", "label": "REAL"},
        {"video_id": "cdf:id3_id1_0000", "archive_path": "Celeb-synthesis/id3_id1_0000.mp4",
         "label": "FAKE"},
        {"video_id": "cdf:id3_id4_0000", "archive_path": "Celeb-synthesis/id3_id4_0000.mp4",
         "label": "FAKE"},
    ])
    selected = fresh.select_celebdf(path, {"cdf:id1"}, "seed")
    assert {video["name"] for video in selected} == {"id3_0000", "00000", "id3_id4_0000"}
    youtube = next(video for video in selected if video["name"] == "00000")
    assert youtube["tokens"] == ""


def test_dff_reals_and_fakes_share_source_photos(tmp_path):
    root = tmp_path / "DeepFakeFace"
    root.mkdir()
    names = [f"61/{index}_1980-01-01_2009.jpg" for index in range(5)]
    for archive in fresh.DFF_ARCHIVES:
        folder = archive.removesuffix(".zip")
        with zipfile.ZipFile(root / archive, "w") as stream:
            for name in names:
                stream.writestr(f"{folder}/{name}", b"x")
    rows = fresh._dff_rows(tmp_path, "seed")
    sources = {}
    for row in rows:
        sources.setdefault(row["fake_method"] or "real", set()).add(
            row["source_path"].split("/", 1)[1])
    assert len(sources) == 4 and len({frozenset(value) for value in sources.values()}) == 1
    assert {row["identity_id"] for row in rows if row["fake_method"] == "insightface"} == {
        f"dff-{index}" for index in range(5)}


def test_sfhq_generator_names():
    assert fresh.SFHQ_GENERATOR.match("FLUX1_dev_image_0000110").group(1) == "FLUX1_dev"
    assert fresh.SFHQ_GENERATOR.match("SDXL_image_0000001").group(1) == "SDXL"
