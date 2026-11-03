import csv
import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from deepfake_detection.data import fresh_v4
from deepfake_detection.data.format_match import read_encodings
from deepfake_detection.data.fresh_v3 import generator_of


def _download(root, archive: str, records: list[dict]) -> None:
    folder = root / archive.removeprefix("dir:")
    (folder / "rows").mkdir(parents=True)
    lines = []
    for number, record in enumerate(records):
        digest = hashlib.sha256(f"{archive}{number}".encode()).hexdigest()
        path = f"images/{record.get('label', 'real')}/{digest[:2]}/{digest}.png"
        (folder / path).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8)).save(folder / path)
        lines.append(json.dumps({"status": "stored", "path": path, "image_sha256": digest,
                                 "format": "JPEG", "jpeg_quality": 90, "subsampling": "4:2:0",
                                 **record}))
    lines.append(json.dumps({"status": "no_face", "image_sha256": "0" * 64}))
    (folder / "rows" / "unit_0000.jsonl").write_text("\n".join(lines) + "\n")


def _generated(root, generator: str, count: int) -> None:
    folder = root / fresh_v4.GENERATED_ARCHIVE.removeprefix("dir:") / generator
    folder.mkdir(parents=True)
    lines = []
    for index in range(count):
        Image.new("RGB", (8, 8), (index, 0, 0)).save(folder / f"{index:05d}.png")
        lines.append(json.dumps({"index": index, "generator": generator}))
    (folder / "metadata.jsonl").write_text("\n".join(lines) + "\n")


def test_catalog_builds_both_slices_and_matches_formats_only_where_primary(tmp_path,
                                                                          monkeypatch):
    monkeypatch.setattr(fresh_v4, "SEEN_REALS", 6)
    monkeypatch.setattr(fresh_v4, "UNSEEN_REALS", 5)
    for generator in ("sdxl_base", "sd35_medium"):
        _generated(tmp_path, generator, 3)
    _download(tmp_path, fresh_v4.UNSPLASH_ARCHIVE, [{}] * 2)
    _download(tmp_path, fresh_v4.OPENIMAGES_ARCHIVE, [{}] * 9)
    _download(tmp_path, fresh_v4.OPENFAKE_TEST_ARCHIVE,
              [{"label": "fake", "model": "z-image-turbo", "format": "PNG"}] * 3
              + [{"label": "real", "model": "imagenet"}] * 2)
    result = fresh_v4.build_catalog(tmp_path, tmp_path / "catalog.csv", tmp_path / "enc.csv",
                                    generators=("sdxl_base", "sd35_medium"))
    assert result["counts"] == {
        "v4_photos_seen/fake-sd35_medium": 3, "v4_photos_seen/fake-sdxl_base": 3,
        "v4_photos_seen/real-openimages": 4, "v4_photos_seen/real-unsplash": 2,
        "v4_photos_unseen/fake-z_image_turbo": 3, "v4_photos_unseen/real-imagenet": 2,
        "v4_photos_unseen/real-openimages": 3}
    assert result["unused_openimages_faces"] == 2
    with (tmp_path / "catalog.csv").open(newline="") as stream:
        rows = {row["image_id"]: row for row in csv.DictReader(stream)}
    encodings = read_encodings(tmp_path / "enc.csv")
    assert {rows[image]["source_domain"] for image in encodings} == {"v4_photos_seen"}
    assert {rows[image]["label"] for image in encodings} == {"FAKE"}
    assert len(encodings) == 6 and set(encodings.values()) == {(90, "4:2:0")}
    assert {generator_of(row["video_id"]) for row in rows.values()} >= {"sdxl_base", "unsplash"}
    assert {row["preprocess"] for row in rows.values()} == {
        "canonical:v4_photos_seen", "canonical:v4_photos_unseen"}
    again = fresh_v4.build_catalog(tmp_path, tmp_path / "again.csv", tmp_path / "enc2.csv",
                                   generators=("sdxl_base", "sd35_medium"))
    assert again["sha256"] == result["sha256"]
    assert again["encodings_sha256"] == result["encodings_sha256"]


def test_generated_images_must_all_have_metadata(tmp_path):
    _generated(tmp_path, "sdxl_base", 2)
    folder = tmp_path / fresh_v4.GENERATED_ARCHIVE.removeprefix("dir:") / "sdxl_base"
    Image.new("RGB", (8, 8)).save(folder / "00007.png")
    with pytest.raises(ValueError, match="without metadata"):
        fresh_v4.generated_records(tmp_path, ("sdxl_base",))


def test_drop_leaks_removes_reference_matches_and_one_of_each_copy(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    from deepfake_detection.data.openworld import EFS, _row

    def photo(name, label, source):
        return _row(domain=fresh_v4.SEEN, method=fresh_v4.SEEN if label == "fake" else "",
                    family=EFS, split="test", video=f"{label}-{source}-{name}",
                    frame=f"{name}.png", entities=(name,), archive="dir:x",
                    member=f"{name}.png", preprocess=f"canonical:{fresh_v4.SEEN}", identity=name)

    def fingerprinted(name, source_id="", file="", pixel=""):
        return {"source_id": source_id or name, "relative_path": f"{name}.png", "split": "test",
                "status": "ok", "file_sha256": file or f"file-{name}",
                "pixel_sha256": pixel or f"pixel-{name}", "dhash64": f"dhash-{name}"}

    photos = {"a": photo("a", "fake", "sdxl_base"), "b": photo("b", "real", "unsplash"),
              "c": photo("c", "fake", "sd35_medium"), "d": photo("d", "real", "openimages"),
              "e": photo("e", "real", "openimages"), "f": photo("f", "fake", "realvisxl_v5")}
    catalog = tmp_path / "catalog.csv"
    fresh_v4._write_catalog(catalog, list(photos.values()))
    preliminary, earlier = tmp_path / "preliminary", tmp_path / "earlier"
    manifests = {
        preliminary: [fingerprinted(name, row["image_id"],
                                    file="shared-file" if name == "b" else "",
                                    pixel="shared-pixel" if name in "de" else "")
                      for name, row in photos.items()],
        # An earlier image with b's bytes, and one that c nearly copies.
        earlier: [fingerprinted("ref_b", file="shared-file"), fingerprinted("ref_c")],
    }
    basis = np.eye(8, dtype=np.float32)

    def near(index, other):
        vector = basis[index] + 0.02 * basis[other]
        return vector / np.linalg.norm(vector)

    vectors = {"a": basis[0], "b": basis[1], "c": near(2, 7), "d": basis[3], "e": basis[4],
               "f": near(0, 5), "ref_b": basis[6], "ref_c": basis[2]}

    def thumbnails(paths, workers=8):
        return np.stack([vectors[Path(path).stem] for path in paths])

    monkeypatch.setattr(fresh_v4, "verify_manifest", lambda directory: (True, []))
    monkeypatch.setattr(fresh_v4, "_read_manifest", lambda directory: manifests[directory])
    monkeypatch.setattr(fresh_v4, "thumbnails", thumbnails)
    output = tmp_path / "kept.csv"
    result = fresh_v4.drop_leaks(catalog, preliminary, tmp_path / "data",
                                 [(earlier, tmp_path / "earlier-data")], output)
    with output.open(newline="") as stream:
        kept = {row["image_id"] for row in csv.DictReader(stream)}
    ids = {name: row["image_id"] for name, row in photos.items()}
    assert len(kept) == 2 and len(kept & {ids["a"], ids["f"]}) == 1
    assert len(kept & {ids["d"], ids["e"]}) == 1
    assert result["hits"] == {"file_sha256": 1, "thumbnail_correlation": 1,
                              "within_v4:pixel_sha256": 1,
                              "within_v4:thumbnail_correlation": 1}
    assert result["removed"]["v4_photos_seen/real-unsplash"] == 1
    assert result["reference_images"] == 2
