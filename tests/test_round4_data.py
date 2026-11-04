import csv
import hashlib
import json

import pytest

from deepfake_detection.data import openworld, round4
from deepfake_detection.data.format_match import read_encodings


def _download(root, archive, records):
    folder = root / archive.removeprefix("dir:")
    (folder / "rows").mkdir(parents=True)
    lines = []
    for number, record in enumerate(records):
        digest = hashlib.sha256(f"{archive}{number}".encode()).hexdigest()
        lines.append(json.dumps({"status": "stored", "image_sha256": digest,
                                 "path": f"images/{record['label']}/{digest}.png", **record}))
    lines.append(json.dumps({"status": "no_face", "image_sha256": "0" * 64}))
    (folder / "rows" / "unit_0000.jsonl").write_text("\n".join(lines) + "\n")


def _jpeg(quality):
    return {"format": "JPEG", "jpeg_quality": quality, "subsampling": "4:2:0"}


PNG = {"format": "PNG"}


def _sources(root):
    _download(root, round4.OPENFAKE_ARCHIVE,
              [{"label": "fake", "group": "flux1", **PNG}] * 4
              + [{"label": "fake", "group": "sdxl", **_jpeg(95)}] * 2
              + [{"label": "fake", "group": "kolors", **PNG}]
              + [{"label": "fake", "group": "other:amateur-v6", **_jpeg(90)}]
              + [{"label": "fake", "group": "flux1", "public_figures": "Donald Trump", **PNG}]
              + [{"label": "real", "group": "laion", **_jpeg(75)}] * 3
              + [{"label": "real", "group": "pexels", **_jpeg(95)}] * 2
              + [{"label": "real", "group": "laion", "public_figures": "Prince William",
                  **_jpeg(75)}])
    _download(root, round4.SFHQ_ARCHIVE,
              [{"label": "fake", "group": "flux1_dev", **_jpeg(90)}] * 2
              + [{"label": "real", "group": "ffhq", "format": "WEBP"}] * 3)


EXPECTED = {"openfake_flux1", "openfake_sdxl", "openfake_rare", "sfhq_flux1_dev"}


def test_catalog_caps_families_pools_rare_ones_and_holds_back_public_figures(tmp_path):
    _sources(tmp_path)
    output, encodings = tmp_path / "catalog.csv", tmp_path / "encodings.csv"
    result = round4.build_additions_catalog(tmp_path, output, encodings, cap=3, minimum=2,
                                            methods=EXPECTED)
    assert result["counts"] == {
        "round4_openfake/openfake_flux1": 3, "round4_openfake/openfake_rare": 2,
        "round4_openfake/openfake_sdxl": 2, "round4_openfake/real-laion": 3,
        "round4_openfake/real-pexels": 2, "round4_sfhq/real-ffhq": 3,
        "round4_sfhq/sfhq_flux1_dev": 2}
    assert result["rare_families"] == ["kolors", "other:amateur-v6"]
    assert result["held_back_public_figures"] == {"fake": 1, "real": 1}
    with output.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert {row["split"] for row in rows} == {"train"}
    assert {row["lineage_video_ids"].split(":")[0] for row in rows if row["label"] == "REAL"} \
        == {"laion", "pexels", "ffhq"}
    assert {row["preprocess"] for row in rows} == {"canonical:round4_openfake",
                                                   "canonical:round4_sfhq"}
    # OpenFake's reals are the reference, so only its PNG fakes are encoded; SFHQ's
    # reference is its fakes, so every FFHQ real is encoded at their quality.
    matching = result["format_matching"]
    assert matching["canonical:round4_openfake"]["reference"] == "REAL"
    assert matching["canonical:round4_openfake"]["encoded"] == {"FAKE": 4}
    assert matching["canonical:round4_sfhq"]["reference"] == "FAKE"
    by_id = {row["image_id"]: row for row in rows}
    pairs = read_encodings(encodings)
    assert {by_id[image]["source_domain"] for image, pair in pairs.items()
            if pair == (90, "4:2:0")} >= {"round4_sfhq"}
    assert sum(by_id[image]["label"] == "REAL" for image in pairs) == 3
    again = round4.build_additions_catalog(tmp_path, tmp_path / "again.csv",
                                           tmp_path / "again_enc.csv", cap=3, minimum=2,
                                           methods=EXPECTED)
    assert again["sha256"] == result["sha256"]
    assert again["encodings_sha256"] == result["encodings_sha256"]


def test_catalog_refuses_methods_outside_the_protocol(tmp_path):
    _sources(tmp_path)
    with pytest.raises(ValueError, match="unexpected"):
        round4.build_additions_catalog(tmp_path, tmp_path / "c.csv", tmp_path / "e.csv",
                                       cap=3, minimum=2)


def test_rare_families_share_one_method():
    records = ([{"label": "fake", "group": "flux1"}] * 3 + [{"label": "fake", "group": "kolors"}]
               + [{"label": "real", "group": "laion"}] * 5)
    assert round4.openfake_methods(records, minimum=2) == {
        "flux1": "openfake_flux1", "kolors": "openfake_rare"}


def test_round4_protocol_adds_only_training_methods():
    round3 = openworld.method_roles(openworld.PROTOCOL_ROUND3)
    round4_roles = openworld.method_roles(openworld.PROTOCOL_ROUND4)
    assert round4_roles["train"] - round3["train"] == openworld.ROUND4_TRAIN_METHODS
    assert round4_roles["validation"] == round3["validation"]
    assert round4_roles["test"] == round3["test"]
    names = [method for methods in round4_roles.values() for method in methods]
    assert len(names) == len(set(names))
    assert len(openworld.ROUND4_TRAIN_METHODS) == 21


def _manifest_row(source_id, **changes):
    row = {"source_id": source_id, "relative_path": f"images/{source_id}.jpg",
           "label": "REAL", "split": "train", "file_sha256": f"f{source_id}",
           "pixel_sha256": f"p{source_id}", "source_domain": "ff", "fake_method": "",
           "video_id": f"ff:{source_id}", "lineage_video_ids": f"ff:{source_id}"}
    return {**row, **changes}


def test_base_unchanged_finds_missing_changed_and_stray_rows():
    base = [_manifest_row("a"), _manifest_row("b"), _manifest_row("c")]
    added = _manifest_row("d", source_domain="round4_openfake")
    assert round4.base_unchanged([*base, added], base) == []
    rows = [base[0], _manifest_row("b", pixel_sha256="other"), added,
            _manifest_row("e", split="validation", source_domain="round4_sfhq")]
    assert round4.base_unchanged(rows, base) == [
        "1 round 3 rows are missing", "1 round 3 rows changed pixel_sha256",
        "1 added rows are not round 4 training photos"]
