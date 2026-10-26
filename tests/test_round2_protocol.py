import csv
import json

from deepfake_detection.data import round2
from deepfake_detection.data.openworld import COLUMNS, EFS, _row


def _write_catalog(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _generation(root, generator, seed):
    folder = root / generator
    folder.mkdir(parents=True)
    (folder / "00000.png").write_bytes(b"generated")
    (folder / "provenance.json").write_text(json.dumps({
        "generator": generator, "count": 1, "seed": seed,
    }))
    (folder / "metadata.jsonl").write_text(json.dumps({
        "index": 0, "generator": generator, "seed": seed, "prompt": "a portrait",
    }) + "\n")


def _reconstruction(root, generator, source, seed):
    folder = root / generator
    target = folder / "images" / "ff" / "real" / "train" / "001" / "frame.png"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"reconstructed")
    autoencoder = generator.removeprefix("recon_")
    (folder / "provenance.json").write_text(json.dumps({
        "autoencoder": autoencoder, "count": 1, "seed": seed,
    }))
    (folder / "metadata.jsonl").write_text(json.dumps({
        "output": target.relative_to(folder).as_posix(),
        "source_sample_id": "source-sample",
        "source_relative_path": source,
    }) + "\n")


def test_round2_catalog_pairs_every_new_method_with_reals(tmp_path, monkeypatch):
    seed = "round2-test"
    train_real = _row(
        domain="ff", method="", family=EFS, split="train", video="001", frame="000.png",
        entities=("ff:001",), archive="real/train.zip", member="ff/001/000.png",
        preprocess="copy", identity="ff:001",
    )
    validation_real = _row(
        domain="whichfaceisreal", method="", family=EFS, split="validation",
        video="face-1", frame="real.jpg", entities=("whichface:1",),
        archive="test/whichfaceisreal.zip", member="real/real.jpg",
        preprocess="canonical:whichfaceisreal", identity="whichface:1",
    )
    base = tmp_path / "base.csv"
    _write_catalog(base, [train_real, validation_real])
    generated = tmp_path / "generated"
    for method in ("kandinsky22", "sana600m", "stable_cascade_lite"):
        _generation(generated, method, seed)
    for method in ("recon_movq", "recon_dcae"):
        _reconstruction(generated, method, train_real["relative_path"], seed)
    monkeypatch.setattr(round2, "T2I_COUNTS", {"kandinsky22": 1, "sana600m": 1})
    monkeypatch.setattr(round2, "RECON_COUNTS", {"recon_movq": 1, "recon_dcae": 1})
    monkeypatch.setattr(round2, "CASCADE_COUNT", 1)

    output = tmp_path / "additions.csv"
    result = round2.build_additions_catalog(base, generated, output, seed=seed)
    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    assert result["rows"] == 10
    methods = {row["fake_method"] for row in rows if row["label"] == "FAKE"}
    assert methods == {
        "kandinsky22", "sana600m", "stable_cascade_lite", "recon_movq", "recon_dcae",
    }
    for domain in {row["source_domain"] for row in rows}:
        assert {row["label"] for row in rows if row["source_domain"] == domain} == {
            "REAL", "FAKE",
        }
    assert all(row["preprocess"].startswith("canonical:") for row in rows)
    copied = [row for row in rows if row["label"] == "REAL"]
    assert all(row["identity_id"] for row in copied)


def test_prune_fresh_overlaps_removes_complete_addition_video_group(tmp_path, monkeypatch):
    addition_rows = [
        _row(
            domain="round2_sana600m", method="", family=EFS, split="train",
            video=video, frame=f"{video}.jpg", entities=(f"ff:{video}",),
            archive="dir:data", member=f"{video}.jpg", preprocess="canonical:test",
            identity=f"ff:{video}",
        )
        for video in ("keep", "drop")
    ]
    catalog = tmp_path / "additions.csv"
    _write_catalog(catalog, addition_rows)
    columns = ["source_id", "status", "file_sha256", "pixel_sha256", "dhash64"]

    def write_manifest(directory, rows):
        directory.mkdir()
        with (directory / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    preliminary, fresh = tmp_path / "preliminary", tmp_path / "fresh"
    write_manifest(preliminary, [
        {"source_id": row["image_id"], "status": "ok", "file_sha256": f"f{i}",
         "pixel_sha256": f"p{i}", "dhash64": f"d{i}"}
        for i, row in enumerate(addition_rows)
    ])
    write_manifest(fresh, [
        {"source_id": "fresh", "status": "ok", "file_sha256": "fresh-file",
         "pixel_sha256": "fresh-pixel", "dhash64": "d1"}
    ])
    monkeypatch.setattr(round2, "verify_manifest", lambda _path: (True, []))
    output = tmp_path / "pruned.csv"
    result = round2.prune_fresh_overlaps(catalog, preliminary, fresh, output)
    with output.open(newline="", encoding="utf-8") as stream:
        kept = list(csv.DictReader(stream))
    assert [row["video_id"] for row in kept] == ["round2_sana600m:keep"]
    assert result["removed_rows"] == 1
    assert result["collision_rows"] == {"dhash64": 1}


def test_merge_rejects_a_destination_collision(tmp_path):
    row = _row(
        domain="ff", method="", family=EFS, split="train", video="001", frame="000.png",
        entities=("ff:001",), archive="real/train.zip", member="ff/001/000.png",
        preprocess="copy", identity="ff:001",
    )
    base, additions = tmp_path / "base.csv", tmp_path / "additions.csv"
    _write_catalog(base, [row])
    _write_catalog(additions, [row])
    try:
        round2.merge_catalogs(base, additions, tmp_path / "merged.csv")
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("a duplicate destination must fail")
