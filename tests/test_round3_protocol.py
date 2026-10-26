import csv
import hashlib
import json

import pytest

from deepfake_detection.data import round3
from deepfake_detection.data.openworld import COLUMNS, EFS, _row


def _write_catalog(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _read(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _fairface_metadata(path, ages):
    path.write_text("".join(
        json.dumps({"index": index, "split": "train", "file": f"train/{index:06d}.jpg",
                    "age": age, "gender": 0, "race": index % 7}) + "\n"
        for index, age in enumerate(ages)))


def test_selection_keeps_adults_in_disjoint_pools(tmp_path, monkeypatch):
    monkeypatch.setattr(round3, "POOLS", {"kandinsky22": 2, "recon_movq": 1})
    metadata = tmp_path / "metadata.jsonl"
    _fairface_metadata(metadata, [0, 1, 2, 3, 4, 5, 8])
    output = tmp_path / "selection.json"
    result = round3.select_fairface(metadata, output)
    spec = json.loads(output.read_text())
    chosen = [entry for entries in spec["pools"].values() for entry in entries]
    assert result["eligible"] == 4
    assert len({entry["id"] for entry in chosen}) == 3
    assert all(entry["age"] >= 3 for entry in chosen)
    # Selection is reproducible and refuses to change silently.
    assert round3.select_fairface(metadata, output) == result
    with pytest.raises(RuntimeError):
        round3.select_fairface(metadata, output, seed="other")


def _t2i(root, generator):
    folder = root / generator
    folder.mkdir(parents=True)
    (folder / "00000.png").write_bytes(b"generated")
    (folder / "provenance.json").write_text(json.dumps({
        "generator": generator, "count": 1, "seed": round3.ROUND2_SEED}))
    (folder / "metadata.jsonl").write_text(json.dumps({
        "index": 0, "generator": generator, "seed": round3.ROUND2_SEED,
        "prompt": "a portrait"}) + "\n")


def _reconstruction(root, generator, entry, selection_sha256):
    folder = root / generator
    output = entry["file"].replace(".jpg", ".png")
    (folder / output).parent.mkdir(parents=True)
    (folder / output).write_bytes(b"reconstructed")
    (folder / "provenance.json").write_text(json.dumps({
        "autoencoder": generator.removeprefix("recon_"), "count": 1, "pool": generator,
        "selection_sha256": selection_sha256}))
    (folder / "metadata.jsonl").write_text(json.dumps({
        "output": output, "source_id": entry["id"], "source_file": entry["file"]}) + "\n")


def test_catalog_pairs_every_round3_collection_with_fairface_reals(tmp_path, monkeypatch):
    pools = {name: 1 for name in ("kandinsky22", "sana600m", "recon_movq", "recon_dcae")}
    monkeypatch.setattr(round3, "POOLS", pools)
    monkeypatch.setattr(round3, "T2I_COUNTS", {"kandinsky22": 1, "sana600m": 1})
    monkeypatch.setattr(round3, "RECON_COUNTS", {"recon_movq": 1, "recon_dcae": 1})
    metadata = tmp_path / "metadata.jsonl"
    _fairface_metadata(metadata, [3, 4, 5, 6])
    selection = tmp_path / "selection.json"
    round3.select_fairface(metadata, selection)
    digest = hashlib.sha256(selection.read_bytes()).hexdigest()
    spec = json.loads(selection.read_text())
    round2_generated, round3_generated = tmp_path / "round2", tmp_path / "round3"
    for generator in ("kandinsky22", "sana600m"):
        _t2i(round2_generated, generator)
    for generator in ("recon_movq", "recon_dcae"):
        _reconstruction(round3_generated, generator, spec["pools"][generator][0], digest)

    output = tmp_path / "additions.csv"
    result = round3.build_additions_catalog(selection, round2_generated, round3_generated,
                                            output)
    rows = _read(output)
    assert result["rows"] == 8
    for domain in {row["source_domain"] for row in rows}:
        reals = [row for row in rows if row["source_domain"] == domain
                 and row["label"] == "REAL"]
        assert len(reals) == 1
        assert reals[0]["archive"] == round3.FAIRFACE_ARCHIVE
        assert reals[0]["lineage_video_ids"].startswith("fairface:")
        assert reals[0]["preprocess"] == f"canonical:{domain}"
    recon = [row for row in rows if row["source_domain"] == "round3_recon_movq"]
    assert {row["lineage_video_ids"] for row in recon} == {
        f"fairface:{spec['pools']['recon_movq'][0]['id']}"}


def test_combine_drops_only_the_replaced_round2_pairings(tmp_path):
    def row(domain, method, video):
        return _row(domain=domain, method=method, family=EFS, split="train", video=video,
                    frame="0.png", entities=("ff:001",), archive="dir:x", member=f"{video}.png",
                    preprocess=f"canonical:{domain}")

    round2_rows = [row("round2_kandinsky22", "kandinsky22", "fake-0"),
                   row("round2_kandinsky22", "", "real-0"),
                   row("round2_recon_movq", "recon_movq", "fake-1"),
                   row("round2_recon_movq", "", "real-1")]
    round3_rows = [row("round3_kandinsky22", "kandinsky22", "fake-2")]
    first, second = tmp_path / "round2.csv", tmp_path / "round3.csv"
    _write_catalog(first, round2_rows)
    _write_catalog(second, round3_rows)
    output = tmp_path / "combined.csv"
    result = round3.combine_additions(first, second, output)
    assert result["round2_rows_replaced"] == 2
    assert {row["source_domain"] for row in _read(output)} == {
        "round2_recon_movq", "round3_kandinsky22"}


def test_nearest_distances_counts_differing_bits():
    distances = round3.nearest_distances(["ffff000000000000", "0000000000000000"],
                                         ["ffff000000000001", "00000000000000ff"])
    assert distances.tolist() == [1, 8]
    assert round3.nearest_distances(["0000000000000000"], []).tolist() == [64]
