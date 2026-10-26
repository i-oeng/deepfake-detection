import csv
import json

import pytest
from PIL import Image

from scripts import generate_round2 as round2


def _image(path, color=(10, 20, 30)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)


def _manifest(path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_prompts_and_torch_generators_are_deterministic() -> None:
    assert round2.prompt("seed", 4) == round2.prompt("seed", 4)
    assert round2.prompt("seed", 4) != round2.prompt("seed", 5)
    pytest.importorskip("torch")
    assert round2._generator("seed", "model", 4).initial_seed() == round2._generator(
        "seed", "model", 4
    ).initial_seed()


def test_decode_stripes_cover_the_last_latent_row() -> None:
    assert round2._tile_starts(96, tile_size=48, overlap=16) == [0, 32, 48]
    assert round2._tile_starts(40, tile_size=48, overlap=16) == [0]
    with pytest.raises(ValueError, match="overlap"):
        round2._tile_starts(96, tile_size=48, overlap=48)


def test_output_provenance_is_immutable_and_precedes_payloads(tmp_path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    expected = {"schema_version": 1, "generator": "example", "count": 2}
    round2._ensure_provenance(output, expected)
    round2._ensure_provenance(output, expected)
    assert json.loads((output / "provenance.json").read_text(encoding="utf-8")) == expected

    with pytest.raises(ValueError, match="provenance differs"):
        round2._ensure_provenance(output, {**expected, "count": 3})

    orphaned = tmp_path / "orphaned"
    orphaned.mkdir()
    (orphaned / "00000.png").write_bytes(b"payload")
    with pytest.raises(ValueError, match="payloads but no provenance"):
        round2._ensure_provenance(orphaned, expected)


def test_t2i_metadata_recovers_after_image_write_and_deduplicates(tmp_path) -> None:
    output = tmp_path / "t2i"
    output.mkdir()
    _image(output / "00000.png")
    settings = {"revision": "abc", "size": 512}

    records = round2._sync_t2i_metadata(output, "generator", "seed", 2, settings)
    expected = round2._t2i_record("generator", "seed", 0, settings)
    assert records == {0: expected}

    metadata = output / "metadata.jsonl"
    metadata.write_text(
        json.dumps(expected) + "\n" + json.dumps(expected) + "\n", encoding="utf-8"
    )
    round2._sync_t2i_metadata(output, "generator", "seed", 2, settings)
    assert len(metadata.read_text(encoding="utf-8").splitlines()) == 1

    metadata.write_text(json.dumps({**expected, "revision": "changed"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="configuration differs"):
        round2._sync_t2i_metadata(output, "generator", "seed", 2, settings)


def test_training_real_selection_is_exact_safe_and_reproducible(tmp_path) -> None:
    data_root = tmp_path / "data"
    rows = [
        {"split": "train", "label": "REAL", "sample_id": "real-a", "relative_path": "a.png"},
        {"split": "train", "label": "REAL", "sample_id": "real-b", "relative_path": "b.png"},
        {"split": "train", "label": "FAKE", "sample_id": "fake", "relative_path": "f.png"},
        {"split": "validation", "label": "REAL", "sample_id": "dev", "relative_path": "v.png"},
    ]
    for row in rows:
        _image(data_root / row["relative_path"])
    manifest = tmp_path / "manifest.csv"
    _manifest(manifest, rows)

    first = round2._train_reals(manifest, data_root, 2, "seed")
    second = round2._train_reals(manifest, data_root, 2, "seed")
    assert [row["sample_id"] for row in first] == [row["sample_id"] for row in second]
    assert {row["sample_id"] for row in first} == {"real-a", "real-b"}
    with pytest.raises(ValueError, match="requested 3 training reals"):
        round2._train_reals(manifest, data_root, 3, "seed")


def test_reconstruction_metadata_recovers_from_an_existing_image(tmp_path) -> None:
    output = tmp_path / "recon"
    output.mkdir()
    row = {
        "sample_id": "real-a",
        "relative_path": "ff/video/frame.jpg",
        "_path": str(tmp_path / "source.jpg"),
    }
    _image(output / "ff" / "video" / "frame.png")
    records = round2._sync_recon_metadata(
        output, "movq", "seed", [row], "revision", "model"
    )
    assert list(records) == ["ff/video/frame.png"]
    assert records["ff/video/frame.png"]["source_sample_id"] == "real-a"


def test_cli_rejects_nonpositive_work_sizes(tmp_path) -> None:
    with pytest.raises(SystemExit):
        round2.main(
            [
                "t2i",
                "--generator",
                "sana600m",
                "--count",
                "0",
                "--output",
                str(tmp_path),
            ]
        )
