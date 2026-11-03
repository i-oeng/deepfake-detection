import io
import random

import pytest
from PIL import Image, JpegImagePlugin

from deepfake_detection.data.format_match import (
    Source,
    allocate,
    assign,
    encode_jpeg,
    read_encodings,
    write_encodings,
)


def _source(image_id: str, label: str, fmt: str = "PNG", quality: int | None = None,
            subsampling: str | None = None, collection: str = "openfake") -> Source:
    return Source(image_id, collection, label, f"sha-{image_id}", fmt, quality, subsampling)


def test_allocate_splits_by_the_largest_remainder() -> None:
    assert allocate({"a": 1, "b": 1, "c": 1}, 4) == {"a": 2, "b": 1, "c": 1}
    assert allocate({"a": 0.7, "b": 0.3}, 3) == {"a": 2, "b": 1}
    assert allocate({"a": 1}, 0) == {}
    with pytest.raises(ValueError):
        allocate({}, 2)


def test_the_other_label_takes_the_complement_of_its_own_jpeg_sources() -> None:
    sources = [_source(f"r{i}", "REAL", "JPEG", 75, "4:2:0") for i in range(6)]
    sources += [_source(f"r{i}", "REAL", "JPEG", 95, "4:4:4") for i in range(6, 8)]
    sources += [_source("r8", "REAL", "WEBP")]  # the reference label draws from the reference
    sources += [_source(f"f{i}", "FAKE", "JPEG", 95, "4:4:4") for i in range(2)]
    sources += [_source(f"f{i}", "FAKE", "PNG") for i in range(2, 8)]
    assignments, summary = assign(sources)
    assert {assignments[f"f{i}"] for i in range(2, 8)} == {(75, "4:2:0")}
    assert assignments["r8"] in {(75, "4:2:0"), (95, "4:4:4")}
    assert not {f"r{i}" for i in range(8)} & set(assignments)  # JPEG sources stay as they are
    record = summary["openfake"]
    assert record["reference"] == "REAL" and record["encoded"] == {"FAKE": 6, "REAL": 1}
    assert record["pair_distance"] < 0.06
    shuffled = sources[:]
    random.Random(3).shuffle(shuffled)
    assert assign(shuffled) == (assignments, summary)


def test_mirror_case_and_collections_without_jpeg() -> None:
    sources = [_source(f"s{i}", "FAKE", "JPEG", 90, "4:2:0", "sfhq") for i in range(3)]
    sources += [_source(f"h{i}", "REAL", "WEBP", collection="sfhq") for i in range(4)]
    sources += [_source(f"g{i}", label, "PNG", collection="generated")
                for i, label in enumerate(("REAL", "FAKE"))]
    assignments, summary = assign(sources)
    assert assignments == {f"h{i}": (90, "4:2:0") for i in range(4)}
    assert summary["sfhq"]["reference"] == "FAKE" and summary["sfhq"]["pair_distance"] == 0
    assert summary["generated"]["reference"] is None
    with pytest.raises(ValueError, match="unique"):
        assign(sources + sources[:1])


def test_encode_jpeg_and_the_encodings_file_round_trip(tmp_path) -> None:
    from scripts.download_openfake_round4 import jpeg_quality

    buffer = io.BytesIO()
    Image.new("RGBA", (40, 24), (200, 30, 30, 128)).save(buffer, format="PNG")
    encoded = Image.open(io.BytesIO(encode_jpeg(buffer.getvalue(), (75, "4:2:0"))))
    assert encoded.format == "JPEG" and encoded.size == (40, 24)
    assert jpeg_quality(encoded.quantization[0]) == 75
    assert JpegImagePlugin.get_sampling(encoded) == 2  # 4:2:0
    digest = write_encodings(tmp_path / "e.csv", {"b": (90, "4:4:4"), "a": (75, "4:2:0")})
    assert len(digest) == 64
    assert read_encodings(tmp_path / "e.csv") == {"a": (75, "4:2:0"), "b": (90, "4:4:4")}
    (tmp_path / "bad.csv").write_text("image_id,jpeg_quality,subsampling\nx,0,4:2:0\n")
    with pytest.raises(ValueError, match="invalid"):
        read_encodings(tmp_path / "bad.csv")


def test_materialize_reads_an_assigned_source_as_jpeg(tmp_path) -> None:
    from deepfake_detection.data.openworld import _source

    (tmp_path / "src").mkdir()
    Image.new("RGB", (16, 16), (10, 120, 200)).save(tmp_path / "src" / "a.png")
    assert _source(str(tmp_path), "dir:src", "a.png", None)[:4] == b"\x89PNG"
    assert _source(str(tmp_path), "dir:src", "a.png", (80, "4:4:4"))[:2] == b"\xff\xd8"
