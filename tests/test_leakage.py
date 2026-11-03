import numpy as np
from PIL import Image

from deepfake_detection.data import leakage


def _row(file: str = "", pixel: str = "", dhash: str = "") -> dict[str, str]:
    return {"file_sha256": file, "pixel_sha256": pixel, "dhash64": dhash}


def test_find_leaks_reports_exact_fingerprints_first_then_near_copies(tmp_path, monkeypatch):
    monkeypatch.setattr(leakage, "thumbnails", lambda paths, workers=8: np.stack(
        [leakage.thumbnail(path) for path in paths]))
    rng = np.random.default_rng(0)
    for name in ("a", "b", "c"):
        pixels = rng.integers(0, 256, (16, 16), dtype=np.uint8).repeat(16, 0).repeat(16, 1)
        Image.fromarray(pixels).convert("RGB").save(tmp_path / f"{name}.png")
    with Image.open(tmp_path / "a.png") as image:  # a resized, re-encoded copy of a
        image.resize((200, 200)).save(tmp_path / "a_copy.jpg", quality=80)
    references = [_row("f-a", "p-a", "d-a"), _row("f-b", "p-b", "")]
    reference_paths = [str(tmp_path / "a.png"), str(tmp_path / "b.png")]
    queries = [_row("f-x", "p-x", "d-a"), _row("f-y", "p-y", "d-y"), _row("f-z", "", "")]
    query_paths = [str(tmp_path / name) for name in ("c.png", "a_copy.jpg", "c.png")]
    reasons, correlation = leakage.find_leaks(queries, query_paths, references,
                                              reference_paths)
    # An empty fingerprint never matches, even where a reference value is empty too.
    assert reasons == ["dhash64", "thumbnail_correlation", ""]
    assert correlation[1] >= leakage.NEAR_DUPLICATE_CORRELATION > correlation[0]
    assert leakage.histogram(correlation)[">=0.99"] == 1
