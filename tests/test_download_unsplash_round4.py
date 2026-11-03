import io

import pytest

from scripts.download_unsplash_round4 import BATCH, HOST, choose_photos, preview_check, rank


def _photo(photo_id: str, submitted: str = "2019-03-28 10:00:00", url: str | None = None,
           width: str = "4000", height: str = "6000") -> dict:
    return {"photo_id": photo_id, "photo_image_url": url or f"{HOST}photo-{photo_id}",
            "photo_submitted_at": submitted, "photo_width": width, "photo_height": height,
            "photographer_username": f"by-{photo_id}"}


def test_choose_photos_keeps_pre_cutoff_cdn_photos_in_seeded_order() -> None:
    photos = [_photo(f"p{i}") for i in range(BATCH + 3)]
    photos += [
        _photo("late", submitted="2022-08-01 00:00:00"),
        _photo("odd", url="https://images.unsplash.company"),
        _photo("bare", url=HOST),
        _photo("query", url=f"{HOST}photo-x?w=10"),
        _photo("nosize", width=""),
        _photo("p0"),  # a duplicate id keeps its first row
    ]
    rows = choose_photos(photos)
    assert [row["photo_id"] for row in rows] == sorted((f"p{i}" for i in range(BATCH + 3)),
                                                       key=rank)
    assert [row["unit"] for row in rows].count("unsplash_0000") == BATCH
    assert rows[-1]["unit"] == "unsplash_0001"
    assert rows[0]["width"] == 4000 and rows[0]["label"] == "real"
    assert choose_photos(list(reversed(photos[:BATCH + 3]))) == rows


class _Detector:
    def __init__(self, faces: list) -> None:
        self.faces, self.seen = faces, None

    def detect(self, rgb):
        self.seen = rgb.shape
        return self.faces


def test_preview_check_measures_faces_in_source_pixels() -> None:
    import numpy as np
    from PIL import Image

    from deepfake_detection.inference.face import Face

    buffer = io.BytesIO()
    Image.new("RGB", (213, 320)).save(buffer, format="JPEG")
    preview, points = buffer.getvalue(), np.zeros((5, 2))
    # The source is 4000 x 6000, so one preview pixel is 18.75 source pixels.
    detector = _Detector([Face((10.0, 10.0, 3.0, 3.0), 0.9, points)])  # 56 source px
    assert preview_check(preview, 4000, 6000, detector) == {
        "preview_faces": 1, "preview_best_score": 0.9, "preview_pass": True}
    assert detector.seen == (320, 213, 3)
    tiny = Face((10.0, 10.0, 2.0, 2.0), 0.9, points)  # 37.5 source px
    unsure = Face((10.0, 10.0, 30.0, 30.0), 0.55, points)
    result = preview_check(preview, 4000, 6000, _Detector([unsure, tiny]))
    assert (result["preview_pass"], result["preview_best_score"]) == (False, 0.9)
    assert not preview_check(preview, 4000, 6000, _Detector([]))["preview_pass"]


def test_fetch_retries_then_raises_and_reports_missing_photos() -> None:
    httpx = pytest.importorskip("httpx")
    from scripts.download_unsplash_round4 import fetch

    replies = {"/gone": [404], "/flaky": [503, 200], "/down": [500] * 3}

    def handler(request):
        status = replies[request.url.path].pop(0)
        return httpx.Response(status, content=b"jpeg" if status == 200 else b"")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert fetch(client, "https://cdn.test/gone", pause=0) is None
        assert fetch(client, "https://cdn.test/flaky", pause=0) == b"jpeg"
        with pytest.raises(httpx.HTTPStatusError):
            fetch(client, "https://cdn.test/down", retries=3, pause=0)
