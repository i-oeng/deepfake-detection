import pandas as pd

from scripts.download_openfake_round4 import choose_rows, crop_box, rank


def _rows(key, hashes):
    return pd.DataFrame({"key": [key] * len(hashes), "image_sha256": hashes,
                         "file": "f", "row_group": 0, "position": range(len(hashes))})


def test_choose_rows_takes_quota_in_seeded_rank_order() -> None:
    fake, rare = ("fake", "flux1"), ("fake", "kolors")
    candidates = pd.concat([_rows(fake, [f"h{i}" for i in range(10)]), _rows(rare, ["r0", "r1"])])
    chosen = choose_rows(candidates, {fake: 3})
    picked = chosen[chosen["key"] == fake]["image_sha256"].tolist()
    assert picked == sorted([f"h{i}" for i in range(10)], key=rank)[:3]
    assert sorted(chosen[chosen["key"] == rare]["image_sha256"]) == ["r0", "r1"]
    shuffled = candidates.sample(frac=1.0, random_state=7)
    again = choose_rows(shuffled, {fake: 3})
    assert sorted(again["image_sha256"]) == sorted(chosen["image_sha256"])


def test_crop_box_is_a_clipped_square_around_the_face() -> None:
    assert crop_box((40, 40, 20, 20), 200, 200) == (25, 25, 75, 75)
    assert crop_box((0, 0, 40, 20), 100, 100) == (0, 0, 70, 60)


def test_pixel_budget_holds_back_images_beyond_its_capacity() -> None:
    import threading
    import time

    from scripts.download_openfake_round4 import PixelBudget

    budget, order = PixelBudget(100.0), []
    first_inside, release_first = threading.Event(), threading.Event()

    def large() -> None:
        with budget.hold(80.0):
            order.append("large in")
            first_inside.set()
            release_first.wait(5)
            order.append("large out")

    def second() -> None:
        with budget.hold(50.0):
            order.append("second in")

    a = threading.Thread(target=large)
    a.start()
    first_inside.wait(5)
    b = threading.Thread(target=second)
    b.start()
    time.sleep(0.2)
    assert order == ["large in"]  # 80 + 50 exceeds 100, so the second image waits
    release_first.set()
    a.join(5)
    b.join(5)
    assert order == ["large in", "large out", "second in"]
    with budget.hold(250.0):  # larger than the whole budget: runs once nothing else holds it
        pass


def test_source_encoding_recovers_libjpeg_quality_and_subsampling() -> None:
    import io

    from PIL import Image

    from scripts.download_openfake_round4 import source_encoding

    pixels = Image.radial_gradient("L").convert("RGB")
    for quality, sampling, name in ((35, 2, "4:2:0"), (75, 2, "4:2:0"), (90, 0, "4:4:4"),
                                    (95, 1, "4:2:2")):
        buffer = io.BytesIO()
        pixels.save(buffer, "JPEG", quality=quality, subsampling=sampling)
        with Image.open(buffer) as image:
            assert source_encoding(image) == {"jpeg_quality": quality, "subsampling": name}
    buffer = io.BytesIO()
    pixels.save(buffer, "PNG")
    with Image.open(buffer) as image:
        assert source_encoding(image) == {}


class _Detector:
    def __init__(self, faces: list) -> None:
        self.faces, self.seen = faces, None

    def detect(self, rgb):
        self.seen = rgb.shape
        return self.faces


def test_locate_face_detects_on_a_320_px_copy_and_measures_in_source_pixels() -> None:
    import numpy as np

    from deepfake_detection.inference.face import Face
    from scripts.download_openfake_round4 import locate_face

    rgb = np.zeros((1000, 800, 3), dtype=np.uint8)  # detected at scale 0.32
    points = np.full((5, 2), 16.0)
    detector = _Detector([Face((32.0, 16.0, 32.0, 24.0), 0.9, points)])
    status, info = locate_face(rgb, detector)
    assert detector.seen == (320, 256, 3)
    assert status == "found"
    assert info["face_box"] == [100.0, 50.0, 100.0, 75.0]
    assert info["landmarks"][0] == [50.0, 50.0]
    small = Face((0.0, 0.0, 16.0, 16.0), 0.9, points)  # 50 px in the source: too small
    assert locate_face(rgb, _Detector([small]))[0] == "face_too_small"
    status, info = locate_face(rgb, _Detector([Face((0.0, 0.0, 40.0, 40.0), 0.6, points)]))
    assert (status, info["best_score"]) == ("low_confidence", 0.6)
    assert locate_face(rgb, _Detector([]))[0] == "no_face"
    detector = _Detector([])
    assert locate_face(np.zeros((200, 300, 3), dtype=np.uint8), detector)[1][
        "detection_scale"] == 1.0
    assert detector.seen == (200, 300, 3)  # already small enough: detected as is


def test_test_rows_keep_photo_generators_and_reals_keyed_by_model() -> None:
    from scripts.download_openfake_round4 import test_rows

    rows = pd.DataFrame({
        "split": ["core/test"] * 5 + ["core/train"],
        "person": [True, True, False, False, True, True],
        "label": ["fake", "fake", "real", "fake", "fake", "fake"],
        "model": ["z-image-turbo", "sora-2", "docci", "gpt-image-2", "veo-3", "flux.1-dev"],
    })
    # reals need no person prompt; fakes need one and must be photo generators
    assert test_rows(rows)["key"].tolist() == [("fake", "z-image-turbo"), ("real", "docci")]
