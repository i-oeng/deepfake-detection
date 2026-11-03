import pytest

from scripts.download_openimages_round4 import FACE_LABEL, choose_images, face_shares, rank


def _box(image: str, x: tuple, y: tuple, label: str = FACE_LABEL, group: str = "0",
         depiction: str = "0") -> dict:
    return {"ImageID": image, "LabelName": label, "XMin": str(x[0]), "XMax": str(x[1]),
            "YMin": str(y[0]), "YMax": str(y[1]), "IsGroupOf": group, "IsDepiction": depiction}


def _image(image: str, rotation: str = "0.0") -> dict:
    return {"ImageID": image, "Subset": "test", "Rotation": rotation, "Author": "a",
            "License": "https://creativecommons.org/licenses/by/2.0/", "OriginalURL": "u"}


def test_face_shares_take_the_largest_real_face_and_flag_depictions() -> None:
    largest, depicted = face_shares([
        _box("a", (0.1, 0.2), (0.1, 0.4)),  # shorter side 0.1
        _box("a", (0.1, 0.4), (0.1, 0.3)),  # shorter side 0.2: the largest
        _box("a", (0.0, 0.9), (0.0, 0.9), group="1"),  # a group box is no face
        _box("a", (0.0, 0.9), (0.0, 0.9), label="/m/01g317"),  # a person box
        _box("b", (0.0, 0.5), (0.0, 0.5), depiction="1"),
    ])
    assert largest == {"a": pytest.approx(0.2)} and depicted == {"b"}


def test_choose_images_keeps_upright_large_real_faces_in_seeded_order() -> None:
    largest = {"a": 0.2, "b": 0.07, "c": 0.05, "d": 0.3, "e": 0.3, "f": 0.1}
    images = [_image("a"), _image("b", ""), _image("c"), _image("d"), _image("e", "90.0"),
              _image("f"), _image("g")]
    rows = choose_images(images, largest, depicted={"f"})
    assert [row["image_id"] for row in rows] == sorted("abd", key=rank)
    assert rows[0]["url"].endswith(f"/test/{rows[0]['image_id']}.jpg")
    assert choose_images(list(reversed(images)), largest, {"f"}) == rows
