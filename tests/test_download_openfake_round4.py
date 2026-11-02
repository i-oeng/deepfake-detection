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
