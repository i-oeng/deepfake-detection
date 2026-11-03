import io
import zipfile

import pytest

from scripts.download_sfhq_t2i_round4 import (
    FFHQ_USED,
    choose_members,
    choose_shards,
    model_of,
    rank,
    read_member,
)


def test_choose_members_takes_a_seeded_quota_per_generator_minus_validation() -> None:
    names = [f"images/SDXL_image_{i:07d}.jpg" for i in range(10)]
    names += [f"images/DALLE3_image_{i:07d}.jpg" for i in range(2)] + ["images/"]
    chosen = choose_members(names, excluded={"SDXL_image_0000003.jpg"}, quota=4)
    sdxl = [n for n in chosen if model_of(n) == "SDXL"]
    eligible = [n for n in names[:10] if not n.endswith("0000003.jpg")]
    assert sdxl == sorted(sorted(eligible, key=rank)[:4])
    assert [n for n in chosen if model_of(n) == "DALLE3"] == names[10:12]
    assert choose_members(list(reversed(names)), {"SDXL_image_0000003.jpg"}, 4) == chosen


def test_choose_shards_avoids_shards_other_splits_use() -> None:
    shards = choose_shards(9)
    assert len(shards) == 9 and not set(shards) & FFHQ_USED
    assert all(s.endswith("000.tar") for s in shards)


@pytest.mark.parametrize("compression", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
def test_read_member_returns_checked_bytes(compression) -> None:
    buffer = io.BytesIO()
    payloads = {"images/a_image_1.jpg": b"x" * 5000, "images/b_image_2.jpg": bytes(range(256))}
    with zipfile.ZipFile(buffer, "w", compression) as archive:
        for name, data in payloads.items():
            archive.writestr(name, data)
    with zipfile.ZipFile(buffer) as archive:
        infos = archive.infolist()
    for info in infos:
        assert read_member(buffer, info) == payloads[info.filename]
    broken = infos[0]
    broken.CRC ^= 1
    with pytest.raises(ValueError, match="CRC"):
        read_member(buffer, broken)
