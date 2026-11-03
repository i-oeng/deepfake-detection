import json
import re

import pytest
from PIL import Image

import scripts.generate_round4 as round4
from scripts.generate_round2 import prompt as round2_prompt


def test_prompts_are_deterministic_adult_and_new() -> None:
    prompts = [round4.prompt(index) for index in range(500)]
    assert prompts == [round4.prompt(index) for index in range(500)]
    assert all(any(age in text for age in round4.AGES) for text in prompts)
    assert not any(re.search(r"\b(child|kid|teen|boy|girl)\b", text) for text in prompts)
    assert len(set(prompts)) == len(prompts)
    assert not set(prompts) & {round2_prompt("round2-20261008", index) for index in range(500)}


def test_sizes_are_photographic_buckets_near_one_megapixel() -> None:
    sizes = {round4.image_size(index) for index in range(200)}
    assert sizes == set(round4.SIZES)
    assert all(w % 64 == 0 and h % 64 == 0 and abs(w * h / 1024**2 - 1) < 0.06
               for w, h in sizes)


def test_records_pin_settings_and_noise_differs_by_generator() -> None:
    first = round4.record("sdxl_base", 7)
    assert first["revision"] == round4.GENERATORS["sdxl_base"]["revision"]
    assert (first["prompt"], first["width"]) == (round4.record("sd35_medium", 7)["prompt"],
                                                 round4.record("sd35_medium", 7)["width"])
    assert first["noise_seed"] != round4.record("realvisxl_v5", 7)["noise_seed"]
    assert json.loads(json.dumps(first)) == first
    assert all(re.fullmatch(r"[0-9a-f]{40}", settings["revision"])
               for settings in round4.GENERATORS.values())


def test_metadata_sync_and_pending_resume_a_partial_run(tmp_path) -> None:
    Image.new("RGB", (8, 8)).save(tmp_path / "00002.png")
    records = round4.sync_metadata(tmp_path, "sdxl_base")
    assert list(records) == [2] and records[2] == round4.record("sdxl_base", 2)
    assert round4.pending(tmp_path, 5, None) == [0, 1, 3, 4]
    assert round4.pending(tmp_path, 5, 2) == [0, 1]
    with pytest.raises(ValueError, match="contract"):
        round4.sync_metadata(tmp_path, "realvisxl_v5")


def test_a_busy_gpu_stops_the_run_before_any_model_loads(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(round4, "_gpu_free", lambda: int(2e9))

    def no_load(*args):
        raise AssertionError("a model was loaded")

    monkeypatch.setattr(round4, "_sdxl", no_load)
    assert round4.generate("sdxl_base", tmp_path / "out", 3, None) == 3
    assert json.loads((tmp_path / "out" / "provenance.json").read_text())["seed"] == round4.SEED
