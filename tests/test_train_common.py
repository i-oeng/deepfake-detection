import json
from collections import defaultdict

import pytest

from deepfake_detection.train.common import (
    create_run_directory,
    run_id,
    sampling_weights,
    split_rows,
)


def _row(label: str, method: str, video: str, split: str = "train") -> dict[str, str]:
    return {"label": label, "fake_method": method, "video_id": video, "split": split}


def test_sampling_weights_balance_labels_methods_and_videos():
    rows = [
        _row("REAL", "", "r1"), _row("REAL", "", "r1"), _row("REAL", "", "r2"),
        _row("FAKE", "a", "f1"), _row("FAKE", "a", "f1"), _row("FAKE", "a", "f1"),
        _row("FAKE", "b", "f2"), _row("FAKE", "b", "f3"),
    ]
    weights = sampling_weights(rows)

    by_label: dict[str, float] = defaultdict(float)
    by_method: dict[str, float] = defaultdict(float)
    by_video: dict[str, float] = defaultdict(float)
    for row, weight in zip(rows, weights, strict=True):
        by_label[row["label"]] += weight
        by_method[row["fake_method"]] += weight
        by_video[row["video_id"]] += weight
    assert by_label == pytest.approx({"REAL": 0.5, "FAKE": 0.5})
    assert by_method["a"] == pytest.approx(by_method["b"])
    assert by_video["r1"] == pytest.approx(by_video["r2"])
    assert by_video["f2"] == pytest.approx(by_video["f3"])


def test_sampling_weights_require_fake_rows():
    with pytest.raises(ValueError, match="no fake methods"):
        sampling_weights([_row("REAL", "", "r1")])


def test_split_rows_groups_by_split_in_fixed_order():
    rows = [_row("REAL", "", "v", split) for split in ("test", "train", "validation", "train")]
    splits = split_rows(rows)
    assert list(splits) == ["train", "validation", "test"]
    assert [len(part) for part in splits.values()] == [2, 1, 1]


def test_run_id_is_deterministic_and_input_sensitive():
    config = {"seed": 1, "lr_head": 0.001}
    base = run_id(config, "abc", "sha")
    assert base == run_id(dict(reversed(config.items())), "abc", "sha")
    assert len(base) == 20
    assert base != run_id({**config, "seed": 2}, "abc", "sha")
    assert base != run_id(config, "abd", "sha")
    assert base != run_id(config, "abc", "shb")


def test_create_run_directory_writes_config_and_refuses_reuse(tmp_path):
    config = {"output_root": "out", "seed": 1}
    directory = create_run_directory(config, tmp_path, "run1", default_output_root="unused")
    assert directory == tmp_path / "out" / "run1"
    assert json.loads((directory / "config.json").read_text()) == config
    with pytest.raises(FileExistsError):
        create_run_directory(config, tmp_path, "run1", default_output_root="unused")
