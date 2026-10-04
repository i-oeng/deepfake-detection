from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def load_normalizer():
    script = Path(__file__).parents[1] / "scripts/normalize_df40.py"
    spec = importlib.util.spec_from_file_location("normalize_df40", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def protocol(fake_path: str, fake_split: str, real_groups: dict[str, dict]) -> dict:
    return {
        "protocol": {
            "method_fake": {
                fake_split: {"fake-group": {"frames": [fake_path]}},
            },
            "source_real": real_groups,
        }
    }


def test_normalize_is_deterministic_and_excludes_unavailable_paths(tmp_path: Path) -> None:
    normalizer = load_normalizer()
    ff_real_groups = {
        split: {
            f"real-{index}": {"frames": [f"ff/real/{index:03d}.jpg"]}
            for index in range(start, stop)
        }
        for split, start, stop in (("train", 0, 4), ("val", 4, 6), ("test", 6, 8))
    }

    for method, (filename, _family, domain, allowed_splits) in normalizer.METHODS.items():
        fake_split = sorted(allowed_splits)[0]
        fake_path = f"{domain}/{method}/fake.jpg"
        real_groups = ff_real_groups if method == "simswap" else {}
        if method == "starganv2":
            real_groups = {"test": {"celeba-real": {"frames": ["celeba/real/001.jpg"]}}}
        (tmp_path / filename).write_text(
            json.dumps(protocol(fake_path, fake_split, real_groups)),
            encoding="utf-8",
        )

    excluded = {"ff/simswap/fake.jpg"}
    first = normalizer.normalize(tmp_path, excluded)
    second = normalizer.normalize(tmp_path, excluded)

    assert first == second
    assert excluded.isdisjoint({row["relative_path"] for row in first})
    real_ff_splits = {
        row["split"]
        for row in first
        if row["source_domain"] == "ff" and row["label"] == "REAL"
    }
    assert real_ff_splits == {
        "train",
        "validation",
        "test",
    }
    assert len({row["image_id"] for row in first}) == len(first)
    assert len({row["relative_path"] for row in first}) == len(first)
