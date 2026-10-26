import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("cv2")

from deepfake_detection.train.openworld import (  # noqa: E402
    SBI_METHOD,
    alignment_uniformity,
    open_world_report,
    sampling_plan,
    source_fingerprint,
    with_pseudo_fakes,
)
from deepfake_detection.train.sbi import face_mask, self_blend  # noqa: E402


def _rows():
    rows = []
    for method in ("a", "b"):
        for video in range(3):
            rows.append({"label": "FAKE", "fake_method": method, "video_id": f"{method}{video}"})
    for video in range(4):
        rows.append({"label": "REAL", "fake_method": "", "video_id": f"r{video}"})
    return rows


def test_pseudo_fakes_receive_the_configured_share_of_fake_mass():
    rows, weights = with_pseudo_fakes(_rows(), 0.4)
    assert sum(row["fake_method"] == SBI_METHOD for row in rows) == 4
    mass = {}
    for row, weight in zip(rows, weights, strict=True):
        key = row["fake_method"] or "real"
        mass[key] = mass.get(key, 0.0) + weight
    assert mass["real"] == pytest.approx(0.5)
    assert mass[SBI_METHOD] == pytest.approx(0.2)
    assert mass["a"] == pytest.approx(0.15) and mass["b"] == pytest.approx(0.15)
    plain, plain_weights = with_pseudo_fakes(_rows(), 0.0)
    assert len(plain) == 10 and sum(plain_weights) == pytest.approx(1.0)


def test_self_blend_changes_only_the_face_region():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
    for seed in range(5):
        blended = self_blend(image, np.random.default_rng(seed))
        assert blended.shape == image.shape and blended.dtype == np.uint8
        changed = np.abs(blended.astype(int) - image.astype(int)).sum(axis=2) > 0
        # Corners lie outside every face mask.
        assert not changed[:16, :16].any() and not changed[-16:, -16:].any()
    mask = face_mask(256, np.random.default_rng(1))
    assert mask.max() == pytest.approx(1.0) and mask[128, 128] > 0.5


def test_alignment_and_uniformity():
    import torch

    features = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    alignment, uniformity = alignment_uniformity(features, labels)
    assert float(alignment) == pytest.approx(0.0)
    assert float(uniformity) < 0


def test_source_fingerprint_changes_with_training_source(tmp_path, monkeypatch):
    from deepfake_detection.train import openworld

    monkeypatch.setattr(openworld, "TRAINING_SOURCES", ("train.py", "data.py"))
    (tmp_path / "train.py").write_text("first")
    (tmp_path / "data.py").write_text("data")
    before, files = source_fingerprint(tmp_path)
    assert set(files) == {"train.py", "data.py"}

    (tmp_path / "train.py").write_text("second")
    after, _ = source_fingerprint(tmp_path)
    assert after != before


def _namespaced_rows():
    rows = [{"label": "FAKE", "fake_method": "a", "video_id": f"a{video}",
             "lineage_video_ids": f"ff:{video:03d}"} for video in range(2)]
    # Two FF++ clips with several frames each, and many single-photo groups.
    rows += [{"label": "REAL", "fake_method": "", "video_id": f"clip{video}",
              "lineage_video_ids": f"ff:{video:03d}"} for video in range(2) for _ in range(3)]
    rows += [{"label": "REAL", "fake_method": "", "video_id": f"photo{index}",
              "lineage_video_ids": f"fairface:fairface-{index:06d}"} for index in range(18)]
    return rows


def test_sampling_plan_balances_real_sources_only_when_configured():
    _, _, plain = sampling_plan(_namespaced_rows(), {"sbi_share": 0.3})
    assert plain["real/fairface"] == pytest.approx(0.5 * 18 / 20)
    _, weights, mass = sampling_plan(_namespaced_rows(),
                                     {"sbi_share": 0.3, "real_balance": "lineage_namespace"})
    assert sum(weights) == pytest.approx(1.0)
    assert mass["real/ff"] == pytest.approx(0.25)
    assert mass["real/fairface"] == pytest.approx(0.25)
    assert mass["sbi/ff"] == pytest.approx(mass["sbi/fairface"])
    assert mass["fake/ff"] == pytest.approx(plain["fake/ff"])
    with pytest.raises(ValueError, match="real_balance"):
        sampling_plan(_namespaced_rows(), {"real_balance": "source_domain"})


def test_open_world_report_lists_real_only_domains_without_scoring_them():
    records = []
    for domain, labels in (("openfake", (0, 1)), ("lfw", (0,))):
        for label in labels:
            for video in range(2):
                records.append({
                    "source_domain": domain, "video_id": f"{domain}:{label}-{video}",
                    "fake_method": "openfake" if label else "", "label": label,
                    "score": 0.2 + 0.6 * label + 0.01 * video,
                    "manipulation_family": "entire_face_synthesis" if label else "real",
                    "lineage_video_ids": f"{domain}:{video}",
                })
    report = open_world_report(records)
    assert set(report["domain_macro_auroc"]) == {"openfake"}
    assert report["single_class_domains"] == ["lfw"]
    assert report["score"] == 1.0
