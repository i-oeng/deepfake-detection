import random

from scripts import compare_round4_photos as compare


def _record(domain, video, label, score, method=""):
    return {"source_domain": domain, "video_id": f"{domain}:{video}", "label": label,
            "fake_method": method if label else "", "score": score, "lineage_video_ids": ""}


def _domain(domain, rng, fakes=20, reals=20, shift=0.3, method="m", name=None):
    rows = [_record(domain, f"real-src-{index:06d}", 0, rng.random() * 0.7)
            for index in range(reals)]
    rows += [_record(domain, f"fake-{name or method}-{index:06d}", 1,
                     min(1.0, rng.random() * 0.7 + shift), method) for index in range(fakes)]
    return rows


def _sets(seed=0):
    rng = random.Random(seed)
    validation = _domain("ff", rng) + [_record("cdf", f"real-c-{i}", 0, rng.random() * 0.6)
                                       for i in range(20)]
    fresh_v2 = [row for domain in ("dff", "celebdf_video", "sfhq_t2i")
                for row in _domain(domain, rng)] + _domain("ignored", rng)
    fresh_v3 = (_domain("openfake", rng, method="openfake", name="flux.1-dev")
                + _domain("openfake", rng, reals=0, method="openfake", name="z-image-turbo")
                + [_record("lfw", f"real-l-{i}", 0, rng.random() * 0.6) for i in range(20)]
                + _domain("openfake_reddit", rng, reals=5))
    return validation, fresh_v2, fresh_v3


def test_targets_keep_only_train_generators_and_the_target_domains():
    _, fresh_v2, fresh_v3 = _sets()
    kept = compare.targets(fresh_v2, fresh_v3, {"flux.1-dev"})
    domains = {row["source_domain"] for row in kept}
    assert domains == {"dff", "celebdf_video", "sfhq_t2i", "openfake", "lfw", "openfake_reddit"}
    openfake = [row for row in kept if row["source_domain"] == "openfake" and row["label"]]
    assert len(openfake) == 20
    assert all("flux.1-dev" in row["video_id"] for row in openfake)


def test_evaluate_locks_one_threshold_within_the_real_cap():
    validation, fresh_v2, fresh_v3 = _sets()
    result = compare.evaluate(validation, compare.targets(fresh_v2, fresh_v3, {"flux.1-dev"}))
    assert set(result["target_auroc"]) == set(compare.TARGETS)
    locked = result["threshold"]
    assert locked["validation_real_fpr"] <= 0.10
    assert "openfake_reddit" not in locked["real_sources"]
    assert {"ff", "cdf", "lfw", "openfake"} <= set(locked["real_sources"])
    accuracy = result["balanced_accuracy"]["dff"]
    assert 0.5 <= accuracy["balanced_accuracy"] <= 1.0
    assert accuracy["videos"] == {"fake": 20, "real": 20}
    assert set(result["real_fpr"]) >= {"ff", "cdf", "lfw", "openfake_reddit"}


def test_decide_needs_a_higher_target_mean_and_both_guardrails():
    def model(mean):
        return {"target_mean_auroc": mean}

    def guard(ff, bias):
        return {"domain_mean_auroc": {"ff": ff}, "source_bias": {"mean_auroc": bias}}

    base = guard(0.97, 0.66)
    assert compare.decide(model(0.7), model(0.8), base, guard(0.965, 0.68))["advances"]
    assert not compare.decide(model(0.7), model(0.8), base, guard(0.955, 0.66))["advances"]
    assert not compare.decide(model(0.7), model(0.8), base, guard(0.97, 0.70))["advances"]
    assert not compare.decide(model(0.7), model(0.7), base, base)["advances"]
