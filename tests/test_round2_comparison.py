from scripts.compare_round2 import ensemble, paired_domain_bootstrap


def _predictions(delta=0.0):
    rows = []
    for domain in ("a", "b"):
        for index, (label, method, score) in enumerate((
            (0, "", 0.2), (0, "", 0.3), (1, "fake", 0.7 + delta),
            (1, "fake", 0.8 + delta),
        )):
            rows.append({
                "sample_id": f"{domain}-{index}",
                "source_id": f"source-{domain}-{index}",
                "split": "test",
                "video_id": f"video-{domain}-{index}",
                "source_domain": domain,
                "fake_method": method,
                "manipulation_family": "real" if not label else "synthesis",
                "label": label,
                "lineage_video_ids": f"source:{domain}:{index}",
                "logit": score,
                "score": score,
            })
    return rows


def test_ensemble_averages_aligned_logits(tmp_path):
    import json

    paths = []
    for seed, shift in enumerate((0.0, 0.2)):
        path = tmp_path / f"seed-{seed}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in _predictions(shift)))
        paths.append(path)
    records, seeds = ensemble(paths)
    assert len(seeds) == 2
    fake = next(row for row in records if row["sample_id"] == "a-2")
    assert abs(fake["logit"] - 0.8) < 1e-12


def test_paired_bootstrap_uses_all_domains():
    baseline = _predictions()
    candidate = _predictions(0.1)
    result = paired_domain_bootstrap(baseline, candidate, repeats=100, seed=1)
    assert result["replicates"] == 100
    assert "domain" in result["unit"]
