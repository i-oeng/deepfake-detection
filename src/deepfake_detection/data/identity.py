"""Audit name-derived Celeb-DF identity clues across frozen splits."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from .manifest import verify_manifest

IDENTITY_TOKEN = re.compile(r"(?<![A-Za-z0-9])id[0-9]+")
SPLITS = ("train", "validation", "test")


def candidate_ids(video_id: str) -> set[str]:
    """Return all idNN tokens, including both sides of a swapped-face name."""
    return set(IDENTITY_TOKEN.findall(video_id))


def audit_candidate_identity_tokens(manifest_dir: str | Path) -> dict:
    """Report possible Celeb-DF subject reuse without claiming face verification."""
    directory = Path(manifest_dir)
    valid, errors = verify_manifest(directory)
    if not valid:
        raise ValueError(f"invalid immutable manifest: {errors}")
    metadata = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    with (directory / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"source_domain", "video_id", "split", "label", "status"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("manifest lacks candidate-identity audit columns")
        rows = [row for row in reader if row["source_domain"] == "cdf"]
    if not rows:
        raise ValueError("manifest has no Celeb-DF source-domain rows")

    tokens_by_split: dict[str, set[str]] = defaultdict(set)
    covered = Counter()
    totals = Counter()
    decoded: list[tuple[dict[str, str], set[str]]] = []
    for row in rows:
        if row["status"] != "ok" or row["split"] not in SPLITS:
            raise ValueError("candidate-identity audit requires valid decoded split rows")
        split = row["split"]
        tokens = candidate_ids(row["video_id"])
        totals[split] += 1
        covered[split] += bool(tokens)
        tokens_by_split[split].update(tokens)
        decoded.append((row, tokens))

    pairs = {}
    for left, right in combinations(SPLITS, 2):
        shared = tokens_by_split[left] & tokens_by_split[right]
        affected = Counter(
            (row["split"], row["label"])
            for row, tokens in decoded
            if row["split"] in {left, right} and tokens & shared
        )
        pairs[f"{left}_vs_{right}"] = {
            "shared_token_count": len(shared),
            "shared_tokens": sorted(shared),
            "affected_rows": sum(affected.values()),
            "affected_rows_by_split_label": [
                {"split": split, "label": label, "rows": count}
                for (split, label), count in sorted(affected.items())
            ],
        }
    return {
        "schema_version": 1,
        "manifest_id": metadata["manifest_id"],
        "manifest_sha256": metadata["manifest_sha256"],
        "scope": "source_domain=cdf; all idNN tokens in video_id",
        "interpretation": (
            "Name-derived candidate identity overlap only. Tokens can refer to source or "
            "target in manipulations; visual identity matching is still required."
        ),
        "row_coverage": {
            split: {"total": totals[split], "with_token": covered[split],
                    "without_token": totals[split] - covered[split],
                    "distinct_tokens": len(tokens_by_split[split])}
            for split in SPLITS
        },
        "split_pairs": pairs,
    }


def write_candidate_identity_report(report: dict, output: str | Path) -> Path:
    path = Path(output)
    content = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"refusing to overwrite changed identity audit: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path
