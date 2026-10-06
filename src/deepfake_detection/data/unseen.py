"""Strict lineage checks for the DF40 unseen-method benchmark."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from .identity import candidate_ids
from .manifest import verify_manifest

METHODS = {
    "train": {"simswap", "blendface", "wav2lip", "fomm", "sadtalker", "stylegan2", "sd21", "dit"},
    "validation": {"facedancer", "mraa", "stylegan3"},
    "test": {"uniface", "mcnet", "rddm"},
}


def audit_unseen_rows(rows: list[dict[str, str]]) -> dict:
    """Require documented parents and disjoint source clips and identities.

    Name-derived identities are useful leakage signals but not proof of identity
    separation. Missing identity evidence leaves the protocol unapproved.
    """
    required = {
        "sample_id",
        "split",
        "label",
        "fake_method",
        "video_id",
        "source_domain",
        "lineage_video_ids",
        "identity_id",
        "target_identity_id",
        "pixel_sha256",
        "dhash64",
        "status",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError("manifest lacks unseen-method audit columns")
    failures: list[str] = []
    counts: Counter[tuple[str, str]] = Counter()
    links: dict[str, dict[str, set[str]]] = {
        name: defaultdict(set) for name in ("video", "parent", "identity", "pixel", "dhash")
    }
    missing_parent = missing_identity = 0
    for row in rows:
        split = row["split"]
        if split not in METHODS or row["status"] != "ok":
            failures.append(f"invalid split or decode status: {row['sample_id']}")
            continue
        method = row["fake_method"] if row["label"] == "FAKE" else "real"
        if row["label"] == "FAKE" and method not in METHODS[split]:
            failures.append(f"method {method} is forbidden in {split}")
        counts[(split, method)] += 1
        domain = row["source_domain"]
        links["video"][f"{domain}:{row['video_id']}"].add(split)
        parents = [part for part in row["lineage_video_ids"].split("|") if part]
        if not parents:
            missing_parent += 1
        for parent in parents:
            links["parent"][parent].add(split)
        # Both roles are identity evidence. Keeping only source identity here
        # would allow an explicit target person to occur in multiple splits
        # whenever its name does not match Celeb-DF's idNN convention.
        identities = {
            identity.strip()
            for column in ("identity_id", "target_identity_id")
            for identity in row[column].split("|")
            if identity.strip()
        }
        identities.update(candidate_ids(row["video_id"]))
        identities.update(candidate_ids(row["target_identity_id"]))
        # FF++ does not publish person IDs in these catalogs. Each source clip
        # is the auditable identity unit; CDF keeps explicit idNN tokens.
        if not identities and domain == "ff":
            identities.update(parents)
        if not identities:
            missing_identity += 1
        for identity in identities:
            links["identity"][f"{domain}:{identity}"].add(split)
        for kind, column in (("pixel", "pixel_sha256"), ("dhash", "dhash64")):
            if row[column]:
                links[kind][row[column]].add(split)
    for split, methods in METHODS.items():
        for method in sorted(methods):
            if counts[(split, method)] == 0:
                failures.append(f"missing {split}/{method} samples")
        if counts[(split, "real")] == 0:
            failures.append(f"missing {split}/real samples")
    overlap = {
        kind: {key: sorted(splits) for key, splits in values.items() if len(splits) > 1}
        for kind, values in links.items()
    }
    for kind, values in overlap.items():
        if values:
            failures.append(f"{len(values)} cross-split {kind} groups")
    if missing_parent:
        failures.append(f"{missing_parent} rows have no source-video lineage")
    if missing_identity:
        failures.append(f"{missing_identity} rows have no identity evidence")
    return {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "counts": {f"{split}/{method}": count for (split, method), count in sorted(counts.items())},
        "overlap": {kind: dict(sorted(values.items())) for kind, values in overlap.items()},
        "missing_parent_rows": missing_parent,
        "missing_identity_rows": missing_identity,
        "identity_note": (
            "Name tokens and metadata are candidate identities; visual verification is required."
        ),
    }


def audit_unseen_manifest(directory: Path) -> dict:
    valid, errors = verify_manifest(directory)
    if not valid:
        raise ValueError(f"invalid immutable manifest: {errors}")
    with (directory / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    result = audit_unseen_rows(rows)
    result["manifest_id"] = directory.name
    result["manifest_sha256"] = json.loads((directory / "manifest.json").read_text())[
        "manifest_sha256"
    ]
    return result
