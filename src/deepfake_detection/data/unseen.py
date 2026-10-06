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


REQUIRED = {
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
LINK_KINDS = ("video", "parent", "identity", "pixel", "dhash")


def row_links(row: dict[str, str]) -> dict[str, set[str]]:
    """Return the lineage, identity, and content keys that must not cross splits."""
    domain = row["source_domain"]
    parents = {part for part in row["lineage_video_ids"].split("|") if part}
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
    # is the auditable identity unit; CDF keeps explicit idNN tokens. CelebA
    # images have no identity metadata, so each image lineage stands in.
    if not identities and domain in {"ff", "celeba"}:
        identities.update(parents)
    return {
        "video": {f"{domain}:{row['video_id']}"},
        "parent": parents,
        "identity": {f"{domain}:{identity}" for identity in identities},
        "pixel": {row["pixel_sha256"]} if row["pixel_sha256"] else set(),
        "dhash": {row["dhash64"]} if row["dhash64"] else set(),
    }


def audit_unseen_rows(rows: list[dict[str, str]]) -> dict:
    """Require documented parents and disjoint source clips and identities.

    Name-derived identities are useful leakage signals but not proof of identity
    separation. Missing identity evidence leaves the protocol unapproved.
    """
    if not rows or not REQUIRED.issubset(rows[0]):
        raise ValueError("manifest lacks unseen-method audit columns")
    failures: list[str] = []
    counts: Counter[tuple[str, str]] = Counter()
    links: dict[str, dict[str, set[str]]] = {name: defaultdict(set) for name in LINK_KINDS}
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
        keys = row_links(row)
        missing_parent += not keys["parent"]
        missing_identity += not keys["identity"]
        for kind, values in keys.items():
            for value in values:
                links[kind][value].add(split)
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


# Minimum independent videos (CelebA: images) per held-out protocol cell.
HELDOUT_TARGETS = {
    **{
        (domain, method): count
        for method in ("simswap", "blendface", "wav2lip", "sadtalker", "stylegan2", "sd21", "dit")
        for domain, count in (("ff", 20), ("cdf", 10))
    },
    ("celeba", "starganv2"): 300,
    ("ff", "real"): 60,
    ("cdf", "real"): 30,
    ("celeba", "real"): 300,
}


def audit_heldout_rows(
    rows: list[dict[str, str]], reference_rows: list[dict[str, str]]
) -> dict:
    """Gate an evaluation-only manifest against a frozen protocol's train and validation."""
    if not rows or not REQUIRED.issubset(rows[0]):
        raise ValueError("manifest lacks held-out audit columns")
    forbidden: dict[str, set[str]] = {kind: set() for kind in LINK_KINDS}
    for row in reference_rows:
        if row["split"] in {"train", "validation"}:
            for kind, values in row_links(row).items():
                forbidden[kind].update(values)
    failures: list[str] = []
    overlap: dict[str, set[str]] = {kind: set() for kind in LINK_KINDS}
    videos: dict[tuple[str, str], set[str]] = defaultdict(set)
    missing_parent = missing_identity = 0
    for row in rows:
        if row["split"] != "test" or row["status"] != "ok":
            failures.append(f"invalid split or decode status: {row['sample_id']}")
            continue
        method = row["fake_method"] if row["label"] == "FAKE" else "real"
        if (row["source_domain"], method) not in HELDOUT_TARGETS:
            failures.append(f"unexpected cell {row['source_domain']}/{method}")
        videos[(row["source_domain"], method)].add(row["video_id"])
        keys = row_links(row)
        missing_parent += not keys["parent"]
        missing_identity += not keys["identity"]
        for kind, values in keys.items():
            overlap[kind].update(values & forbidden[kind])
    for (domain, method), target in sorted(HELDOUT_TARGETS.items()):
        if len(videos[(domain, method)]) < target:
            failures.append(
                f"{domain}/{method}: {len(videos[(domain, method)])} of {target} videos"
            )
    for kind, values in overlap.items():
        if values:
            failures.append(f"{len(values)} {kind} keys shared with frozen train or validation")
    if missing_parent:
        failures.append(f"{missing_parent} rows have no source-video lineage")
    if missing_identity:
        failures.append(f"{missing_identity} rows have no identity evidence")
    return {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "videos": {f"{d}/{m}": len(v) for (d, m), v in sorted(videos.items())},
        "overlap": {kind: sorted(values) for kind, values in overlap.items()},
        "missing_parent_rows": missing_parent,
        "missing_identity_rows": missing_identity,
    }


def audit_heldout_manifest(directory: Path, reference: Path) -> dict:
    rows_by_directory = []
    for path in (directory, reference):
        valid, errors = verify_manifest(path)
        if not valid:
            raise ValueError(f"invalid immutable manifest {path}: {errors}")
        with (path / "manifest.csv").open(newline="", encoding="utf-8") as stream:
            rows_by_directory.append(list(csv.DictReader(stream)))
    result = audit_heldout_rows(*rows_by_directory)
    for name, path in (("manifest", directory), ("reference", reference)):
        result[f"{name}_id"] = path.name
        result[f"{name}_sha256"] = json.loads((path / "manifest.json").read_text())[
            "manifest_sha256"
        ]
    return result
