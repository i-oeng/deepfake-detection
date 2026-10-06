"""Normalize the official DeepfakeBench DF40 JSON catalogs.

The published JSON files index frames by dataset, class, split, and sometimes
compression before the video record. They contain paths from the authors'
machines, so the normalized catalog retains those paths as provenance and
assigns a portable destination under ``images/`` for later materialization.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import ZipFile

METHODS = {
    "simswap": ("simswap", "face_swap"),
    "wav2lip": ("wav2lip", "face_reenactment"),
    "stylegan2": ("stylegan2", "entire_face_synthesis"),
    "sd2.1": ("sd21", "entire_face_synthesis"),
    "blendface": ("blendface", "face_swap"),
    "uniface": ("uniface", "face_swap"),
    "facedancer": ("facedancer", "face_swap"),
    "fomm": ("fomm", "face_reenactment"),
    "mraa": ("mraa", "face_reenactment"),
    "mcnet": ("mcnet", "face_reenactment"),
    "sadtalker": ("sadtalker", "face_reenactment"),
    "dit": ("dit", "entire_face_synthesis"),
    "stylegan3": ("stylegan3", "entire_face_synthesis"),
    "rddm": ("rddm", "entire_face_synthesis"),
    "starganv2": ("starganv2", "face_edit"),
}
SPLITS = {"train": "train", "val": "validation", "validation": "validation", "test": "test"}
COMPRESSIONS = {"c23", "c40", "raw"}
COLUMNS = (
    "image_id",
    "relative_path",
    "label",
    "split",
    "official_splits",
    "fake_method",
    "manipulation_family",
    "source_domain",
    "video_id",
    "identity_id",
    "frame_index",
    "compression",
    "source_json",
    "source_path",
    "source_video_id",
    "target_identity_id",
    "lineage_video_ids",
)


def _method_and_domain(path: Path) -> tuple[str, str, str]:
    stem = path.stem.casefold()
    if stem.endswith("_cdf"):
        method_key, domain = stem[:-4], "cdf"
    elif stem.endswith("_ff"):
        method_key, domain = stem[:-3], "ff"
    elif stem == "starganv2":
        method_key, domain = stem, "celeba"
    else:
        raise ValueError(f"unsupported DF40 catalog filename: {path.name}")
    if method_key not in METHODS:
        raise ValueError(f"unsupported DF40 method in {path.name}")
    method, family = METHODS[method_key]
    return method, family, domain


def _records(node: Any, keys: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], dict]]:
    if not isinstance(node, dict):
        raise ValueError(f"expected an object at {'/'.join(keys) or '<root>'}")
    if "frames" in node:
        yield keys, node
        return
    for key, value in node.items():
        yield from _records(value, (*keys, str(key)))


def _field(keys: tuple[str, ...], mapping: dict[str, str], name: str) -> str:
    matches = {mapping[key.casefold()] for key in keys if key.casefold() in mapping}
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {name} in {'/'.join(keys)}")
    return matches.pop()


def _label(record: dict, keys: tuple[str, ...]) -> str:
    candidates = [str(record.get("label", "")), *keys]
    labels = set()
    for candidate in candidates:
        lowered = candidate.casefold()
        if re.search(r"(^|[_-])real($|[_-])", lowered):
            labels.add("REAL")
        elif re.search(r"(^|[_-])fake($|[_-])", lowered):
            labels.add("FAKE")
    if len(labels) != 1:
        raise ValueError(f"missing or conflicting real/fake label in {'/'.join(keys)}")
    return labels.pop()


def _safe_part(value: str, name: str) -> str:
    if not value or value in {".", ".."} or any(char in value for char in '/\\:<>"|?*\x00'):
        raise ValueError(f"unsafe {name}: {value!r}")
    return value


def _frame_index(frame_name: str, position: int) -> str:
    match = re.search(r"(\d+)$", Path(frame_name).stem)
    return str(int(match.group(1))) if match else str(position)


def _pilot_eval_split(seed: str, domain: str, video_name: str) -> str:
    """Keep each evaluation video in one split across all method catalogs."""
    token = f"df40-pilot-eval-v1\0{seed}\0{domain}\0{video_name}".encode()
    return "validation" if hashlib.sha256(token).digest()[0] < 128 else "test"


def _lineage_video_ids(domain: str, video_name: str, record: dict) -> str:
    """Record every plausible parent; FF swap names do not establish direction."""
    explicit = str(record.get("source_video_id", "")).strip()
    if explicit:
        parents = [part.strip() for part in explicit.split("|") if part.strip()]
    elif domain == "ff":
        # FF++ swap names such as 001_002 refer to two source clips. Keeping
        # both prevents a target/source role guess from hiding leakage.
        parents = [part for part in video_name.split("_") if part]
    else:
        parents = [video_name]
    return "|".join(sorted({f"{domain}:{part}" for part in parents}))


def normalize_df40(
    input_dir: Path, output: Path, *, eval_split_seed: str | None = None
) -> dict[str, Any]:
    """Write one deterministic row per unique official frame path.

    Repeated real references across method JSONs are collapsed. Conflicting
    label or train/evaluation claims fail so they cannot contaminate a pilot.
    An explicit seed repartitions the overlapping official validation/test
    pool by domain and video name, without changing the official train split.
    """
    sources = sorted(input_dir.glob("*.json"), key=lambda path: path.name.casefold())
    if not sources:
        raise FileNotFoundError(f"no JSON catalogs found in {input_dir}")

    rows: dict[str, dict[str, str]] = {}
    official_splits: dict[str, set[str]] = {}
    evaluation_groups: dict[str, set[str]] = {}
    for source in sources:
        method, family, domain = _method_and_domain(source)
        with source.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        for keys, record in _records(payload):
            official_split = _field(keys, SPLITS, "split")
            compression = next((key for key in keys if key.casefold() in COMPRESSIONS), "")
            label = _label(record, keys)
            frames = record["frames"]
            if not isinstance(frames, list) or not all(isinstance(p, str) for p in frames):
                raise ValueError(f"invalid frames list in {source.name}: {'/'.join(keys)}")
            video_name = _safe_part(keys[-1], "video name")
            video_id = f"{domain}:{video_name}"
            split = official_split
            if eval_split_seed is not None and official_split != "train":
                # The published JSON lists the same evaluation frames under
                # both val and test. Assign the entire video consistently,
                # including its appearances in other method catalogs.
                split = _pilot_eval_split(eval_split_seed, domain, video_name)
                evaluation_groups.setdefault(video_id, set()).add(official_split)
            for position, original in enumerate(sorted(frames)):
                normalized = original.replace("\\", "/")
                frame_name = _safe_part(PurePosixPath(normalized).name, "frame name")
                if not normalized or not frame_name:
                    raise ValueError(f"empty frame path in {source.name}")
                # An absolute source path is provenance, never a local read target.
                path_token = hashlib.sha256(normalized.encode()).hexdigest()[:12]
                relative = PurePosixPath(
                    "images",
                    domain,
                    method if label == "FAKE" else "real",
                    split,
                    compression or "none",
                    video_name,
                    f"{path_token}_{frame_name}",
                ).as_posix()
                image_id = hashlib.sha256(relative.encode()).hexdigest()[:24]
                row = {
                    "image_id": image_id,
                    "relative_path": relative,
                    "label": label,
                    "split": split,
                    "official_splits": "",
                    "fake_method": method if label == "FAKE" else "",
                    "manipulation_family": family if label == "FAKE" else "real",
                    "source_domain": domain,
                    "video_id": video_id,
                    # Official JSON does not consistently provide person identities.
                    "identity_id": str(record.get("identity_id", "")),
                    "frame_index": _frame_index(frame_name, position),
                    "compression": compression,
                    "source_json": source.name,
                    "source_path": normalized,
                    "source_video_id": str(record.get("source_video_id", "")),
                    "target_identity_id": str(record.get("target_identity_id", "")),
                    "lineage_video_ids": _lineage_video_ids(domain, video_name, record),
                }
                existing = rows.get(normalized)
                if existing is not None:
                    if (
                        existing["label"],
                        existing["split"],
                        existing["source_domain"],
                        existing["fake_method"],
                    ) != (
                        label,
                        split,
                        domain,
                        row["fake_method"],
                    ):
                        raise ValueError(f"conflicting official frame assignment: {normalized}")
                    official_splits[normalized].add(official_split)
                    continue
                rows[normalized] = row
                official_splits[normalized] = {official_split}

    if not rows:
        raise ValueError("official catalogs contain no frame paths")
    # Sort by the portable path to make the result independent of JSON key order.
    for path, row in rows.items():
        row["official_splits"] = "|".join(sorted(official_splits[path]))
    ordered = sorted(rows.values(), key=lambda row: row["relative_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(ordered)
    return {
        "catalog": str(output),
        "sources": len(sources),
        "rows": len(ordered),
        "eval_split_seed": eval_split_seed,
        "overlapping_eval_paths": sum(
            splits == {"validation", "test"} for splits in official_splits.values()
        ),
        "evaluation_groups": len(evaluation_groups),
    }


# These method archives are present on xixi, while their official JSON files
# are absent. ZIP member paths provide video/frame lineage without guessed
# source or target roles; the resulting catalog records archive provenance.
ARCHIVE_METHODS = {
    "fomm": ("train/fomm.zip", "face_reenactment"),
    "facedancer": ("test/facedancer.zip", "face_swap"),
    "mraa": ("test/MRAA.zip", "face_reenactment"),
    "stylegan3": ("test/StyleGAN3.zip", "entire_face_synthesis"),
    "uniface": ("test/uniface.zip", "face_swap"),
    "mcnet": ("test/mcnet.zip", "face_reenactment"),
    "rddm": ("test/RDDM.zip", "entire_face_synthesis"),
}

UNSEEN_METHOD_SPLITS = {
    "simswap": "train",
    "blendface": "train",
    "wav2lip": "train",
    "fomm": "train",
    "sadtalker": "train",
    "stylegan2": "train",
    "sd21": "train",
    "dit": "train",
    "facedancer": "validation",
    "mraa": "validation",
    "stylegan3": "validation",
    "uniface": "test",
    "mcnet": "test",
    "rddm": "test",
}


def _source_entities(domain: str, video_id: str) -> tuple[str, ...]:
    """Return conservative source identity/clip keys derived from published names."""
    name = video_id.removeprefix(f"{domain}:")
    if domain == "cdf":
        identities = sorted(set(re.findall(r"(?<![A-Za-z0-9])id[0-9]+", name)))
        # Opaque CDF names do not prove identity. They are excluded from this
        # protocol instead of being promoted to person identifiers.
        return tuple(f"cdf:{identity}" for identity in identities)
    # FF++ swap names begin with two clip IDs; reenactment and real names begin
    # with one. Later numeric fields are timestamps or method-specific counters.
    match = re.match(r"^(\d+)(?:_(\d+))?", name)
    if match:
        return tuple(f"ff:{value}" for value in match.groups() if value is not None)
    return (f"ff:video:{name}",)


GROUP_TARGETS = {
    "test": {"ff": 15, "cdf": 6},
    "validation": {"ff": 15, "cdf": 8},
    "train": {"ff": 75},
}


def _rank(seed: str, *parts: str) -> str:
    return hashlib.sha256("\0".join((seed, *parts)).encode()).hexdigest()


def _portable_row(row: dict[str, str], split: str, entities: tuple[str, ...]) -> dict[str, str]:
    output = {column: row.get(column, "") for column in COLUMNS}
    output["split"] = split
    output["lineage_video_ids"] = "|".join(entities)
    frame_name = PurePosixPath(output["source_path"]).name
    token = hashlib.sha256(output["source_path"].encode()).hexdigest()[:12]
    video_name = output["video_id"].removeprefix(f"{output['source_domain']}:")
    output["relative_path"] = PurePosixPath(
        "images",
        output["source_domain"],
        output["fake_method"] or "real",
        split,
        output["compression"] or "none",
        video_name,
        f"{token}_{frame_name}",
    ).as_posix()
    output["image_id"] = hashlib.sha256(output["relative_path"].encode()).hexdigest()[:24]
    return output


def partition_df40_unseen(catalog_path: Path, output: Path, *, seed: str) -> dict[str, Any]:
    """Select source-disjoint video groups before the frame-level sampler runs."""
    with catalog_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not set(COLUMNS).issubset(reader.fieldnames or ()):
            raise ValueError("extended catalog lacks DF40 lineage columns")
        source = list(reader)
    fake_groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    real_groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in source:
        if row["label"] == "FAKE":
            target = UNSEEN_METHOD_SPLITS.get(row["fake_method"])
            if target is None:
                continue
            # Training methods use the authors' training pool; development and
            # test methods use their test archives. This avoids moving the same
            # image between the source protocol pools.
            if target == "train" and row["split"] != "train":
                continue
            if target != "train" and not row["source_json"].startswith("archive:test/"):
                continue
            fake_groups.setdefault(
                (row["fake_method"], row["source_domain"], row["video_id"]), []
            ).append(row)
        else:
            real_groups.setdefault((row["source_domain"], row["video_id"]), []).append(row)

    entity_owner: dict[str, str] = {}
    accepted: dict[tuple[str, str, str], str] = {}
    selected_group_counts = Counter()
    # Test and development reserve identities first. Training then uses the
    # remaining FF++ pool, preventing the large training set from starving eval.
    for split in ("test", "validation", "train"):
        methods = [method for method, owner in UNSEEN_METHOD_SPLITS.items() if owner == split]
        for domain, target_count in GROUP_TARGETS[split].items():
            for method in methods:
                candidates = [key for key in fake_groups if key[:2] == (method, domain)]
                chosen = 0
                while chosen < target_count:
                    eligible = []
                    for key in candidates:
                        if key in accepted:
                            continue
                        entities = _source_entities(domain, key[2])
                        if not entities:
                            continue
                        owners = {entity_owner.get(entity) for entity in entities}
                        if owners - {None, split}:
                            continue
                        new_count = sum(entity not in entity_owner for entity in entities)
                        eligible.append(
                            (new_count, _rank(seed, split, method, domain, key[2]), key)
                        )
                    if not eligible:
                        target = f"{split}/{method}/{domain}"
                        raise ValueError(
                            f"cannot select {target_count} disjoint groups for {target}"
                        )
                    _, _, key = min(eligible)
                    entities = _source_entities(domain, key[2])
                    for entity in entities:
                        entity_owner[entity] = split
                    accepted[key] = split
                    chosen += 1
                selected_group_counts[(split, method, domain)] = chosen

    # Reserve enough independent real videos for 1:1 frame balancing at the
    # configured eight-frame cap. Prefer identities already used by fakes.
    real_group_targets = {
        (split, domain): len([m for m, owner in UNSEEN_METHOD_SPLITS.items() if owner == split])
        * groups
        for split, domains in GROUP_TARGETS.items()
        for domain, groups in domains.items()
    }
    assigned_real: dict[tuple[str, str], str] = {}
    for (split, domain), target_count in real_group_targets.items():
        candidates = [key for key in real_groups if key[0] == domain]
        while (
            sum(owner == split and key[0] == domain for key, owner in assigned_real.items())
            < target_count
        ):
            eligible = []
            for key in candidates:
                if key in assigned_real:
                    continue
                entities = _source_entities(domain, key[1])
                if not entities:
                    continue
                owners = {entity_owner.get(entity) for entity in entities}
                if owners - {None, split}:
                    continue
                new_count = sum(entity not in entity_owner for entity in entities)
                eligible.append((new_count, _rank(seed, "real", split, domain, key[1]), key))
            if not eligible:
                raise ValueError(f"cannot reserve {target_count} real groups for {split}/{domain}")
            _, _, key = min(eligible)
            entities = _source_entities(domain, key[1])
            for entity in entities:
                entity_owner[entity] = split
            assigned_real[key] = split

    rows = []
    for key, split in accepted.items():
        entities = _source_entities(key[1], key[2])
        rows.extend(_portable_row(row, split, entities) for row in fake_groups[key])
    for key, split in assigned_real.items():
        entities = _source_entities(key[0], key[1])
        rows.extend(_portable_row(row, split, entities) for row in real_groups[key])
    rows.sort(key=lambda row: row["relative_path"])
    if len({row["relative_path"] for row in rows}) != len(rows):
        raise ValueError("partition produced duplicate paths")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter((row["split"], row["label"], row["fake_method"] or "real") for row in rows)
    return {
        "catalog": str(output),
        "rows": len(rows),
        "seed": seed,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "counts": {"/".join(key): value for key, value in sorted(counts.items())},
        "selected_video_groups": {
            "/".join(key): value for key, value in sorted(selected_group_counts.items())
        },
        "entity_counts": dict(sorted(Counter(entity_owner.values()).items())),
    }


SYNTHESIS = "entire_face_synthesis"
# FOMM is published only in DF40's training archive, so it has no held-out videos.
HELDOUT_METHODS = (
    "simswap", "blendface", "wav2lip", "sadtalker", "stylegan2", "sd21", "dit", "starganv2",
)


def _heldout_entities(domain: str, video_id: str) -> tuple[str, ...]:
    # CelebA images are single frames; each image is its own lineage unit.
    if domain == "celeba":
        return (f"celeba:{video_id.removeprefix('celeba:')}",)
    return _source_entities(domain, video_id)


def partition_df40_heldout(
    catalog_path: Path,
    reference_manifest: Path,
    output: Path,
    *,
    downloads_root: Path | None = None,
    exclude_videos: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Collect evaluation-archive videos whose sources never reach frozen train or validation.

    Every row is assigned to ``test``. The frame sampler later chooses videos and
    frames from this pool, so this step only decides eligibility. ``exclude_videos``
    removes video IDs that a previous content audit found colliding with the
    reference, so the sampler can choose replacements.
    """
    owned: set[str] = set()
    owned_images: set[tuple[str, str]] = set()
    with reference_manifest.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["split"] in {"train", "validation"}:
                owned.update(_source_entities(row["source_domain"], row["video_id"]))
                owned.update(part for part in row["lineage_video_ids"].split("|") if part)
                if row["manipulation_family"] == SYNTHESIS:
                    owned_images.add((row["fake_method"], row["frame_index"]))
    with catalog_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not set(COLUMNS).issubset(reader.fieldnames or ()):
            raise ValueError("extended catalog lacks DF40 lineage columns")
        source = list(reader)
    rows: dict[str, dict[str, str]] = {}
    excluded: Counter[str] = Counter()
    for row in source:
        if row["label"] == "FAKE":
            # The authors' training pool stays unused, as in the frozen protocol.
            if row["fake_method"] not in HELDOUT_METHODS or row["split"] == "train":
                continue
        elif row["label"] != "REAL":
            continue
        entities = _heldout_entities(row["source_domain"], row["video_id"])
        if not entities or owned.intersection(entities):
            excluded[row["source_domain"]] += 1
            continue
        # Synthesis folders mimic source videos, but the archives reuse each
        # generated image, numbered by frame index, under different folders.
        if (row["fake_method"], row["frame_index"]) in owned_images:
            excluded["synthesis_image"] += 1
            continue
        if row["video_id"] in exclude_videos:
            excluded["content_collision"] += 1
            continue
        portable = _portable_row(row, "test", entities)
        # Real frames are listed once per method catalog; keep one copy.
        rows.setdefault(portable["relative_path"], portable)
    empty: set[str] = set()
    if downloads_root is not None:
        from .materialize import find_empty_members

        empty = find_empty_members(list(rows.values()), downloads_root)
    ordered = sorted(
        (row for path, row in rows.items() if path not in empty),
        key=lambda row: row["relative_path"],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(ordered)
    videos = Counter(
        (row["source_domain"], row["fake_method"] or "real")
        for row in {(r["source_domain"], r["fake_method"], r["video_id"]): r
                    for r in ordered}.values()
    )
    return {
        "catalog": str(output),
        "rows": len(ordered),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "reference_entities": len(owned),
        "excluded_rows_by_domain": dict(sorted(excluded.items())),
        "empty_archive_members": sorted(empty),
        "eligible_videos": {"/".join(key): value for key, value in sorted(videos.items())},
    }


def extend_df40_from_archives(
    catalog_path: Path, downloads_root: Path, output: Path, *, eval_split_seed: str
) -> dict[str, Any]:
    """Add absent methods from their published ZIP members, preserving provenance."""
    with catalog_path.open(newline="", encoding="utf-8") as stream:
        existing = list(csv.DictReader(stream))
    if not existing:
        raise ValueError("base DF40 catalog is empty")
    rows = [{column: row.get(column, "") for column in COLUMNS} for row in existing]
    for row in rows:
        if not row["lineage_video_ids"]:
            domain = row["source_domain"]
            video_name = row["video_id"].removeprefix(f"{domain}:")
            row["lineage_video_ids"] = _lineage_video_ids(domain, video_name, row)
    seen_paths = {row["relative_path"] for row in rows}
    added = {}
    for method, (archive_name, family) in ARCHIVE_METHODS.items():
        if any(row["fake_method"] == method for row in rows):
            raise ValueError(f"method {method} is already in the base catalog")
        archive_path = downloads_root / archive_name
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)
        count = 0
        with ZipFile(archive_path) as archive:
            for member in sorted(archive.namelist()):
                if not member.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                parts = PurePosixPath(member).parts
                if len(parts) < 3 or any(part in {"", ".", ".."} for part in parts):
                    raise ValueError(f"unsafe archive member in {archive_name}: {member}")
                domains = {part for part in parts if part in {"ff", "cdf"}}
                if len(domains) > 1:
                    raise ValueError(f"ambiguous domain in {archive_name}: {member}")
                domain = domains.pop() if domains else "ff" if method == "fomm" else ""
                if not domain:
                    raise ValueError(f"missing source domain in {archive_name}: {member}")
                video_name = _safe_part(parts[-2], "archive video name")
                frame_name = _safe_part(parts[-1], "archive frame name")
                split = (
                    "train"
                    if archive_name.startswith("train/")
                    else _pilot_eval_split(eval_split_seed, domain, video_name)
                )
                source_path = f"deepfakes_detection_datasets/DF40/{member}"
                token = hashlib.sha256(source_path.encode()).hexdigest()[:12]
                relative = PurePosixPath(
                    "images",
                    domain,
                    method,
                    split,
                    "none",
                    video_name,
                    f"{token}_{frame_name}",
                ).as_posix()
                if relative in seen_paths:
                    raise ValueError(f"duplicate portable path: {relative}")
                seen_paths.add(relative)
                rows.append(
                    {
                        "image_id": hashlib.sha256(relative.encode()).hexdigest()[:24],
                        "relative_path": relative,
                        "label": "FAKE",
                        "split": split,
                        "official_splits": "train" if split == "train" else "test",
                        "fake_method": method,
                        "manipulation_family": family,
                        "source_domain": domain,
                        "video_id": f"{domain}:{video_name}",
                        "identity_id": "",
                        "frame_index": _frame_index(frame_name, 0),
                        "compression": "",
                        "source_json": f"archive:{archive_name}",
                        "source_path": source_path,
                        "source_video_id": "",
                        "target_identity_id": "",
                        "lineage_video_ids": _lineage_video_ids(domain, video_name, {}),
                    }
                )
                count += 1
        added[method] = count
    rows.sort(key=lambda row: row["relative_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return {
        "catalog": str(output),
        "rows": len(rows),
        "added": added,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
