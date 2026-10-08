"""Fresh open-world test set (v2) from sources that protocol v1 never used.

v1's test split was scored once and is spent. This set measures later changes:

- ``celebdf_video``: Celeb-DF v2's own synthesis method on raw videos, through
  the same YuNet crop pipeline as inference. Reals are Celeb-real and the
  YouTube-real clips v1 excluded. Videos showing any Celeb-DF identity that v1
  used for validation are left out, so model selection never saw those people.
- ``dff``: DeepFakeFace Stable Diffusion 1.5 text-to-image, Stable Diffusion
  inpainting, and InsightFace swaps, against the IMDB-WIKI photos they derive from.
- ``sfhq_t2i``: curated FLUX.1 and SDXL faces (SFHQ-T2I sample) against two FFHQ
  shards.

Every row is a ``test`` row. Rows whose content matches any v1 image are dropped.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from .identity import IDENTITY_TOKEN
from .openworld import COLUMNS, EFS, FE, FS, _image_members, _rank, _row, _safe

SPLIT = "test"
FRAMES_PER_VIDEO = 16
CELEBDF_FAKE_VIDEOS = 600
DFF_PER_CLASS = 1000
DFF_ARCHIVES = {
    "wiki.zip": ("", ""),
    "text2img.zip": ("sd15_text2img", EFS),
    "inpainting.zip": ("sd_inpainting", FE),
    "insight.zip": ("insightface", FS),
}
SFHQ_ARCHIVE = "SyntheticFacesHighQuality-T2I/small-sample.zip"
FFHQ_FOLDER = "dir:ffhq"
SFHQ_GENERATOR = re.compile(r"^(.+?)_image_\d+$")


def validation_identities(reference_manifest: Path) -> set[str]:
    """Celeb-DF identity units (``cdf:idN``) that v1 used for model selection."""
    with reference_manifest.open(newline="", encoding="utf-8") as stream:
        return {
            token
            for row in csv.DictReader(stream)
            if row["split"] == "validation" and row["source_domain"] == "cdf"
            for token in row["lineage_video_ids"].split("|")
            if re.fullmatch(r"cdf:id\d+", token)
        }


def select_celebdf(catalog: Path, excluded: set[str], seed: str) -> list[dict[str, str]]:
    """All eligible real videos and a hash-ranked sample of synthesis videos."""
    with catalog.open(newline="", encoding="utf-8") as stream:
        videos = list(csv.DictReader(stream))
    eligible = []
    for video in videos:
        name = PurePosixPath(video["archive_path"]).stem
        tokens = {f"cdf:{token}" for token in IDENTITY_TOKEN.findall(name)}
        if tokens & excluded:
            continue
        eligible.append({**video, "name": name, "tokens": "|".join(sorted(tokens))})
    real = [video for video in eligible if video["label"] == "REAL"]
    fake = sorted((video for video in eligible if video["label"] == "FAKE"),
                  key=lambda video: _rank(seed, "celebdf", video["name"]))
    return real + fake[:CELEBDF_FAKE_VIDEOS]


def _celebdf_rows(downloads_root: Path, celebdf_root: Path, reference: Path, seed: str,
                  workers: int) -> tuple[list[dict[str, str]], dict]:
    from deepfake_detection.inference import crops

    selected = select_celebdf(celebdf_root / "video_catalog.csv",
                              validation_identities(reference), seed)
    videos = downloads_root / "celebdf_videos"
    with ZipFile(celebdf_root / "Celeb-DF-v2.zip") as archive:
        for video in selected:
            target = videos / video["archive_path"]
            member = archive.getinfo(video["archive_path"])
            if target.is_file() and target.stat().st_size == member.file_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".part")
            temporary.write_bytes(archive.read(member))
            temporary.replace(target)
    by_id = {video["video_id"]: video for video in selected}
    artifact = crops.build_crops(
        [(video["video_id"], videos / video["archive_path"]) for video in selected],
        downloads_root / "celebdf_crops", frames=FRAMES_PER_VIDEO, workers=workers,
    )
    rows = []
    excluded: Counter[str] = Counter()
    for crop in crops.read_crops(artifact):
        video = by_id[crop["media_id"]]
        if crop["status"] != "ok":
            excluded[f"{video['label']}/{crop['reason']}"] += 1
            continue
        tokens = tuple(video["tokens"].split("|")) if video["tokens"] else ()
        # YouTube-real clips carry no identity labels; each clip is its own unit.
        entities = tokens or (f"cdf-youtube:{video['name']}",)
        rows.append(_row(
            domain="celebdf_video",
            method="celeb_synthesis" if video["label"] == "FAKE" else "",
            family=FS, split=SPLIT, video=_safe(video["name"], "video name"),
            frame=PurePosixPath(crop["crop_path"]).name, entities=entities,
            archive=f"dir:celebdf_crops/{artifact.name}", member=crop["crop_path"],
            preprocess="copy",
            identity="|".join(entity.split(":", 1)[1] for entity in entities),
        ))
    return rows, {"crop_artifact": artifact.name, "videos": dict(Counter(
        video["label"] for video in selected)), "excluded_frames": dict(excluded)}


def _dff_rows(downloads_root: Path, seed: str) -> list[dict[str, str]]:
    """The same source photos for the reals and every fake method.

    Each DeepFakeFace fake keeps the path of the IMDB-WIKI photo it derives from
    (``<method>/61/19242061_1982-08-16_2009.jpg``), so one ranked set of paths
    gives matched real and fake images of the same people.
    """
    listings = {archive: {PurePosixPath(*PurePosixPath(member).parts[1:]).as_posix(): member
                          for member in _image_members(downloads_root / "DeepFakeFace" / archive)}
                for archive in DFF_ARCHIVES}
    shared = set.intersection(*(set(listing) for listing in listings.values()))
    chosen = sorted(shared, key=lambda name: _rank(seed, "dff", name))[:DFF_PER_CLASS]
    rows = []
    for archive, (method, family) in DFF_ARCHIVES.items():
        for name in chosen:
            member = listings[archive][name]
            parts = PurePosixPath(member).parts
            stem = PurePosixPath(member).stem
            # IMDB-WIKI names begin with the person ID; fakes keep their source name.
            person = stem.split("_")[0]
            rows.append(_row(
                domain="dff", method=method, family=family, split=SPLIT,
                video=_safe(f"{method or 'real'}-{'-'.join(parts[-2:-1])}-{stem}", "video"),
                frame=parts[-1], entities=(f"dff:{person}",),
                archive=f"DeepFakeFace/{archive}", member=member,
                preprocess="canonical:dff", identity=f"dff-{person}",
            ))
    return rows


def _sfhq_rows(downloads_root: Path) -> list[dict[str, str]]:
    rows = []
    for member in _image_members(downloads_root / SFHQ_ARCHIVE):
        stem = PurePosixPath(member).stem
        match = SFHQ_GENERATOR.match(stem)
        if not match:
            raise ValueError(f"cannot name the generator of {member}")
        method = match.group(1).casefold()
        rows.append(_row(
            domain="sfhq_t2i", method=method, family=EFS, split=SPLIT, video=f"fake-{stem}",
            frame=PurePosixPath(member).name, entities=(f"sfhq:{stem}",),
            archive=SFHQ_ARCHIVE, member=member, preprocess="canonical:sfhq_t2i",
            identity=f"sfhq-{stem}",
        ))
    folder = downloads_root / FFHQ_FOLDER.removeprefix("dir:")
    for path in sorted(path for path in folder.rglob("*")
                       if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")):
        member = path.relative_to(folder).as_posix()
        rows.append(_row(
            domain="sfhq_t2i", method="", family=EFS, split=SPLIT, video=f"real-{path.stem}",
            frame=path.name, entities=(f"ffhq:{path.stem}",), archive=FFHQ_FOLDER,
            member=member, preprocess="canonical:sfhq_t2i", identity=f"ffhq-{path.stem}",
        ))
    return rows


def build_catalog(downloads_root: Path, celebdf_root: Path, reference_manifest: Path,
                  output: Path, *, seed: str, workers: int = 12) -> dict:
    celebdf, celebdf_summary = _celebdf_rows(downloads_root, celebdf_root,
                                             reference_manifest, seed, workers)
    rows = celebdf + _dff_rows(downloads_root, seed) + _sfhq_rows(downloads_root)
    rows.sort(key=lambda row: row["relative_path"])
    if len({row["relative_path"] for row in rows}) != len(rows):
        raise ValueError("catalog produced duplicate destination paths")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter((row["source_domain"], row["fake_method"] or "real") for row in rows)
    return {
        "catalog": str(output),
        "rows": len(rows),
        "seed": seed,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "counts": {"/".join(key): value for key, value in sorted(counts.items())},
        "celebdf": celebdf_summary,
    }


def drop_reference_matches(catalog: Path, manifest_dir: Path, reference_dir: Path,
                           output: Path) -> dict:
    """Remove rows whose file, pixel, or perceptual hash occurs in the reference manifest.

    Individual images are removed rather than whole videos: the reference is
    never scored together with this set, so only duplicated content matters.
    """
    from .manifest import verify_manifest
    from .prune import FINGERPRINTS

    for directory in (manifest_dir, reference_dir):
        valid, errors = verify_manifest(directory)
        if not valid:
            raise ValueError(f"invalid manifest {directory}: {errors}")
    reference: dict[str, set[str]] = defaultdict(set)
    with (reference_dir / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            for column in FINGERPRINTS:
                reference[column].add(row[column])
    with (manifest_dir / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        manifest = {row["source_id"]: row for row in csv.DictReader(stream)}
    with catalog.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if {row["image_id"] for row in rows} != set(manifest):
        raise ValueError("manifest does not describe this catalog")
    removed: Counter[str] = Counter()
    kept = []
    for row in rows:
        fingerprints = manifest[row["image_id"]]
        hits = [column for column in FINGERPRINTS if fingerprints[column] in reference[column]]
        if hits:
            removed[f"{row['source_domain']}/{row['fake_method'] or 'real'}/{hits[0]}"] += 1
            continue
        kept.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(kept)
    return {
        "catalog": str(output),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "rows": len(kept),
        "removed": dict(sorted(removed.items())),
        "reference_manifest_id": reference_dir.name,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    catalog = commands.add_parser("catalog")
    catalog.add_argument("--downloads-root", type=Path, required=True)
    catalog.add_argument("--celebdf-root", type=Path, required=True)
    catalog.add_argument("--reference-manifest", type=Path, required=True)
    catalog.add_argument("--output", type=Path, required=True)
    catalog.add_argument("--seed", required=True)
    catalog.add_argument("--workers", type=int, default=12)
    deduplicate = commands.add_parser("drop-reference-matches")
    deduplicate.add_argument("--catalog", type=Path, required=True)
    deduplicate.add_argument("--manifest-dir", type=Path, required=True)
    deduplicate.add_argument("--reference-dir", type=Path, required=True)
    deduplicate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "catalog":
        result = build_catalog(args.downloads_root, args.celebdf_root, args.reference_manifest,
                               args.output, seed=args.seed, workers=args.workers)
    else:
        result = drop_reference_matches(args.catalog, args.manifest_dir, args.reference_dir,
                                        args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
