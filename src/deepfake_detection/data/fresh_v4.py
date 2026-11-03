"""Fresh open-world test set (v4) for round 4: AI-generated photos of people.

Two slices from sources no earlier split used (``reports/openworld/round4_protocol.md``):

- ``v4_photos_seen`` (primary): faces generated on xixi for this set by
  ``scripts/generate_round4.py`` from three training families (SDXL base 1.0,
  RealVisXL V5.0, SD 3.5 Medium), against every Unsplash Lite face and Open
  Images faces up to ``SEEN_REALS``.
- ``v4_photos_unseen`` (secondary): OpenFake's ``core/test`` faces, whose
  generators never train, against its own ImageNet and DOCCI faces topped up with
  the next Open Images faces to ``UNSEEN_REALS``.

The real crops and OpenFake faces come from the round 4 downloads
(``scripts/download_*_round4.py``), which already applied the face policy and
keep each source's format, JPEG quality and chroma subsampling. Open Images faces
are taken in the order of a seeded hash of their SHA-256.

``v4_photos_seen`` is format matched (``data.format_match``): the generated PNGs
are JPEG-encoded with the reals' quality and subsampling pairs, so the primary
slice cannot be solved by source format. ``v4_photos_unseen`` keeps its sources
as distributed, so the source-format gap (PNG-source against JPEG-source fakes)
measures how far a model relies on format.

Every row is a ``test`` row; ``openworld materialize --size-at-original-resolution
--encodings`` then canonicalizes each slice as one collection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path, PurePosixPath

from .format_match import Source, assign, write_encodings
from .openworld import COLUMNS, EFS, _rank, _row

SPLIT = "test"
SEED = "fresh-v4-20261103"
SEEN, UNSEEN = "v4_photos_seen", "v4_photos_unseen"
GENERATED_ARCHIVE = "dir:v4_photos_seen_generated"
OPENFAKE_TEST_ARCHIVE = "dir:openfake_round4_test_det320"
UNSPLASH_ARCHIVE = "dir:unsplash_round4_det320"
OPENIMAGES_ARCHIVE = "dir:openimages_round4_det320"
SEEN_GENERATORS = ("sdxl_base", "realvisxl_v5", "sd35_medium")
SEEN_REALS = 1000
UNSEEN_REALS = 1000


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9.+]+", "_", value.casefold()).strip("_") or "unknown"


def _folder(downloads_root: Path, archive: str) -> Path:
    return downloads_root / archive.removeprefix("dir:")


def stored_records(folder: Path) -> list[dict]:
    """The stored faces of one round 4 download, in a stable order."""
    records = []
    for path in sorted((folder / "rows").glob("*.jsonl")):
        with path.open(encoding="utf-8") as stream:
            records += [json.loads(line) for line in stream if line.strip()]
    stored = [record for record in records if record["status"] == "stored"]
    for record in stored:
        relative = PurePosixPath(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe stored path in {folder}: {relative}")
    hashes = Counter(record["image_sha256"] for record in stored)
    duplicated = [digest for digest, count in hashes.items() if count > 1]
    if duplicated:
        raise ValueError(f"{len(duplicated)} images stored twice in {folder}")
    return sorted(stored, key=lambda record: record["image_sha256"])


def _face_row(slice_name: str, label: str, source: str, record: dict,
              archive: str) -> dict[str, str]:
    key = record["image_sha256"][:16]
    entity = f"{slice_name}:{source}:{key}"
    return _row(domain=slice_name, method=slice_name if label == "fake" else "", family=EFS,
                split=SPLIT, video=f"{label}-{source}-{key}",
                frame=PurePosixPath(record["path"]).name, entities=(entity,),
                archive=archive, member=record["path"], preprocess=f"canonical:{slice_name}",
                identity=entity)


def _source(row: dict[str, str], record: dict) -> Source:
    return Source(row["image_id"], row["preprocess"], row["label"], record["image_sha256"],
                  record.get("format") or "", record.get("jpeg_quality"),
                  record.get("subsampling"))


def generated_records(downloads_root: Path, generators=SEEN_GENERATORS) -> dict[str, list]:
    """Each generator's finished images as records like the downloads' (format PNG)."""
    result = {}
    for generator in generators:
        folder = _folder(downloads_root, GENERATED_ARCHIVE) / generator
        with (folder / "metadata.jsonl").open(encoding="utf-8") as stream:
            metadata = [json.loads(line) for line in stream if line.strip()]
        records = []
        for entry in metadata:
            if entry["generator"] != generator:
                raise ValueError(f"{folder}: metadata names {entry['generator']}")
            image = folder / f"{int(entry['index']):05d}.png"
            if not image.is_file():
                raise FileNotFoundError(f"metadata lists a missing image: {image}")
            records.append({"path": f"{generator}/{image.name}", "format": "PNG",
                            "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest()})
        if len(records) != len(list(folder.glob("*.png"))):
            raise ValueError(f"{folder}: images without metadata")
        if not records:
            raise ValueError(f"{folder}: no generated images")
        result[generator] = records
    return result


def build_catalog(downloads_root: Path, output: Path, encodings: Path, *, seed: str = SEED,
                  generators=SEEN_GENERATORS) -> dict:
    rows: list[dict[str, str]] = []
    sources: list[Source] = []

    def add(slice_name: str, label: str, source: str, record: dict, archive: str) -> None:
        row = _face_row(slice_name, label, source, record, archive)
        rows.append(row)
        if slice_name == SEEN:
            sources.append(_source(row, record))

    for generator, records in generated_records(downloads_root, generators).items():
        for record in records:
            add(SEEN, "fake", _slug(generator), record, GENERATED_ARCHIVE)
    unsplash = stored_records(_folder(downloads_root, UNSPLASH_ARCHIVE))
    openimages = sorted(stored_records(_folder(downloads_root, OPENIMAGES_ARCHIVE)),
                        key=lambda record: _rank(seed, "openimages", record["image_sha256"]))
    seen_fill = max(0, SEEN_REALS - len(unsplash))
    for record in unsplash[:SEEN_REALS]:
        add(SEEN, "real", "unsplash", record, UNSPLASH_ARCHIVE)
    for record in openimages[:seen_fill]:
        add(SEEN, "real", "openimages", record, OPENIMAGES_ARCHIVE)
    openfake = stored_records(_folder(downloads_root, OPENFAKE_TEST_ARCHIVE))
    for record in openfake:
        add(UNSEEN, record["label"], _slug(record["model"]), record, OPENFAKE_TEST_ARCHIVE)
    openfake_reals = sum(record["label"] == "real" for record in openfake)
    topup = openimages[seen_fill:seen_fill + max(0, UNSEEN_REALS - openfake_reals)]
    for record in topup:
        add(UNSEEN, "real", "openimages", record, OPENIMAGES_ARCHIVE)
    if len({row["relative_path"] for row in rows}) != len(rows):
        raise ValueError("catalog produced duplicate destination paths")
    rows.sort(key=lambda row: row["relative_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    assignments, matching = assign(sources)
    counts = Counter((row["source_domain"], row["video_id"].split(":", 1)[1].rsplit("-", 1)[0])
                     for row in rows)
    return {
        "catalog": str(output), "rows": len(rows), "seed": seed,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "counts": {"/".join(key): value for key, value in sorted(counts.items())},
        "encodings": str(encodings), "encodings_sha256": write_encodings(encodings, assignments),
        "format_matching": matching,
        "unused_openimages_faces": len(openimages) - seen_fill - len(topup),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    catalog = commands.add_parser("catalog")
    catalog.add_argument("--downloads-root", type=Path, required=True)
    catalog.add_argument("--output", type=Path, required=True)
    catalog.add_argument("--encodings", type=Path, required=True)
    catalog.add_argument("--seed", default=SEED)
    args = parser.parse_args(argv)
    result = build_catalog(args.downloads_root, args.output, args.encodings, seed=args.seed)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
