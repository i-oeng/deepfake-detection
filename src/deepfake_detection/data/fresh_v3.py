"""Fresh open-world test set (v3) for round 3, from sources no earlier split used.

Fresh v2 was scored once at the end of round 2 and then inspected for the
real-source diagnosis, so it no longer measures round 3 cleanly. This set does:

- ``openfake``: OpenFakeTiny (all three published splits; none trains here). Real
  LAION, Pexels, ImageNet, and DOCCI photos against images from current
  generators (FLUX.1 and FLUX.2, SDXL and its fine-tunes, SD 1.5, SD 3.5,
  Midjourney 6 and 7, GPT Image, Imagen, Ideogram, Qwen-Image, Z-Image, video
  models, and others). Generators from a training family are left out: SD 2.1
  and SD-Turbo (DF40 trains on SD 2.1) and a random-weight Sana.
- ``openfake_reddit``: OpenFakeTiny's Reddit split, in-the-wild posts labeled
  real or AI-generated.
- ``lfw``: LFW news photos, one per person and real only. It measures false
  positives on a real source that neither training nor validation has seen.

Every row is a ``test`` row. Both classes of each collection pass through the
canonical face crop, so faces at least 64 source pixels across remain.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path, PurePosixPath

from .openworld import COLUMNS, EFS, _rank, _row

SPLIT = "test"
OPENFAKE_ARCHIVE = "dir:round3_sources/openfake"
LFW_ARCHIVE = "dir:round3_sources/lfw"
OPENFAKE_CORE = ("core-train", "core-validation", "core-test")
OPENFAKE_REDDIT = "reddit-test"
# Same family as a training generator: DF40's SD 2.1, and round 2's Sana.
EXCLUDED_GENERATORS = frozenset({"sd-2.1", "sd-turbo", "tiny-random-sana"})
LFW_PEOPLE = 2000
LFW_NAME = re.compile(r"^(.+)_(\d{4})\.(?:jpg|jpeg|png)$", re.IGNORECASE)


def _records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9.+-]+", "-", value.casefold()).strip("-") or "unknown"


def openfake_rows(downloads_root: Path) -> tuple[list[dict[str, str]], dict]:
    folder = downloads_root / OPENFAKE_ARCHIVE.removeprefix("dir:")
    rows, skipped = [], Counter()
    for record in _records(folder / "metadata.jsonl"):
        split, label = record["split"], record["label"].casefold()
        if label not in {"real", "fake"}:
            raise ValueError(f"unexpected OpenFake label: {record['label']!r}")
        if split in OPENFAKE_CORE:
            domain, method = "openfake", "openfake"
        elif split == OPENFAKE_REDDIT:
            domain, method = "openfake_reddit", "reddit_ai"
        else:
            raise ValueError(f"unexpected OpenFake split: {split}")
        if label == "fake" and record["model"] in EXCLUDED_GENERATORS:
            skipped[record["model"]] += 1
            continue
        index = int(record["index"])
        source = _slug(record["model"])
        entity = f"openfake:{index:06d}"
        rows.append(_row(
            domain=domain, method=method if label == "fake" else "", family=EFS,
            split=SPLIT, video=f"{label}-{source}-{index:06d}",
            frame=PurePosixPath(record["file"]).name, entities=(entity,),
            archive=OPENFAKE_ARCHIVE, member=record["file"],
            preprocess=f"canonical:{domain}", identity=entity,
        ))
    return rows, {"excluded_generators": dict(sorted(skipped.items()))}


def lfw_rows(downloads_root: Path, seed: str) -> list[dict[str, str]]:
    """The first photo of each of ``LFW_PEOPLE`` hash-ranked people."""
    folder = downloads_root / LFW_ARCHIVE.removeprefix("dir:")
    first: dict[str, tuple[int, dict]] = {}
    for record in _records(folder / "metadata.jsonl"):
        match = LFW_NAME.match(record["filename"])
        if not match:
            raise ValueError(f"cannot name the person in {record['filename']}")
        person, number = match.group(1), int(match.group(2))
        if person not in first or number < first[person][0]:
            first[person] = (number, record)
    people = sorted(first, key=lambda person: _rank(seed, "lfw", person))[:LFW_PEOPLE]
    rows = []
    for person in people:
        record = first[person][1]
        entity = f"lfw:{person}"
        rows.append(_row(
            domain="lfw", method="", family=EFS, split=SPLIT,
            video=f"real-{_slug(person)}", frame=PurePosixPath(record["file"]).name,
            entities=(entity,), archive=LFW_ARCHIVE, member=record["file"],
            preprocess="canonical:lfw", identity=entity,
        ))
    return rows


def build_catalog(downloads_root: Path, output: Path, *, seed: str) -> dict:
    openfake, summary = openfake_rows(downloads_root)
    rows = openfake + lfw_rows(downloads_root, seed)
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
        "catalog": str(output), "rows": len(rows), "seed": seed,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "counts": {"/".join(key): value for key, value in sorted(counts.items())},
        **summary,
    }


def generator_of(video_id: str) -> str:
    """The generator (or real source) slug that ``openfake_rows`` put in a video ID."""
    name = video_id.split(":", 1)[1]
    return name.split("-", 1)[1].rsplit("-", 1)[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    catalog = commands.add_parser("catalog")
    catalog.add_argument("--downloads-root", type=Path, required=True)
    catalog.add_argument("--output", type=Path, required=True)
    catalog.add_argument("--seed", required=True)
    args = parser.parse_args(argv)
    result = build_catalog(args.downloads_root, args.output, seed=args.seed)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
