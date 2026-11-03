"""Select and download Open Images photos with large faces: fresh v4's second real source.

No other split uses Open Images. Its validation and test images (41,620 and
125,436 Flickr photos, all CC BY 2.0, uploaded years before modern image
generators) have exhaustive, human-verified boxes. ``select`` reads the box and
image tables (``{split}-annotations-bbox.csv`` from ``openimages/v5`` and
``{split}-images-with-rotation.csv`` from ``openimages/2018_04``) and keeps a photo
when its largest "Human face" box (``FACE_LABEL``), not a group box and not a
depiction, spans at least ``MIN_FACE_SHARE`` of the frame on both sides; photos
that also show a depicted face (a drawing, painting, statue or screen) or need a
rotation are dropped. Photos are ordered by a seeded hash of their id into units of
``BATCH``, so a partial download is a random sample.

``download`` fetches each photo from the CVDF mirror (1024 px on the long side) and
applies the face policy and storage of ``download_openfake_round4.py``, exactly as
for OpenFake. Records are written per unit, so an interrupted download resumes.

Run from the repository root with the same ``PYTHONPATH`` and allocator settings as
``download_openfake_round4.py``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import time
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    from scripts.download_openfake_round4 import (
        MIN_FREE_BYTES,
        _detector,
        prepare_output,
        store_face,
        summarize_download,
        write_records,
    )
    from scripts.download_unsplash_round4 import fetch
except ModuleNotFoundError:  # run as ``python scripts/download_openimages_round4.py``
    from download_openfake_round4 import (
        MIN_FREE_BYTES,
        _detector,
        prepare_output,
        store_face,
        summarize_download,
        write_records,
    )
    from download_unsplash_round4 import fetch

SPLITS = ("validation", "test")
FACE_LABEL = "/m/0dzct"  # "Human face" in oidv7-class-descriptions-boxable.csv
MIN_FACE_SHARE = 0.06  # about 40 px at 1024 px; the face policy then needs 64
SEED = "round4-openimages-20261103"
BATCH = 200
MIRROR = "https://open-images-dataset.s3.amazonaws.com"
FIELDS = ("unit", "image_id", "split", "url", "face_share", "author", "license",
          "original_url", "label", "model", "group")


def rank(image_id: str, seed: str = SEED) -> str:
    return hashlib.sha256(f"{seed}\0{image_id}".encode()).hexdigest()


def face_shares(boxes) -> tuple[dict[str, float], set[str]]:
    """Each image's largest real face (its shorter side, as a share of the frame), and
    the images that also show a depicted face."""
    largest: dict[str, float] = {}
    depicted: set[str] = set()
    for box in boxes:
        if box["LabelName"] != FACE_LABEL:
            continue
        if box["IsDepiction"] == "1":
            depicted.add(box["ImageID"])
            continue
        if box["IsGroupOf"] == "1":
            continue
        share = min(float(box["XMax"]) - float(box["XMin"]),
                    float(box["YMax"]) - float(box["YMin"]))
        largest[box["ImageID"]] = max(share, largest.get(box["ImageID"], 0.0))
    return largest, depicted


def choose_images(images: list[dict], largest: dict[str, float], depicted: set[str],
                  min_share: float = MIN_FACE_SHARE, seed: str = SEED) -> list[dict]:
    """Upright photos with a large real face and no depicted one, in seeded rank order."""
    kept = [image for image in images
            if largest.get(image["ImageID"], 0.0) >= min_share
            and image["ImageID"] not in depicted and image["Rotation"] in ("", "0.0")]
    kept.sort(key=lambda image: rank(image["ImageID"], seed))
    return [{"unit": f"openimages_{index // BATCH:04d}", "image_id": image["ImageID"],
             "split": image["Subset"],
             "url": f"{MIRROR}/{image['Subset']}/{image['ImageID']}.jpg",
             "face_share": round(largest[image["ImageID"]], 4), "author": image["Author"],
             "license": image["License"], "original_url": image["OriginalURL"],
             "label": "real", "model": "openimages", "group": "openimages"}
            for index, image in enumerate(kept)]


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def select(tables: Path, output: Path) -> dict:
    images, largest, depicted, hashes = [], {}, set(), {}
    for split in SPLITS:
        for name in (f"{split}-annotations-bbox.csv", f"{split}-images-with-rotation.csv"):
            hashes[name] = hashlib.sha256((tables / name).read_bytes()).hexdigest()
        shares, shown = face_shares(_read(tables / f"{split}-annotations-bbox.csv"))
        largest.update(shares)
        depicted |= shown
        images += _read(tables / f"{split}-images-with-rotation.csv")
    rows = choose_images(images, largest, depicted)
    output.mkdir(parents=True, exist_ok=True)
    selection = output / "selection.csv"
    with selection.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "source": {"dataset": "Open Images (boxes v5, images 2018_04)", "mirror": MIRROR,
                   "tables_sha256": hashes},
        "seed": SEED, "min_face_share": MIN_FACE_SHARE, "images": len(images),
        "with_real_face": len(largest), "with_depicted_face": len(depicted),
        "selected": dict(sorted(Counter(row["split"] for row in rows).items())),
        "licenses": dict(Counter(row["license"] for row in rows)),
        "units": len({row["unit"] for row in rows}),
        "selection_sha256": hashlib.sha256(selection.read_bytes()).hexdigest(),
    }
    (output / "selection.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                                           encoding="utf-8")
    return summary


def fetch_unit(client, unit: str, rows: list[dict], output: Path) -> dict:
    records, streamed = [], 0
    for row in rows:
        data = fetch(client, row["url"])
        if data is None:
            records.append({**row, "status": "not_found"})
            continue
        streamed += len(data)
        digest = hashlib.sha256(data).hexdigest()
        record = {**row, "image_sha256": digest}
        directory = output / "images" / row["label"] / row["group"] / digest[:2]
        record.update(store_face(data, directory, digest))
        if "path" in record:
            record["path"] = Path(record["path"]).relative_to(output).as_posix()
        records.append(record)
    write_records(output, unit, records)
    return {"records": records, "streamed": streamed}


def download(selection: Path, output: Path, workers: int, limit: int | None) -> dict:
    import httpx

    with selection.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    units: dict = defaultdict(list)
    for row in rows:
        units[row["unit"]].append(row)
    done = prepare_output(output, selection)
    todo = [(unit, members) for unit, members in sorted(units.items())
            if f"{unit}.jsonl" not in done]
    print(f"{len(units) - len(todo)} of {len(units)} units already done", flush=True)
    if limit is not None:
        todo = todo[:limit]
    _detector()  # fetch and verify the pinned YuNet model once, before the threads start
    warnings.filterwarnings("ignore", message="Palette images with Transparency")
    started, streamed, stored, statuses = time.time(), 0, 0, Counter()
    with httpx.Client(timeout=120, follow_redirects=True,
                      limits=httpx.Limits(max_connections=workers * 2)) as client:

        def run(job):
            unit, members = job
            if shutil.disk_usage(output).free < MIN_FREE_BYTES:
                raise RuntimeError(f"less than {MIN_FREE_BYTES / 1e9:.0f} GB free under "
                                   f"{output}; stopped (rerun resumes)")
            return fetch_unit(client, unit, members, output)

        with ThreadPoolExecutor(workers) as pool:
            for count, result in enumerate(pool.map(run, todo), 1):
                streamed += result["streamed"]
                for record in result["records"]:
                    statuses[record["status"]] += 1
                    stored += record.get("bytes", 0)
                rate = streamed / max(1.0, time.time() - started)
                print(f"{count}/{len(todo)} units, {streamed / 1e9:.1f} GB streamed "
                      f"({rate / 1e6:.0f} MB/s), {stored / 1e9:.2f} GB stored; "
                      f"{dict(statuses)}", flush=True)
    return summarize_download(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    choose = commands.add_parser("select")
    choose.add_argument("--tables", type=Path, required=True,
                        help="directory with the box and image tables of both splits")
    choose.add_argument("--output", type=Path, default=Path("data/subsets/openimages_round4"))
    fetch_parser = commands.add_parser("download")
    fetch_parser.add_argument("--selection", type=Path,
                              default=Path("data/subsets/openimages_round4/selection.csv"))
    fetch_parser.add_argument("--output", type=Path,
                              default=Path("data/raw/openimages_round4_det320"))
    fetch_parser.add_argument("--workers", type=int, default=8)
    fetch_parser.add_argument("--limit-units", type=int, help="fetch only this many units")
    args = parser.parse_args()
    if args.command == "select":
        print(json.dumps(select(args.tables, args.output), indent=2))
    else:
        print(json.dumps(download(args.selection, args.output, args.workers, args.limit_units),
                         indent=2))


if __name__ == "__main__":
    main()
