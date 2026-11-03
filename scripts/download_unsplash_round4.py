"""Select and download Unsplash Lite photos with faces: fresh v4's new real source.

No other split uses Unsplash. The Lite dataset (``unsplash-research-dataset-lite-latest.zip``,
dataset 1.4.0, 25,000 photos) ships tables, not images: ``photos.tsv000`` gives each
photo's URL on Unsplash's image CDN. Its terms allow downloading and storing the
photos and internal machine-learning use, and forbid redistributing them, so the
photos and crops stay on xixi like the other data.

``select`` keeps every photo submitted before ``CUTOFF``, before Stable Diffusion's
public release, so no kept photo can come from a modern image generator. Photos are
ordered by a seeded hash of their id into units of ``BATCH``, so a partial download
is a random sample.

``download`` fetches, for each photo, a preview at most ``PREVIEW_SIDE`` px on its
long side (about 25 KB) and runs YuNet on it. Only a photo whose preview shows a
face that could pass the face policy (any face scoring at least ``PREVIEW_SCORE``
with a short side of at least ``PREVIEW_SHORT_SIDE`` source pixels, both looser than
the policy) is fetched at full resolution. There the face policy and storage of
``download_openfake_round4.py`` decide, exactly as for OpenFake (320 px detection,
lossless crops, ``policy.json``). The bare CDN URL serves the full-resolution photo
re-encoded as a JPEG (about quality 89, 4:2:0, metadata stripped); each record keeps
the quality and subsampling found. Records are written per unit, so an interrupted
download resumes; a photo the CDN no longer has is recorded as ``not_found``, and any
other failure stops the unit after ``RETRIES`` attempts so a rerun redoes it.

Run from the repository root with the same ``PYTHONPATH`` and allocator settings as
``download_openfake_round4.py``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
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
except ModuleNotFoundError:  # run as ``python scripts/download_unsplash_round4.py``
    from download_openfake_round4 import (
        MIN_FREE_BYTES,
        _detector,
        prepare_output,
        store_face,
        summarize_download,
        write_records,
    )

CUTOFF = "2022-08-01"  # Stable Diffusion was released on 2022-08-22
SEED = "round4-unsplash-20261103"
BATCH = 200
HOST = "https://images.unsplash.com/"
PREVIEW_SIDE = 320
PREVIEW_QUERY = f"fit=max&w={PREVIEW_SIDE}&h={PREVIEW_SIDE}&fm=jpg&q=90"
PREVIEW_SCORE = 0.6  # the policy needs 0.80 on its own 320 px copy
PREVIEW_SHORT_SIDE = 48.0  # source pixels; the policy needs 64
RETRIES = 4
FIELDS = ("unit", "photo_id", "url", "width", "height", "submitted_at", "photographer",
          "label", "model", "group")


def rank(photo_id: str, seed: str = SEED) -> str:
    return hashlib.sha256(f"{seed}\0{photo_id}".encode()).hexdigest()


def read_photos(path: Path) -> list[dict]:
    """``photos.tsv000``: tab separated, unquoted, with long description fields."""
    csv.field_size_limit(2**31 - 1)
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t", quoting=csv.QUOTE_NONE))


def choose_photos(photos: list[dict], cutoff: str = CUTOFF, seed: str = SEED) -> list[dict]:
    """Photos submitted before ``cutoff`` with a CDN URL and a size, in seeded rank order."""
    kept: dict[str, dict] = {}
    for photo in photos:
        url = photo["photo_image_url"]
        if (photo["photo_submitted_at"] >= cutoff or not url.startswith(HOST)
                or len(url) == len(HOST) or "?" in url
                or not (photo["photo_width"].isdigit() and photo["photo_height"].isdigit())):
            continue
        kept.setdefault(photo["photo_id"], photo)
    ordered = sorted(kept.values(), key=lambda photo: rank(photo["photo_id"], seed))
    return [{"unit": f"unsplash_{index // BATCH:04d}", "photo_id": photo["photo_id"],
             "url": photo["photo_image_url"], "width": int(photo["photo_width"]),
             "height": int(photo["photo_height"]), "submitted_at": photo["photo_submitted_at"],
             "photographer": photo["photographer_username"], "label": "real",
             "model": "unsplash", "group": "unsplash"} for index, photo in enumerate(ordered)]


def select(photos_path: Path, output: Path) -> dict:
    photos = read_photos(photos_path)
    rows = choose_photos(photos)
    output.mkdir(parents=True, exist_ok=True)
    selection = output / "selection.csv"
    with selection.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "source": {"dataset": "Unsplash Lite 1.4.0",
                   "archive": "unsplash-research-dataset-lite-latest.zip",
                   "photos_sha256": hashlib.sha256(photos_path.read_bytes()).hexdigest()},
        "seed": SEED, "cutoff": CUTOFF, "photos": len(photos), "selected": len(rows),
        "photographers": len({row["photographer"] for row in rows}),
        "units": len({row["unit"] for row in rows}),
        "selection_sha256": hashlib.sha256(selection.read_bytes()).hexdigest(),
    }
    (output / "selection.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                                           encoding="utf-8")
    return summary


def preview_check(data: bytes, width: int, height: int, detector) -> dict:
    """Whether a preview shows a face that could pass the face policy at full resolution."""
    import numpy as np
    from PIL import Image

    rgb = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    faces = detector.detect(rgb)
    scale = max(width, height) / max(rgb.shape[:2])  # source pixels per preview pixel
    possible = [face for face in faces if face.score >= PREVIEW_SCORE
                and min(face.box[2], face.box[3]) * scale >= PREVIEW_SHORT_SIDE]
    return {"preview_faces": len(faces),
            "preview_best_score": round(max((face.score for face in faces), default=0.0), 4),
            "preview_pass": bool(possible)}


def fetch(client, url: str, retries: int = RETRIES, pause: float = 2.0) -> bytes | None:
    """The response body, or None when the CDN no longer has the photo (404 or 410).

    Transport errors and other statuses are retried with a growing pause and then
    raised, so the unit is not written and a rerun fetches it again.
    """
    import httpx

    for attempt in range(retries):
        try:
            response = client.get(url)
            if response.status_code in (404, 410):
                return None
            response.raise_for_status()
            return response.content
        except httpx.HTTPError:
            if attempt == retries - 1:
                raise
            time.sleep(pause * 2**attempt)
    raise AssertionError("unreachable")


def fetch_unit(client, unit: str, rows: list[dict], output: Path) -> dict:
    records, streamed = [], 0
    for row in rows:
        record = dict(row)
        preview = fetch(client, f"{row['url']}?{PREVIEW_QUERY}")
        if preview is None:
            records.append({**record, "status": "not_found"})
            continue
        streamed += len(preview)
        try:
            record.update(preview_check(preview, int(row["width"]), int(row["height"]),
                                        _detector()))
        except Exception as error:  # any undecodable preview is recorded, not raised
            records.append({**record, "status": "preview_decode_error",
                            "error": type(error).__name__})
            continue
        if not record["preview_pass"]:
            records.append({**record, "status": "preview_rejected"})
            continue
        data = fetch(client, row["url"])
        if data is None:
            records.append({**record, "status": "not_found"})
            continue
        streamed += len(data)
        digest = hashlib.sha256(data).hexdigest()
        record["image_sha256"] = digest
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
    choose.add_argument("--photos", type=Path, required=True,
                        help="photos.tsv000 from the Unsplash Lite archive")
    choose.add_argument("--output", type=Path, default=Path("data/subsets/unsplash_round4"))
    fetch_parser = commands.add_parser("download")
    fetch_parser.add_argument("--selection", type=Path,
                              default=Path("data/subsets/unsplash_round4/selection.csv"))
    fetch_parser.add_argument("--output", type=Path,
                              default=Path("data/raw/unsplash_round4_det320"))
    fetch_parser.add_argument("--workers", type=int, default=8)
    fetch_parser.add_argument("--limit-units", type=int, help="fetch only this many units")
    args = parser.parse_args()
    if args.command == "select":
        print(json.dumps(select(args.photos, args.output), indent=2))
    else:
        print(json.dumps(download(args.selection, args.output, args.workers, args.limit_units),
                         indent=2))


if __name__ == "__main__":
    main()
