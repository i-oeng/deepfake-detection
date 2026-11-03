"""Select and download open-world round 4's OpenFake training images.

``select`` recomputes a download plan of ``scripts/openfake_metadata.py`` from its
metadata cache, picks the rows inside the planned row groups, and writes
``selection.csv`` with a summary. Within each quota key, rows are taken in the
order of a seeded hash of their image SHA-256, so the choice is deterministic and
independent of row order. Every row in those row groups whose prompt names an LFW
public figure is also kept, for the celebrity collection.

``download`` streams the planned row groups, keeps only the selected rows, checks
each image against its published SHA-256, and applies the inference face policy
(``deepfake_detection.inference.face``: largest face with score at least 0.80 and
short side at least 64 px) the way ``deepfake_detection.data.openworld`` does:
faces are detected on a copy whose long side is ``DETECTION_SIDE`` (320 px), where
YuNet finds faces that fill a large frame, and the size limit is applied in source
pixels. (The first round 4 download detected at full resolution and kept about half
as many Pexels photos.) A passing image is stored as a lossless PNG crop at source
resolution, a square ``CROP_MARGIN`` times the face box, or as its original bytes
when that square would cover most of the image. Each record also keeps the
source's JPEG quality and chroma subsampling, so PNG sources can later be encoded
to match the JPEG ones. Canonicalization (alignment, scale matching, JPEG 90)
happens later, from these files. Each row group's records are written once it
finishes, so an interrupted download resumes; ``policy.json`` stops a resume into
an output written under another face policy.

Run from the repository root with ``PYTHONPATH=.genlibs:src``. Many threads that
allocate and free multi-megabyte buffers otherwise make glibc and Arrow keep freed
memory (xixi reached 30 GB and was killed), so also set ``MALLOC_ARENA_MAX=2
MALLOC_MMAP_THRESHOLD_=1048576 ARROW_DEFAULT_MEMORY_POOL=system``.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import gc
import hashlib
import io
import json
import math
import shutil
import threading
import time
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

try:
    from scripts.openfake_metadata import (
        OPENFAKE,
        RemoteFile,
        _load,
        plan_download,
        plan_quotas,
        row_group_costs,
        training_rows,
        training_units,
    )
except ModuleNotFoundError:  # run as ``python scripts/download_openfake_round4.py``
    from openfake_metadata import (
        OPENFAKE,
        RemoteFile,
        _load,
        plan_download,
        plan_quotas,
        row_group_costs,
        training_rows,
        training_units,
    )

SEED = "round4-openfake-20261102"
CROP_MARGIN = 2.5
WHOLE_IMAGE_SHARE = 0.8
MIN_FREE_BYTES = 10 * 10**9  # stop before the shared disk fills
FIELDS = ("file", "row_group", "position", "image_sha256", "label", "model", "group", "roles",
          "public_figures")
# IJG (libjpeg) luminance quantization table at quality 50, row-major.
IJG_LUMA = (16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
            14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
            18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
            49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99)
SUBSAMPLING = {0: "4:4:4", 1: "4:2:2", 2: "4:2:0"}


def rank(image_sha256: str) -> str:
    return hashlib.sha256(f"{SEED}\0{image_sha256}".encode()).hexdigest()


def choose_rows(candidates, quotas: dict):
    """Take each key's quota of rows in hash-rank order; a key without a quota keeps all."""
    import pandas as pd

    ordered = candidates.assign(rank=candidates["image_sha256"].map(rank)).sort_values(
        ["rank", "file", "row_group", "position"])
    parts = [part.head(quotas.get(key, len(part)))
             for key, part in ordered.groupby("key", sort=True)]
    return pd.concat(parts) if parts else ordered.head(0)


def crop_box(face_box, width: int, height: int, margin: float = CROP_MARGIN) -> tuple:
    """A square of ``margin`` times the face box's longer side, centered on it, clipped."""
    x, y, w, h = face_box
    half = margin * max(w, h) / 2
    cx, cy = x + w / 2, y + h / 2
    return (max(0, math.floor(cx - half)), max(0, math.floor(cy - half)),
            min(width, math.ceil(cx + half)), min(height, math.ceil(cy + half)))


def jpeg_quality(luma_table) -> int:
    """The IJG quality whose luminance table has the closest sum (order does not matter).

    Exact for files written by libjpeg (Pillow, most software); for camera or
    editor tables it is the IJG quality of the same overall coarseness.
    """
    def total(quality: int) -> int:
        scale = 5000 // quality if quality < 50 else 200 - 2 * quality
        return sum(min(255, max(1, (value * scale + 50) // 100)) for value in IJG_LUMA)

    target = sum(luma_table)
    return min(range(1, 101), key=lambda quality: abs(total(quality) - target))


def source_encoding(image) -> dict:
    """JPEG quality and chroma subsampling of a JPEG source; nothing for other formats."""
    from PIL import JpegImagePlugin

    tables = getattr(image, "quantization", None)
    if image.format not in {"JPEG", "MPO"} or not tables:
        return {}
    sampling = JpegImagePlugin.get_sampling(image)
    return {"jpeg_quality": jpeg_quality(tables[0]),
            "subsampling": SUBSAMPLING.get(sampling, "other")}


def face_policy() -> str:
    from deepfake_detection.data.openworld import DETECTION_SIDE
    from deepfake_detection.inference.face import DETECTOR_ID, POLICY_ID

    return f"{DETECTOR_ID}-det{DETECTION_SIDE}-origsize-{POLICY_ID}"


def locate_face(rgb, detector) -> tuple[str, dict]:
    """The inference face policy on a ``DETECTION_SIDE`` copy, measured in source pixels.

    Returns ``found`` or the rejection reason, with the chosen face's box and
    landmarks (or the best score when no face is confident enough).
    """
    import numpy as np

    from deepfake_detection.data.openworld import DETECTION_SIDE
    from deepfake_detection.inference.face import MIN_SCORE, MIN_SHORT_SIDE

    height, width = rgb.shape[:2]
    scale = min(1.0, DETECTION_SIDE / max(width, height))
    if scale < 1.0:
        import cv2

        size = (round(width * scale), round(height * scale))
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    faces = detector.detect(rgb)
    info = {"faces_detected": len(faces), "detection_scale": round(scale, 6)}
    if not faces:
        return "no_face", info
    confident = [face for face in faces if face.score >= MIN_SCORE]
    if not confident:
        return "low_confidence", {**info, "best_score": round(max(f.score for f in faces), 4)}
    face = confident[0]
    box = [value / scale for value in face.box]
    info.update(face_box=[round(v, 2) for v in box], score=round(face.score, 4),
                landmarks=np.round(face.landmarks / scale, 2).tolist())
    if min(box[2], box[3]) < MIN_SHORT_SIDE:
        return "face_too_small", info
    return "found", info


def select(cache: Path, plan: str, output: Path) -> dict:
    paths = sorted(cache.glob("*.parquet"))
    rows = _load(paths, ["label", "model", "image_sha256", "row_group", "person",
                         "public_figures"])
    rows["position"] = rows.groupby(["file", "row_group"], sort=False).cumcount()
    training = training_rows(rows)
    units, available = training_units(training)
    costs = row_group_costs(paths)
    quotas = plan_quotas(available, plan)
    result = plan_download(units, costs, quotas)
    picked = set(result["picked"])

    def in_picked(frame):
        return frame[[(f, int(g)) in picked
                      for f, g in zip(frame["file"], frame["row_group"], strict=True)]]

    quota_rows = choose_rows(in_picked(training), quotas)
    celebrity = in_picked(training_rows(rows.assign(person=rows["public_figures"] != "")))
    selected = {}
    for role, frame in (("quota", quota_rows), ("celebrity", celebrity)):
        for row in frame.itertuples(index=False):
            entry = selected.setdefault(row.image_sha256, {
                "file": row.file, "row_group": int(row.row_group), "position": int(row.position),
                "image_sha256": row.image_sha256, "label": row.label, "model": row.model,
                "group": row.key[1], "roles": [], "public_figures": row.public_figures})
            if role not in entry["roles"]:
                entry["roles"].append(role)
    entries = sorted(selected.values(), key=lambda e: (e["file"], e["row_group"], e["position"]))
    output.mkdir(parents=True, exist_ok=True)
    selection = output / "selection.csv"
    with selection.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, FIELDS, lineterminator="\n")
        writer.writeheader()
        for entry in entries:
            writer.writerow({**entry, "roles": "+".join(entry["roles"])})
    counts = Counter(f"{e['label']}/{e['group']}" for e in entries if "quota" in e["roles"])
    celebrities = Counter(e["label"] for e in entries if "celebrity" in e["roles"])
    summary = {
        "source": {"repo": OPENFAKE[0], "revision": OPENFAKE[1]},
        "plan": plan, "seed": SEED,
        "quotas": {f"{label}/{name}": q for (label, name), q in sorted(quotas.items())},
        "row_groups": len(picked), "streamed_image_gb": round(result["bytes"] / 1e9, 1),
        "selected_rows": len(entries),
        "quota_rows": dict(sorted(counts.items())),
        "celebrity_rows": dict(sorted(celebrities.items())),
        "selection_sha256": hashlib.sha256(selection.read_bytes()).hexdigest(),
    }
    (output / "selection.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                                           encoding="utf-8")
    return summary


_local = threading.local()


def _detector():
    from deepfake_detection.inference.face import FaceDetector

    if not hasattr(_local, "detector"):
        _local.detector = FaceDetector()
    return _local.detector


def _rss_bytes() -> int:
    """This process's resident memory on Linux; 0 where /proc is unavailable."""
    try:
        with open("/proc/self/status", encoding="ascii") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def _release_memory() -> None:
    """Return freed memory to the operating system (Python, Arrow, and glibc)."""
    gc.collect()
    try:
        import pyarrow as pa

        pa.default_memory_pool().release_unused()
    except ImportError:
        pass
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


class PixelBudget:
    """Bounds the megapixels being decoded and cropped at the same time.

    Decoded images (up to 45 MP in OpenFake), not the number of workers, set peak
    memory, so concurrent large images wait. An image larger than the whole
    budget runs alone. A new image also waits while the process holds more than
    ``max_rss`` bytes, releasing freed memory to the operating system meanwhile.
    """

    def __init__(self, megapixels: float, max_rss: int = 16 * 10**9) -> None:
        self.capacity = megapixels
        self.max_rss = max_rss
        self._used = 0.0
        self._released = 0.0
        self._condition = threading.Condition()

    @contextmanager
    def hold(self, megapixels: float):
        with self._condition:
            while self._used and (self._used + megapixels > self.capacity
                                  or _rss_bytes() > self.max_rss):
                if _rss_bytes() > self.max_rss and time.monotonic() - self._released > 5:
                    _release_memory()
                    self._released = time.monotonic()
                self._condition.wait(timeout=1.0)
            self._used += megapixels
        try:
            yield
        finally:
            with self._condition:
                self._used -= megapixels
                self._condition.notify_all()


BUDGET = PixelBudget(120.0)


def store_face(data: bytes, directory: Path, name: str) -> dict:
    """Apply the face policy to one image and store its face region; return the record."""
    import numpy as np
    from PIL import Image

    try:
        image = Image.open(io.BytesIO(data))  # reads the header only
        image_format, (width, height) = image.format, image.size
        record = {"width": width, "height": height, "format": image_format,
                  **source_encoding(image)}
    except Exception as error:  # any undecodable payload is recorded, not raised
        return {"status": "decode_error", "error": type(error).__name__}
    with BUDGET.hold(width * height / 1e6):
        try:
            image = image.convert("RGB")
        except Exception as error:  # any undecodable payload is recorded, not raised
            return {**record, "status": "decode_error", "error": type(error).__name__}
        status, found = locate_face(np.asarray(image), _detector())
        record.update(found)
        if status != "found":
            return {**record, "status": status}
        return _store(image, data, record, directory, name)


def _store(image, data: bytes, record: dict, directory: Path, name: str) -> dict:
    width, height, image_format = record["width"], record["height"], record["format"]
    box = crop_box(record["face_box"], width, height)
    directory.mkdir(parents=True, exist_ok=True)
    if (box[2] - box[0]) * (box[3] - box[1]) >= WHOLE_IMAGE_SHARE * width * height:
        path = directory / f"{name}.{(image_format or 'bin').lower()}"
        path.write_bytes(data)
        box, stored = (0, 0, width, height), "original"
    else:
        path = directory / f"{name}.png"
        image.crop(box).save(path, compress_level=6)
        stored = "crop"
    return {**record, "status": "stored", "stored": stored, "crop_box": list(box),
            "path": path.as_posix(), "bytes": path.stat().st_size}


def fetch_group(client, file: str, group: int, wanted: list[dict], output: Path) -> dict:
    import pyarrow.parquet as pq

    if shutil.disk_usage(output).free < MIN_FREE_BYTES:
        raise RuntimeError(f"less than {MIN_FREE_BYTES / 1e9:.0f} GB free under {output}; "
                           "stopped (rerun resumes)")
    parquet = pq.ParquetFile(RemoteFile(client, *OPENFAKE, file), pre_buffer=True)
    table = parquet.read_row_group(group, columns=["image", "image_sha256"])
    images = table.column("image")
    hashes = table.column("image_sha256").to_pylist()
    records = []
    for row in wanted:
        index = int(row["position"])
        data = images[index].as_py()["bytes"]  # one image at a time, not the whole group
        record = dict(row)
        if hashes[index] != row["image_sha256"]:
            record["status"] = "selection_mismatch"
        elif hashlib.sha256(data).hexdigest() != row["image_sha256"]:
            record["status"] = "hash_mismatch"
        else:
            group_dir = row["group"].replace(":", "_")  # "other:<model>" is not a portable name
            directory = output / "images" / row["label"] / group_dir / row["image_sha256"][:2]
            record.update(store_face(data, directory, row["image_sha256"]))
            if "path" in record:
                record["path"] = Path(record["path"]).relative_to(output).as_posix()
        records.append(record)
    target = output / "rows" / f"{file.replace('/', '__')}__{group:03d}.jsonl"
    partial = target.with_suffix(".partial")
    partial.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records),
                       encoding="utf-8")
    partial.replace(target)
    image_bytes = parquet.metadata.row_group(group).column(0).total_compressed_size
    return {"records": records, "streamed": image_bytes}


def download(selection: Path, output: Path, workers: int, limit: int | None) -> dict:
    import httpx

    with selection.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    groups: dict = defaultdict(list)
    for row in rows:
        groups[(row["file"], int(row["row_group"]))].append(row)
    (output / "rows").mkdir(parents=True, exist_ok=True)
    policy = {"face_policy": face_policy(), "crop_margin": CROP_MARGIN,
              "whole_image_share": WHOLE_IMAGE_SHARE,
              "selection_sha256": hashlib.sha256(selection.read_bytes()).hexdigest()}
    marker = output / "policy.json"
    if marker.exists() and json.loads(marker.read_text(encoding="utf-8")) != policy:
        raise RuntimeError(f"{output} was written under another policy or selection "
                           f"({marker}); use a new output directory")
    if not marker.exists():
        if any((output / "rows").iterdir()):
            raise RuntimeError(f"{output} has records but no {marker.name}; "
                               "use a new output directory")
        marker.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    done = {path.name for path in (output / "rows").glob("*.jsonl")}
    todo = [(file, group, wanted) for (file, group), wanted in sorted(groups.items())
            if f"{file.replace('/', '__')}__{group:03d}.jsonl" not in done]
    print(f"{len(groups) - len(todo)} of {len(groups)} row groups already done", flush=True)
    if limit is not None:
        todo = todo[:limit]
    _detector()  # fetch and verify the pinned YuNet model once, before the threads start
    warnings.filterwarnings("ignore", message="Palette images with Transparency")
    print(f"fetching {len(todo)} row groups", flush=True)
    started, streamed, stored, statuses = time.time(), 0, 0, Counter()
    with httpx.Client(timeout=300, limits=httpx.Limits(max_connections=workers * 2)) as client:
        with ThreadPoolExecutor(workers) as pool:
            jobs = pool.map(lambda job: fetch_group(client, *job, output), todo)
            for count, result in enumerate(jobs, 1):
                streamed += result["streamed"]
                for record in result["records"]:
                    statuses[record["status"]] += 1
                    stored += record.get("bytes", 0)
                if count % 25 == 0 or count == len(todo):
                    rate = streamed / max(1.0, time.time() - started)
                    left = (len(todo) - count) * streamed / count / max(rate, 1.0)
                    print(f"{count}/{len(todo)} row groups, {streamed / 1e9:.1f} GB streamed "
                          f"({rate / 1e6:.0f} MB/s), {stored / 1e9:.2f} GB stored, "
                          f"about {left / 3600:.1f} h left; {dict(statuses)}", flush=True)
    return summarize_download(output)


def summarize_download(output: Path) -> dict:
    by_key: dict = defaultdict(Counter)
    stored_bytes = 0
    for path in sorted((output / "rows").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            by_key[f"{record['label']}/{record['group']}"][record["status"]] += 1
            stored_bytes += record.get("bytes", 0)
    marker = output / "policy.json"
    policy = json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else {}
    summary = {**policy, "stored_gb": round(stored_bytes / 1e9, 2),
               "by_key": {key: dict(counts) for key, counts in sorted(by_key.items())}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                                         encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    choose = commands.add_parser("select")
    choose.add_argument("--cache", type=Path, required=True,
                        help="the openfake metadata cache of scripts/openfake_metadata.py")
    choose.add_argument("--plan", default="B")
    choose.add_argument("--output", type=Path, default=Path("data/subsets/openfake_round4"))
    fetch = commands.add_parser("download")
    fetch.add_argument("--selection", type=Path,
                       default=Path("data/subsets/openfake_round4/selection.csv"))
    fetch.add_argument("--output", type=Path, default=Path("data/raw/openfake_round4_det320"))
    fetch.add_argument("--workers", type=int, default=12)
    fetch.add_argument("--pixel-budget", type=float, default=BUDGET.capacity,
                       help="megapixels decoded and cropped at once")
    fetch.add_argument("--max-rss-gb", type=float, default=BUDGET.max_rss / 1e9,
                       help="admit no new image while the process holds more memory")
    fetch.add_argument("--limit-groups", type=int, help="fetch only this many row groups")
    args = parser.parse_args()
    if args.command == "select":
        print(json.dumps(select(args.cache, args.plan, args.output), indent=2))
    else:
        BUDGET.capacity = args.pixel_budget
        BUDGET.max_rss = int(args.max_rss_gb * 1e9)
        print(json.dumps(download(args.selection, args.output, args.workers,
                                  args.limit_groups), indent=2))


if __name__ == "__main__":
    main()
