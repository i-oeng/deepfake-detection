"""Select and download open-world round 4's SFHQ-T2I fakes and their FFHQ reals.

``select`` lists ``SyntheticFacesHighQuality.zip`` (22.9 GB) from its central
directory over HTTP range requests, drops every image of ``small-sample.zip``
(fresh v2's ``sfhq_t2i``, now validation), and takes up to ``QUOTA`` images per
generator in the order of a seeded hash of the member name. Reals come from the
same 1024 px FFHQ mirror fresh v2 used: ``FFHQ_SHARDS`` shards of 1,000 images,
chosen by seeded hash among those no split uses.

``download`` fetches each chosen zip member with one range request and checks its
CRC, streams each chosen FFHQ shard, and applies the face policy and storage of
``download_openfake_round4.py`` (320 px detection, lossless crops, ``policy.json``).
An image whose SHA-256 matches a ``small-sample.zip`` image is recorded as
``validation_duplicate`` and not stored. Records are written per unit (``BATCH``
zip members or one shard), so an interrupted download resumes.

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
import struct
import tarfile
import threading
import time
import warnings
import zipfile
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

try:
    from scripts.download_openfake_round4 import (
        MIN_FREE_BYTES,
        _detector,
        prepare_output,
        store_face,
        summarize_download,
        write_records,
    )
    from scripts.openfake_metadata import RemoteFile
except ModuleNotFoundError:  # run as ``python scripts/download_sfhq_t2i_round4.py``
    from download_openfake_round4 import (
        MIN_FREE_BYTES,
        _detector,
        prepare_output,
        store_face,
        summarize_download,
        write_records,
    )
    from openfake_metadata import RemoteFile

SFHQ = ("bitmind/SyntheticFacesHighQuality-T2I", "c7f422bdcaf4ffe1ced6da49198845c874221a03")
ARCHIVE = "SyntheticFacesHighQuality.zip"
FFHQ = ("gaunernst/ffhq-1024-wds", "d74f1f1f59e3bbe975bee29872b9bef827314577")
FFHQ_SHARD_COUNT = 70  # 00000.tar to 69000.tar, 1,000 images each (checked 2026-11-03)
# whichfaceisreal (v1 validation) uses FFHQ 00000 to 01001; fresh v2 uses 35000 and 50000.
FFHQ_USED = frozenset({"00000.tar", "01000.tar", "35000.tar", "50000.tar"})
FFHQ_SHARDS = 9
QUOTA = 2000
SEED = "round4-sfhq-20261103"
BATCH = 200
FIELDS = ("unit", "source", "member", "label", "model", "group")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
LOCAL_HEADER = 0x04034B50


def rank(name: str, seed: str = SEED) -> str:
    return hashlib.sha256(f"{seed}\0{name}".encode()).hexdigest()


def model_of(member: str) -> str:
    """``images/FLUX1_schnell_image_0000001.jpg`` is a FLUX1_schnell image."""
    return PurePosixPath(member).name.split("_image_")[0]


def choose_members(names: list[str], excluded: set[str], quota: int = QUOTA,
                   seed: str = SEED) -> list[str]:
    """Up to ``quota`` images per generator, in seeded rank order, minus ``excluded`` names."""
    by_model: dict = defaultdict(list)
    for name in names:
        if name.lower().endswith(IMAGE_SUFFIXES) and PurePosixPath(name).name not in excluded:
            by_model[model_of(name)].append(name)
    chosen = []
    for model in sorted(by_model):
        chosen += sorted(by_model[model], key=lambda name: rank(name, seed))[:quota]
    return sorted(chosen)


def choose_shards(count: int = FFHQ_SHARDS, used: frozenset = FFHQ_USED,
                  seed: str = SEED) -> list[str]:
    shards = [f"{index * 1000:05d}.tar" for index in range(FFHQ_SHARD_COUNT)]
    return sorted(sorted((s for s in shards if s not in used),
                         key=lambda shard: rank(shard, seed))[:count])


def read_member(raw, info: zipfile.ZipInfo) -> bytes:
    """One zip member's bytes, read with one range request where the header allows it."""
    name_length = len(info.filename.encode("utf-8" if info.flag_bits & 0x800 else "cp437"))
    raw.seek(info.header_offset)
    blob = raw.read(30 + name_length + 512 + info.compress_size)
    if struct.unpack("<I", blob[:4])[0] != LOCAL_HEADER:
        raise ValueError(f"no local header for {info.filename}")
    name_length, extra_length = struct.unpack("<HH", blob[26:30])
    start = 30 + name_length + extra_length
    data = blob[start:start + info.compress_size]
    if len(data) < info.compress_size:  # a long local extra field: read the rest
        raw.seek(info.header_offset + start + len(data))
        data += raw.read(info.compress_size - len(data))
    if info.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(data, -15)
    elif info.compress_type != zipfile.ZIP_STORED:
        raise ValueError(f"unsupported compression {info.compress_type}: {info.filename}")
    if zlib.crc32(data) != info.CRC:
        raise ValueError(f"CRC mismatch: {info.filename}")
    return data


def _archive(client) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BufferedReader(RemoteFile(client, *SFHQ, ARCHIVE),
                                             buffer_size=4 << 20))


def select(small_sample: Path, output: Path) -> dict:
    import httpx

    with zipfile.ZipFile(small_sample) as sample:
        excluded = {PurePosixPath(name).name for name in sample.namelist()
                    if name.lower().endswith(IMAGE_SUFFIXES)}
        hashes = sorted(hashlib.sha256(sample.read(name)).hexdigest()
                        for name in sample.namelist() if name.lower().endswith(IMAGE_SUFFIXES))
    with httpx.Client(timeout=300) as client:
        names = [info.filename for info in _archive(client).infolist() if not info.is_dir()]
    members = choose_members(names, excluded)
    shards = choose_shards()
    rows = [{"unit": f"sfhq_{index // BATCH:04d}", "source": "sfhq", "member": member,
             "label": "fake", "model": model_of(member), "group": model_of(member).lower()}
            for index, member in enumerate(members)]
    rows += [{"unit": f"ffhq_{PurePosixPath(shard).stem}", "source": "ffhq", "member": shard,
              "label": "real", "model": "ffhq", "group": "ffhq"} for shard in shards]
    output.mkdir(parents=True, exist_ok=True)
    selection = output / "selection.csv"
    with selection.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (output / "small_sample_sha256.txt").write_text("\n".join(hashes) + "\n", encoding="ascii")
    summary = {
        "sources": {"sfhq": {"repo": SFHQ[0], "revision": SFHQ[1], "file": ARCHIVE},
                    "ffhq": {"repo": FFHQ[0], "revision": FFHQ[1]}},
        "seed": SEED, "quota": QUOTA, "archive_images": len(names),
        "excluded_small_sample": len(excluded),
        "fakes": dict(sorted(Counter(model_of(m) for m in members).items())),
        "ffhq_shards": shards,
        "selection_sha256": hashlib.sha256(selection.read_bytes()).hexdigest(),
    }
    (output / "selection.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                                           encoding="utf-8")
    return summary


_local = threading.local()


def _record(row: dict, data: bytes, output: Path, validation: set[str]) -> dict:
    digest = hashlib.sha256(data).hexdigest()
    record = {**row, "image_sha256": digest}
    if digest in validation:
        return {**record, "status": "validation_duplicate"}
    directory = output / "images" / row["label"] / row["group"] / digest[:2]
    record.update(store_face(data, directory, digest))
    if "path" in record:
        record["path"] = Path(record["path"]).relative_to(output).as_posix()
    return record


def fetch_sfhq(client, infos: dict, unit: str, rows: list[dict], output: Path,
               validation: set[str]) -> dict:
    if not hasattr(_local, "archive"):
        _local.archive = RemoteFile(client, *SFHQ, ARCHIVE)
    records, streamed = [], 0
    for row in rows:
        info = infos[row["member"]]
        records.append(_record(row, read_member(_local.archive, info), output, validation))
        streamed += info.compress_size
    write_records(output, unit, records)
    return {"records": records, "streamed": streamed}


def fetch_ffhq(client, unit: str, row: dict, output: Path, validation: set[str]) -> dict:
    raw = RemoteFile(client, *FFHQ, row["member"])
    records = []
    with tarfile.open(fileobj=io.BufferedReader(raw, buffer_size=8 << 20), mode="r|") as shard:
        for member in shard:
            if not member.isfile() or not member.name.lower().endswith(IMAGE_SUFFIXES):
                continue
            data = shard.extractfile(member).read()
            entry = {**row, "member": f"{row['member']}/{member.name}"}
            records.append(_record(entry, data, output, validation))
    write_records(output, unit, records)
    return {"records": records, "streamed": raw.size}


def download(selection: Path, output: Path, workers: int, limit: int | None) -> dict:
    import httpx

    with selection.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    hashes = selection.with_name("small_sample_sha256.txt").read_text(encoding="ascii")
    validation = set(hashes.split())
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
    with httpx.Client(timeout=300, limits=httpx.Limits(max_connections=workers * 2)) as client:
        infos = {info.filename: info for info in _archive(client).infolist()}

        def run(job):
            unit, members = job
            if shutil.disk_usage(output).free < MIN_FREE_BYTES:
                raise RuntimeError(f"less than {MIN_FREE_BYTES / 1e9:.0f} GB free under "
                                   f"{output}; stopped (rerun resumes)")
            if members[0]["source"] == "ffhq":
                return fetch_ffhq(client, unit, members[0], output, validation)
            return fetch_sfhq(client, infos, unit, members, output, validation)

        # Shards first: each streams 1 GB, so they should not be the last jobs left.
        todo.sort(key=lambda job: job[1][0]["source"] != "ffhq")
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
    choose.add_argument("--small-sample", type=Path, required=True,
                        help="fresh v2's small-sample.zip, whose images stay in validation")
    choose.add_argument("--output", type=Path, default=Path("data/subsets/sfhq_round4"))
    fetch = commands.add_parser("download")
    fetch.add_argument("--selection", type=Path,
                       default=Path("data/subsets/sfhq_round4/selection.csv"))
    fetch.add_argument("--output", type=Path, default=Path("data/raw/sfhq_round4_det320"))
    fetch.add_argument("--workers", type=int, default=12)
    fetch.add_argument("--limit-units", type=int, help="fetch only this many units")
    args = parser.parse_args()
    if args.command == "select":
        print(json.dumps(select(args.small_sample, args.output), indent=2))
    else:
        print(json.dumps(download(args.selection, args.output, args.workers, args.limit_units),
                         indent=2))


if __name__ == "__main__":
    main()
