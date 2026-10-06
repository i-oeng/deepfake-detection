"""Extract only a verified DF40 subset from the authors' processed ZIPs."""

from __future__ import annotations

import csv
import os
import shutil
import tempfile
import zlib
from collections import Counter
from contextlib import ExitStack
from pathlib import Path, PurePosixPath
from zipfile import ZipFile, ZipInfo

from .subset import verify_subset

SOURCE_PREFIX = "deepfakes_detection_datasets/"
FAKE_ARCHIVES = {
    "blendface": "blendface.zip",
    "dit": "DiT.zip",
    "facedancer": "facedancer.zip",
    "fomm": "fomm.zip",
    "mcnet": "mcnet.zip",
    "mraa": "MRAA.zip",
    "rddm": "RDDM.zip",
    "sadtalker": "sadtalker.zip",
    "sd21": "sd2.1.zip",
    "simswap": "simswap.zip",
    "starganv2": "starganv2.zip",
    "stylegan2": "StyleGAN2.zip",
    "stylegan3": "StyleGAN3.zip",
    "uniface": "uniface.zip",
    "wav2lip": "wav2lip.zip",
}
REAL_ARCHIVES = {
    "ff": "real/FaceForensics++_real_data_for_DF40.zip",
    "cdf": "real/Celeb-DF-v2_real_data_for_DF40.zip",
    "celeba": "test/starganv2.zip",
}
REQUIRED_COLUMNS = {
    "image_id", "relative_path", "source_path", "split", "label", "fake_method",
    "source_domain",
}


def _safe_parts(value: str, name: str) -> tuple[str, ...]:
    if not value or "\\" in value or PurePosixPath(value).is_absolute():
        raise ValueError(f"unsafe {name}: {value!r}")
    parts = tuple(value.split("/"))
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"unsafe {name}: {value!r}")
    return parts


def _archive_path(row: dict[str, str], downloads_root: Path) -> Path:
    if row["label"] == "REAL":
        try:
            name = REAL_ARCHIVES[row["source_domain"]]
        except KeyError as error:
            raise ValueError(f"unsupported real domain: {row['source_domain']}") from error
    elif row["label"] == "FAKE":
        try:
            filename = FAKE_ARCHIVES[row["fake_method"]]
        except KeyError as error:
            raise ValueError(f"unsupported fake method: {row['fake_method']}") from error
        if row["split"] not in {"train", "validation", "test"}:
            raise ValueError(f"unsupported split: {row['split']}")
        name = f"{'train' if row['split'] == 'train' else 'test'}/{filename}"
    else:
        raise ValueError(f"unsupported label: {row['label']}")
    return downloads_root / name


def _candidate_members(row: dict[str, str]) -> set[str]:
    source = row["source_path"]
    _safe_parts(source, "source path")
    if not source.startswith(SOURCE_PREFIX):
        raise ValueError(f"unsupported source path: {source}")
    relative = source.removeprefix(SOURCE_PREFIX)
    if row["label"] == "REAL":
        if row["source_domain"] == "celeba":
            relative = relative.removeprefix("DF40/")
        return {relative}

    if relative.startswith("DF40_train/"):
        relative = relative.removeprefix("DF40_train/")
    elif relative.startswith("DF40/"):
        relative = relative.removeprefix("DF40/")
    else:
        raise ValueError(f"unsupported fake source path: {source}")

    # The published method ZIPs use several layouts: some omit the method or
    # domain folder, while synthesis ZIPs rename the source-video folder.
    candidates = {relative, relative.partition("/")[2]}
    for candidate in tuple(candidates):
        for original, archived in (
            ("/YouTube-real/", "/Fake_from_Youtube-real/"),
            ("/Celeb-real/", "/Fake_from_Celeb-real/"),
        ):
            if original in candidate:
                candidates.add(candidate.replace(original, archived, 1))
    for candidate in tuple(candidates):
        for domain in ("ff", "cdf"):
            marker = f"/{domain}/"
            if marker in candidate:
                candidates.add(candidate.replace(marker, "/", 1))
    return {candidate for candidate in candidates if candidate}


def _resolve_member(archive: ZipFile, row: dict[str, str]) -> ZipInfo:
    candidates = _candidate_members(row)
    matches = [archive.getinfo(name) for name in sorted(candidates) if name in archive.NameToInfo]
    if len(matches) != 1 or matches[0].is_dir():
        raise ValueError(
            f"expected one image in {archive.filename} for {row['source_path']}; "
            f"found {len(matches)}"
        )
    if matches[0].file_size == 0:
        raise ValueError(f"empty ZIP member in {archive.filename}: {matches[0].filename}")
    return matches[0]


def find_empty_members(rows: list[dict[str, str]], downloads_root: Path) -> set[str]:
    """Return relative paths whose published archive member has zero bytes.

    DF40 ships a few empty images. Catalog builders drop them before sampling so
    a selected video never fails extraction. Unresolvable rows are left to
    ``materialize_df40``, which rejects them.
    """
    empty: set[str] = set()
    with ExitStack() as stack:
        archives: dict[Path, ZipFile | None] = {}
        for row in rows:
            path = _archive_path(row, downloads_root)
            if path not in archives:
                archives[path] = stack.enter_context(ZipFile(path)) if path.is_file() else None
            archive = archives[path]
            if archive is None:
                continue
            names = [name for name in _candidate_members(row) if name in archive.NameToInfo]
            if len(names) == 1 and archive.getinfo(names[0]).file_size == 0:
                empty.add(row["relative_path"])
    return empty


def _matches_existing(path: Path, member: ZipInfo) -> bool:
    if path.stat().st_size != member.file_size:
        return False
    crc = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            crc = zlib.crc32(chunk, crc)
    return crc == member.CRC


def _current_umask() -> int:
    mask = os.umask(0)
    os.umask(mask)
    return mask


def _extract_one(archive: ZipFile, member: ZipInfo, destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".part", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as output, archive.open(member) as source:
            shutil.copyfileobj(source, output, length=1024 * 1024)
        if temporary.stat().st_size != member.file_size:
            raise OSError(f"incomplete ZIP member: {member.filename}")
        # mkstemp always creates 0600; apply the caller's umask so a shared
        # project group can read extracted images like any other output.
        os.chmod(temporary, 0o666 & ~_current_umask())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return member.file_size


def materialize_df40(
    subset_dir: Path, downloads_root: Path, data_root: Path, *, dry_run: bool = False
) -> dict[str, object]:
    """Resolve every selected image before writing, then extract atomically."""
    valid, errors = verify_subset(subset_dir)
    if not valid:
        raise ValueError("invalid subset artifact: " + "; ".join(errors))
    with (subset_dir / "subset.csv").open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames or ()))
        if missing:
            raise ValueError("subset is missing columns: " + ", ".join(missing))
        rows = list(reader)
    if not rows:
        raise ValueError("subset is empty")

    root = data_root.resolve()
    archives: dict[Path, ZipFile] = {}
    resolved: list[tuple[ZipFile, ZipInfo, Path]] = []
    destinations: set[Path] = set()
    archive_counts: Counter[str] = Counter()
    with ExitStack() as stack:
        for row in rows:
            parts = _safe_parts(row["relative_path"], "destination path")
            if parts[0] != "images":
                raise ValueError(f"destination must be under images/: {row['relative_path']}")
            destination = root.joinpath(*parts).resolve()
            if not destination.is_relative_to(root):
                raise ValueError(f"destination escapes data root: {row['relative_path']}")
            if destination in destinations:
                raise ValueError(f"duplicate destination: {row['relative_path']}")
            destinations.add(destination)
            archive_path = _archive_path(row, downloads_root)
            if archive_path not in archives:
                if not archive_path.is_file():
                    raise FileNotFoundError(f"source archive is missing: {archive_path}")
                archives[archive_path] = stack.enter_context(ZipFile(archive_path))
            archive = archives[archive_path]
            member = _resolve_member(archive, row)
            resolved.append((archive, member, destination))
            archive_counts[str(archive_path.relative_to(downloads_root))] += 1

        written = reused = bytes_written = 0
        if not dry_run:
            for archive, member, destination in resolved:
                if destination.exists():
                    if not _matches_existing(destination, member):
                        raise RuntimeError(f"refusing to overwrite changed image: {destination}")
                    reused += 1
                    continue
                bytes_written += _extract_one(archive, member, destination)
                written += 1
    return {
        "subset_id": subset_dir.name,
        "selected": len(resolved),
        "written": written,
        "reused": reused,
        "bytes_written": bytes_written,
        "dry_run": dry_run,
        "archive_counts": dict(sorted(archive_counts.items())),
    }
