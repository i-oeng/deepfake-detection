"""Build and verify content-addressed dataset manifests."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

from deepfake_detection import __version__

from .config import DatasetSpec

SCHEMA_VERSION = 1
CORE_COLUMNS = (
    "sample_id",
    "dataset_name",
    "dataset_version",
    "source_id",
    "path_query",
    "relative_path",
    "label",
    "split",
    "status",
    "file_size_bytes",
    "width",
    "height",
    "mode",
    "image_format",
    "file_sha256",
    "pixel_sha256",
    "dhash64",
    "error_type",
)


@dataclass(frozen=True)
class ManifestArtifact:
    """Paths and identifiers for one immutable manifest."""

    manifest_id: str
    manifest_sha256: str
    directory: Path
    manifest_path: Path
    metadata_path: Path
    checksums_path: Path
    row_count: int
    reused_existing: bool


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _decoded_pixel_sha256(image: Image.Image) -> str:
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    digest = hashlib.sha256()
    digest.update(rgb.width.to_bytes(8, "big"))
    digest.update(rgb.height.to_bytes(8, "big"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def _dhash64(image: Image.Image) -> str:
    grayscale = ImageOps.exif_transpose(image).convert("L")
    resized = grayscale.resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(resized.tobytes())
    value = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            value = (value << 1) | int(pixels[offset + column] > pixels[offset + column + 1])
    return f"{value:016x}"


def read_metadata(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    if suffix in {".jsonl", ".ndjson"}:
        records = []
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise TypeError(f"JSON line {line_number} is not an object")
                records.append(record)
        return pd.DataFrame.from_records(records).fillna("").astype(str)
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if isinstance(payload, dict) and isinstance(payload.get("records"), list):
            payload = payload["records"]
        elif isinstance(payload, dict):
            payload = [payload]
        if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
            raise TypeError(
                "JSON metadata must be a flat list of objects or contain a records list"
            )
        return pd.DataFrame.from_records(payload).fillna("").astype(str)
    raise ValueError(f"unsupported metadata format: {path.suffix}")


def _validate_metadata(frame: pd.DataFrame, spec: DatasetSpec) -> None:
    missing = sorted(set(spec.required_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"metadata is missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("metadata contains no rows")


def _image_index(spec: DatasetSpec) -> dict[str, list[Path]] | None:
    if spec.lookup_by_stem_column is None:
        return None
    index: dict[str, list[Path]] = {}
    for path in spec.data_root.glob(spec.image_glob):
        if path.is_file() and path.suffix.lower() in spec.image_extensions:
            index.setdefault(path.stem.casefold(), []).append(path)
    for paths in index.values():
        paths.sort(key=lambda value: value.as_posix())
    return index


def _inside_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolve_candidate(
    row: dict[str, str],
    spec: DatasetSpec,
    index: dict[str, list[Path]] | None,
) -> tuple[str, Path | None, str | None]:
    """Return path query, unique path, or a resolution error status."""

    root = spec.data_root.resolve()
    if spec.lookup_by_stem_column:
        key = row.get(spec.lookup_by_stem_column, "").strip()
        query = f"stem:{key}"
        if not key:
            return query, None, "missing_path_value"
        matches = (index or {}).get(key.casefold(), [])
        if not matches:
            return query, None, "missing_image"
        if len(matches) > 1:
            return query, None, "ambiguous_path"
        candidate = matches[0].resolve()
    else:
        if spec.path_column:
            rendered = row.get(spec.path_column, "").strip()
            if not rendered:
                return "", None, "missing_path_value"
        else:
            try:
                rendered = str(spec.path_template).format_map(row)
            except KeyError as error:
                raise ValueError(
                    f"path_template references missing column {error.args[0]!r}"
                ) from error
        query = rendered.replace("\\", "/")
        if any(character in rendered for character in "*?["):
            matches = sorted(
                (
                    path.resolve()
                    for path in root.glob(rendered)
                    if path.is_file() and path.suffix.lower() in spec.image_extensions
                ),
                key=lambda value: value.as_posix(),
            )
            if not matches:
                return query, None, "missing_image"
            if len(matches) > 1:
                return query, None, "ambiguous_path"
            candidate = matches[0]
        else:
            candidate = Path(rendered)
            if not candidate.is_absolute():
                candidate = root / candidate
            candidate = candidate.resolve()

    if not _inside_root(candidate, root):
        return query, None, "outside_data_root"
    if not candidate.is_file():
        return query, None, "missing_image"
    return query, candidate, None


def _stable_sample_id(spec: DatasetSpec, row: dict[str, str], path_query: str) -> str:
    source_id = row.get(spec.id_column, "").strip()
    if source_id:
        identity = source_id
    else:
        canonical_row = json.dumps(row, sort_keys=True, separators=(",", ":"))
        identity = f"{path_query}\0{canonical_row}"
    payload = f"{spec.name}\0{spec.version}\0{identity}".encode()
    return hashlib.sha256(payload).hexdigest()


def _inspect_one(
    row: dict[str, str],
    spec: DatasetSpec,
    index: dict[str, list[Path]] | None,
    carry_columns: tuple[str, ...],
) -> dict[str, str | int]:
    path_query, path, resolution_error = _resolve_candidate(row, spec, index)
    result: dict[str, str | int] = {
        "sample_id": _stable_sample_id(spec, row, path_query),
        "dataset_name": spec.name,
        "dataset_version": spec.version,
        "source_id": row.get(spec.id_column, "").strip(),
        "path_query": path_query,
        "relative_path": "",
        "label": row.get(spec.label_column, "").strip(),
        "split": row.get(spec.split_column, "").strip(),
        "status": resolution_error or "ok",
        "file_size_bytes": "",
        "width": "",
        "height": "",
        "mode": "",
        "image_format": "",
        "file_sha256": "",
        "pixel_sha256": "",
        "dhash64": "",
        "error_type": "",
    }
    for column in carry_columns:
        result[column] = row.get(column, "").strip()

    if path is None:
        return result

    result["relative_path"] = path.relative_to(spec.data_root.resolve()).as_posix()
    result["file_size_bytes"] = path.stat().st_size
    result["file_sha256"] = _sha256_file(path)
    try:
        with Image.open(path) as image:
            result["width"], result["height"] = image.size
            result["mode"] = image.mode
            result["image_format"] = image.format or ""
            image.load()
            result["pixel_sha256"] = _decoded_pixel_sha256(image)
            result["dhash64"] = _dhash64(image)
    except Exception as error:  # Pillow raises format-specific exception types.
        result["status"] = "decode_error"
        result["error_type"] = type(error).__name__
    return result


def _carry_columns(spec: DatasetSpec) -> tuple[str, ...]:
    requested = (*spec.metadata_columns, *spec.group_columns, *spec.leakage_check_columns)
    columns = tuple(dict.fromkeys(requested))
    collisions = sorted(set(columns) & set(CORE_COLUMNS))
    if collisions:
        raise ValueError(
            "metadata columns collide with reserved manifest columns: " + ", ".join(collisions)
        )
    return columns


def _canonical_rows(
    frame: pd.DataFrame,
    spec: DatasetSpec,
    index: dict[str, list[Path]] | None,
    carry_columns: tuple[str, ...],
) -> list[dict[str, str | int]]:
    records = [
        {str(key): "" if value is None else str(value) for key, value in record.items()}
        for record in frame.to_dict(orient="records")
    ]
    inspect = lambda row: _inspect_one(row, spec, index, carry_columns)  # noqa: E731
    with ThreadPoolExecutor(max_workers=spec.workers) as pool:
        rows = list(pool.map(inspect, records))
    rows.sort(
        key=lambda row: (
            str(row["relative_path"]),
            str(row["path_query"]),
            str(row["sample_id"]),
        )
    )
    return rows


def _write_manifest_csv(
    path: Path,
    rows: Iterable[dict[str, str | int]],
    columns: tuple[str, ...],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _artifact_from_existing(directory: Path, expected_sha256: str) -> ManifestArtifact:
    manifest_path = directory / "manifest.csv"
    metadata_path = directory / "manifest.json"
    checksums_path = directory / "SHA256SUMS"
    if not all(path.is_file() for path in (manifest_path, metadata_path, checksums_path)):
        raise RuntimeError(f"incomplete immutable artifact already exists: {directory}")
    actual_sha256 = _sha256_file(manifest_path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(f"refusing to overwrite modified immutable manifest: {manifest_path}")
    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    return ManifestArtifact(
        manifest_id=str(metadata["manifest_id"]),
        manifest_sha256=actual_sha256,
        directory=directory,
        manifest_path=manifest_path,
        metadata_path=metadata_path,
        checksums_path=checksums_path,
        row_count=int(metadata["row_count"]),
        reused_existing=True,
    )


def build_manifest(spec: DatasetSpec) -> ManifestArtifact:
    """Build a deterministic manifest and store it under its content hash."""

    if not spec.metadata_path.is_file():
        raise FileNotFoundError(f"metadata file not found: {spec.metadata_path}")
    if not spec.data_root.is_dir():
        raise FileNotFoundError(f"dataset root not found: {spec.data_root}")

    frame = read_metadata(spec.metadata_path)
    _validate_metadata(frame, spec)
    carry_columns = _carry_columns(spec)
    index = _image_index(spec)
    rows = _canonical_rows(frame, spec, index, carry_columns)
    columns = (*CORE_COLUMNS, *carry_columns)

    staging_root = spec.output_root / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix="manifest-", suffix=".csv", dir=staging_root)
    os.close(handle)
    temporary_path = Path(temporary_name)
    try:
        _write_manifest_csv(temporary_path, rows, columns)
        manifest_sha256 = _sha256_file(temporary_path)
        manifest_id = manifest_sha256[:20]
        directory = spec.output_root / spec.name / manifest_id
        if directory.exists():
            return _artifact_from_existing(directory, manifest_sha256)

        directory.parent.mkdir(parents=True, exist_ok=True)
        staging_directory = Path(tempfile.mkdtemp(prefix="artifact-", dir=staging_root))
        try:
            staging_manifest = staging_directory / "manifest.csv"
            shutil.move(str(temporary_path), staging_manifest)
            metadata = {
                "schema_version": SCHEMA_VERSION,
                "builder_version": __version__,
                "dataset_name": spec.name,
                "dataset_version": spec.version,
                "manifest_id": manifest_id,
                "manifest_sha256": manifest_sha256,
                "row_count": len(rows),
                "columns": list(columns),
                "config_sha256": spec.fingerprint(),
                "source_metadata": _display_path(spec.metadata_path, spec.project_root),
                "source_metadata_sha256": _sha256_file(spec.metadata_path),
            }
            staging_metadata = staging_directory / "manifest.json"
            staging_metadata.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            staging_checksums = staging_directory / "SHA256SUMS"
            staging_checksums.write_text(
                f"{manifest_sha256}  manifest.csv\n"
                f"{_sha256_file(staging_metadata)}  manifest.json\n",
                encoding="utf-8",
            )
            try:
                staging_directory.rename(directory)
            except FileExistsError:
                return _artifact_from_existing(directory, manifest_sha256)
        finally:
            if staging_directory.exists():
                shutil.rmtree(staging_directory)

        manifest_path = directory / "manifest.csv"
        metadata_path = directory / "manifest.json"
        checksums_path = directory / "SHA256SUMS"
        return ManifestArtifact(
            manifest_id=manifest_id,
            manifest_sha256=manifest_sha256,
            directory=directory,
            manifest_path=manifest_path,
            metadata_path=metadata_path,
            checksums_path=checksums_path,
            row_count=len(rows),
            reused_existing=False,
        )
    finally:
        temporary_path.unlink(missing_ok=True)


def verify_manifest(directory: str | Path) -> tuple[bool, list[str]]:
    """Verify all checksums and the manifest ID without changing the artifact."""

    root = Path(directory)
    errors: list[str] = []
    metadata_path = root / "manifest.json"
    sums_path = root / "SHA256SUMS"
    if not metadata_path.is_file():
        errors.append("manifest.json is missing")
    if not sums_path.is_file():
        errors.append("SHA256SUMS is missing")
    if errors:
        return False, errors

    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    expected: dict[str, str] = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        digest, separator, filename = line.partition("  ")
        if not separator:
            errors.append(f"invalid checksum line: {line!r}")
            continue
        expected[filename] = digest

    for filename, digest in expected.items():
        path = root / filename
        if not path.is_file():
            errors.append(f"{filename} is missing")
        elif _sha256_file(path) != digest:
            errors.append(f"checksum mismatch: {filename}")

    manifest_path = root / "manifest.csv"
    if manifest_path.is_file():
        actual = _sha256_file(manifest_path)
        if actual != metadata.get("manifest_sha256"):
            errors.append("manifest SHA-256 does not match manifest.json")
        if actual[:20] != metadata.get("manifest_id"):
            errors.append("manifest ID is not derived from manifest.csv")
        if root.name != metadata.get("manifest_id"):
            errors.append("directory name does not match manifest ID")
    return not errors, errors
