"""Deterministic, group-safe subset selection before large dataset downloads."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from deepfake_detection import __version__

from .config import expand_path, find_project_root
from .manifest import read_metadata

SUBSET_SCHEMA_VERSION = 1
SELECTION_COLUMNS = ("selection_group", "selection_role")


@dataclass(frozen=True)
class SplitRule:
    fake_methods: tuple[str, ...]
    max_images_per_method: int
    max_groups_per_method: int
    max_frames_per_group: int

    def __post_init__(self) -> None:
        if not self.fake_methods:
            raise ValueError("each split rule must select at least one fake method")
        if self.max_images_per_method < 1:
            raise ValueError("max_images_per_method must be at least 1")
        if self.max_groups_per_method < 1:
            raise ValueError("max_groups_per_method must be at least 1")
        if self.max_frames_per_group < 1:
            raise ValueError("max_frames_per_group must be at least 1")


@dataclass(frozen=True)
class SubsetSpec:
    name: str
    source_dataset: str
    project_root: Path
    source_metadata: Path
    output_root: Path
    seed: str
    id_column: str
    path_column: str
    label_column: str
    split_column: str
    method_column: str
    family_column: str
    domain_column: str
    group_columns: tuple[str, ...]
    frame_order_column: str | None
    fake_label: str
    real_label: str
    real_to_fake_ratio: float
    require_disjoint_train_test_methods: bool
    strict_quotas: bool
    split_rules: tuple[tuple[str, SplitRule], ...]
    config_path: Path | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("subset name must not be empty")
        if not self.group_columns:
            raise ValueError("at least one group fallback column is required")
        if self.real_to_fake_ratio < 0:
            raise ValueError("real_to_fake_ratio must be non-negative")
        if not self.split_rules:
            raise ValueError("at least one split rule is required")

    @property
    def required_columns(self) -> tuple[str, ...]:
        columns = {
            self.id_column,
            self.path_column,
            self.label_column,
            self.split_column,
            self.method_column,
            self.family_column,
            self.domain_column,
            *self.group_columns,
        }
        if self.frame_order_column:
            columns.add(self.frame_order_column)
        return tuple(sorted(columns))

    @property
    def splits(self) -> dict[str, SplitRule]:
        return dict(self.split_rules)

    def fingerprint(self) -> str:
        payload = {
            "name": self.name,
            "source_dataset": self.source_dataset,
            "seed": self.seed,
            "id_column": self.id_column,
            "path_column": self.path_column,
            "label_column": self.label_column,
            "split_column": self.split_column,
            "method_column": self.method_column,
            "family_column": self.family_column,
            "domain_column": self.domain_column,
            "group_columns": self.group_columns,
            "frame_order_column": self.frame_order_column,
            "fake_label": self.fake_label,
            "real_label": self.real_label,
            "real_to_fake_ratio": self.real_to_fake_ratio,
            "require_disjoint_train_test_methods": self.require_disjoint_train_test_methods,
            "strict_quotas": self.strict_quotas,
            "split_rules": {
                split: asdict(rule) for split, rule in self.split_rules
            },
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SubsetArtifact:
    subset_id: str
    subset_sha256: str
    directory: Path
    subset_path: Path
    metadata_path: Path
    checksums_path: Path
    row_count: int
    reused_existing: bool


def _string_tuple(mapping: dict[str, Any], key: str) -> tuple[str, ...]:
    value = mapping.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"{key} must be a list of strings")
    return tuple(value)


def load_subset_spec(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> SubsetSpec:
    resolved_config = Path(config_path).resolve()
    with resolved_config.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise TypeError("subset config must be a YAML mapping")
    subset = raw.get("subset", raw)
    if not isinstance(subset, dict):
        raise TypeError("the subset section must be a mapping")

    root = (
        Path(project_root).resolve()
        if project_root is not None
        else find_project_root(resolved_config.parent)
    )
    rules_raw = subset.get("splits")
    if not isinstance(rules_raw, dict):
        raise TypeError("splits must be a mapping")
    rules: list[tuple[str, SplitRule]] = []
    for split, rule_raw in rules_raw.items():
        if not isinstance(rule_raw, dict):
            raise TypeError(f"split rule {split!r} must be a mapping")
        rules.append(
            (
                str(split),
                SplitRule(
                    fake_methods=_string_tuple(rule_raw, "fake_methods"),
                    max_images_per_method=int(rule_raw["max_images_per_method"]),
                    max_groups_per_method=int(rule_raw["max_groups_per_method"]),
                    max_frames_per_group=int(rule_raw["max_frames_per_group"]),
                ),
            )
        )

    frame_order = subset.get("frame_order_column")
    return SubsetSpec(
        name=str(subset["name"]),
        source_dataset=str(subset["source_dataset"]),
        project_root=root,
        source_metadata=expand_path(str(subset["source_metadata"]), root),
        output_root=expand_path(str(subset.get("output_root", "data/subsets")), root),
        seed=str(subset.get("seed", "0")),
        id_column=str(subset["id_column"]),
        path_column=str(subset["path_column"]),
        label_column=str(subset["label_column"]),
        split_column=str(subset["split_column"]),
        method_column=str(subset["method_column"]),
        family_column=str(subset["family_column"]),
        domain_column=str(subset["domain_column"]),
        group_columns=_string_tuple(subset, "group_columns"),
        frame_order_column=str(frame_order) if frame_order else None,
        fake_label=str(subset.get("fake_label", "FAKE")),
        real_label=str(subset.get("real_label", "REAL")),
        real_to_fake_ratio=float(subset.get("real_to_fake_ratio", 1.0)),
        require_disjoint_train_test_methods=bool(
            subset.get("require_disjoint_train_test_methods", True)
        ),
        strict_quotas=bool(subset.get("strict_quotas", True)),
        split_rules=tuple(rules),
        config_path=resolved_config,
    )


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _stable_rank(seed: str, *parts: str) -> str:
    payload = "\0".join((seed, *parts)).encode()
    return hashlib.sha256(payload).hexdigest()


def _method_key(value: str) -> str:
    return value.strip().casefold()


def _frame_key(row: pd.Series | dict[str, str], spec: SubsetSpec) -> tuple[Any, ...]:
    raw = str(row.get(spec.frame_order_column, "")) if spec.frame_order_column else ""
    try:
        numeric: Decimal | None = Decimal(raw) if raw else None
    except InvalidOperation:
        numeric = None
    return (
        numeric is None,
        numeric if numeric is not None else Decimal(0),
        raw,
        str(row.get(spec.path_column, "")),
        str(row.get(spec.id_column, "")),
    )


def _even_positions(length: int, limit: int) -> list[int]:
    if length <= limit:
        return list(range(length))
    if limit == 1:
        return [length // 2]
    return [round(index * (length - 1) / (limit - 1)) for index in range(limit)]


def _derive_group(row: pd.Series, spec: SubsetSpec) -> str:
    for column in spec.group_columns:
        value = str(row[column]).strip()
        if value:
            return f"{column}:{value}"
    return f"{spec.id_column}:{str(row[spec.id_column]).strip()}"


def _prepare_catalog(frame: pd.DataFrame, spec: SubsetSpec) -> pd.DataFrame:
    missing = sorted(set(spec.required_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"source metadata is missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("source metadata contains no rows")

    catalog = frame.fillna("").astype(str)
    for column in catalog.columns:
        catalog[column] = catalog[column].str.strip()
    for column in (spec.id_column, spec.path_column):
        empty = int((catalog[column] == "").sum())
        if empty:
            raise ValueError(f"source metadata has {empty} empty values in {column!r}")
        duplicated = catalog[column][catalog[column].duplicated(keep=False)]
        if not duplicated.empty:
            examples = ", ".join(sorted(duplicated.unique())[:5])
            raise ValueError(f"source metadata has duplicate {column!r} values: {examples}")

    catalog = catalog.copy()
    catalog["_selection_group"] = catalog.apply(_derive_group, axis=1, spec=spec)
    overlaps = (
        catalog.groupby("_selection_group", sort=True)[spec.split_column]
        .nunique(dropna=False)
    )
    overlapping = overlaps[overlaps > 1]
    if not overlapping.empty:
        examples = ", ".join(overlapping.index.astype(str).tolist()[:5])
        raise ValueError(f"group values cross source splits: {examples}")
    # The fallback selection group alone is insufficient: a video ID may be
    # unique while the same person appears in another split under another ID.
    for column in spec.group_columns:
        values = catalog.loc[catalog[column] != "", [column, spec.split_column]]
        overlap = values.groupby(column, sort=True)[spec.split_column].nunique()
        repeated = overlap[overlap > 1]
        if not repeated.empty:
            examples = ", ".join(repeated.index.astype(str).tolist()[:5])
            raise ValueError(f"{column} values cross source splits: {examples}")
    return catalog


def _validate_method_protocol(catalog: pd.DataFrame, spec: SubsetSpec) -> None:
    rules = spec.splits
    if spec.require_disjoint_train_test_methods and "train" in rules and "test" in rules:
        train = {_method_key(value) for value in rules["train"].fake_methods}
        test = {_method_key(value) for value in rules["test"].fake_methods}
        overlap = sorted(train & test)
        if overlap:
            raise ValueError("train and test fake methods overlap: " + ", ".join(overlap))

    for split, rule in rules.items():
        available = {
            _method_key(value)
            for value in catalog.loc[
                (catalog[spec.split_column] == split)
                & (catalog[spec.label_column] == spec.fake_label),
                spec.method_column,
            ]
            if value
        }
        missing = [method for method in rule.fake_methods if _method_key(method) not in available]
        if missing:
            raise ValueError(
                f"split {split!r} does not contain configured methods: {', '.join(missing)}"
            )


def _sample_group_frames(
    group: pd.DataFrame,
    spec: SubsetSpec,
    limit: int,
) -> pd.DataFrame:
    ordered_indices = sorted(
        group.index.tolist(),
        key=lambda index: _frame_key(group.loc[index], spec),
    )
    chosen_positions = _even_positions(len(ordered_indices), min(limit, len(ordered_indices)))
    return group.loc[[ordered_indices[position] for position in chosen_positions]].copy()


def _sample_fake_method(
    candidates: pd.DataFrame,
    spec: SubsetSpec,
    split: str,
    method: str,
    rule: SplitRule,
) -> tuple[pd.DataFrame, str | None]:
    groups = sorted(
        candidates["_selection_group"].unique().tolist(),
        key=lambda group: (_stable_rank(spec.seed, split, method, str(group)), str(group)),
    )
    parts = []
    remaining = rule.max_images_per_method
    for group in groups[: rule.max_groups_per_method]:
        if remaining <= 0:
            break
        sampled = _sample_group_frames(
            candidates.loc[candidates["_selection_group"] == group],
            spec,
            min(rule.max_frames_per_group, remaining),
        )
        sampled["selection_role"] = "fake"
        parts.append(sampled)
        remaining -= len(sampled)
    shortage = None
    if remaining:
        selected_count = rule.max_images_per_method - remaining
        shortage = (
            f"{split}/{method}: selected {selected_count} of "
            f"{rule.max_images_per_method} requested images"
        )
    selected = pd.concat(parts, ignore_index=False) if parts else candidates.iloc[0:0].copy()
    return selected, shortage


def _sample_real_domain(
    candidates: pd.DataFrame,
    spec: SubsetSpec,
    split: str,
    domain: str,
    target: int,
    frames_per_group: int,
) -> tuple[pd.DataFrame, int]:
    groups = sorted(
        candidates["_selection_group"].unique().tolist(),
        key=lambda group: (_stable_rank(spec.seed, split, "real", domain, str(group)), str(group)),
    )
    remaining = target
    parts = []
    for group in groups:
        if remaining <= 0:
            break
        sampled = _sample_group_frames(
            candidates.loc[candidates["_selection_group"] == group],
            spec,
            min(frames_per_group, remaining),
        )
        sampled["selection_role"] = "real"
        parts.append(sampled)
        remaining -= len(sampled)
    selected = pd.concat(parts, ignore_index=False) if parts else candidates.iloc[0:0].copy()
    return selected, max(remaining, 0)


def _selection_counts(frame: pd.DataFrame, spec: SubsetSpec) -> list[dict[str, Any]]:
    columns = [spec.split_column, spec.label_column, spec.method_column, spec.domain_column]
    counts = frame.groupby(columns, dropna=False, sort=True).size().reset_index(name="images")
    return [
        {column: str(row[column]) for column in columns} | {"images": int(row["images"])}
        for _, row in counts.iterrows()
    ]


def _write_subset_csv(
    path: Path,
    rows: list[dict[str, str]],
    columns: tuple[str, ...],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _canonical_rows(frame: pd.DataFrame, spec: SubsetSpec) -> list[dict[str, str]]:
    output = frame.drop(columns=["_selection_group"]).rename(
        columns={"selection_group_output": "selection_group"}
    )
    records = [
        {str(key): str(value) for key, value in record.items()}
        for record in output.to_dict(orient="records")
    ]
    split_order = {split: index for index, split in enumerate(spec.splits)}
    records.sort(
        key=lambda row: (
            split_order.get(row[spec.split_column], len(split_order)),
            row[spec.label_column] != spec.fake_label,
            _method_key(row[spec.method_column]),
            row["selection_group"],
            _frame_key(row, spec),
        )
    )
    return records


def _artifact_from_existing(directory: Path, expected_sha256: str) -> SubsetArtifact:
    subset_path = directory / "subset.csv"
    metadata_path = directory / "subset.json"
    checksums_path = directory / "SHA256SUMS"
    if not all(path.is_file() for path in (subset_path, metadata_path, checksums_path)):
        raise RuntimeError(f"incomplete immutable subset artifact exists: {directory}")
    actual = _sha256_file(subset_path)
    if actual != expected_sha256:
        raise RuntimeError(f"refusing to overwrite modified immutable subset: {subset_path}")
    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    return SubsetArtifact(
        subset_id=str(metadata["subset_id"]),
        subset_sha256=actual,
        directory=directory,
        subset_path=subset_path,
        metadata_path=metadata_path,
        checksums_path=checksums_path,
        row_count=int(metadata["row_count"]),
        reused_existing=True,
    )


def build_subset(spec: SubsetSpec) -> SubsetArtifact:
    """Select a deterministic catalog subset without reading image payloads."""

    if not spec.source_metadata.is_file():
        raise FileNotFoundError(f"source metadata not found: {spec.source_metadata}")
    source = read_metadata(spec.source_metadata)
    catalog = _prepare_catalog(source, spec)
    _validate_method_protocol(catalog, spec)

    selected_parts: list[pd.DataFrame] = []
    shortfalls: list[str] = []
    for split, rule in spec.split_rules:
        split_rows = catalog.loc[catalog[spec.split_column] == split]
        fake_parts = []
        for configured_method in rule.fake_methods:
            method_rows = split_rows.loc[
                (split_rows[spec.label_column] == spec.fake_label)
                & (
                    split_rows[spec.method_column].map(_method_key)
                    == _method_key(configured_method)
                )
            ]
            sampled, shortage = _sample_fake_method(
                method_rows,
                spec,
                split,
                configured_method,
                rule,
            )
            if shortage:
                shortfalls.append(shortage)
            fake_parts.append(sampled)
        selected_fake = pd.concat(fake_parts, ignore_index=False)
        selected_parts.append(selected_fake)

        domain_counts = selected_fake[spec.domain_column].value_counts(sort=False).sort_index()
        for domain, fake_count in domain_counts.items():
            target = math.ceil(int(fake_count) * spec.real_to_fake_ratio)
            real_candidates = split_rows.loc[
                (split_rows[spec.label_column] == spec.real_label)
                & (split_rows[spec.domain_column] == domain)
            ]
            sampled_real, remaining = _sample_real_domain(
                real_candidates,
                spec,
                split,
                str(domain),
                target,
                rule.max_frames_per_group,
            )
            if remaining:
                shortfalls.append(
                    f"{split}/REAL/{domain}: missing {remaining} of {target} balanced images"
                )
            selected_parts.append(sampled_real)

    if shortfalls and spec.strict_quotas:
        raise ValueError("subset quotas could not be satisfied: " + "; ".join(shortfalls))

    selected = pd.concat(selected_parts, ignore_index=False)
    if selected.empty:
        raise ValueError("subset selection produced no rows")
    selected = selected.copy()
    selected["selection_group_output"] = selected["_selection_group"]
    duplicate_ids = selected[spec.id_column][selected[spec.id_column].duplicated(keep=False)]
    if not duplicate_ids.empty:
        examples = ", ".join(sorted(duplicate_ids.unique())[:5])
        raise RuntimeError(f"selection contains duplicate sample IDs: {examples}")

    rows = _canonical_rows(selected, spec)
    columns = tuple(column for column in source.columns if column not in SELECTION_COLUMNS)
    columns = (*columns, *SELECTION_COLUMNS)

    staging_root = spec.output_root / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix="subset-", suffix=".csv", dir=staging_root)
    os.close(handle)
    temporary_path = Path(temporary_name)
    try:
        _write_subset_csv(temporary_path, rows, columns)
        subset_sha256 = _sha256_file(temporary_path)
        subset_id = subset_sha256[:20]
        directory = spec.output_root / spec.source_dataset / spec.name / subset_id
        if directory.exists():
            return _artifact_from_existing(directory, subset_sha256)

        directory.parent.mkdir(parents=True, exist_ok=True)
        staging_directory = Path(tempfile.mkdtemp(prefix="artifact-", dir=staging_root))
        try:
            staging_subset = staging_directory / "subset.csv"
            shutil.move(str(temporary_path), staging_subset)
            metadata = {
                "schema_version": SUBSET_SCHEMA_VERSION,
                "builder_version": __version__,
                "purpose": "pre-download deterministic selection",
                "subset_name": spec.name,
                "source_dataset": spec.source_dataset,
                "subset_id": subset_id,
                "subset_sha256": subset_sha256,
                "row_count": len(rows),
                "config_sha256": spec.fingerprint(),
                "source_metadata": _display_path(spec.source_metadata, spec.project_root),
                "source_metadata_sha256": _sha256_file(spec.source_metadata),
                "quota_shortfalls": shortfalls,
                "selection_counts": _selection_counts(selected, spec),
            }
            staging_metadata = staging_directory / "subset.json"
            staging_metadata.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            staging_checksums = staging_directory / "SHA256SUMS"
            staging_checksums.write_text(
                f"{subset_sha256}  subset.csv\n"
                f"{_sha256_file(staging_metadata)}  subset.json\n",
                encoding="utf-8",
            )
            try:
                staging_directory.rename(directory)
            except FileExistsError:
                return _artifact_from_existing(directory, subset_sha256)
        finally:
            if staging_directory.exists():
                shutil.rmtree(staging_directory)

        return SubsetArtifact(
            subset_id=subset_id,
            subset_sha256=subset_sha256,
            directory=directory,
            subset_path=directory / "subset.csv",
            metadata_path=directory / "subset.json",
            checksums_path=directory / "SHA256SUMS",
            row_count=len(rows),
            reused_existing=False,
        )
    finally:
        temporary_path.unlink(missing_ok=True)


def verify_subset(directory: str | Path) -> tuple[bool, list[str]]:
    root = Path(directory)
    errors: list[str] = []
    metadata_path = root / "subset.json"
    checksums_path = root / "SHA256SUMS"
    if not metadata_path.is_file():
        errors.append("subset.json is missing")
    if not checksums_path.is_file():
        errors.append("SHA256SUMS is missing")
    if errors:
        return False, errors

    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    expected: dict[str, str] = {}
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
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

    subset_path = root / "subset.csv"
    if subset_path.is_file():
        actual = _sha256_file(subset_path)
        if actual != metadata.get("subset_sha256"):
            errors.append("subset SHA-256 does not match subset.json")
        if actual[:20] != metadata.get("subset_id"):
            errors.append("subset ID is not derived from subset.csv")
        if root.name != metadata.get("subset_id"):
            errors.append("directory name does not match subset ID")
    return not errors, errors
