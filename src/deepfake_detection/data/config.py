"""Typed configuration for dataset manifests and audits."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
)


@dataclass(frozen=True)
class QualityGates:
    """Explicit conditions that make an audit fail."""

    max_missing_images: int = 0
    max_decode_errors: int = 0
    max_ambiguous_paths: int = 0
    max_cross_split_exact_groups: int = 0
    max_cross_split_pixel_groups: int = 0
    max_cross_split_perceptual_groups: int = 0
    max_group_overlap_values: int = 0
    fail_on_unexpected_labels: bool = True
    fail_on_unexpected_splits: bool = True
    fail_on_suspicious_leakage: bool = False
    leakage_purity_threshold: float = 0.98

    def __post_init__(self) -> None:
        if not 0.0 <= self.leakage_purity_threshold <= 1.0:
            raise ValueError("leakage_purity_threshold must be between 0 and 1")


@dataclass(frozen=True)
class DatasetSpec:
    """Normalized dataset configuration with absolute filesystem paths."""

    name: str
    version: str
    project_root: Path
    data_root: Path
    metadata_path: Path
    output_root: Path
    report_root: Path
    id_column: str
    label_column: str
    split_column: str
    path_column: str | None = None
    path_template: str | None = None
    lookup_by_stem_column: str | None = None
    image_glob: str = "images/**/*"
    metadata_columns: tuple[str, ...] = ()
    group_columns: tuple[str, ...] = ()
    leakage_check_columns: tuple[str, ...] = ()
    expected_labels: tuple[str, ...] = ()
    expected_splits: tuple[str, ...] = ()
    image_extensions: tuple[str, ...] = DEFAULT_IMAGE_EXTENSIONS
    workers: int = 4
    quality_gates: QualityGates = field(default_factory=QualityGates)
    config_path: Path | None = None

    def __post_init__(self) -> None:
        selectors = (self.path_column, self.path_template, self.lookup_by_stem_column)
        if sum(value is not None for value in selectors) != 1:
            raise ValueError(
                "configure exactly one of path_column, path_template, or "
                "lookup_by_stem_column"
            )
        if not self.name.strip():
            raise ValueError("dataset name must not be empty")
        if self.workers < 1:
            raise ValueError("workers must be at least 1")

    @property
    def required_columns(self) -> tuple[str, ...]:
        columns = {
            self.id_column,
            self.label_column,
            self.split_column,
            *self.metadata_columns,
            *self.group_columns,
            *self.leakage_check_columns,
        }
        if self.path_column:
            columns.add(self.path_column)
        if self.lookup_by_stem_column:
            columns.add(self.lookup_by_stem_column)
        return tuple(sorted(columns))

    def fingerprint(self) -> str:
        """Hash the semantic config while excluding machine-specific absolute paths."""

        payload = {
            "name": self.name,
            "version": self.version,
            "id_column": self.id_column,
            "label_column": self.label_column,
            "split_column": self.split_column,
            "path_column": self.path_column,
            "path_template": self.path_template,
            "lookup_by_stem_column": self.lookup_by_stem_column,
            "image_glob": self.image_glob,
            "metadata_columns": self.metadata_columns,
            "group_columns": self.group_columns,
            "leakage_check_columns": self.leakage_check_columns,
            "expected_labels": self.expected_labels,
            "expected_splits": self.expected_splits,
            "image_extensions": self.image_extensions,
            "quality_gates": asdict(self.quality_gates),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def _project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd().resolve()


def _expand_path(value: str, base: Path) -> Path:
    expanded = os.path.expanduser(os.path.expandvars(value))
    path = Path(expanded)
    return (path if path.is_absolute() else base / path).resolve()


def _tuple_of_strings(mapping: dict[str, Any], key: str) -> tuple[str, ...]:
    value = mapping.get(key, [])
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"{key} must be a list of strings")
    return tuple(value)


def load_dataset_spec(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> DatasetSpec:
    """Load and validate a YAML dataset specification."""

    resolved_config = Path(config_path).resolve()
    with resolved_config.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)

    if not isinstance(raw, dict):
        raise TypeError("dataset config must be a YAML mapping")
    dataset = raw.get("dataset", raw)
    if not isinstance(dataset, dict):
        raise TypeError("the dataset section must be a mapping")

    root = (
        Path(project_root).resolve()
        if project_root is not None
        else _project_root(resolved_config.parent)
    )
    data_root = _expand_path(str(dataset["data_root"]), root)
    metadata_value = str(dataset["metadata_path"])
    metadata_path = _expand_path(metadata_value, data_root)
    output_root = _expand_path(str(dataset.get("output_root", "data/manifests")), root)
    report_root = _expand_path(str(dataset.get("report_root", "reports/data_audit")), root)

    gates_raw = dataset.get("quality_gates", {}) or {}
    if not isinstance(gates_raw, dict):
        raise TypeError("quality_gates must be a mapping")

    extensions = _tuple_of_strings(dataset, "image_extensions")
    if not extensions:
        extensions = DEFAULT_IMAGE_EXTENSIONS
    extensions = tuple(
        extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        for extension in extensions
    )

    return DatasetSpec(
        name=str(dataset["name"]),
        version=str(dataset.get("version", "unversioned")),
        project_root=root,
        data_root=data_root,
        metadata_path=metadata_path,
        output_root=output_root,
        report_root=report_root,
        id_column=str(dataset["id_column"]),
        label_column=str(dataset["label_column"]),
        split_column=str(dataset["split_column"]),
        path_column=dataset.get("path_column"),
        path_template=dataset.get("path_template"),
        lookup_by_stem_column=dataset.get("lookup_by_stem_column"),
        image_glob=str(dataset.get("image_glob", "images/**/*")),
        metadata_columns=_tuple_of_strings(dataset, "metadata_columns"),
        group_columns=_tuple_of_strings(dataset, "group_columns"),
        leakage_check_columns=_tuple_of_strings(dataset, "leakage_check_columns"),
        expected_labels=_tuple_of_strings(dataset, "expected_labels"),
        expected_splits=_tuple_of_strings(dataset, "expected_splits"),
        image_extensions=extensions,
        workers=int(dataset.get("workers", 4)),
        quality_gates=QualityGates(**gates_raw),
        config_path=resolved_config,
    )
