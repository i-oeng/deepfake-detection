"""Deterministic integrity, leakage, and split audits for canonical manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DatasetSpec

AUDIT_SCHEMA_VERSION = 1


class AuditFailure(RuntimeError):
    """Raised by callers that require every configured quality gate to pass."""


@dataclass(frozen=True)
class AuditArtifact:
    report_path: Path
    passed: bool
    reused_existing: bool


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _counts(series: pd.Series) -> dict[str, int]:
    normalized = series.fillna("").astype(str).replace("", "<MISSING>")
    counts = normalized.value_counts(dropna=False).sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def _cross_table(frame: pd.DataFrame, first: str, second: str) -> dict[str, dict[str, int]]:
    table = pd.crosstab(
        frame[first].fillna("").astype(str).replace("", "<MISSING>"),
        frame[second].fillna("").astype(str).replace("", "<MISSING>"),
        dropna=False,
    )
    table = table.sort_index().sort_index(axis=1)
    return {
        str(index): {str(column): int(value) for column, value in row.items()}
        for index, row in table.iterrows()
    }


def _duplicate_summary(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    usable = frame.loc[frame[column].fillna("").astype(str) != ""].copy()
    groups = []
    for digest, group in usable.groupby(column, sort=True):
        if len(group) < 2:
            continue
        splits = sorted(
            value for value in group["split"].fillna("").astype(str).unique() if value
        )
        groups.append(
            {
                "fingerprint": str(digest),
                "samples": int(len(group)),
                "splits": splits,
                "cross_split": len(splits) > 1,
                "sample_ids": sorted(group["sample_id"].astype(str).tolist())[:10],
                "relative_paths": sorted(group["relative_path"].astype(str).tolist())[:10],
            }
        )
    cross_split = [group for group in groups if group["cross_split"]]
    return {
        "duplicate_groups": len(groups),
        "duplicate_samples": sum(group["samples"] for group in groups),
        "cross_split_groups": len(cross_split),
        "cross_split_samples": sum(group["samples"] for group in cross_split),
        "examples": groups[:20],
    }


def _group_overlap(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    usable = frame.loc[frame[column].fillna("").astype(str) != ""]
    overlapping = []
    for value, group in usable.groupby(column, sort=True):
        splits = sorted(item for item in group["split"].astype(str).unique() if item)
        if len(splits) > 1:
            overlapping.append(
                {
                    "value": str(value),
                    "splits": splits,
                    "samples": int(len(group)),
                }
            )
    return {
        "overlapping_values": len(overlapping),
        "overlapping_samples": sum(group["samples"] for group in overlapping),
        "examples": overlapping[:20],
    }


def _label_purity(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    usable = frame.loc[frame["label"].fillna("").astype(str) != ""].copy()
    if usable.empty:
        return {
            "cardinality": 0,
            "missing": 0,
            "majority_baseline": None,
            "label_purity": None,
            "purity_lift": None,
            "skipped_reason": "no labeled rows",
        }
    values = usable[column].fillna("").astype(str).replace("", "<MISSING>")
    cardinality = int(values.nunique(dropna=False))
    maximum_cardinality = max(50, int(len(usable) * 0.2))
    if cardinality > maximum_cardinality:
        return {
            "cardinality": cardinality,
            "missing": int((values == "<MISSING>").sum()),
            "majority_baseline": None,
            "label_purity": None,
            "purity_lift": None,
            "skipped_reason": "high-cardinality identifier-like column",
        }
    table = pd.crosstab(values, usable["label"].astype(str))
    purity = float(table.max(axis=1).sum() / table.to_numpy().sum())
    majority = float(usable["label"].value_counts(normalize=True).max())
    return {
        "cardinality": cardinality,
        "missing": int((values == "<MISSING>").sum()),
        "majority_baseline": round(majority, 8),
        "label_purity": round(purity, 8),
        "purity_lift": round(purity - majority, 8),
        "skipped_reason": None,
    }


def _gate_results(report: dict[str, Any], spec: DatasetSpec) -> tuple[list[str], list[str]]:
    gates = spec.quality_gates
    failures: list[str] = []
    warnings: list[str] = []
    statuses = report["summary"]["status_counts"]

    missing = sum(
        statuses.get(status, 0)
        for status in ("missing_image", "missing_path_value", "outside_data_root")
    )
    if missing > gates.max_missing_images:
        failures.append(f"missing or unsafe image paths: {missing} > {gates.max_missing_images}")
    decode_errors = statuses.get("decode_error", 0)
    if decode_errors > gates.max_decode_errors:
        failures.append(f"decode errors: {decode_errors} > {gates.max_decode_errors}")
    ambiguous = statuses.get("ambiguous_path", 0)
    if ambiguous > gates.max_ambiguous_paths:
        failures.append(f"ambiguous image paths: {ambiguous} > {gates.max_ambiguous_paths}")

    duplicate_limits = {
        "file_sha256": gates.max_cross_split_exact_groups,
        "pixel_sha256": gates.max_cross_split_pixel_groups,
        "dhash64": gates.max_cross_split_perceptual_groups,
    }
    for column, limit in duplicate_limits.items():
        actual = report["duplicates"][column]["cross_split_groups"]
        if actual > limit:
            failures.append(f"cross-split {column} groups: {actual} > {limit}")

    overlap_count = sum(
        value["overlapping_values"] for value in report["group_overlap"].values()
    )
    if overlap_count > gates.max_group_overlap_values:
        failures.append(
            f"configured group values crossing splits: {overlap_count} > "
            f"{gates.max_group_overlap_values}"
        )

    if report["unexpected_labels"] and gates.fail_on_unexpected_labels:
        failures.append("unexpected labels: " + ", ".join(report["unexpected_labels"]))
    if report["unexpected_splits"] and gates.fail_on_unexpected_splits:
        failures.append("unexpected splits: " + ", ".join(report["unexpected_splits"]))

    for column, result in report["metadata_leakage_indicators"].items():
        purity = result["label_purity"]
        lift = result["purity_lift"]
        if purity is None or lift is None:
            continue
        if purity >= gates.leakage_purity_threshold and lift >= 0.05:
            message = (
                f"metadata column {column!r} nearly determines the label "
                f"(purity={purity:.3f}, lift={lift:.3f})"
            )
            if gates.fail_on_suspicious_leakage:
                failures.append(message)
            else:
                warnings.append(message)
    return sorted(failures), sorted(warnings)


def audit_manifest(manifest_path: str | Path, spec: DatasetSpec) -> dict[str, Any]:
    """Return a deterministic audit report for one canonical manifest."""

    path = Path(manifest_path)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    required = {
        "sample_id",
        "relative_path",
        "label",
        "split",
        "status",
        "file_sha256",
        "pixel_sha256",
        "dhash64",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"manifest is missing required columns: {', '.join(missing)}")

    expected_labels = set(spec.expected_labels)
    actual_labels = set(frame["label"].astype(str).unique())
    expected_splits = set(spec.expected_splits)
    actual_splits = set(frame["split"].astype(str).unique())

    manifest_sha256 = _sha256_file(path)
    audit_identity = (
        f"{AUDIT_SCHEMA_VERSION}\0{manifest_sha256}\0{spec.fingerprint()}".encode()
    )
    audit_id = hashlib.sha256(audit_identity).hexdigest()[:20]
    report: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "dataset_name": spec.name,
        "dataset_version": spec.version,
        "audit_id": audit_id,
        "config_sha256": spec.fingerprint(),
        "manifest_id": manifest_sha256[:20],
        "manifest_sha256": manifest_sha256,
        "summary": {
            "rows": int(len(frame)),
            "status_counts": _counts(frame["status"]),
            "label_counts": _counts(frame["label"]),
            "split_counts": _counts(frame["split"]),
            "label_by_split": _cross_table(frame, "split", "label"),
        },
        "missingness": {
            column: int((frame[column].fillna("").astype(str) == "").sum())
            for column in sorted(frame.columns)
        },
        "duplicates": {
            "relative_path": _duplicate_summary(frame, "relative_path"),
            "file_sha256": _duplicate_summary(frame, "file_sha256"),
            "pixel_sha256": _duplicate_summary(frame, "pixel_sha256"),
            "dhash64": _duplicate_summary(frame, "dhash64"),
        },
        "group_overlap": {
            column: _group_overlap(frame, column) for column in spec.group_columns
        },
        "metadata_leakage_indicators": {
            column: _label_purity(frame, column) for column in spec.leakage_check_columns
        },
        "unexpected_labels": sorted(actual_labels - expected_labels) if expected_labels else [],
        "unexpected_splits": sorted(actual_splits - expected_splits) if expected_splits else [],
    }
    failures, warnings = _gate_results(report, spec)
    report["quality_gates"] = {
        "passed": not failures,
        "failures": failures,
        "warnings": warnings,
    }
    return report


def write_audit_report(
    report: dict[str, Any],
    spec: DatasetSpec,
    output_path: str | Path | None = None,
) -> AuditArtifact:
    """Write a deterministic report once and refuse later mutation."""

    path = (
        Path(output_path)
        if output_path is not None
        else spec.report_root
        / spec.name
        / report["manifest_id"]
        / report["audit_id"]
        / "audit.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    reused = False
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"refusing to overwrite changed immutable audit report: {path}")
        reused = True
    else:
        with path.open("xb") as stream:
            stream.write(payload)
    return AuditArtifact(
        report_path=path,
        passed=bool(report["quality_gates"]["passed"]),
        reused_existing=reused,
    )
