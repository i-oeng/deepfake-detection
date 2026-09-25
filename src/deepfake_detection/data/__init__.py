"""Dataset discovery, immutable manifests, and audit reports."""

from .audit import AuditFailure, audit_manifest, write_audit_report
from .config import DatasetSpec, load_dataset_spec
from .manifest import ManifestArtifact, build_manifest, verify_manifest
from .subset import (
    SubsetArtifact,
    SubsetSpec,
    build_subset,
    load_subset_spec,
    verify_subset,
)

__all__ = [
    "AuditFailure",
    "DatasetSpec",
    "ManifestArtifact",
    "SubsetArtifact",
    "SubsetSpec",
    "audit_manifest",
    "build_manifest",
    "build_subset",
    "load_dataset_spec",
    "load_subset_spec",
    "verify_manifest",
    "verify_subset",
    "write_audit_report",
]
