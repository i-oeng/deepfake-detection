"""Command-line interface for immutable manifests and dataset audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import audit_manifest, write_audit_report
from .config import load_dataset_spec
from .df40 import normalize_df40
from .manifest import build_manifest, verify_manifest
from .materialize import materialize_df40
from .prune import prune_cross_split
from .subset import build_subset, load_subset_spec, verify_subset


def _build(args: argparse.Namespace) -> int:
    spec = load_dataset_spec(args.config)
    artifact = build_manifest(spec)
    report = audit_manifest(artifact.manifest_path, spec)
    audit_artifact = write_audit_report(report, spec)
    print(
        json.dumps(
            {
                "manifest_id": artifact.manifest_id,
                "manifest": str(artifact.manifest_path),
                "manifest_reused": artifact.reused_existing,
                "rows": artifact.row_count,
                "audit": str(audit_artifact.report_path),
                "audit_passed": audit_artifact.passed,
                "failures": report["quality_gates"]["failures"],
                "warnings": report["quality_gates"]["warnings"],
            },
            indent=2,
        )
    )
    return 0 if audit_artifact.passed or args.allow_audit_failures else 2


def _audit(args: argparse.Namespace) -> int:
    spec = load_dataset_spec(args.config)
    report = audit_manifest(args.manifest, spec)
    artifact = write_audit_report(report, spec, args.output)
    print(
        json.dumps(
            {
                "audit": str(artifact.report_path),
                "audit_passed": artifact.passed,
                "failures": report["quality_gates"]["failures"],
                "warnings": report["quality_gates"]["warnings"],
            },
            indent=2,
        )
    )
    return 0 if artifact.passed or args.allow_failures else 2


def _verify(args: argparse.Namespace) -> int:
    valid, errors = verify_manifest(args.manifest_dir)
    print(json.dumps({"valid": valid, "errors": errors}, indent=2))
    return 0 if valid else 2


def _subset(args: argparse.Namespace) -> int:
    spec = load_subset_spec(args.config)
    artifact = build_subset(spec)
    print(
        json.dumps(
            {
                "subset_id": artifact.subset_id,
                "subset": str(artifact.subset_path),
                "rows": artifact.row_count,
                "reused": artifact.reused_existing,
            },
            indent=2,
        )
    )
    return 0


def _verify_subset(args: argparse.Namespace) -> int:
    valid, errors = verify_subset(args.subset_dir)
    print(json.dumps({"valid": valid, "errors": errors}, indent=2))
    return 0 if valid else 2


def _normalize_df40(args: argparse.Namespace) -> int:
    result = normalize_df40(
        args.input_dir, args.output, eval_split_seed=args.eval_split_seed
    )
    print(json.dumps(result, indent=2))
    return 0


def _materialize_df40(args: argparse.Namespace) -> int:
    result = materialize_df40(
        args.subset_dir, args.downloads_root, args.data_root, dry_run=args.dry_run
    )
    print(json.dumps(result, indent=2))
    return 0


def _prune_cross_split(args: argparse.Namespace) -> int:
    result = prune_cross_split(args.subset_dir, args.manifest_dir, args.output_root)
    print(json.dumps(result, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deepfake-data",
        description="Build and audit content-addressed deepfake dataset manifests.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build an immutable manifest and audit it")
    build.add_argument("--config", type=Path, required=True)
    build.add_argument(
        "--allow-audit-failures",
        action="store_true",
        help="Return success after writing artifacts even if quality gates fail.",
    )
    build.set_defaults(handler=_build)

    audit = subparsers.add_parser("audit", help="Audit an existing manifest")
    audit.add_argument("--config", type=Path, required=True)
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--output", type=Path)
    audit.add_argument(
        "--allow-failures",
        action="store_true",
        help="Return success even if quality gates fail.",
    )
    audit.set_defaults(handler=_audit)

    verify = subparsers.add_parser("verify", help="Verify an immutable manifest's checksums")
    verify.add_argument("--manifest-dir", type=Path, required=True)
    verify.set_defaults(handler=_verify)

    subset = subparsers.add_parser(
        "subset",
        help="Create a deterministic pre-download subset from a metadata catalog",
    )
    subset.add_argument("--config", type=Path, required=True)
    subset.set_defaults(handler=_subset)

    verify_subset_parser = subparsers.add_parser(
        "verify-subset",
        help="Verify an immutable subset artifact's checksums",
    )
    verify_subset_parser.add_argument("--subset-dir", type=Path, required=True)
    verify_subset_parser.set_defaults(handler=_verify_subset)

    normalize = subparsers.add_parser(
        "normalize-df40", help="Normalize official DF40 JSON frame catalogs"
    )
    normalize.add_argument("--input-dir", type=Path, required=True)
    normalize.add_argument("--output", type=Path, required=True)
    normalize.add_argument(
        "--eval-split-seed",
        help="Explicitly repartition overlapping official val/test videos for a pilot",
    )
    normalize.set_defaults(handler=_normalize_df40)

    materialize = subparsers.add_parser(
        "materialize-df40", help="Extract only the verified DF40 pilot images from ZIPs"
    )
    materialize.add_argument("--subset-dir", type=Path, required=True)
    materialize.add_argument("--downloads-root", type=Path, required=True)
    materialize.add_argument("--data-root", type=Path, required=True)
    materialize.add_argument("--dry-run", action="store_true")
    materialize.set_defaults(handler=_materialize_df40)
    prune = subparsers.add_parser(
        "prune-cross-split", help="Remove colliding video groups from an audited subset"
    )
    prune.add_argument("--subset-dir", type=Path, required=True)
    prune.add_argument("--manifest-dir", type=Path, required=True)
    prune.add_argument("--output-root", type=Path, required=True)
    prune.set_defaults(handler=_prune_cross_split)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
