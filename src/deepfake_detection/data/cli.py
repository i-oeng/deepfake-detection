"""Command-line interface for immutable manifests and dataset audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import audit_manifest, write_audit_report
from .config import load_dataset_spec
from .df40 import (
    extend_df40_from_archives,
    normalize_df40,
    partition_df40_heldout,
    partition_df40_unseen,
)
from .identity import audit_candidate_identity_tokens, write_candidate_identity_report
from .manifest import build_manifest, verify_manifest
from .materialize import materialize_df40
from .prune import prune_cross_split
from .subset import build_subset, load_subset_spec, verify_subset
from .unseen import audit_heldout_manifest, audit_unseen_manifest


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
    result = normalize_df40(args.input_dir, args.output, eval_split_seed=args.eval_split_seed)
    print(json.dumps(result, indent=2))
    return 0


def _extend_df40(args: argparse.Namespace) -> int:
    result = extend_df40_from_archives(
        args.catalog, args.downloads_root, args.output, eval_split_seed=args.eval_split_seed
    )
    print(json.dumps(result, indent=2))
    return 0


def _partition_df40(args: argparse.Namespace) -> int:
    result = partition_df40_unseen(args.catalog, args.output, seed=args.seed)
    print(json.dumps(result, indent=2))
    return 0


def _partition_heldout(args: argparse.Namespace) -> int:
    excluded = frozenset()
    if args.exclude_videos:
        # One video ID per line; text after "#" records why it was excluded.
        lines = args.exclude_videos.read_text(encoding="utf-8").splitlines()
        excluded = frozenset(filter(None, (line.split("#")[0].strip() for line in lines)))
    result = partition_df40_heldout(
        args.catalog, args.reference_manifest, args.output,
        downloads_root=args.downloads_root, exclude_videos=excluded,
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


def _identity_tokens(args: argparse.Namespace) -> int:
    report = audit_candidate_identity_tokens(args.manifest_dir)
    if args.output:
        write_candidate_identity_report(report, args.output)
    print(json.dumps(report, indent=2))
    return 0


def _audit_unseen(args: argparse.Namespace) -> int:
    return _write_gate(audit_unseen_manifest(args.manifest_dir), args.output)


def _audit_heldout(args: argparse.Namespace) -> int:
    return _write_gate(
        audit_heldout_manifest(args.manifest_dir, args.reference_manifest_dir), args.output
    )


def _write_gate(result: dict, output: Path | None) -> int:
    if output:
        content = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if output.exists() and output.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"refusing to overwrite changed protocol audit: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


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
    extend = subparsers.add_parser(
        "extend-df40-archives",
        help="Add methods absent from official JSON using ZIP member lineage",
    )
    extend.add_argument("--catalog", type=Path, required=True)
    extend.add_argument("--downloads-root", type=Path, required=True)
    extend.add_argument("--output", type=Path, required=True)
    extend.add_argument("--eval-split-seed", required=True)
    extend.set_defaults(handler=_extend_df40)
    partition = subparsers.add_parser(
        "partition-df40-unseen",
        help="Assign source identities and clips to the frozen unseen-method split",
    )
    partition.add_argument("--catalog", type=Path, required=True)
    partition.add_argument("--output", type=Path, required=True)
    partition.add_argument("--seed", required=True)
    partition.set_defaults(handler=_partition_df40)

    heldout = subparsers.add_parser(
        "partition-df40-heldout",
        help="Collect evaluation-archive videos disjoint from a frozen protocol's training",
    )
    heldout.add_argument("--catalog", type=Path, required=True)
    heldout.add_argument("--reference-manifest", type=Path, required=True)
    heldout.add_argument("--output", type=Path, required=True)
    heldout.add_argument(
        "--downloads-root", type=Path, help="Drop rows whose DF40 archive member is empty"
    )
    heldout.add_argument(
        "--exclude-videos", type=Path, help="File of video IDs with audited content collisions"
    )
    heldout.set_defaults(handler=_partition_heldout)

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
    identity = subparsers.add_parser(
        "identity-tokens", help="Audit candidate Celeb-DF subject tokens across splits"
    )
    identity.add_argument("--manifest-dir", type=Path, required=True)
    identity.add_argument("--output", type=Path)
    identity.set_defaults(handler=_identity_tokens)
    unseen = subparsers.add_parser("audit-unseen", help="Gate the frozen unseen-method manifest")
    unseen.add_argument("--manifest-dir", type=Path, required=True)
    unseen.add_argument("--output", type=Path)
    unseen.set_defaults(handler=_audit_unseen)
    gate = subparsers.add_parser(
        "audit-heldout", help="Gate an evaluation-only manifest against a frozen protocol"
    )
    gate.add_argument("--manifest-dir", type=Path, required=True)
    gate.add_argument("--reference-manifest-dir", type=Path, required=True)
    gate.add_argument("--output", type=Path)
    gate.set_defaults(handler=_audit_heldout)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
