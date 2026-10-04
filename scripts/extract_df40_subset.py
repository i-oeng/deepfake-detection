#!/usr/bin/env python3
"""Extract only images selected by a DF40 subset from a method archive."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--kind", required=True, help="fake method slug or real:<domain>")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="permit a partial archive; a later archive must supply the remaining paths",
    )
    return parser.parse_args()


def selected_paths(subset: Path, kind: str) -> list[str]:
    selected: list[str] = []
    with subset.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if kind.startswith("real:"):
                domain = kind.split(":", 1)[1]
                include = row["label"] == "REAL" and row["source_domain"] == domain
            else:
                include = row["label"] == "FAKE" and row["fake_method"] == kind
            if include:
                selected.append(row["relative_path"].replace("\\", "/").lstrip("./"))
    if not selected:
        raise RuntimeError(f"subset contains no paths for {kind!r}")
    if len(selected) != len(set(selected)):
        raise RuntimeError(f"subset contains duplicate paths for {kind!r}")
    return selected


def safe_destination(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        raise RuntimeError(f"unsafe selected path: {relative}")
    destination = root.joinpath(*pure.parts)
    resolved_root = root.resolve()
    resolved_parent = destination.parent.resolve()
    if resolved_parent != resolved_root and resolved_root not in resolved_parent.parents:
        raise RuntimeError(f"selected path escapes output root: {relative}")
    return destination


def suffix_index(paths: list[str]) -> dict[str, str | None]:
    index: dict[str, str | None] = {}
    for target in paths:
        parts = PurePosixPath(target).parts
        for start in range(max(0, len(parts) - 9), len(parts) - 1):
            suffix = "/".join(parts[start:])
            existing = index.get(suffix)
            if existing is None and suffix in index:
                continue
            if existing is not None and existing != target:
                index[suffix] = None
            else:
                index[suffix] = target
    return index


def match_member(member: str, index: dict[str, str | None]) -> str | None:
    normalized = member.replace("\\", "/").lstrip("./")
    parts = PurePosixPath(normalized).parts
    best: tuple[int, str] | None = None
    for start in range(max(0, len(parts) - 9), len(parts) - 1):
        suffix = "/".join(parts[start:])
        target = index.get(suffix)
        if target is not None:
            score = len(parts) - start
            if best is None or score > best[0]:
                best = (score, target)
    return None if best is None else best[1]


def main() -> None:
    args = parse_args()
    targets = selected_paths(args.subset, args.kind)
    target_set = set(targets)
    index = suffix_index(targets)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    existing = {
        target
        for target in targets
        if (path := safe_destination(output_root, target)).is_file() and path.stat().st_size > 0
    }
    extracted: set[str] = set(existing)
    duplicate_members: dict[str, list[str]] = defaultdict(list)

    with zipfile.ZipFile(args.archive) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"archive CRC check failed at {bad}")
        for info in archive.infolist():
            if info.is_dir():
                continue
            target = match_member(info.filename, index)
            if target is None or target not in target_set:
                continue
            if target in extracted:
                if target not in existing:
                    duplicate_members[target].append(info.filename)
                continue
            destination = safe_destination(output_root, target)
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_name(destination.name + ".partial")
            with archive.open(info) as source, partial.open("wb") as sink:
                shutil.copyfileobj(source, sink, length=1024 * 1024)
                sink.flush()
                os.fsync(sink.fileno())
            if partial.stat().st_size != info.file_size:
                partial.unlink(missing_ok=True)
                raise RuntimeError(f"size mismatch while extracting {target}")
            partial.replace(destination)
            extracted.add(target)

    missing = sorted(target_set - extracted)
    result = {
        "kind": args.kind,
        "selected": len(targets),
        "preexisting": len(existing),
        "extracted": len(extracted) - len(existing),
        "missing": len(missing),
        "duplicate_member_targets": len(duplicate_members),
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    if missing:
        print(json.dumps({"missing_examples": missing[:20]}, indent=2), flush=True)
        if not args.allow_missing:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
