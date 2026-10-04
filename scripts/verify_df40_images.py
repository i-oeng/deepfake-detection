#!/usr/bin/env python3
"""Verify that every selected DF40 pilot image exists and decodes."""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image


def check(item: tuple[str, Path]) -> tuple[str, str | None]:
    relative, path = item
    if not path.is_file():
        return relative, "missing"
    if path.stat().st_size == 0:
        return relative, "empty"
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as error:  # noqa: BLE001 - report exact decoder failures
        return relative, f"decode_error:{type(error).__name__}:{error}"
    return relative, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    with args.subset.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    items = [(row["relative_path"], args.data_root / row["relative_path"]) for row in rows]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(check, items))
    failures = [(relative, error) for relative, error in results if error]
    report = {
        "subset": str(args.subset),
        "selected": len(rows),
        "valid_images": len(rows) - len(failures),
        "failures": len(failures),
        "failure_examples": failures[:20],
    }
    print(json.dumps(report, indent=2))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
