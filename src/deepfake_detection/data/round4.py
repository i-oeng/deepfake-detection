"""Build and audit the training additions for open-world round 4: AI-generated photos.

Part B of ``reports/openworld/round4_protocol.md``. Round 3's training data stays
unchanged; round 4 adds faces from two published datasets, both already through the
face policy on download (``scripts/download_openfake_round4.py``,
``scripts/download_sfhq_t2i_round4.py``):

- ``round4_openfake``: OpenFake's train split. Each generator family is one fake
  method with up to ``FAMILY_CAP`` faces, taken in the order of a seeded hash;
  families with fewer than ``RARE_MINIMUM`` faces share the method
  ``openfake_rare``, so a few dozen images are not sampled as often as a whole
  family. The reals are every LAION and Pexels face.
- ``round4_sfhq``: SFHQ-T2I, one fake method per model, against FFHQ faces.

Rows whose prompt names a public figure (``public_figures`` in the download
records) are held back: they belong to the celebrity collections (part A), which
wait for the identity audit. Each real source (LAION, Pexels, FFHQ) is its own
lineage namespace, so ``real_balance: lineage_namespace`` gives it an equal share
of the real sampling mass next to FF++ and FairFace.

Each collection is format matched (``data.format_match``) before canonicalization,
and every addition goes through round 3's leakage pruning against validation, test
and the fresh test sets. ``scripts/rebuild-round4`` runs the whole build.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from .format_match import Source, assign, write_encodings
from .fresh_v4 import _slug, stored_records
from .openworld import (
    EFS,
    PROTOCOL_ROUND3,
    PROTOCOL_ROUND4,
    ROUND4_TRAIN_METHODS,
    _rank,
    _row,
)
from .round2 import _read_manifest, _write_catalog
from .round2 import freeze_training_config as freeze_round2
from .round3 import _write_json, prune_evaluation_overlaps
from .round3 import audit as audit_round3

SEED = "round4-20261104"
SPLIT = "train"
OPENFAKE, SFHQ = "round4_openfake", "round4_sfhq"
OPENFAKE_ARCHIVE = "dir:openfake_round4_det320"
SFHQ_ARCHIVE = "dir:sfhq_round4_det320"
FAMILY_CAP = 2000
RARE_MINIMUM = 200
RARE = "rare"
# Columns of a round 3 manifest row that must not change in round 4.
KEPT_COLUMNS = ("relative_path", "label", "split", "file_sha256", "pixel_sha256",
                "source_domain", "fake_method", "video_id", "lineage_video_ids")


def _folder(downloads_root: Path, archive: str) -> Path:
    return downloads_root / archive.removeprefix("dir:")


def _face_row(domain: str, method: str, source: str, record: dict,
              archive: str) -> dict[str, str]:
    label = "fake" if method else "real"
    key = record["image_sha256"][:16]
    # The real source is the lineage namespace that real_balance weighs.
    entity = f"{source}:{key}"
    return _row(domain=domain, method=method, family=EFS, split=SPLIT,
                video=f"{label}-{source}-{key}", frame=PurePosixPath(record["path"]).name,
                entities=(entity,), archive=archive, member=record["path"],
                preprocess=f"canonical:{domain}", identity=entity)


def _capped(records: list[dict], method: str, seed: str, cap: int) -> list[dict]:
    ordered = sorted(records, key=lambda record: _rank(seed, method, record["image_sha256"]))
    return ordered[:cap]


def openfake_methods(records: list[dict], minimum: int = RARE_MINIMUM) -> dict[str, str]:
    """Fake method of each OpenFake family: its own, or the pooled rare method."""
    sizes = Counter(record["group"] for record in records if record["label"] == "fake")
    return {group: f"openfake_{_slug(group) if size >= minimum else RARE}"
            for group, size in sorted(sizes.items())}


def build_additions_catalog(downloads_root: Path, output: Path, encodings: Path, *,
                            seed: str = SEED, cap: int = FAMILY_CAP,
                            minimum: int = RARE_MINIMUM,
                            methods: set[str] | frozenset[str] = frozenset(ROUND4_TRAIN_METHODS),
                            ) -> dict:
    """The round 4 photo additions and their format-matching encodings."""
    rows: list[dict[str, str]] = []
    sources: list[Source] = []

    def add(domain: str, method: str, source: str, record: dict, archive: str) -> None:
        row = _face_row(domain, method, source, record, archive)
        rows.append(row)
        sources.append(Source(row["image_id"], row["preprocess"], row["label"],
                              record["image_sha256"], record.get("format") or "",
                              record.get("jpeg_quality"), record.get("subsampling")))

    openfake = stored_records(_folder(downloads_root, OPENFAKE_ARCHIVE))
    held_back = Counter(record["label"] for record in openfake if record.get("public_figures"))
    usable = [record for record in openfake if not record.get("public_figures")]
    family_method = openfake_methods(usable, minimum)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in usable:
        if record["label"] == "fake":
            grouped[family_method[record["group"]]].append(record)
    for method, members in sorted(grouped.items()):
        for record in _capped(members, method, seed, cap):
            add(OPENFAKE, method, _slug(record["group"]), record, OPENFAKE_ARCHIVE)
    for record in usable:
        if record["label"] == "real":
            add(OPENFAKE, "", _slug(record["group"]), record, OPENFAKE_ARCHIVE)

    sfhq = stored_records(_folder(downloads_root, SFHQ_ARCHIVE))
    grouped = defaultdict(list)
    for record in sfhq:
        if record["label"] == "fake":
            grouped[f"sfhq_{_slug(record['group'])}"].append(record)
        else:
            add(SFHQ, "", _slug(record["group"]), record, SFHQ_ARCHIVE)
    for method, members in sorted(grouped.items()):
        for record in _capped(members, method, seed, cap):
            add(SFHQ, method, method.removeprefix("sfhq_"), record, SFHQ_ARCHIVE)

    found = {row["fake_method"] for row in rows if row["fake_method"]}
    if found != set(methods):
        raise ValueError(f"methods differ from the round 4 protocol: missing "
                         f"{sorted(set(methods) - found)}, "
                         f"unexpected {sorted(found - set(methods))}")
    if len({row["relative_path"] for row in rows}) != len(rows):
        raise ValueError("catalog produced duplicate destination paths")
    rows.sort(key=lambda row: row["relative_path"])
    _write_catalog(output, rows)
    assignments, matching = assign(sources)
    counts = Counter(f"{row['source_domain']}/"
                     f"{row['fake_method'] or 'real-' + row['identity_id'].split(':')[0]}"
                     for row in rows)
    pooled = sorted(group for group, method in family_method.items()
                    if method == f"openfake_{RARE}")
    return {"catalog": str(output), "rows": len(rows), "seed": seed, "family_cap": cap,
            "rare_minimum": minimum, "rare_families": pooled,
            "held_back_public_figures": dict(sorted(held_back.items())),
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "counts": dict(sorted(counts.items())),
            "encodings": str(encodings),
            "encodings_sha256": write_encodings(encodings, assignments),
            "format_matching": matching}


def base_unchanged(rows: list[dict[str, str]], base: list[dict[str, str]]) -> list[str]:
    """Failures if a round 3 row is missing or changed, or an added row is not a photo."""
    by_source = {row["source_id"]: row for row in rows}
    failures = []
    missing = [row["source_id"] for row in base if row["source_id"] not in by_source]
    if missing:
        failures.append(f"{len(missing)} round 3 rows are missing")
    changed = Counter(column for row in base if row["source_id"] in by_source
                      for column in KEPT_COLUMNS
                      if by_source[row["source_id"]][column] != row[column])
    failures += [f"{count} round 3 rows changed {column}"
                 for column, count in sorted(changed.items())]
    base_ids = {row["source_id"] for row in base}
    added = [row for row in rows if row["source_id"] not in base_ids]
    stray = [row for row in added
             if row["split"] != SPLIT or row["source_domain"] not in {OPENFAKE, SFHQ}]
    if stray:
        failures.append(f"{len(stray)} added rows are not round 4 training photos")
    return failures


def audit(manifest_dir: Path, data_root: Path, base_dir: Path, references, *,
          workers: int = 8) -> dict:
    """Round 3's lineage and separation audit under the round 4 protocol, plus a check
    that every round 3 row is unchanged."""
    result = audit_round3(manifest_dir, data_root, base_dir, references, workers=workers,
                          protocol=PROTOCOL_ROUND4)
    unchanged = base_unchanged(_read_manifest(manifest_dir), _read_manifest(base_dir))
    result["failures"] = sorted(set(result["failures"]) | set(unchanged))
    result["passed"] = not result["failures"]
    result["round4_manifest_id"] = result.pop("round3_manifest_id")
    result["round3_rows_unchanged"] = not unchanged
    return result


def freeze_training_config(base_config: Path, manifest_dir: Path, output: Path) -> dict:
    """Pin the E9 recipe to the round 4 photo manifest as E10-photos; nothing else
    changes."""
    return freeze_round2(base_config, manifest_dir, output, inherit=PROTOCOL_ROUND3,
                         protocol=PROTOCOL_ROUND4)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    catalog = commands.add_parser("catalog")
    catalog.add_argument("--downloads-root", type=Path, required=True)
    catalog.add_argument("--output", type=Path, required=True)
    catalog.add_argument("--encodings", type=Path, required=True)
    prune = commands.add_parser("prune")
    prune.add_argument("--catalog", type=Path, required=True)
    check = commands.add_parser("audit")
    check.add_argument("--base-manifest-dir", type=Path, required=True)
    for command in (prune, check):
        command.add_argument("--manifest-dir", type=Path, required=True)
        command.add_argument("--data-root", type=Path, required=True)
        command.add_argument("--reference", nargs=2, type=Path, action="append", required=True,
                             metavar=("MANIFEST_DIR", "DATA_ROOT"))
        command.add_argument("--workers", type=int, default=8)
        command.add_argument("--output", type=Path, required=True)
    freeze = commands.add_parser("freeze-config")
    freeze.add_argument("--base-config", type=Path, required=True)
    freeze.add_argument("--manifest-dir", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "catalog":
        result = build_additions_catalog(args.downloads_root, args.output, args.encodings)
    elif args.command == "prune":
        result = prune_evaluation_overlaps(args.catalog, args.manifest_dir, args.data_root,
                                           [tuple(pair) for pair in args.reference],
                                           args.output, workers=args.workers)
    elif args.command == "audit":
        result = audit(args.manifest_dir, args.data_root, args.base_manifest_dir,
                       [tuple(pair) for pair in args.reference], workers=args.workers)
        _write_json(args.output, result)
    else:
        result = freeze_training_config(args.base_config, args.manifest_dir, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("passed", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
