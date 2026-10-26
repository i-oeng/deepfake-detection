"""Build and audit the frozen additions for open-world training round 2.

Raw generated images are never put directly in the training manifest.  This
module pairs every new fake source with real sources in the same preprocessing
collection.  ``openworld materialize`` then detects, aligns, scale matches, and
JPEG encodes both classes together before this module merges them with v1.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path, PurePosixPath

import yaml

from .config import find_project_root
from .manifest import verify_manifest
from .openworld import (
    COLUMNS,
    EFS,
    PROTOCOL_ROUND2,
    PROTOCOL_V1,
    _rank,
    _row,
    method_roles,
)
from .unseen import audit_unseen_manifest

SEED = "round2-20261008"
T2I_COUNTS = {"kandinsky22": 2000, "sana600m": 2000}
RECON_COUNTS = {"recon_movq": 2000, "recon_dcae": 2000}
CASCADE_COUNT = 1000
AUTOENCODER_FAMILY = "autoencoder_reconstruction"
FINGERPRINTS = ("file_sha256", "pixel_sha256", "dhash64")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not set(COLUMNS).issubset(reader.fieldnames or ()):
            raise ValueError(f"catalog lacks required columns: {path}")
        return list(reader)


def _read_manifest(path: Path) -> list[dict[str, str]]:
    if path.is_dir():
        path = path / "manifest.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _read_jsonl(path: Path, key: str) -> dict[str, dict]:
    records: dict[str, dict] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                value = str(record[key])
            except (json.JSONDecodeError, KeyError) as error:
                raise ValueError(f"invalid metadata at {path}:{line_number}") from error
            if value in records:
                raise ValueError(f"duplicate {key}={value!r} in {path}")
            records[value] = record
    return records


def _write_catalog(path: Path, rows: list[dict[str, str]]) -> None:
    rows.sort(key=lambda row: (row["relative_path"], row["image_id"]))
    for column in ("image_id", "relative_path"):
        values = [row[column] for row in rows]
        if len(values) != len(set(values)):
            raise ValueError(f"catalog produced duplicate {column}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _entities(row: dict[str, str]) -> tuple[str, ...]:
    values = tuple(part for part in row["lineage_video_ids"].split("|") if part)
    if not values:
        raise ValueError(f"real source has no lineage: {row['image_id']}")
    return values


def _identity_evidence(row: dict[str, str]) -> str:
    """Preserve explicit IDs, falling back to the source lineage identity unit."""
    return row["identity_id"] or "|".join(_entities(row))


def _source_archive(archive: str) -> str:
    if not archive or archive.startswith(("dir:", "/")) or ".." in PurePosixPath(archive).parts:
        raise ValueError(f"unsafe base archive: {archive!r}")
    return f"df40/downloads/{archive}"


def _select_reals(
    base: list[dict[str, str]], split: str, count: int, seed: str, domain: str | None = None
) -> list[dict[str, str]]:
    eligible = [
        row
        for row in base
        if row["split"] == split
        and row["label"] == "REAL"
        and (domain is None or row["source_domain"] == domain)
        and row["archive"]
        and row["member"]
    ]
    eligible.sort(key=lambda row: _rank(seed, "paired-real", split, domain or "*", row["image_id"]))
    if len(eligible) < count:
        raise ValueError(f"only {len(eligible)} eligible {split}/{domain or '*'} reals for {count}")
    return eligible[:count]


def _paired_real(row: dict[str, str], domain: str, split: str, preprocess: str) -> dict[str, str]:
    return _row(
        domain=domain,
        method="",
        family=EFS,
        split=split,
        video=f"real-{row['image_id']}",
        frame=PurePosixPath(row["member"]).name,
        entities=_entities(row),
        archive=_source_archive(row["archive"]),
        member=row["member"],
        preprocess=preprocess,
        identity=_identity_evidence(row),
    )


def _validate_generation(folder: Path, generator: str, count: int, seed: str) -> dict[str, dict]:
    provenance_path = folder / "provenance.json"
    metadata_path = folder / "metadata.jsonl"
    if not provenance_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"generation is incomplete: {folder}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    expected = {"generator": generator, "count": count, "seed": seed}
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise ValueError(f"generation contract differs for {folder}")
    records = _read_jsonl(metadata_path, "index")
    if set(records) != {str(index) for index in range(count)}:
        raise ValueError(f"generation metadata is incomplete: {folder}")
    for index in range(count):
        image = folder / f"{index:05d}.png"
        record = records[str(index)]
        if (
            not image.is_file()
            or record.get("generator") != generator
            or record.get("seed") != seed
        ):
            raise ValueError(f"generated sample is incomplete: {image}")
    return records


def _t2i_rows(
    base: list[dict[str, str]], generated_root: Path, archive_root: str,
    generator: str, count: int, seed: str, split: str,
) -> list[dict[str, str]]:
    folder = generated_root / generator
    records = _validate_generation(folder, generator, count, seed)
    domain = generator if split == "validation" else f"round2_{generator}"
    preprocess = f"canonical:{domain}"
    fake = []
    for index in range(count):
        member = f"{index:05d}.png"
        identity = f"synthetic:{generator}:{index:05d}"
        fake.append(_row(
            domain=domain,
            method=generator,
            family=EFS,
            split=split,
            video=f"fake-{index:05d}",
            frame=member,
            entities=(identity,),
            archive=f"dir:{archive_root}/{generator}",
            member=member,
            preprocess=preprocess,
            identity=identity,
        ))
        if records[str(index)]["prompt"] == "":
            raise ValueError(f"empty prompt for {generator}/{index}")
    comparator = "whichfaceisreal" if split == "validation" else None
    real = [
        _paired_real(row, domain, split, preprocess)
        for row in _select_reals(base, split, count, seed + generator, comparator)
    ]
    return fake + real


def _validate_reconstruction(
    folder: Path, generator: str, count: int, seed: str
) -> list[dict]:
    provenance_path = folder / "provenance.json"
    metadata_path = folder / "metadata.jsonl"
    if not provenance_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"reconstruction is incomplete: {folder}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("count") != count or provenance.get("seed") != seed:
        raise ValueError(f"reconstruction contract differs for {folder}")
    if f"recon_{provenance.get('autoencoder')}" != generator:
        raise ValueError(f"reconstruction method differs for {folder}")
    records = list(_read_jsonl(metadata_path, "output").values())
    if len(records) != count:
        raise ValueError(f"reconstruction metadata is incomplete: {folder}")
    for record in records:
        relative = PurePosixPath(record["output"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe reconstruction output: {relative}")
        if not folder.joinpath(*relative.parts).is_file():
            raise FileNotFoundError(f"missing reconstruction: {relative}")
    return sorted(records, key=lambda record: record["source_sample_id"])


def _reconstruction_rows(
    base: list[dict[str, str]], generated_root: Path, archive_root: str,
    generator: str, count: int, seed: str,
) -> list[dict[str, str]]:
    folder = generated_root / generator
    records = _validate_reconstruction(folder, generator, count, seed)
    by_relative = {row["relative_path"]: row for row in base}
    domain = f"round2_{generator}"
    preprocess = f"canonical:{domain}"
    rows = []
    for record in records:
        source = by_relative.get(record["source_relative_path"])
        if source is None or source["split"] != "train" or source["label"] != "REAL":
            raise ValueError(f"reconstruction source is not a v1 training real: {record}")
        entities = _entities(source)
        source_id = source["image_id"]
        fake_member = record["output"]
        rows.append(_row(
            domain=domain,
            method=generator,
            family=AUTOENCODER_FAMILY,
            split="train",
            video=f"fake-{source_id}",
            frame=PurePosixPath(fake_member).name,
            entities=entities,
            archive=f"dir:{archive_root}/{generator}",
            member=fake_member,
            preprocess=preprocess,
            identity=_identity_evidence(source),
        ))
        rows.append(_row(
            domain=domain,
            method="",
            family=AUTOENCODER_FAMILY,
            split="train",
            video=f"real-{source_id}",
            frame=PurePosixPath(source["relative_path"]).name,
            entities=entities,
            archive="dir:df40_openworld",
            member=source["relative_path"],
            preprocess=preprocess,
            identity=_identity_evidence(source),
        ))
    return rows


def build_additions_catalog(
    base_catalog: Path,
    generated_root: Path,
    output: Path,
    *,
    archive_root: str = "df40/downloads/round2_generated",
    seed: str = SEED,
) -> dict:
    """Describe complete raw outputs and their matched real comparators."""
    base = _read_csv(base_catalog)
    rows: list[dict[str, str]] = []
    for generator, count in T2I_COUNTS.items():
        rows += _t2i_rows(base, generated_root, archive_root, generator, count, seed, "train")
    for generator, count in RECON_COUNTS.items():
        rows += _reconstruction_rows(base, generated_root, archive_root, generator, count, seed)
    rows += _t2i_rows(
        base, generated_root, archive_root, "stable_cascade_lite", CASCADE_COUNT, seed,
        "validation",
    )
    _write_catalog(output, rows)
    counts = Counter((row["split"], row["fake_method"] or "real") for row in rows)
    return {
        "catalog": str(output),
        "rows": len(rows),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "counts": {"/".join(key): value for key, value in sorted(counts.items())},
    }


def merge_catalogs(base_catalog: Path, additions_catalog: Path, output: Path) -> dict:
    """Merge materialized additions with the unchanged frozen v1 catalog."""
    base, additions = _read_csv(base_catalog), _read_csv(additions_catalog)
    rows = base + additions
    _write_catalog(output, rows)
    counts = Counter((row["split"], row["fake_method"] or "real") for row in rows)
    return {
        "catalog": str(output),
        "rows": len(rows),
        "base_rows": len(base),
        "addition_rows": len(additions),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "counts": {"/".join(key): value for key, value in sorted(counts.items())},
    }


def prune_fresh_overlaps(
    catalog: Path, manifest_dir: Path, fresh_manifest_dir: Path, output: Path
) -> dict:
    """Remove addition video groups whose fingerprints occur in frozen fresh-v2.

    The preliminary manifest supplies fingerprints after canonical preprocessing.
    Pruning whole video groups keeps neighboring frames together and uses the final
    test set only for leakage removal, never for model or hyperparameter selection.
    """
    for directory in (manifest_dir, fresh_manifest_dir):
        valid, errors = verify_manifest(directory)
        if not valid:
            raise ValueError(f"invalid immutable manifest {directory}: {errors}")
    additions = _read_csv(catalog)
    manifest = _read_manifest(manifest_dir)
    fresh = _read_manifest(fresh_manifest_dir)
    by_source = {row["source_id"]: row for row in manifest}
    if len(by_source) != len(manifest):
        raise ValueError("preliminary manifest has duplicate source IDs")
    addition_ids = {row["image_id"] for row in additions}
    if len(addition_ids) != len(additions) or not addition_ids.issubset(by_source):
        raise ValueError("preliminary manifest does not cover additions exactly")
    reserved = {
        column: {row[column] for row in fresh if row[column]}
        for column in FINGERPRINTS
    }
    collisions: Counter[str] = Counter()
    remove_groups: set[tuple[str, str]] = set()
    for row in additions:
        fingerprinted = by_source[row["image_id"]]
        if fingerprinted["status"] != "ok":
            raise ValueError(f"addition failed decoding: {row['image_id']}")
        for column in FINGERPRINTS:
            value = fingerprinted[column]
            if value and value in reserved[column]:
                collisions[column] += 1
                remove_groups.add((row["split"], row["video_id"]))
    kept = [
        row for row in additions
        if (row["split"], row["video_id"]) not in remove_groups
    ]
    if not kept:
        raise ValueError("fresh-v2 overlap pruning would remove every addition")
    _write_catalog(output, kept)
    return {
        "catalog": str(output),
        "rows": len(kept),
        "removed_rows": len(additions) - len(kept),
        "removed_video_groups": len(remove_groups),
        "collision_rows": dict(sorted(collisions.items())),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


def freeze_training_config(base_config: Path, manifest_dir: Path, output: Path) -> dict:
    """Pin the E7 recipe to the immutable Round 2 manifest as E8."""
    valid, errors = verify_manifest(manifest_dir)
    if not valid:
        raise ValueError(f"invalid Round 2 manifest: {errors}")
    metadata = json.loads((manifest_dir / "manifest.json").read_text(encoding="utf-8"))
    source = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    if not isinstance(source, dict) or not isinstance(source.get("experiment"), dict):
        raise ValueError("base config needs an experiment mapping")
    experiment = dict(source["experiment"])
    if experiment.get("protocol") != PROTOCOL_V1:
        raise ValueError("Round 2 must inherit a v1 open-world recipe")
    project_root = find_project_root(output.resolve().parent)
    resolved_manifest = manifest_dir.resolve()
    try:
        relative_manifest = resolved_manifest.relative_to(project_root.resolve()).as_posix()
    except ValueError as error:
        # xixi shares immutable manifests through the worktree's data/manifests
        # symlink. Preserve that portable project-relative path after verifying
        # that it resolves to the exact immutable directory supplied by build.
        linked = (
            project_root / "data" / "manifests" / manifest_dir.parent.name / manifest_dir.name
        )
        if not linked.is_dir() or linked.resolve() != resolved_manifest:
            raise ValueError(
                f"manifest is outside the project data link: {manifest_dir}"
            ) from error
        relative_manifest = linked.relative_to(project_root).as_posix()
    experiment.update({
        "protocol": PROTOCOL_ROUND2,
        "manifest_dir": relative_manifest,
        "manifest_id": metadata["manifest_id"],
        "manifest_sha256": metadata["manifest_sha256"],
    })
    payload = yaml.safe_dump({"experiment": experiment}, sort_keys=False)
    if output.exists() and output.read_text(encoding="utf-8") != payload:
        raise RuntimeError(f"refusing to overwrite changed training config: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")
    return {
        "config": str(output),
        "manifest_id": metadata["manifest_id"],
        "manifest_sha256": metadata["manifest_sha256"],
    }


def audit_fresh_separation(round2: Path, base: Path, fresh: Path) -> dict:
    """Check only Round 2 additions against the untouched fresh-v2 manifest."""
    rows = _read_manifest(round2)
    base_ids = {row["source_id"] for row in _read_manifest(base)}
    additions = [row for row in rows if row["source_id"] not in base_ids]
    reference = _read_manifest(fresh)
    failures = []
    overlap: dict[str, list[str]] = {}
    for column in FINGERPRINTS:
        reserved = {row[column] for row in reference if row[column]}
        hits = sorted({row[column] for row in additions if row[column] in reserved})
        overlap[column] = hits
        if hits:
            failures.append(f"{len(hits)} {column} values overlap fresh v2")
    fresh_parents = {
        value
        for row in reference
        for value in row["lineage_video_ids"].split("|")
        if value
    }
    parent_hits = sorted({
        value
        for row in additions
        for value in row["lineage_video_ids"].split("|")
        if value and value in fresh_parents
    })
    overlap["lineage_video_ids"] = parent_hits
    if parent_hits:
        failures.append(f"{len(parent_hits)} lineage values overlap fresh v2")
    result = audit_unseen_manifest(round2, method_roles(PROTOCOL_ROUND2))
    failures.extend(result["failures"])
    return {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "round2_manifest_id": round2.name,
        "base_manifest_id": base.name,
        "fresh_manifest_id": fresh.name,
        "addition_rows": len(additions),
        "overlap": overlap,
        "lineage_audit": result,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    catalog = commands.add_parser("catalog")
    catalog.add_argument("--base-catalog", type=Path, required=True)
    catalog.add_argument("--generated-root", type=Path, required=True)
    catalog.add_argument("--archive-root", default="df40/downloads/round2_generated")
    catalog.add_argument("--output", type=Path, required=True)
    catalog.add_argument("--seed", default=SEED)
    merge = commands.add_parser("merge")
    merge.add_argument("--base-catalog", type=Path, required=True)
    merge.add_argument("--additions-catalog", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    prune = commands.add_parser("prune-fresh")
    prune.add_argument("--catalog", type=Path, required=True)
    prune.add_argument("--manifest-dir", type=Path, required=True)
    prune.add_argument("--fresh-manifest-dir", type=Path, required=True)
    prune.add_argument("--output", type=Path, required=True)
    freeze = commands.add_parser("freeze-config")
    freeze.add_argument("--base-config", type=Path, required=True)
    freeze.add_argument("--manifest-dir", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--manifest-dir", type=Path, required=True)
    audit.add_argument("--base-manifest-dir", type=Path, required=True)
    audit.add_argument("--fresh-manifest-dir", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "catalog":
        result = build_additions_catalog(
            args.base_catalog, args.generated_root, args.output,
            archive_root=args.archive_root, seed=args.seed,
        )
    elif args.command == "merge":
        result = merge_catalogs(args.base_catalog, args.additions_catalog, args.output)
    elif args.command == "prune-fresh":
        result = prune_fresh_overlaps(
            args.catalog, args.manifest_dir, args.fresh_manifest_dir, args.output
        )
    elif args.command == "freeze-config":
        result = freeze_training_config(args.base_config, args.manifest_dir, args.output)
    else:
        result = audit_fresh_separation(
            args.manifest_dir, args.base_manifest_dir, args.fresh_manifest_dir
        )
        payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output.exists() and args.output.read_text(encoding="utf-8") != payload:
            previous = json.loads(args.output.read_text(encoding="utf-8"))
            if previous.get("passed"):
                raise RuntimeError(f"refusing to overwrite passed audit: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("passed", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
