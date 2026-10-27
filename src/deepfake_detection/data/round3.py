"""Build and audit the frozen additions for open-world training round 3.

Up to round 2 every training real is an FF++ video frame, and the E8 detector
separates FF++ reals from real photos of every other source at a video AUROC of
0.93-0.98 (``train.source_bias``). Its round 2 text-to-image fakes were paired
with FF++ frames, so "not an FF++ frame" was enough to call them fake. Round 3
keeps every round 2 method and changes the real side only:

- Kandinsky 2.2 and Sana faces (the round 2 generations) are paired with
  FairFace photos instead of FF++ frames.
- MoVQ and DC-AE reconstructions of FairFace photos are added next to the
  round 2 reconstructions of FF++ frames, so real photos appear on both sides of
  the label and only autoencoder traces separate the pairs.

FairFace (YFCC-100M Flickr photos, CC BY 4.0) appears in no earlier split. Only
the adult age bins are used, matching the adult-only generation prompts. Every
addition is pruned against the validation and test images it could leak into,
including near-duplicates, because FFHQ (validation and fresh v2 reals) is also
drawn from Flickr. A 64-bit dHash cannot find those among aligned faces: the
nearest unrelated evaluation face lies within 6 bits for a quarter of training
images. Near-duplicates are instead found by the correlation of 32 x 32
grayscale thumbnails, with the threshold set from unrelated images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path, PurePosixPath

from .manifest import verify_manifest
from .openworld import EFS, PROTOCOL_ROUND2, PROTOCOL_ROUND3, _rank, _row, method_roles
from .round2 import (
    AUTOENCODER_FAMILY,
    FINGERPRINTS,
    _read_csv,
    _read_jsonl,
    _read_manifest,
    _validate_generation,
    _write_catalog,
)
from .round2 import freeze_training_config as freeze_round2
from .unseen import audit_unseen_manifest

SEED = "round3-20261027"
ROUND2_SEED = "round2-20261008"
FAIRFACE_ARCHIVE = "dir:round3_sources/fairface"
# FairFace age labels 0-8 are 0-2, 3-9, 10-19, 20-29, ..., 70+.
ADULT_AGE_BINS = frozenset(range(3, 9))
T2I_COUNTS = {"kandinsky22": 2000, "sana600m": 2000}
RECON_COUNTS = {"recon_movq": 2000, "recon_dcae": 2000}
POOLS = {**T2I_COUNTS, **RECON_COUNTS}
# Round 2 collections whose fakes round 3 re-pairs with FairFace photos.
REPLACED_DOMAINS = frozenset({"round2_kandinsky22", "round2_sana600m"})
THUMBNAIL_SIDE = 32
# Thumbnail correlation at or above which two images count as one (calibrated in
# reports/openworld/round3_protocol.md).
NEAR_DUPLICATE_CORRELATION = 0.99
# Share of E9 training images given a black alignment-padding margin.
PADDING_AUGMENT = 0.5


def select_fairface(metadata: Path, output: Path, *, seed: str = SEED) -> dict:
    """Assign disjoint, hash-ranked pools of adult FairFace photos to round 3 uses."""
    records = _read_jsonl(metadata, "index")
    eligible = sorted(
        (record for record in records.values()
         if record["split"] == "train" and int(record["age"]) in ADULT_AGE_BINS),
        key=lambda record: _rank(seed, "fairface", record["file"]),
    )
    if len(eligible) < sum(POOLS.values()):
        raise ValueError(f"only {len(eligible)} adult FairFace photos for {sum(POOLS.values())}")
    pools, start = {}, 0
    for pool, count in POOLS.items():
        pools[pool] = [{"id": f"fairface-{int(record['index']):06d}", "file": record["file"],
                        "age": int(record["age"]), "gender": int(record["gender"]),
                        "race": int(record["race"])}
                       for record in eligible[start:start + count]]
        start += count
    payload = json.dumps({"seed": seed, "source": FAIRFACE_ARCHIVE, "pools": pools},
                         indent=2, sort_keys=True) + "\n"
    if output.exists() and output.read_text(encoding="utf-8") != payload:
        raise RuntimeError(f"refusing to overwrite a different selection: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")
    return {"selection": str(output), "eligible": len(eligible),
            "pools": {pool: len(entries) for pool, entries in pools.items()},
            "sha256": hashlib.sha256(payload.encode()).hexdigest()}


def _fairface_real(entry: dict, domain: str, family: str) -> dict[str, str]:
    return _row(
        domain=domain, method="", family=family, split="train",
        video=f"real-{entry['id']}", frame=PurePosixPath(entry["file"]).name,
        entities=(f"fairface:{entry['id']}",), archive=FAIRFACE_ARCHIVE,
        member=entry["file"], preprocess=f"canonical:{domain}", identity=entry["id"],
    )


def _t2i_rows(pool: list[dict], generated_root: Path, archive_root: str,
              generator: str) -> list[dict[str, str]]:
    count = T2I_COUNTS[generator]
    _validate_generation(generated_root / generator, generator, count, ROUND2_SEED)
    domain = f"round3_{generator}"
    rows = []
    for index in range(count):
        member = f"{index:05d}.png"
        identity = f"synthetic:{generator}:{index:05d}"
        rows.append(_row(
            domain=domain, method=generator, family=EFS, split="train",
            video=f"fake-{index:05d}", frame=member, entities=(identity,),
            archive=f"dir:{archive_root}/{generator}", member=member,
            preprocess=f"canonical:{domain}", identity=identity,
        ))
    return rows + [_fairface_real(entry, domain, EFS) for entry in pool]


def _validate_pool_reconstruction(folder: Path, generator: str, pool: list[dict],
                                  selection_sha256: str) -> dict[str, dict]:
    provenance_path, metadata_path = folder / "provenance.json", folder / "metadata.jsonl"
    if not provenance_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"reconstruction is incomplete: {folder}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    expected = {"pool": generator, "count": len(pool), "selection_sha256": selection_sha256}
    if any(provenance.get(key) != value for key, value in expected.items()) or \
            f"recon_{provenance.get('autoencoder')}" != generator:
        raise ValueError(f"reconstruction contract differs for {folder}")
    records = {record["source_id"]: record
               for record in _read_jsonl(metadata_path, "output").values()}
    if set(records) != {entry["id"] for entry in pool}:
        raise ValueError(f"reconstruction metadata is incomplete: {folder}")
    for record in records.values():
        relative = PurePosixPath(record["output"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe reconstruction output: {relative}")
        if not folder.joinpath(*relative.parts).is_file():
            raise FileNotFoundError(f"missing reconstruction: {relative}")
    return records


def _reconstruction_rows(pool: list[dict], generated_root: Path, archive_root: str,
                         generator: str, selection_sha256: str) -> list[dict[str, str]]:
    records = _validate_pool_reconstruction(generated_root / generator, generator, pool,
                                            selection_sha256)
    domain = f"round3_{generator}"
    rows = []
    for entry in pool:
        output = records[entry["id"]]["output"]
        rows.append(_row(
            domain=domain, method=generator, family=AUTOENCODER_FAMILY, split="train",
            video=f"fake-{entry['id']}", frame=PurePosixPath(output).name,
            entities=(f"fairface:{entry['id']}",), archive=f"dir:{archive_root}/{generator}",
            member=output, preprocess=f"canonical:{domain}", identity=entry["id"],
        ))
        rows.append(_fairface_real(entry, domain, AUTOENCODER_FAMILY))
    return rows


def build_additions_catalog(selection: Path, round2_generated: Path, round3_generated: Path,
                            output: Path, *,
                            round2_archive: str = "df40/downloads/round2_generated",
                            round3_archive: str = "df40/downloads/round3_generated") -> dict:
    """Describe the new round 3 rows; round 2 rows that stay are merged separately."""
    spec = json.loads(selection.read_text(encoding="utf-8"))
    digest = hashlib.sha256(selection.read_bytes()).hexdigest()
    ids = [entry["id"] for entries in spec["pools"].values() for entry in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("FairFace pools overlap")
    rows: list[dict[str, str]] = []
    for generator in T2I_COUNTS:
        rows += _t2i_rows(spec["pools"][generator], round2_generated, round2_archive, generator)
    for generator in RECON_COUNTS:
        rows += _reconstruction_rows(spec["pools"][generator], round3_generated,
                                     round3_archive, generator, digest)
    _write_catalog(output, rows)
    counts = Counter((row["source_domain"], row["fake_method"] or "real") for row in rows)
    return {"catalog": str(output), "rows": len(rows), "selection_sha256": digest,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "counts": {"/".join(key): value for key, value in sorted(counts.items())}}


def combine_additions(round2_additions: Path, round3_additions: Path, output: Path) -> dict:
    """Round 2 additions except the replaced pairings, followed by the round 3 rows.

    The result is merged with the unchanged v1 catalog by ``round2 merge``.
    """
    round2, round3 = _read_csv(round2_additions), _read_csv(round3_additions)
    kept = [row for row in round2 if row["source_domain"] not in REPLACED_DOMAINS]
    if len(kept) == len(round2):
        raise ValueError("round 2 additions contain none of the replaced pairings")
    rows = kept + round3
    _write_catalog(output, rows)
    counts = Counter((row["split"], row["source_domain"], row["fake_method"] or "real")
                     for row in rows)
    return {"catalog": str(output), "rows": len(rows), "round2_rows_kept": len(kept),
            "round2_rows_replaced": len(round2) - len(kept), "round3_rows": len(round3),
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "counts": {"/".join(key): value for key, value in sorted(counts.items())}}


def thumbnail(path: str):
    """A zero-mean, unit-norm 32 x 32 grayscale vector; dot products are correlations."""
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        small = image.convert("L").resize((THUMBNAIL_SIDE, THUMBNAIL_SIDE),
                                          Image.Resampling.BOX)
    vector = np.asarray(small, dtype=np.float32).ravel()
    vector -= vector.mean()
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def thumbnails(paths: list[str], workers: int = 8):
    import numpy as np

    if not paths:
        return np.zeros((0, THUMBNAIL_SIDE * THUMBNAIL_SIDE), dtype=np.float32)
    with ProcessPoolExecutor(workers) as pool:
        return np.stack(list(pool.map(thumbnail, paths, chunksize=64)))


def nearest_correlation(queries, references, chunk: int = 1024):
    """Highest correlation of each query thumbnail with any reference thumbnail."""
    import numpy as np

    if not len(references):
        return np.full(len(queries), -1.0, dtype=np.float32)
    if not len(queries):
        return np.zeros(0, dtype=np.float32)
    return np.concatenate([(queries[start:start + chunk] @ references.T).max(axis=1)
                           for start in range(0, len(queries), chunk)])


Reference = tuple[Path, Path]


def _evaluation_rows(manifest_dir: Path, data_root: Path,
                     references: list[Reference]) -> list[dict[str, str]]:
    """Every row a model is scored on, with its image path in ``_path``.

    These are the non-train rows of ``manifest_dir`` (``_own`` set) and all rows of
    each reference (test-only) manifest.
    """
    rows = [{**row, "_path": str(data_root / row["relative_path"]), "_own": True}
            for row in _read_manifest(manifest_dir) if row["split"] != "train"]
    for directory, root in references:
        rows += [{**row, "_path": str(root / row["relative_path"]), "_own": False}
                 for row in _read_manifest(directory)]
    return rows


def _histogram(values) -> dict[str, int]:
    edges = (0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99)
    return {f">={edge}": int((values >= edge).sum()) for edge in edges}


def prune_evaluation_overlaps(catalog: Path, manifest_dir: Path, data_root: Path,
                              references: list[Reference], output: Path, *,
                              threshold: float = NEAR_DUPLICATE_CORRELATION,
                              workers: int = 8) -> dict:
    """Remove addition groups that match any evaluation image exactly or nearly.

    ``manifest_dir`` is a preliminary round 3 build that supplies fingerprints after
    canonical preprocessing; its validation and test rows, and every row of the
    reference (test-only) manifests, are evaluation images. A whole video group goes
    when any of its images matches, so a reconstruction leaves with its real.

    A validation addition is itself an evaluation row of the preliminary build, so
    it is checked against the references alone, as in round 2; the lineage audit
    already keeps validation and test apart.
    """
    import numpy as np

    for directory in (manifest_dir, *(directory for directory, _ in references)):
        valid, errors = verify_manifest(directory)
        if not valid:
            raise ValueError(f"invalid immutable manifest {directory}: {errors}")
    additions = _read_csv(catalog)
    by_source = {row["source_id"]: row for row in _read_manifest(manifest_dir)}
    if not {row["image_id"] for row in additions} <= set(by_source):
        raise ValueError("preliminary manifest does not cover the additions")
    fingerprinted = [by_source[row["image_id"]] for row in additions]
    if any(row["status"] != "ok" for row in fingerprinted):
        raise ValueError("an addition failed decoding")
    evaluation = _evaluation_rows(manifest_dir, data_root, references)
    scopes = {True: evaluation, False: [row for row in evaluation if not row["_own"]]}
    reserved = {training: {column: {row[column] for row in rows if row[column]}
                           for column in FINGERPRINTS}
                for training, rows in scopes.items()}
    queries = thumbnails([str(data_root / row["relative_path"]) for row in fingerprinted],
                         workers)
    candidates = thumbnails([row["_path"] for row in evaluation], workers)
    own = np.array([row["_own"] for row in evaluation], dtype=bool)
    training = np.array([row["split"] == "train" for row in additions], dtype=bool)
    correlation = np.full(len(additions), -1.0, dtype=np.float32)
    correlation[training] = nearest_correlation(queries[training], candidates)
    correlation[~training] = nearest_correlation(queries[~training], candidates[~own])
    hits: Counter[str] = Counter()
    remove: set[tuple[str, str]] = set()
    for row, fingerprint, value in zip(additions, fingerprinted, correlation, strict=True):
        scope = reserved[row["split"] == "train"]
        exact = [column for column in FINGERPRINTS if fingerprint[column] in scope[column]]
        if exact or value >= threshold:
            hits[exact[0] if exact else "thumbnail_correlation"] += 1
            remove.add((row["split"], row["video_id"]))
    kept = [row for row in additions if (row["split"], row["video_id"]) not in remove]
    _write_catalog(output, kept)
    return {"catalog": str(output), "rows": len(kept), "removed_rows": len(additions) - len(kept),
            "removed_video_groups": len(remove), "hits": dict(sorted(hits.items())),
            "near_duplicate_correlation": threshold,
            "nearest_correlation_counts": _histogram(correlation),
            "references": [directory.name for directory, _ in references],
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


def audit(manifest_dir: Path, data_root: Path, base_dir: Path, references: list[Reference], *,
          threshold: float = NEAR_DUPLICATE_CORRELATION, workers: int = 8) -> dict:
    """Lineage audit of the round 3 manifest and separation of its training rows."""
    rows = _read_manifest(manifest_dir)
    base_ids = {row["source_id"] for row in _read_manifest(base_dir)}
    training = [row for row in rows if row["split"] == "train"]
    evaluation = _evaluation_rows(manifest_dir, data_root, references)
    failures, overlap = [], {}
    for column in FINGERPRINTS:
        reserved = {row[column] for row in evaluation if row[column]}
        found = sorted({row[column] for row in training if row[column] in reserved})
        overlap[column] = found
        if found:
            failures.append(f"{len(found)} training {column} values occur in evaluation")
    additions = [row for row in training if row["source_id"] not in base_ids]
    correlation = nearest_correlation(
        thumbnails([str(data_root / row["relative_path"]) for row in additions], workers),
        thumbnails([row["_path"] for row in evaluation], workers))
    near = int((correlation >= threshold).sum())
    if near:
        failures.append(f"{near} additions correlate with an evaluation image at "
                        f">= {threshold}")
    parents = {value for row in evaluation
               for value in row["lineage_video_ids"].split("|") if value}
    shared = sorted({value for row in additions
                     for value in row["lineage_video_ids"].split("|") if value in parents})
    overlap["lineage_video_ids"] = shared
    if shared:
        failures.append(f"{len(shared)} addition lineage values occur in evaluation")
    lineage = audit_unseen_manifest(manifest_dir, method_roles(PROTOCOL_ROUND3))
    failures.extend(lineage["failures"])
    namespaces = Counter(row["lineage_video_ids"].split("|")[0].split(":")[0]
                         for row in training if row["label"] == "REAL")
    return {"passed": not failures, "failures": sorted(set(failures)),
            "round3_manifest_id": manifest_dir.name, "base_manifest_id": base_dir.name,
            "reference_manifest_ids": [directory.name for directory, _ in references],
            "addition_rows": len(additions), "near_duplicate_correlation": threshold,
            "nearest_correlation_counts": _histogram(correlation), "overlap": overlap,
            "training_real_namespaces": dict(sorted(namespaces.items())),
            "lineage_audit": lineage}


def freeze_training_config(base_config: Path, manifest_dir: Path, output: Path) -> dict:
    """Pin the E8 recipe to the round 3 manifest as E9.

    E9 balances FF++ and FairFace reals and adds alignment-padding augmentation; the
    rest of the E8 recipe, budget, and seeds is unchanged.
    """
    return freeze_round2(base_config, manifest_dir, output, inherit=PROTOCOL_ROUND2,
                         protocol=PROTOCOL_ROUND3,
                         settings={"real_balance": "lineage_namespace",
                                   "padding_augment": PADDING_AUGMENT})


def _write_json(path: Path, result: dict) -> None:
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("passed"):
            raise RuntimeError(f"refusing to overwrite passed audit: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    choose = commands.add_parser("select-fairface")
    choose.add_argument("--metadata", type=Path, required=True)
    choose.add_argument("--output", type=Path, required=True)
    catalog = commands.add_parser("catalog")
    catalog.add_argument("--selection", type=Path, required=True)
    catalog.add_argument("--round2-generated", type=Path, required=True)
    catalog.add_argument("--round3-generated", type=Path, required=True)
    catalog.add_argument("--output", type=Path, required=True)
    combine = commands.add_parser("combine")
    combine.add_argument("--round2-additions", type=Path, required=True)
    combine.add_argument("--round3-additions", type=Path, required=True)
    combine.add_argument("--output", type=Path, required=True)
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
    if args.command == "select-fairface":
        result = select_fairface(args.metadata, args.output)
    elif args.command == "catalog":
        result = build_additions_catalog(args.selection, args.round2_generated,
                                         args.round3_generated, args.output)
    elif args.command == "combine":
        result = combine_additions(args.round2_additions, args.round3_additions, args.output)
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
