"""Raw-video cross-dataset protocol P5: Celeb-DF v2's published test list.

``prepare`` extracts only the listed videos from the verified archive and writes
the ``media_id,path`` list for ``deepfake-detect crops``. ``report`` scores each
bundle's per-video decisions, with abstentions excluded (primary) and scored as
0.5 (secondary), on all videos and on videos whose identities are absent from
DF40 Celeb-DF development data, plus the pre-registered paired comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from deepfake_detection.data.identity import candidate_ids
from deepfake_detection.train.evaluation import grouped_bootstrap_difference, report

METHOD = "celebdf_v2"


def prepare(catalog: Path, archive: Path, output_dir: Path) -> Path:
    with catalog.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == "test"]
    output_dir.mkdir(parents=True, exist_ok=True)
    listing = []
    with ZipFile(archive) as source:
        members = {PurePosixPath(name).as_posix(): name for name in source.namelist()}
        for row in rows:
            matches = [name for key, name in members.items() if key.endswith(row["archive_path"])]
            if len(matches) != 1:
                raise ValueError(f"expected one archive member for {row['archive_path']}")
            target = output_dir / row["archive_path"]
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(matches[0]) as stream, target.open("wb") as handle:
                    shutil.copyfileobj(stream, handle)
            listing.append({"media_id": row["video_id"], "path": str(target.resolve())})
    path = output_dir / "inputs.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("media_id", "path"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(listing)
    return path


def _identities(video_id: str) -> set[str]:
    return {f"cdf:{token}" for token in candidate_ids(video_id)}


def development_identities(manifest: Path) -> set[str]:
    with manifest.open(newline="", encoding="utf-8") as stream:
        return {identity for row in csv.DictReader(stream)
                if row["split"] == "validation" and row["source_domain"] == "cdf"
                for identity in _identities(row["video_id"])}


def _rows(results: Path, labels: dict[str, str], *, abstain_as: float | None) -> list[dict]:
    rows = []
    for line in results.read_text(encoding="utf-8").splitlines():
        result = json.loads(line)
        probability = result["probability_fake"]
        if probability is None:
            if abstain_as is None:
                continue
            probability = abstain_as
        video = result["media_id"]
        # Identity components keep a person's real and fake videos together in the
        # bootstrap; YouTube-real clips without idNN tokens are their own group.
        lineage = "|".join(sorted(_identities(video))) or f"cdf:video:{video}"
        fake = labels[video] == "FAKE"
        rows.append({"source_domain": "cdf", "video_id": video, "label": int(fake),
                     "fake_method": METHOD if fake else "", "score": probability,
                     "lineage_video_ids": lineage})
    return rows


def build_report(results: dict[str, Path], catalog: Path, reference_manifest: Path,
                 comparison: tuple[str, str]) -> dict:
    with catalog.open(newline="", encoding="utf-8") as stream:
        labels = {row["video_id"]: row["label"] for row in csv.DictReader(stream)
                  if row["split"] == "test"}
    held = development_identities(reference_manifest)
    disjoint = {video for video in labels if not _identities(video) & held}
    output: dict = {"videos": len(labels), "identity_disjoint_videos": len(disjoint),
                    "development_identities": sorted(held), "bundles": {}}
    for name, path in results.items():
        entry = {}
        for variant, abstain_as in (("excluding_abstentions", None), ("abstain_as_0.5", 0.5)):
            rows = _rows(path, labels, abstain_as=abstain_as)
            entry[variant] = {
                "all": report(rows, domain="cdf")["macro"],
                "identity_disjoint": report(
                    [row for row in rows if row["video_id"] in disjoint], domain="cdf")["macro"],
                "scored_videos": len(rows),
            }
        decisions = [json.loads(line) for line in path.read_text().splitlines()]
        entry["abstentions"] = sum(d["label"] == "ABSTAIN" for d in decisions)
        entry["decisions"] = {label: sum(d["label"] == label for d in decisions)
                              for label in ("FAKE", "REAL", "ABSTAIN")}
        entry["accuracy_at_threshold"] = sum(
            d["label"] == labels[d["media_id"]] for d in decisions if d["label"] != "ABSTAIN"
        ) / max(1, sum(d["label"] != "ABSTAIN" for d in decisions))
        output["bundles"][name] = entry
    baseline, candidate = comparison
    base = _rows(results[baseline], labels, abstain_as=None)
    cand = _rows(results[candidate], labels, abstain_as=None)
    shared = {row["video_id"] for row in base} & {row["video_id"] for row in cand}
    output["h2"] = {
        "baseline": baseline, "candidate": candidate,
        **grouped_bootstrap_difference(
            [row for row in base if row["video_id"] in shared],
            [row for row in cand if row["video_id"] in shared], domain="cdf",
        ),
    }
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--catalog", type=Path, required=True)
    prep.add_argument("--archive", type=Path, required=True)
    prep.add_argument("--output-dir", type=Path, required=True)
    summary = commands.add_parser("report")
    summary.add_argument("--catalog", type=Path, required=True)
    summary.add_argument("--reference-manifest", type=Path, required=True)
    summary.add_argument("--result", action="append", required=True, help="NAME=decisions.jsonl")
    summary.add_argument("--baseline", required=True)
    summary.add_argument("--candidate", required=True)
    summary.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        print(prepare(args.catalog, args.archive, args.output_dir))
        return 0
    results = dict(item.split("=", 1) for item in args.result)
    result = build_report({k: Path(v) for k, v in results.items()}, args.catalog,
                          args.reference_manifest, (args.baseline, args.candidate))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
