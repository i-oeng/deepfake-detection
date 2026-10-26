"""Download the pinned public sources of open-world round 3 and unpack them to files.

None of these sources occurs in any earlier split. CelebA, FFHQ, IMDB-WIKI,
FF++, and Celeb-DF are already used for validation or testing, so the new real
photos come from elsewhere:

- FairFace (CC BY 4.0), 1.25 padding, first training shard: YFCC-100M Flickr
  photos, used as training reals. The wide padding leaves room for the
  1.3-scale canonical crop, which a tight crop would fill with black borders.
- LFW (bitmind mirror): news photos, used only as a test-time real source.
- OpenFakeTiny (ComplexDataLab): real LAION photos and images from current
  text-to-image generators, used only for the fresh v3 test.

Each parquet image is written as ``<source>/<split>/<index>.<ext>`` with its
metadata in ``<source>/metadata.jsonl``. Run with ``PYTHONPATH=.genlibs:src``
for pyarrow.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath

from huggingface_hub import hf_hub_download

FAIRFACE = ("HuggingFaceM4/FairFace", "54d573cdb8b5af490ba8da9da2799628f6e5c496",
            {"train": ["1.25/train-00000-of-00004-e715178553977907.parquet"]})
LFW = ("bitmind/lfw", "274dfbca8b2dc9c6d21ae80f8627c911222eb347",
       {"train": ["data/train-00000-of-00001.parquet"]})
OPENFAKE = ("ComplexDataLab/OpenFakeTiny", "9a7f5a7b37490f1b7d557c285eef52b09a90d0c2",
            {"core-train": ["core/train-00000-of-00002.parquet",
                            "core/train-00001-of-00002.parquet"],
             "core-validation": ["core/validation-00000-of-00002.parquet",
                                 "core/validation-00001-of-00002.parquet"],
             "core-test": ["core/test-00000-of-00001.parquet"],
             "reddit-test": ["reddit/test-00000-of-00001.parquet"]})
SOURCES = {"fairface": FAIRFACE, "lfw": LFW, "openfake": OPENFAKE}
MAGIC = ((b"\xff\xd8\xff", ".jpg"), (b"\x89PNG", ".png"), (b"RIFF", ".webp"), (b"GIF8", ".gif"))


def _suffix(data: bytes) -> str:
    for magic, suffix in MAGIC:
        if data.startswith(magic):
            return suffix
    raise ValueError("unrecognized image encoding")


def unpack(parquet: Path, output: Path, split: str, start: int) -> list[dict]:
    """Write every image of one parquet file; return its metadata records."""
    import pyarrow.parquet as pq

    table = pq.read_table(parquet)
    records = []
    for offset, row in enumerate(table.to_pylist()):
        image = row.pop("image")
        data = image["bytes"]
        index = start + offset
        relative = PurePosixPath(split, f"{index:06d}{_suffix(data)}")
        target = output.joinpath(*relative.parts)
        if not target.is_file() or target.read_bytes() != data:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f"{target.name}.part")
            temporary.write_bytes(data)
            temporary.replace(target)
        records.append({"index": index, "split": split, "file": relative.as_posix(),
                        "parquet": parquet.name, "original_path": image.get("path") or "",
                        **{key: value for key, value in row.items()
                           if isinstance(value, (str, int, float, bool)) or value is None}})
    return records


def download(name: str, output: Path, cache: Path) -> dict:
    repository, revision, splits = SOURCES[name]
    folder = output / name
    records = []
    for split, files in splits.items():
        start = 0
        for filename in files:
            path = Path(hf_hub_download(repository, filename, repo_type="dataset",
                                        revision=revision, local_dir=cache / name))
            unpacked = unpack(path, folder, split, start)
            start += len(unpacked)
            records += unpacked
            print(f"{name} {filename}: {len(unpacked)} images", flush=True)
    with (folder / "metadata.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    return {"repository": repository, "revision": revision, "images": len(records)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--sources", nargs="+", choices=sorted(SOURCES), default=sorted(SOURCES))
    args = parser.parse_args(argv)
    path = args.output / "sources.json"
    summary = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    for name in args.sources:
        summary[name] = download(name, args.output, args.cache)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
