"""Download the pinned public sources of the fresh open-world test set (v2).

None of these generators or real-image sources occurs in the open-world v1
training, validation, or test splits:

- DeepFakeFace (OpenRL): Stable Diffusion 1.5 text-to-image, Stable Diffusion
  inpainting, and InsightFace swaps, with the IMDB-WIKI photos they derive from.
- Synthetic Faces High Quality T2I (bitmind), curated sample: FLUX.1 and SDXL faces.
- FFHQ (WebDataset mirror): two 1,000-image shards of real portraits for the T2I faces.

Celeb-DF v2 raw videos, the fourth source, are already on disk.
"""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path, PurePosixPath

from huggingface_hub import hf_hub_download

DFF = ("OpenRL/DeepFakeFace", "eb2a54fbf2567790bd3dc61d09fdaade6bb6c31c",
       ("wiki.zip", "text2img.zip", "inpainting.zip", "insight.zip"))
SFHQ = ("bitmind/SyntheticFacesHighQuality-T2I", "c7f422bdcaf4ffe1ced6da49198845c874221a03",
        ("small-sample.zip",))
FFHQ = ("gaunernst/ffhq-1024-wds", "d74f1f1f59e3bbe975bee29872b9bef827314577",
        # whichfaceisreal (v1 validation) uses FFHQ 00000-01001; avoid that range.
        ("35000.tar", "50000.tar"))
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


def extract_shard(path: Path, output: Path) -> int:
    """Write each shard image as ``<shard>/<key>.<ext>``; skip non-image entries."""
    count = 0
    with tarfile.open(path) as shard:
        for member in shard.getmembers():
            name = PurePosixPath(member.name).name
            if not member.isfile() or not name.lower().endswith(IMAGE_SUFFIXES):
                continue
            target = output / path.stem / name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(shard.extractfile(member).read())
            count += 1
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = {}
    for repository, revision, files in (DFF, SFHQ):
        for name in files:
            path = hf_hub_download(repository, name, repo_type="dataset", revision=revision,
                                   local_dir=args.output / repository.split("/")[1])
            manifest[f"{repository}/{name}"] = {"revision": revision, "path": path}
            print(path, flush=True)
    repository, revision, shards = FFHQ
    for name in shards:
        path = hf_hub_download(repository, name, repo_type="dataset", revision=revision,
                               local_dir=args.output / "ffhq_shards")
        images = extract_shard(Path(path), args.output / "ffhq")
        manifest[f"{repository}/{name}"] = {"revision": revision, "images": images}
        print(path, images, flush=True)
    (args.output / "sources.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)
                                              + "\n")
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
