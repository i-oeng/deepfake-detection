"""Download the pinned public sources of the fresh open-world test set (v2).

None of these generators or real-image sources occurs in the open-world v1
training, validation, or test splits:

- DeepFakeFace (OpenRL): Stable Diffusion 1.5 text-to-image, Stable Diffusion
  inpainting, and InsightFace swaps, with the IMDB-WIKI photos they derive from.
- Synthetic Faces High Quality T2I (bitmind), curated sample: FLUX.1 and SDXL faces.
- FFHQ (marcosv mirror): a hash-ranked sample of real portraits for the T2I faces.

Celeb-DF v2 raw videos, the fourth source, are already on disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import hf_hub_download

DFF = ("OpenRL/DeepFakeFace", "eb2a54fbf2567790bd3dc61d09fdaade6bb6c31c",
       ("wiki.zip", "text2img.zip", "inpainting.zip", "insight.zip"))
SFHQ = ("bitmind/SyntheticFacesHighQuality-T2I", "c7f422bdcaf4ffe1ced6da49198845c874221a03",
        ("small-sample.zip",))
FFHQ = ("marcosv/ffhq-dataset", "505f94e2ecc6db64e967e8e6c8e2c2079ea0876b")
FFHQ_IMAGES = 70_000
FFHQ_SAMPLE = 1500
SEED = "fresh-test-v2-20261008"


def ffhq_sample() -> list[str]:
    """Hash-ranked FFHQ file names; the mirror stores 10,000 images per Part folder."""
    ranked = sorted(range(FFHQ_IMAGES),
                    key=lambda i: hashlib.sha256(f"{SEED}\0{i}".encode()).hexdigest())
    return [f"Part{i // 10_000 + 1}/{i:05d}.png" for i in sorted(ranked[:FFHQ_SAMPLE])]


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
    repository, revision = FFHQ
    # One request per file: download in parallel; existing files are skipped.
    with ThreadPoolExecutor(16) as pool:
        list(pool.map(lambda name: hf_hub_download(repository, name, repo_type="dataset",
                                                   revision=revision,
                                                   local_dir=args.output / "ffhq"),
                      ffhq_sample()))
    manifest[f"{repository}/sample"] = {"revision": revision, "files": FFHQ_SAMPLE,
                                         "seed": SEED}
    (args.output / "sources.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)
                                              + "\n")
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
