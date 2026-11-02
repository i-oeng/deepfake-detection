"""Summarize OpenFake's metadata for open-world round 4 without downloading images.

Reads only the small columns (label, generator, type, prompt, image hash) of every
parquet shard of ``ComplexDataLab/OpenFake`` at a pinned revision, never the image
bytes, and reports:

- rows per split and generator, and the generators found only in the test split
  (the held-out families of the round 4 draft);
- per generator, prompts that mention a person and prompts that name an LFW
  public figure. Both are text heuristics; YuNet decides on faces later;
- how many OpenFakeTiny images (fresh v3, now validation) reappear in OpenFake;
- how many row groups, and image bytes streamed, two quota plans for
  person-prompt training rows per generator family would need. Whole row groups
  are the unit: the image column has no page index, so single images cannot be
  fetched on their own.

Each shard's metadata is cached, so an interrupted pass resumes. Run with
``PYTHONPATH=.genlibs:src`` for pyarrow, httpx, and huggingface_hub.
"""

from __future__ import annotations

import argparse
import heapq
import io
import json
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

HUB = "https://huggingface.co"
OPENFAKE = ("ComplexDataLab/OpenFake", "3247b942f18d39d1fd476b1077852af033d35d8c")
# The OpenFakeTiny and LFW revisions are those pinned in scripts/download_round3_sources.py.
OPENFAKE_TINY = ("ComplexDataLab/OpenFakeTiny", "9a7f5a7b37490f1b7d557c285eef52b09a90d0c2")
LFW = ("bitmind/lfw", "274dfbca8b2dc9c6d21ae80f8627c911222eb347",
       "data/train-00000-of-00001.parquet")
COLUMNS = ("label", "model", "type", "prompt", "image_sha256")
PERSON = re.compile(
    r"\b(?:man|men|woman|women|person|persons|people|individuals?|boys?|girls?|child|children"
    r"|kids?|bab(?:y|ies)|lad(?:y|ies)|gentlem[ae]n|guys?|couples?|portraits?|selfies?|faces?"
    r"|headshots?|actors?|actress(?:es)?|singers?|rappers?|politicians?|president|athletes?"
    r"|players?|brides?|grooms?|soldiers?|teen(?:ager)?s?|students?|workers?|crowd|family"
    r"|mother|father|son|daughter|wife|husband|he|she|his|her|him)\b",
    re.IGNORECASE,
)
CAPITALIZED_RUN = re.compile(r"[A-Z][\w'’-]*\.?(?:\s+[A-Z][\w'’-]*\.?){1,4}")
POSSESSIVE = re.compile(r"['’]s$")
# Not photo generators, so the round 4 draft keeps them out of photo training:
# video models, anime or illustration models, and a random-weight test model.
EXCLUDED = re.compile(
    r"video|t2v|wan|hunyuan|ltxv|sora|veo|anim|illustrious|noobai|ghibli|^pony$|tiny-random"
)
# Generator families for download quotas; the first matching pattern wins. Fine-tunes
# and LoRAs join their base model's family.
FAMILIES = (
    ("flux2", r"^flux\.2"), ("flux1", r"flux|chroma|mystic"), ("sd3", r"^sd-3"),
    ("sd2", r"^sd-2|^sd-turbo"), ("sd1", r"^sd-1|stable-diffusion-v1|realistic_vision"),
    ("playground", r"playground"),
    ("sdxl", r"sdxl|sd-xl|stable-diffusion-xl|realvisxl|cyberrealistic|pony"),
    ("hidream", r"hidream"), ("qwen", r"qwen"), ("kolors", r"kolors"),
    ("midjourney", r"midjourney"), ("gpt_image", r"gpt-image"), ("dalle", r"dalle"),
    ("ideogram", r"ideogram"), ("imagen", r"imagen"), ("grok", r"grok"),
    ("seedream", r"seedream"), ("nano_banana", r"nano-banana"),
)
# Person-prompt rows to collect: (rows per real source, then rows per fake family
# for families with at least the given number of rows available). Families below
# every threshold keep only the rows that the picked row groups bring along.
PLANS = {
    "A": (15000, ((20000, 3000),)),
    "B": (15000, ((20000, 3000), (6000, 2000))),
}
RETRY_STATUS = {429, 500, 502, 503, 504}


def public_figures(prompt: str, names: set[str]) -> list[str]:
    """LFW full names (two to five capitalized words) that occur in a prompt."""
    found: set[str] = set()
    for run in CAPITALIZED_RUN.finditer(prompt):
        words = [POSSESSIVE.sub("", word.rstrip(".")) for word in run.group(0).split()]
        for size in range(min(5, len(words)), 1, -1):
            for start in range(len(words) - size + 1):
                candidate = " ".join(words[start:start + size])
                if candidate in names:
                    found.add(candidate)
    return sorted(found)


def family(model: str) -> str:
    for name, pattern in FAMILIES:
        if re.search(pattern, model):
            return name
    return f"other:{model}"


def lfw_name(filename: str) -> str:
    """``Aaron_Eckhart_0001.jpg`` becomes ``Aaron Eckhart``."""
    return PurePosixPath(filename).stem.rsplit("_", 1)[0].replace("_", " ")


def plan_download(units: dict, costs: dict, quotas: dict) -> dict:
    """Greedily pick row groups until every key's quota is met or nothing is left.

    ``units`` maps a row group to a Counter of rows per key. Coverage is
    submodular, so stale gains are only recomputed when they reach the top.
    """
    deficit = dict(quotas)

    def gain(unit) -> int:
        return sum(min(count, deficit.get(key, 0)) for key, count in units[unit].items())

    heap = [(-gain(unit), unit) for unit in sorted(units)]
    heapq.heapify(heap)
    picked = []
    while heap and any(deficit.values()):
        stale, unit = heapq.heappop(heap)
        current = gain(unit)
        if current == 0:
            continue
        if heap and current < -heap[0][0] and current < -stale:
            heapq.heappush(heap, (-current, unit))
            continue
        picked.append(unit)
        for key, count in units[unit].items():
            if key in deficit:
                deficit[key] = max(0, deficit[key] - count)
    return {
        "picked": picked,
        "bytes": sum(costs[unit] for unit in picked),
        "shortfall": {key: quotas[key] - left for key, left in deficit.items() if left},
    }


def _send(client, method: str, url: str, **kwargs):
    import httpx

    for attempt in range(7):
        try:
            response = client.request(method, url, follow_redirects=False, **kwargs)
            if response.status_code not in RETRY_STATUS:
                return response
        except httpx.TransportError:
            pass
        time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"giving up on {method} {url}")


class RemoteFile(io.RawIOBase):
    """A read-only, seekable Hub file served by HTTP range requests on its CDN URL.

    The Hub's resolve endpoint is called once per file; byte ranges then go to the
    redirect target, so the many small reads do not count against Hub limits.
    """

    def __init__(self, client, repo: str, revision: str, path: str) -> None:
        super().__init__()
        self._client = client
        self._url = f"{HUB}/datasets/{repo}/resolve/{revision}/{path}"
        self._position = 0
        self._resolve()

    def _resolve(self) -> None:
        response = _send(self._client, "HEAD", self._url)
        if response.is_redirect:
            self._target = response.headers["location"]
            self.size = int(response.headers["x-linked-size"])
        elif response.status_code == 200:
            self._target = self._url
            self.size = int(response.headers["content-length"])
        else:
            raise RuntimeError(f"HEAD {self._url}: {response.status_code}")

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._position, io.SEEK_END: self.size}[whence]
        self._position = base + offset
        return self._position

    def readinto(self, buffer) -> int:
        stop = min(self._position + len(buffer), self.size)
        if stop <= self._position:
            return 0
        for attempt in range(2):
            response = _send(self._client, "GET", self._target,
                             headers={"Range": f"bytes={self._position}-{stop - 1}"})
            if response.status_code == 206:
                break
            if response.status_code == 403 and attempt == 0:
                self._resolve()  # the signed CDN URL expired
                continue
            raise RuntimeError(f"range read on {self._url}: {response.status_code}")
        data = response.content
        buffer[: len(data)] = data
        self._position += len(data)
        return len(data)


def _split(path: str) -> str:
    """``core/train-00000-of-00032-00000.parquet`` belongs to split ``core/train``."""
    config, name = path.split("/", 1)
    return f"{config}/{name.split('-', 1)[0]}"


def scan_shard(client, repo: str, revision: str, path: str, cache: Path, names: set[str]) -> Path:
    """Write one shard's metadata (no image bytes) to the cache and return its path."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    target = cache / (path.replace("/", "__"))
    if target.exists():
        return target
    parquet = pq.ParquetFile(RemoteFile(client, repo, revision, path), pre_buffer=True)
    columns = [name for name in COLUMNS if name in parquet.schema_arrow.names]
    tables, image_bytes = [], []
    for group in range(parquet.metadata.num_row_groups):
        meta = parquet.metadata.row_group(group)
        image_bytes.append(sum(meta.column(i).total_compressed_size
                               for i in range(meta.num_columns)
                               if meta.column(i).path_in_schema.startswith("image.")))
        table = parquet.read_row_group(group, columns=columns)
        tables.append(table.append_column("row_group",
                                          pa.array([group] * table.num_rows, pa.int32())))
    table = pa.concat_tables(tables)
    prompts = table.column("prompt").to_pylist() if "prompt" in columns else [""] * len(table)
    table = table.append_column("person", pa.array([bool(PERSON.search(p or "")) for p in prompts]))
    table = table.append_column("public_figures", pa.array(
        ["|".join(public_figures(p or "", names)) for p in prompts]))
    table = table.replace_schema_metadata({
        "split": _split(path), "file": path, "row_group_image_bytes": json.dumps(image_bytes)})
    partial = target.with_suffix(".partial")
    pq.write_table(table, partial)
    partial.replace(target)
    return target


def _scan(client, repo, revision, cache, names, workers, prefixes=("core/", "reddit/")):
    from huggingface_hub import HfApi

    paths = sorted(path for path in HfApi().list_repo_files(repo, repo_type="dataset",
                                                            revision=revision)
                   if path.endswith(".parquet") and path.startswith(prefixes))
    cache.mkdir(parents=True, exist_ok=True)
    done = 0
    with ThreadPoolExecutor(workers) as pool:
        for _ in pool.map(lambda p: scan_shard(client, repo, revision, p, cache, names), paths):
            done += 1
            if done % 25 == 0 or done == len(paths):
                print(f"{repo}: {done}/{len(paths)} shards", flush=True)
    return [cache / path.replace("/", "__") for path in paths]


def _load(paths: list[Path], columns: list[str]):
    import pandas as pd
    import pyarrow.parquet as pq

    frames = []
    for path in paths:
        table = pq.read_table(path, columns=columns)
        metadata = {key.decode(): value.decode()
                    for key, value in (table.schema.metadata or {}).items()}
        frame = table.to_pandas()
        frame["split"], frame["file"] = metadata["split"], metadata["file"]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _counts(series) -> dict:
    return {str(key): int(value) for key, value in series.value_counts().sort_index().items()}


def summarize(paths: list[Path], tiny_paths: list[Path], lfw_count: int) -> dict:
    import pyarrow.parquet as pq

    rows = _load(paths, ["label", "model", "type", "image_sha256", "row_group", "person",
                         "public_figures"])
    rows["public"] = rows["public_figures"] != ""
    summary: dict = {
        "source": {"repo": OPENFAKE[0], "revision": OPENFAKE[1], "shards": len(paths)},
        "heuristics": {
            "person_prompt": "prompt matches PERSON in scripts/openfake_metadata.py",
            "public_figure": f"prompt contains one of {lfw_count} LFW full names "
                             f"({LFW[0]} revision {LFW[1][:8]})",
        },
        "splits": {},
    }
    for split, group in rows.groupby("split", sort=True):
        models = {}
        for model, part in group.groupby("model", sort=True):
            types = part["type"].dropna().mode()
            models[model] = {
                "label": "/".join(sorted(part["label"].unique())),
                "type": str(types.iat[0]) if len(types) else None,
                "rows": int(len(part)), "person": int(part["person"].sum()),
                "public_figure": int(part["public"].sum()),
            }
        summary["splits"][split] = {
            "rows": int(len(group)), "labels": _counts(group["label"]),
            "person": int(group["person"].sum()),
            "public_figure": int(group["public"].sum()), "models": models,
        }

    def fakes(split: str) -> set[str]:
        part = rows[(rows["split"] == split) & (rows["label"] == "fake")]
        return set(part["model"])

    train, validation, test = fakes("core/train"), fakes("core/validation"), fakes("core/test")
    summary["fake_generators"] = {
        "test_only": sorted(test - train - validation),
        "validation_only": sorted(validation - train - test),
        "train_and_test": sorted(train & test),
        "train_only": sorted(train - test),
    }

    public = rows[rows["public"]]
    names = Counter(name for value in public["public_figures"] for name in value.split("|"))
    summary["public_figures"] = {
        split: {"fake_rows": int((part["label"] == "fake").sum()),
                "real_rows": int((part["label"] == "real").sum()),
                "distinct_names": len({n for v in part["public_figures"] for n in v.split("|")})}
        for split, part in public.groupby("split", sort=True)
    }
    summary["public_figures"]["top_names"] = names.most_common(25)

    if tiny_paths:
        tiny = _load(tiny_paths, [c for c in ("label", "model", "image_sha256")
                                  if c in pq.read_schema(tiny_paths[0]).names])
        if "image_sha256" in tiny:
            matched = rows[rows["image_sha256"].isin(set(tiny["image_sha256"]))]
            summary["openfaketiny_overlap"] = {
                "tiny_rows": int(len(tiny)),
                "tiny_rows_in_openfake": int(tiny["image_sha256"].isin(
                    set(rows["image_sha256"])).sum()),
                "openfake_rows_by_split": _counts(matched["split"]),
            }
        else:
            summary["openfaketiny_overlap"] = {"available": False,
                                               "reason": "OpenFakeTiny has no image_sha256"}

    training = training_rows(rows)
    units, available = training_units(training)
    members: dict = defaultdict(Counter)
    for key, model in zip(training["key"], training["model"], strict=True):
        members[key][model] += 1
    costs = row_group_costs(paths)
    summary["download_plan"] = {
        "excluded_generators": sorted({m for m in rows["model"] if EXCLUDED.search(m)}),
        "person_rows_available": {
            f"{label}/{name}": {"rows": count, "members": dict(members[(label, name)])}
            for (label, name), count in sorted(available.items())},
        "train_image_gb": round(sum(costs.values()) / 1e9, 1),
    }
    for name, (real_quota, tiers) in PLANS.items():
        plan = plan_download(units, costs, plan_quotas(available, name))
        collected = Counter()
        for unit in plan["picked"]:
            collected.update(units[unit])
        summary["download_plan"][f"plan_{name}"] = {
            "real_rows_per_source": real_quota,
            "fake_rows_per_family": {f">= {minimum} available": quota for minimum, quota in tiers},
            "row_groups": len(plan["picked"]),
            "streamed_image_gb": round(plan["bytes"] / 1e9, 1),
            "shards_touched": len({file for file, _ in plan["picked"]}),
            "rows_collected": {f"{label}/{key}": count
                               for (label, key), count in sorted(collected.items())},
        }
    return summary


def training_rows(rows):
    """Person-prompt core/train rows eligible for photo training, with their quota key.

    The key is ``("real", source)`` for reals and ``("fake", family)`` for fakes.
    """
    training = rows[(rows["split"] == "core/train") & rows["person"]]
    keep = (training["label"] == "real") | ~training["model"].str.contains(EXCLUDED)
    training = training[keep].copy()
    training["key"] = [("real", model) if label == "real" else ("fake", family(model))
                       for label, model in zip(training["label"], training["model"],
                                               strict=True)]
    return training


def training_units(training) -> tuple[dict, Counter]:
    """Rows per quota key in each (file, row group), and in total."""
    units: dict = defaultdict(Counter)
    for file, group, key in zip(training["file"], training["row_group"], training["key"],
                                strict=True):
        units[(file, int(group))][key] += 1
    return dict(units), Counter(training["key"])


def row_group_costs(paths: list[Path]) -> dict:
    """Image bytes of every core/train row group, from the cached shard metadata."""
    import pyarrow.parquet as pq

    costs = {}
    for path in paths:
        metadata = pq.read_schema(path).metadata
        if metadata[b"split"] != b"core/train":
            continue
        for group, size in enumerate(json.loads(metadata[b"row_group_image_bytes"])):
            costs[(metadata[b"file"].decode(), group)] = size
    return costs


def plan_quotas(available: Counter, name: str) -> dict:
    """The quota of each key under plan ``name``; families below every tier get none."""
    real_quota, tiers = PLANS[name]
    quotas = {}
    for key, count in available.items():
        if key[0] == "real":
            quotas[key] = real_quota
        else:
            quota = next((q for minimum, q in tiers if count >= minimum), 0)
            if quota:
                quotas[key] = quota
    return quotas


def main() -> None:
    import httpx
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--output", type=Path,
                        default=Path("reports/openworld/openfake_metadata.json"))
    parser.add_argument("--cache", type=Path, default=Path("data/raw/openfake_metadata_cache"))
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    limits = httpx.Limits(max_connections=args.workers * 2)
    with httpx.Client(timeout=120, limits=limits) as client:
        lfw = RemoteFile(client, LFW[0], LFW[1], LFW[2])
        names = {lfw_name(f) for f in pq.read_table(lfw, columns=["filename"])["filename"]
                 .to_pylist()}
        names = {name for name in names if len(name.split()) >= 2}
        paths = _scan(client, *OPENFAKE, args.cache / "openfake", names, args.workers)
        tiny = _scan(client, *OPENFAKE_TINY, args.cache / "openfaketiny", names, args.workers)
    summary = summarize(paths, tiny, len(names))
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
