"""Source format matching: JPEG-encode non-JPEG sources so format cannot reveal the label.

In round 4's OpenFake download, 54% of the fakes come from PNG or WebP files and
98.8% of the reals from JPEG files, so source format alone separates the labels at
a cross-validated AUROC of 0.76. Canonical JPEG 90 does not remove a source's
earlier JPEG compression. Before canonicalization, every image whose source is not
a JPEG is therefore JPEG-encoded once (``reports/openworld/round4_protocol.md``,
"Source format matching"):

- In each collection, the label with more JPEG sources is the reference (ties go
  to the reals). Its JPEG sources give the reference distribution of (quality,
  chroma subsampling) pairs.
- Non-JPEG images of the other label take the complement: the reference
  distribution scaled to that label's size, minus the pairs its own JPEG sources
  already have (clipped at zero), so the label as a whole matches the reference.
- Non-JPEG images of the reference label take the reference distribution.
- JPEG sources are never re-encoded. A collection without JPEG sources is left as
  it is.

Pairs are allocated in exact proportion (largest remainder) and handed out in the
order of a seeded hash of each image's SHA-256, so the assignment is deterministic
and independent of row order. Qualities are libjpeg (IJG) qualities: the download
records estimate each JPEG source's quality as the IJG quality with the closest
luminance table, which encoding at that quality reproduces.
"""

from __future__ import annotations

import csv
import hashlib
import io
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

SEED = "round4-format-20261103"
SUBSAMPLINGS = ("4:4:4", "4:2:2", "4:2:0")
JPEG_FORMATS = frozenset({"JPEG", "MPO"})
FIELDS = ("image_id", "jpeg_quality", "subsampling")

Encoding = tuple[int, str]


@dataclass(frozen=True)
class Source:
    """One catalog image and the encoding of the file it was cut from."""

    image_id: str
    collection: str
    label: str
    sha256: str
    format: str  # Pillow's format name for the source file
    jpeg_quality: int | None = None
    subsampling: str | None = None

    @property
    def jpeg(self) -> bool:
        return self.format.upper() in JPEG_FORMATS

    @property
    def pair(self) -> Encoding | None:
        if not self.jpeg or self.jpeg_quality is None or self.subsampling not in SUBSAMPLINGS:
            return None
        return (int(self.jpeg_quality), self.subsampling)


def _rank(seed: str, source: Source) -> str:
    return hashlib.sha256(f"{seed}\0{source.sha256}\0{source.image_id}".encode()).hexdigest()


def allocate(weights: dict, total: int) -> Counter:
    """Split ``total`` items in proportion to ``weights`` by the largest remainder."""
    weight_sum = sum(weights.values())
    if total <= 0:
        return Counter()
    if weight_sum <= 0:
        raise ValueError("cannot allocate items without positive weights")
    exact = {key: total * weight / weight_sum for key, weight in weights.items()}
    counts = Counter({key: int(value) for key, value in exact.items()})
    left = total - sum(counts.values())
    for key in sorted(exact, key=lambda key: (counts[key] - exact[key], key))[:left]:
        counts[key] += 1
    return +counts


def _distance(first: Counter, second: Counter) -> float:
    """Total variation distance between two pair distributions."""
    a, b = sum(first.values()), sum(second.values())
    if not a or not b:
        return 0.0
    return 0.5 * sum(abs(first[key] / a - second[key] / b) for key in set(first) | set(second))


def assign(sources: list[Source], seed: str = SEED) -> tuple[dict[str, Encoding], dict]:
    """The (quality, subsampling) pair for every non-JPEG source, and a per-collection summary."""
    if len({source.image_id for source in sources}) != len(sources):
        raise ValueError("image ids must be unique")
    by_collection: dict[str, list[Source]] = defaultdict(list)
    for source in sources:
        by_collection[source.collection].append(source)
    assignments: dict[str, Encoding] = {}
    summary: dict[str, dict] = {}
    for collection, members in sorted(by_collection.items()):
        labels = sorted({source.label for source in members})
        jpeg = {label: sum(s.jpeg for s in members if s.label == label) for label in labels}
        pairs = {label: Counter(s.pair for s in members if s.label == label and s.pair)
                 for label in labels}
        reference = max(labels, key=lambda label: (jpeg[label], label.upper() == "REAL"))
        base = pairs[reference]
        record = {"reference": reference if base else None, "jpeg_sources": jpeg,
                  "encoded": {}}
        final = {label: Counter(pairs[label]) for label in labels}
        for label in (labels if base else ()):
            todo = sorted((s for s in members if s.label == label and not s.jpeg),
                          key=lambda source: _rank(seed, source))
            if not todo:
                continue
            weights: dict = dict(base)
            if label != reference:
                scale = sum(s.label == label for s in members) / sum(base.values())
                gap = {pair: count * scale - pairs[label][pair] for pair, count in base.items()}
                weights = {pair: value for pair, value in gap.items() if value > 0} or weights
            counts = allocate(weights, len(todo))
            handed = [pair for pair in sorted(counts) for _ in range(counts[pair])]
            for source, pair in zip(todo, handed, strict=True):
                assignments[source.image_id] = pair
            final[label].update(handed)
            record["encoded"][label] = len(todo)
        if len(labels) == 2 and base:
            record["pair_distance"] = round(_distance(*(final[label] for label in labels)), 4)
        summary[collection] = record
    return assignments, summary


def encode_jpeg(data: bytes, encoding: Encoding) -> bytes:
    """``data`` decoded to RGB (first frame) and JPEG-encoded once at ``encoding``."""
    from PIL import Image

    quality, subsampling = encoding
    with Image.open(io.BytesIO(data)) as image:
        image.seek(0)
        rgb = image.convert("RGB")
    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=quality, subsampling=subsampling)
    return buffer.getvalue()


def write_encodings(path: Path, assignments: dict[str, Encoding]) -> str:
    """Write the assignment as CSV; return its SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(FIELDS)
        for image_id, (quality, subsampling) in sorted(assignments.items()):
            writer.writerow((image_id, quality, subsampling))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_encodings(path: Path) -> dict[str, Encoding]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    encodings = {}
    for row in rows:
        quality, subsampling = int(row["jpeg_quality"]), row["subsampling"]
        if not 1 <= quality <= 100 or subsampling not in SUBSAMPLINGS:
            raise ValueError(f"invalid encoding for {row['image_id']}: {quality} {subsampling}")
        if row["image_id"] in encodings:
            raise ValueError(f"duplicate image id in {path}: {row['image_id']}")
        encodings[row["image_id"]] = (quality, subsampling)
    return encodings
