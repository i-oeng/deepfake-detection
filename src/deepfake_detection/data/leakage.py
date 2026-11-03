"""Leakage control shared by the open-world rounds: exact and near-duplicate matches.

An image leaks into an evaluation set when some evaluation image has the same file
(``file_sha256``), the same decoded pixels (``pixel_sha256``) or the same 64-bit
dHash (``dhash64``), all computed by the manifest build after canonical
preprocessing, or when their 32 x 32 grayscale thumbnails correlate at
``NEAR_DUPLICATE_CORRELATION`` or more. A dHash alone cannot find near-duplicates
among aligned faces (in round 3 the nearest unrelated evaluation face lay within
6 bits for a quarter of training images); the thumbnail correlation, with its
threshold calibrated on unrelated images in round 3, finds resized and
re-encoded copies instead.

Round 3 (``data.round3``) introduced these checks. Round 4 applies the same
functions to fresh v4 and to its training additions.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .prune import FINGERPRINTS
from .round2 import _read_manifest

THUMBNAIL_SIDE = 32
# Thumbnail correlation at or above which two images count as one (calibrated in
# reports/openworld/round3_protocol.md).
NEAR_DUPLICATE_CORRELATION = 0.99
HISTOGRAM_EDGES = (0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99)

# A test-only manifest directory and the data root its relative paths resolve against.
Reference = tuple[Path, Path]


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


def histogram(values) -> dict[str, int]:
    """How many nearest correlations reach each edge, for the audit record."""
    return {f">={edge}": int((values >= edge).sum()) for edge in HISTOGRAM_EDGES}


def reserved_fingerprints(rows: list[dict[str, str]]) -> dict[str, set[str]]:
    return {column: {row[column] for row in rows if row[column]} for column in FINGERPRINTS}


def exact_hits(row: dict[str, str], reserved: dict[str, set[str]]) -> list[str]:
    """The fingerprint columns of ``row`` that occur among ``reserved`` values."""
    return [column for column in FINGERPRINTS if row[column] and row[column] in reserved[column]]


def evaluation_rows(manifest_dir: Path, data_root: Path, references: list[Reference], *,
                    read=_read_manifest) -> list[dict[str, str]]:
    """Every row a model is scored on, with its image path in ``_path``.

    These are the non-train rows of ``manifest_dir`` (``_own`` set) and all rows of
    each reference (test-only) manifest. ``read`` loads one manifest's rows.
    """
    rows = [{**row, "_path": str(data_root / row["relative_path"]), "_own": True}
            for row in read(manifest_dir) if row["split"] != "train"]
    for directory, root in references:
        rows += [{**row, "_path": str(root / row["relative_path"]), "_own": False}
                 for row in read(directory)]
    return rows


def find_leaks(queries: list[dict[str, str]], query_paths: list[str],
               references: list[dict[str, str]], reference_paths: list[str], *,
               threshold: float = NEAR_DUPLICATE_CORRELATION, workers: int = 8):
    """For each query row, why it matches a reference row ("" when it does not).

    The reason is the first matching fingerprint column, else
    ``thumbnail_correlation``. Also returns each query's nearest correlation, for
    the threshold histogram.
    """
    reserved = reserved_fingerprints(references)
    correlation = nearest_correlation(thumbnails(query_paths, workers),
                                      thumbnails(reference_paths, workers))
    reasons = []
    for row, value in zip(queries, correlation, strict=True):
        exact = exact_hits(row, reserved)
        reasons.append(exact[0] if exact else
                       "thumbnail_correlation" if value >= threshold else "")
    return reasons, correlation
