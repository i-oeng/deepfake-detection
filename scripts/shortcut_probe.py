"""Check whether low-level image statistics alone separate real from fake per domain.

For every domain of a manifest, each statistic below is used directly as a score,
and the image-level AUROC is reported as max(AUROC, 1 - AUROC). A value near 0.5
means the statistic carries no label information. A high value flags a domain
whose real and fake images differ in capture or encoding rather than (only) in
manipulation, so detector results on it need that caveat.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from deepfake_detection.train.metrics import auroc

STATISTICS = ("laplacian_variance", "jpeg75_bytes", "high_frequency_ratio", "saturation",
              "luma_std")


def statistics(path: str) -> dict[str, float]:
    import cv2

    cv2.setNumThreads(1)
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    height, width = gray.shape
    ys, xs = np.ogrid[:height, :width]
    radius = np.hypot(ys - height / 2, xs - width / 2)
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=75)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return {
        "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_32F).var()),
        "jpeg75_bytes": float(buffer.tell()),
        "high_frequency_ratio": float(spectrum[radius > min(height, width) / 4].sum()
                                      / spectrum.sum()),
        "saturation": float(hsv[..., 1].mean()),
        "luma_std": float(gray.std()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["validation", "test"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    with args.manifest.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] in args.splits]
    paths = [str(args.data_root / row["relative_path"]) for row in rows]
    with ProcessPoolExecutor(args.workers) as pool:
        values = list(pool.map(statistics, paths, chunksize=64))
    grouped: dict[tuple[str, str], list[tuple[int, dict]]] = defaultdict(list)
    for row, value in zip(rows, values, strict=True):
        grouped[(row["split"], row["source_domain"])].append((int(row["label"] == "FAKE"), value))
    result = {}
    for (split, domain), items in sorted(grouped.items()):
        labels = [label for label, _ in items]
        separability = {}
        for name in STATISTICS:
            area = auroc(labels, [value[name] for _, value in items])
            separability[name] = round(max(area, 1 - area), 4)
        result[f"{split}/{domain}"] = {"images": len(items), "fake": sum(labels),
                                       "max": max(separability.values()), **separability}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for key, value in result.items():
        print(f"{key:32} max={value['max']:.3f}  " + "  ".join(
            f"{name}={value[name]:.3f}" for name in STATISTICS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
