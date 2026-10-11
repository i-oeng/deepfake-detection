"""Parameter count, latency, and peak GPU memory for each benchmark architecture.

Latency is the median of timed forward passes under the same mixed precision used
for evaluation, after warm-up, at batch sizes 1 and 32. A late-fusion detector runs
both branches, so its cost is the sum of the RGB and frequency rows.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import yaml

from .benchmark import BinaryEncoder

CONFIGS = {
    "ResNet18": "configs/training/df40_unseen_resnet18.yaml",
    "ConvNeXt-Tiny": "configs/training/df40_unseen_convnext_tiny.yaml",
    "CLIP ViT-B/16": "configs/training/df40_unseen_clip_vit_b16.yaml",
    "Frequency": "configs/training/df40_unseen_frequency_all_specfilter.yaml",
}


def _time(model: torch.nn.Module, batch: int, device: torch.device, repeats: int) -> dict:
    images = torch.randn(batch, 3, 224, 224, device=device)
    timings = []
    with torch.inference_mode():
        for step in range(repeats + 10):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            with torch.autocast(device_type="cuda"):
                model(images)
            torch.cuda.synchronize(device)
            if step >= 10:
                timings.append(time.perf_counter() - start)
    median = statistics.median(timings)
    return {"batch": batch, "median_ms": median * 1000, "images_per_second": batch / median}


def profile(root: Path, repeats: int = 50) -> dict:
    device = torch.device("cuda")
    result = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "models": {}}
    for name, path in CONFIGS.items():
        config = yaml.safe_load((root / path).read_text(encoding="utf-8"))["experiment"]
        model = BinaryEncoder(config["model"], config).to(device).eval()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        timings = [_time(model, batch, device, repeats) for batch in (1, 32)]
        result["models"][name] = {
            "parameters": sum(p.numel() for p in model.parameters()),
            "latency": timings,
            "peak_memory_mib_batch32": torch.cuda.max_memory_allocated(device) / 2**20,
        }
        del model
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = profile(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
