"""Generate fresh v4's seen-slice fakes: our own photo prompts on three training families.

``t2i`` runs one generator. Each is a family OpenFake's train split contains, run at
its model card's settings from a pinned revision:

- ``sdxl_base``: SDXL base 1.0 alone (OpenFake ``stable-diffusion-xl-base-1.0``),
  default Euler scheduler, 40 steps, guidance 5, no negative prompt.
- ``realvisxl_v5``: RealVisXL V5.0 (OpenFake ``sdxl-realvis-v5``), DPM++ 2M Karras,
  50 steps, guidance 5, the card's negative prompt.
- ``sd35_medium``: Stable Diffusion 3.5 Medium (OpenFake ``sd-3.5``), 40 steps,
  guidance 4.5, no negative prompt. The repository is gated: the Hugging Face
  account whose token xixi uses must have accepted its license.

Prompts come from a grammar written for this test set (``prompt``), not round 2's.
The prompt and image size of index ``i`` derive from ``SEED`` (a namespace no earlier
round used) and ``i`` alone, so every generator renders the same prompts; the
initial noise also depends on the generator. Sizes vary over SDXL's photographic
aspect buckets. Adults only.

Everything fits xixi's 8 GB GPU. The SDXL pipelines use model CPU offload (one
component on the GPU at a time). SD 3.5's T5 encoder alone is larger than the GPU,
so its prompts are first encoded with sequential offload (one layer on the GPU at a
time), the encoders are freed, and the transformer and VAE then denoise with model
offload.

Resumable chunks: each image is written as soon as it is done, an index whose PNG
exists is skipped, and ``--limit`` bounds how many new images one run makes, so a
queued job holds the shared GPU for a bounded time and a rerun continues. Raising
``--count`` later extends the same sequence. ``provenance.json`` pins the generator,
revision, settings, seed and grammar to the output directory. A run exits with
status 3, before loading any model, when the GPU has less than ``MIN_FREE_GPU``
free (another process, such as an Ollama model, holds it), and with status 4 when
the disk has less than ``MIN_FREE_DISK`` free.

Run with the isolated diffusers install of round 2: ``PYTHONPATH=.genlibs:src``.
SD 3.5 also needs ``protobuf`` (6.33.6 on xixi, in ``.genlibs``): its T5 tokenizer
config sets ``add_prefix_space``, so transformers rebuilds the fast tokenizer from
the SentencePiece model on load.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import random
import shutil
import time
from pathlib import Path

try:
    from scripts.generate_round2 import (
        _ensure_provenance,
        _log_cuda_memory,
        _positive_int,
        _read_records,
        _write_records,
    )
except ModuleNotFoundError:  # run as ``python scripts/generate_round4.py``
    from generate_round2 import (
        _ensure_provenance,
        _log_cuda_memory,
        _positive_int,
        _read_records,
        _write_records,
    )

SEED = "round4-v4-photos-seen-20261103"
GRAMMAR = "round4-photo-v1"
SCHEMA_VERSION = 1
MIN_FREE_GPU = 7.0e9  # bytes; an RTX 5060 has about 8.1e9 with only the desktop running
MIN_FREE_DISK = 5e9
ENCODE_BATCH = 8
REALVIS_NEGATIVE = ("bad hands, bad anatomy, ugly, deformed, face asymmetry, eyes asymmetry, "
                    "deformed eyes, deformed mouth, open mouth")
GENERATORS = {
    "sdxl_base": {
        "repo": "stabilityai/stable-diffusion-xl-base-1.0",
        "revision": "462165984030d82259a11f4367a4eed129e94a7b",
        "pipeline": "sdxl", "variant": "fp16", "dtype": "float16", "scheduler": "default",
        "steps": 40, "guidance": 5.0, "negative_prompt": "",
    },
    "realvisxl_v5": {
        "repo": "SG161222/RealVisXL_V5.0",
        "revision": "ac93e0dda1f6d448cae19bbfab8c5e720a5e48bc",
        "pipeline": "sdxl", "variant": "fp16", "dtype": "float16",
        "scheduler": "dpmpp_2m_karras", "steps": 50, "guidance": 5.0,
        "negative_prompt": REALVIS_NEGATIVE,
    },
    "sd35_medium": {
        "repo": "stabilityai/stable-diffusion-3.5-medium",
        "revision": "b940f670f0eda2d07fbb75229e779da1ad11eb80",
        "pipeline": "sd3", "variant": None, "dtype": "bfloat16", "scheduler": "default",
        "steps": 40, "guidance": 4.5, "negative_prompt": "",
    },
}
SIZES = ((1024, 1024), (896, 1152), (1152, 896), (832, 1216), (1216, 832))  # width, height

SHOTS = ("A close-up photo of", "A portrait photo of", "A candid photo of", "A selfie of",
         "A professional headshot of", "A street portrait of", "A snapshot of",
         "A documentary photo of", "An editorial photo of", "A family photo of")
ORIGINS = ("East Asian", "South Asian", "Southeast Asian", "Black", "white", "Hispanic",
           "Middle Eastern", "Pacific Islander", "Central Asian", "North African")
PEOPLE = ("man", "woman")
AGES = ("in their early twenties", "in their late twenties", "in their thirties",
        "in their forties", "in their fifties", "in their sixties", "in their seventies",
        "in their eighties")
EXPRESSIONS = ("smiling", "laughing", "with a calm expression", "looking thoughtful",
               "squinting in the sun", "looking straight at the camera",
               "glancing away from the camera", "mid-conversation", "with a slight frown",
               "grinning")
APPEARANCE = ("", "", "wearing glasses", "with freckles", "with a shaved head", "with braids",
              "with a ponytail", "wearing a headscarf", "wearing a baseball cap",
              "with gray hair", "with curly hair", "with a tattoo on the neck",
              "wearing earrings", "wearing a knit sweater", "wearing a suit",
              "wearing a denim jacket", "wearing a hoodie", "wearing a work uniform")
SCENES = ("at a birthday party", "at a wedding reception", "in a busy cafe",
          "on a city sidewalk", "in a kitchen at home", "at an outdoor market",
          "on a hiking trail", "at the beach", "in an open-plan office", "at a concert",
          "in a park in autumn", "on a commuter train", "in a classroom",
          "at a family dinner", "in a vegetable garden", "in a library",
          "in the stands at a football game", "on an apartment balcony", "in a hotel lobby",
          "in front of a brick wall", "in the driver's seat of a car", "at a street festival",
          "in a photo studio", "in a hospital corridor", "at a bus stop in the rain")
LIGHT = ("natural light", "golden hour light", "overcast daylight", "direct flash",
         "neon signs at night", "soft window light", "warm tungsten light", "dappled shade",
         "backlit by the sun", "harsh noon sun")
CAMERAS = ("", "shot on a smartphone", "shot on a DSLR with an 85mm lens", "35mm film photo",
           "shot on a mirrorless camera", "point-and-shoot camera photo")


def _rng(*parts: object) -> random.Random:
    digest = hashlib.sha256("\0".join(map(str, (SEED, *parts))).encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "little"))


def prompt(index: int) -> str:
    """A deterministic, varied photographic prompt for one adult (the same for every generator)."""
    rng = _rng("prompt", index)
    subject = (f"{rng.choice(SHOTS)} a {rng.choice(ORIGINS)} {rng.choice(PEOPLE)} "
               f"{rng.choice(AGES)}")
    parts = (subject, rng.choice(EXPRESSIONS), rng.choice(APPEARANCE), rng.choice(SCENES),
             rng.choice(LIGHT), rng.choice(CAMERAS))
    return ", ".join(part for part in parts if part)


def image_size(index: int) -> tuple[int, int]:
    """Width and height of index ``index``, from SDXL's photographic aspect buckets."""
    return _rng("size", index).choice(SIZES)


def noise_seed(generator: str, index: int) -> int:
    digest = hashlib.sha256(f"{SEED}\0{generator}\0noise\0{index}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % 2**63


def record(generator: str, index: int) -> dict:
    width, height = image_size(index)
    return {"index": index, "generator": generator, "seed": SEED, "grammar": GRAMMAR,
            "prompt": prompt(index), "width": width, "height": height,
            "noise_seed": noise_seed(generator, index), **GENERATORS[generator]}


def sync_metadata(output: Path, generator: str) -> dict:
    """Check ``metadata.jsonl`` against the contract and add records for PNGs found."""
    path = output / "metadata.jsonl"
    records = _read_records(path, "index")
    for index, existing in records.items():
        if not isinstance(index, int) or index < 0 or existing != record(generator, index):
            raise ValueError(f"metadata differs from this generator's contract: index {index!r}")
    for image in output.glob("*.png"):
        if not image.stem.isdigit():
            raise ValueError(f"unexpected file in the output: {image}")
        records.setdefault(int(image.stem), record(generator, int(image.stem)))
    _write_records(path, records, "index")
    return records


def pending(output: Path, count: int, limit: int | None) -> list[int]:
    todo = [index for index in range(count) if not (output / f"{index:05d}.png").is_file()]
    return todo if limit is None else todo[:limit]


def _torch_generator(generator: str, index: int):
    import torch

    return torch.Generator("cpu").manual_seed(noise_seed(generator, index))


def _sdxl(generator: str, settings: dict, indices: list[int]):
    import torch
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline

    pipe = StableDiffusionXLPipeline.from_pretrained(
        settings["repo"], revision=settings["revision"], variant=settings["variant"],
        torch_dtype=getattr(torch, settings["dtype"]), use_safetensors=True)
    if settings["scheduler"] == "dpmpp_2m_karras":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, algorithm_type="dpmsolver++", solver_order=2,
            use_karras_sigmas=True)
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)

    def run(index: int):
        width, height = image_size(index)
        return pipe(prompt(index), negative_prompt=settings["negative_prompt"] or None,
                    width=width, height=height, num_inference_steps=settings["steps"],
                    guidance_scale=settings["guidance"],
                    generator=_torch_generator(generator, index)).images[0]

    return run


def _sd3(generator: str, settings: dict, indices: list[int]):
    import torch
    from diffusers import StableDiffusion3Pipeline

    dtype = getattr(torch, settings["dtype"])
    source = {"revision": settings["revision"], "torch_dtype": dtype}
    encoder = StableDiffusion3Pipeline.from_pretrained(
        settings["repo"], transformer=None, vae=None, **source)
    encoder.enable_sequential_cpu_offload()
    embeds, negative = {}, None
    with torch.inference_mode():
        for start in range(0, len(indices), ENCODE_BATCH):
            batch = indices[start:start + ENCODE_BATCH]
            positive, negative_embeds, pooled, negative_pooled = encoder.encode_prompt(
                prompt=[prompt(index) for index in batch], prompt_2=None, prompt_3=None,
                do_classifier_free_guidance=True,
                negative_prompt=[settings["negative_prompt"]] * len(batch))
            for position, index in enumerate(batch):
                embeds[index] = (positive[position:position + 1].cpu(),
                                 pooled[position:position + 1].cpu())
            if negative is None:  # one negative prompt: keep its embedding once
                negative = (negative_embeds[:1].cpu(), negative_pooled[:1].cpu())
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
    _log_cuda_memory("prompts_encoded")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        settings["repo"], text_encoder=None, text_encoder_2=None, text_encoder_3=None,
        tokenizer=None, tokenizer_2=None, tokenizer_3=None, **source)
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)

    def run(index: int):
        width, height = image_size(index)
        # The pipeline expects embeddings where encode_prompt would have left them.
        device = pipe._execution_device
        positive, pooled = (tensor.to(device) for tensor in embeds.pop(index))
        return pipe(prompt_embeds=positive, pooled_prompt_embeds=pooled,
                    negative_prompt_embeds=negative[0].to(device),
                    negative_pooled_prompt_embeds=negative[1].to(device),
                    width=width, height=height,
                    num_inference_steps=settings["steps"], guidance_scale=settings["guidance"],
                    generator=_torch_generator(generator, index)).images[0]

    return run


def _gpu_free() -> int:
    import torch

    if not torch.cuda.is_available():
        return 0
    return torch.cuda.mem_get_info()[0]


def generate(generator: str, output: Path, count: int, limit: int | None) -> int:
    settings = GENERATORS[generator]
    output.mkdir(parents=True, exist_ok=True)
    _ensure_provenance(output, {"schema_version": SCHEMA_VERSION, "kind": "t2i",
                                "generator": generator, "seed": SEED, "grammar": GRAMMAR,
                                "settings": settings})
    records = sync_metadata(output, generator)
    todo = pending(output, count, limit)
    if not todo:
        print(f"{generator}: all {count} images exist", flush=True)
        return 0
    free = _gpu_free()
    if free < MIN_FREE_GPU:
        print(f"{generator}: only {free / 1e9:.1f} GB of GPU memory free (need "
              f"{MIN_FREE_GPU / 1e9:.1f}); another process holds the GPU, nothing loaded",
              flush=True)
        return 3
    factory = _sd3 if settings["pipeline"] == "sd3" else _sdxl
    run = factory(generator, settings, todo)
    _log_cuda_memory("loaded")
    started = time.time()
    for done, index in enumerate(todo, 1):
        if shutil.disk_usage(output).free < MIN_FREE_DISK:
            print(f"{generator}: less than {MIN_FREE_DISK / 1e9:.0f} GB of disk free; "
                  "stopped (rerun resumes)", flush=True)
            return 4
        image = run(index)
        target = output / f"{index:05d}.png"
        partial = target.with_name(f"{target.name}.part")
        image.convert("RGB").save(partial, format="PNG")
        partial.replace(target)
        records[index] = record(generator, index)
        _write_records(output / "metadata.jsonl", records, "index")
        if done % 10 == 0 or done == len(todo):
            rate = (time.time() - started) / done
            print(f"{generator}: {done}/{len(todo)} this run, {len(records)}/{count} in all, "
                  f"{rate:.0f} s per image", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    t2i = commands.add_parser("t2i", help="one seen-slice generator")
    t2i.add_argument("--generator", choices=sorted(GENERATORS), required=True)
    t2i.add_argument("--count", type=_positive_int, required=True,
                     help="indices 0 to count - 1 make up the full set")
    t2i.add_argument("--limit", type=_positive_int, help="make at most this many new images")
    t2i.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    return generate(args.generator, args.output, args.count, args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
