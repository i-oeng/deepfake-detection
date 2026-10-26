"""Generate round-2 additions: modern text-to-image faces and autoencoder reconstructions.

Every generator and autoencoder here is absent from the open-world v1 roles and
from the fresh v2 test set (which reserves Stable Diffusion 1.5, SDXL, FLUX.1,
and their autoencoders):

- ``kandinsky22`` (train): Kandinsky 2.2 prior and decoder, MoVQ autoencoder.
- ``sana600m`` (train): Sana 600M at 512 px, DC-AE f32c32 autoencoder.
- ``stable_cascade_lite`` (validation): Stable Cascade lite prior and decoder.
- ``recon`` (train): FF++ training reals passed through MoVQ or DC-AE, giving
  pixel-aligned fakes that differ from their reals only in autoencoder traces.

Outputs are resumable: an index whose PNG exists is skipped. Run with the
isolated diffusers install: ``PYTHONPATH=.genlibs:src``.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import random
from pathlib import Path, PurePosixPath

from PIL import Image

KANDINSKY_PRIOR = ("kandinsky-community/kandinsky-2-2-prior",
                   "9fc51ad5732afc5d031724219d22e6c42179c5a8")
KANDINSKY_DECODER = ("kandinsky-community/kandinsky-2-2-decoder",
                     "9ae140d347fed8ce6e8bb3005dcc1f48543bb8e3")
SANA = ("Efficient-Large-Model/Sana_600M_512px_diffusers",
        "83d7a190bfd1fd070570a793d2dab5c7a3231b9d")
CASCADE_PRIOR = ("stabilityai/stable-cascade-prior", "7ca32c21c3b4d4e35bbb94fcfedfb4fa2259bd91")
CASCADE = ("stabilityai/stable-cascade", "a89f66d459ae653e3b4d4f992a7c3789d0dc4d16")
NEGATIVE = ("cartoon, illustration, painting, drawing, 3d render, anime, deformed face, "
            "text, watermark")
AGES = ("in their twenties", "in their thirties", "in their forties", "in their fifties",
        "in their sixties", "in their seventies")
GENDERS = ("man", "woman")
ORIGINS = ("East Asian", "South Asian", "Southeast Asian", "Black", "white", "Latino",
           "Middle Eastern", "Indigenous")
EXPRESSIONS = ("neutral expression", "slight smile", "laughing", "serious look",
               "talking", "surprised look")
LIGHTING = ("soft window light", "studio lighting", "overcast daylight", "harsh midday sun",
            "warm indoor light", "fluorescent office light", "evening street light")
FRAMING = ("head-and-shoulders portrait", "close-up portrait", "candid photo", "selfie",
           "frame from a news interview", "webcam video call screenshot",
           "passport-style photo")
DETAILS = ("", "wearing glasses", "with a beard", "wearing a hat", "with earrings",
           "with short hair", "with long hair", "with curly hair", "with gray hair")
BACKGROUNDS = ("plain background", "office", "city street", "living room", "park",
               "studio backdrop", "television studio")
SCHEMA_VERSION = 1


def _rng(seed: str, *parts: object) -> random.Random:
    digest = hashlib.sha256("\0".join(map(str, (seed, *parts))).encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "little"))


def prompt(seed: str, index: int) -> str:
    """A deterministic, varied photographic face prompt (adults only)."""
    rng = _rng(seed, "prompt", index)
    detail = rng.choice(DETAILS)
    return ", ".join(part for part in (
        f"A realistic photo, {rng.choice(FRAMING)} of a {rng.choice(ORIGINS)} "
        f"{rng.choice(GENDERS)} {rng.choice(AGES)}",
        rng.choice(EXPRESSIONS), detail, rng.choice(LIGHTING), rng.choice(BACKGROUNDS),
    ) if part)


def _generator(seed: str, name: str, index: int):
    import torch

    value = int.from_bytes(hashlib.sha256(f"{seed}\0{name}\0{index}".encode()).digest()[:8],
                           "little") % 2**63
    return torch.Generator("cpu").manual_seed(value)


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _ensure_provenance(output: Path, expected: dict) -> None:
    """Pin one generation contract to an output directory before writing payloads."""
    path = output / "provenance.json"
    if path.exists():
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError(f"output provenance differs: {path}")
        return
    if any(candidate.is_file() for candidate in output.rglob("*")):
        raise ValueError(f"output has payloads but no provenance: {output}")
    _write_json(path, expected)


def _read_records(path: Path, key: str) -> dict[object, dict]:
    records: dict[object, dict] = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                value = record[key]
            except (json.JSONDecodeError, KeyError) as error:
                raise ValueError(f"invalid metadata at {path}:{line_number}") from error
            if value in records and records[value] != record:
                raise ValueError(f"conflicting metadata for {key}={value!r} in {path}")
            records[value] = record
    return records


def _write_records(path: Path, records: dict[object, dict], key: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    ordered = sorted(records.values(), key=lambda record: str(record[key]).zfill(12))
    temporary.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in ordered),
        encoding="utf-8",
    )
    temporary.replace(path)


def _pending(output: Path, count: int, batch: int) -> list[list[int]]:
    todo = [index for index in range(count) if not (output / f"{index:05d}.png").is_file()]
    return [todo[start:start + batch] for start in range(0, len(todo), batch)]


def _t2i_record(name: str, seed: str, index: int, settings: dict) -> dict:
    return {
        "index": index,
        "generator": name,
        "seed": seed,
        "prompt": prompt(seed, index),
        **settings,
    }


def _sync_t2i_metadata(
    output: Path, name: str, seed: str, count: int, settings: dict
) -> dict[object, dict]:
    path = output / "metadata.jsonl"
    records = _read_records(path, "index")
    for index, record in records.items():
        if not isinstance(index, int) or not 0 <= index < count:
            raise ValueError(f"metadata index outside requested range: {index!r}")
        if record != _t2i_record(name, seed, index, settings):
            raise ValueError(f"metadata configuration differs for index {index}")
    for image in output.glob("*.png"):
        if not image.stem.isdigit() or not 0 <= int(image.stem) < count:
            raise ValueError(f"unexpected generated image: {image}")
        index = int(image.stem)
        records[index] = _t2i_record(name, seed, index, settings)
    _write_records(path, records, "index")
    return records


def _save(
    output: Path,
    name: str,
    seed: str,
    indices: list[int],
    images,
    settings: dict,
    records: dict[object, dict],
) -> None:
    for index, image in zip(indices, images, strict=True):
        temporary = output / f"{index:05d}.png.part"
        image.convert("RGB").save(temporary, format="PNG")
        temporary.replace(output / f"{index:05d}.png")
        records[index] = _t2i_record(name, seed, index, settings)
    _write_records(output / "metadata.jsonl", records, "index")


def _tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    """Cover one dimension completely, including a final edge-aligned tile."""
    if not 0 <= overlap < tile_size:
        raise ValueError("tile overlap must be non-negative and smaller than the tile")
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, tile_size - overlap))
    edge = length - tile_size
    if starts[-1] != edge:
        starts.append(edge)
    return starts


def _decode_movq_tiles(movq, latents, tile_size: int = 32, overlap: int = 8):
    """Decode and feather overlapping latent tiles to bound MoVQ activation memory."""
    import torch

    scale = 2 ** (len(movq.config.block_out_channels) - 1)
    y_starts = _tile_starts(latents.shape[-2], tile_size, overlap)
    x_starts = _tile_starts(latents.shape[-1], tile_size, overlap)
    output_height = latents.shape[-2] * scale
    output_width = latents.shape[-1] * scale
    images = []
    for sample in latents:
        accumulator = torch.zeros((3, output_height, output_width), dtype=torch.float32)
        normalizer = torch.zeros((1, output_height, output_width), dtype=torch.float32)
        for y_start in y_starts:
            for x_start in x_starts:
                # This call sits outside Diffusers' inference-decorated pipeline.
                # Without an explicit context, autograd retains every decoder
                # activation and exhausts the GPU even for a small tile.
                with torch.inference_mode():
                    decoded = movq.decode(
                        sample[
                            None, :, y_start:y_start + tile_size, x_start:x_start + tile_size
                        ].to("cuda", dtype=movq.dtype),
                        force_not_quantize=True,
                    )["sample"][0].float().cpu()
                top, left = y_start * scale, x_start * scale
                bottom, right = top + decoded.shape[-2], left + decoded.shape[-1]
                weight = torch.ones(
                    (1, decoded.shape[-2], decoded.shape[-1]), dtype=torch.float32
                )
                feather = min(overlap * scale, decoded.shape[-2], decoded.shape[-1])
                ramp = torch.linspace(
                    1 / (feather + 1), 1, feather, dtype=torch.float32
                )
                if top > 0:
                    weight[:, :feather] *= ramp[None, :, None]
                if bottom < output_height:
                    weight[:, -feather:] *= ramp.flip(0)[None, :, None]
                if left > 0:
                    weight[:, :, :feather] *= ramp[None, None, :]
                if right < output_width:
                    weight[:, :, -feather:] *= ramp.flip(0)[None, None, :]
                accumulator[:, top:bottom, left:right] += decoded * weight
                normalizer[:, top:bottom, left:right] += weight
                del decoded
                torch.cuda.empty_cache()
        pixels = ((accumulator / normalizer).clamp(-1, 1) + 1).mul(127.5).round().byte()
        images.append(Image.fromarray(pixels.permute(1, 2, 0).numpy()))
    return images


def _log_cuda_memory(stage: str) -> None:
    """Emit compact allocator evidence for long, remotely queued generation jobs."""
    import torch

    free, total = torch.cuda.mem_get_info()
    print(
        f"cuda_memory stage={stage} allocated={torch.cuda.memory_allocated()} "
        f"reserved={torch.cuda.memory_reserved()} free={free} total={total}",
        flush=True,
    )


def generate_kandinsky(output: Path, count: int, seed: str, batch: int) -> None:
    settings = {"revision": KANDINSKY_DECODER[1], "prior_revision": KANDINSKY_PRIOR[1],
                "size": 768, "prior_steps": 25, "decoder_steps": 30, "guidance": 4.0}
    _ensure_provenance(output, {
        "schema_version": SCHEMA_VERSION, "kind": "t2i", "generator": "kandinsky22",
        "seed": seed, "count": count, "negative_prompt": NEGATIVE, "settings": settings,
    })
    records = _sync_t2i_metadata(output, "kandinsky22", seed, count, settings)
    batches = _pending(output, count, batch)
    if not batches:
        return
    import torch
    from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline

    prior = KandinskyV22PriorPipeline.from_pretrained(
        KANDINSKY_PRIOR[0], revision=KANDINSKY_PRIOR[1], torch_dtype=torch.float16).to("cuda")
    # Image embeddings are small; computing them all first keeps one model on the GPU.
    embeds = {}
    for indices in batches:
        result = prior(
            [prompt(seed, i) for i in indices],
            negative_prompt=[NEGATIVE] * len(indices),
            num_inference_steps=25,
            guidance_scale=4.0,
            generator=[_generator(seed, "kandinsky-prior", i) for i in indices],
        )
        for position, index in enumerate(indices):
            embeds[index] = (result.image_embeds[position].cpu(),
                             result.negative_image_embeds[position].cpu())
    # Drop the final result as well as the pipeline so no prior tensors remain
    # live when the substantially larger decoder begins loading.
    del result, prior
    torch.cuda.empty_cache()
    decoder = KandinskyV22Pipeline.from_pretrained(
        KANDINSKY_DECODER[0], revision=KANDINSKY_DECODER[1], torch_dtype=torch.float16
    )
    # Kandinsky's UNet alone nearly fills xixi's 8 GB GPU. Sequential offload
    # moves its submodules independently; model offload still moves the complete
    # UNet and therefore cannot execute even a one-image batch on this machine.
    decoder.enable_sequential_cpu_offload()
    decoder.enable_attention_slicing("max")
    for indices in batches:
        latents = decoder(
            image_embeds=torch.stack([embeds[i][0] for i in indices]).to("cuda"),
            negative_image_embeds=torch.stack([embeds[i][1] for i in indices]).to("cuda"),
            height=768, width=768, num_inference_steps=30, guidance_scale=4.0,
            generator=[_generator(seed, "kandinsky-decoder", i) for i in indices],
            output_type="latent",
        ).images.cpu()
        _log_cuda_memory("denoised")
        # Sequential hooks keep their CPU weight maps alive and can retain CUDA
        # allocations across the stage boundary. Restoring all hooked modules to
        # CPU before loading MoVQ gives the image decoder a clean memory budget.
        decoder.remove_all_hooks()
        gc.collect()
        torch.cuda.empty_cache()
        _log_cuda_memory("unet_offloaded")
        from accelerate import cpu_offload

        cpu_offload(decoder.movq, torch.device("cuda"), offload_buffers=True)
        _log_cuda_memory("movq_hooked")
        images = _decode_movq_tiles(decoder.movq, latents)
        decoder.remove_all_hooks()
        gc.collect()
        torch.cuda.empty_cache()
        decoder.enable_sequential_cpu_offload()
        _save(output, "kandinsky22", seed, indices, images, settings, records)


def generate_sana(output: Path, count: int, seed: str, batch: int) -> None:
    settings = {"revision": SANA[1], "size": 512, "steps": 20, "guidance": 4.5}
    _ensure_provenance(output, {
        "schema_version": SCHEMA_VERSION, "kind": "t2i", "generator": "sana600m",
        "seed": seed, "count": count, "negative_prompt": NEGATIVE, "settings": settings,
    })
    records = _sync_t2i_metadata(output, "sana600m", seed, count, settings)
    batches = _pending(output, count, batch)
    if not batches:
        return
    import torch
    from diffusers import SanaPipeline

    pipe = SanaPipeline.from_pretrained(SANA[0], revision=SANA[1], variant="fp16",
                                        torch_dtype=torch.float16)
    pipe.vae.to(torch.bfloat16)
    pipe.text_encoder.to(torch.bfloat16)
    pipe.enable_model_cpu_offload()
    for indices in batches:
        images = pipe([prompt(seed, i) for i in indices], negative_prompt=[NEGATIVE] * len(indices),
                      height=512, width=512, num_inference_steps=20, guidance_scale=4.5,
                      generator=[_generator(seed, "sana", i) for i in indices]).images
        _save(output, "sana600m", seed, indices, images, settings, records)


def generate_cascade(output: Path, count: int, seed: str, batch: int) -> None:
    settings = {"revision": CASCADE[1], "prior_revision": CASCADE_PRIOR[1], "size": 1024,
                "prior_steps": 20, "decoder_steps": 10, "variant": "lite",
                "prior_guidance": 4.0, "decoder_guidance": 0.0}
    _ensure_provenance(output, {
        "schema_version": SCHEMA_VERSION, "kind": "t2i",
        "generator": "stable_cascade_lite", "seed": seed, "count": count,
        "negative_prompt": NEGATIVE, "settings": settings,
    })
    records = _sync_t2i_metadata(output, "stable_cascade_lite", seed, count, settings)
    batches = _pending(output, count, batch)
    if not batches:
        return
    import torch
    from diffusers import (
        StableCascadeDecoderPipeline,
        StableCascadePriorPipeline,
        StableCascadeUNet,
    )

    prior_unet = StableCascadeUNet.from_pretrained(
        CASCADE_PRIOR[0], subfolder="prior_lite", revision=CASCADE_PRIOR[1], variant="bf16",
        torch_dtype=torch.bfloat16)
    prior = StableCascadePriorPipeline.from_pretrained(
        CASCADE_PRIOR[0], revision=CASCADE_PRIOR[1], prior=prior_unet, variant="bf16",
        torch_dtype=torch.bfloat16)
    prior.enable_model_cpu_offload()
    embeddings = {}
    for indices in batches:
        result = prior(prompt=[prompt(seed, i) for i in indices],
                       negative_prompt=[NEGATIVE] * len(indices), height=1024, width=1024,
                       num_inference_steps=20, guidance_scale=4.0,
                       generator=[_generator(seed, "cascade-prior", i) for i in indices])
        for position, index in enumerate(indices):
            embeddings[index] = result.image_embeddings[position].cpu()
    del prior, prior_unet
    torch.cuda.empty_cache()
    decoder_unet = StableCascadeUNet.from_pretrained(
        CASCADE[0], subfolder="decoder_lite", revision=CASCADE[1], variant="bf16",
        torch_dtype=torch.bfloat16)
    decoder = StableCascadeDecoderPipeline.from_pretrained(
        CASCADE[0], revision=CASCADE[1], decoder=decoder_unet, variant="bf16",
        torch_dtype=torch.bfloat16)
    decoder.enable_model_cpu_offload()
    for indices in batches:
        images = decoder(
            image_embeddings=torch.stack([embeddings[i] for i in indices]).to(
                "cuda", torch.bfloat16),
            prompt=[prompt(seed, i) for i in indices], negative_prompt=[NEGATIVE] * len(indices),
            num_inference_steps=10, guidance_scale=0.0,
            generator=[_generator(seed, "cascade-decoder", i) for i in indices],
        ).images
        _save(output, "stable_cascade_lite", seed, indices, images, settings, records)


def _train_reals(manifest: Path, data_root: Path, count: int, seed: str) -> list[dict]:
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream)
                if row["split"] == "train" and row["label"] == "REAL"]
    rows.sort(key=lambda row: hashlib.sha256(f"{seed}\0{row['sample_id']}".encode()).hexdigest())
    if len(rows) < count:
        raise ValueError(f"requested {count} training reals but manifest has {len(rows)}")
    selected = rows[:count]
    sample_ids: set[str] = set()
    relative_paths: set[str] = set()
    for row in selected:
        relative = PurePosixPath(row["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe source relative path: {relative}")
        if row["sample_id"] in sample_ids or relative.as_posix() in relative_paths:
            raise ValueError("selected reconstruction sources are not unique")
        source = data_root.joinpath(*relative.parts)
        if not source.is_file():
            raise FileNotFoundError(f"reconstruction source is missing: {source}")
        row["_path"] = str(source)
        sample_ids.add(row["sample_id"])
        relative_paths.add(relative.as_posix())
    return selected


def _recon_record(autoencoder: str, seed: str, row: dict, revision: str, model: str) -> dict:
    return {
        "autoencoder": autoencoder,
        "autoencoder_model": model,
        "generator": f"recon_{autoencoder}",
        "revision": revision,
        "seed": seed,
        "source_sample_id": row["sample_id"],
        "source_relative_path": row["relative_path"],
        "output": PurePosixPath(row["relative_path"]).with_suffix(".png").as_posix(),
    }


def _sync_recon_metadata(
    output: Path,
    autoencoder: str,
    seed: str,
    rows: list[dict],
    revision: str,
    model: str,
) -> dict[object, dict]:
    path = output / "metadata.jsonl"
    records = _read_records(path, "output")
    expected = {
        record["output"]: record
        for record in (_recon_record(autoencoder, seed, row, revision, model) for row in rows)
    }
    for relative, record in records.items():
        if relative not in expected or record != expected[relative]:
            raise ValueError(f"reconstruction metadata differs for {relative!r}")
    for relative, record in expected.items():
        if output.joinpath(*PurePosixPath(relative).parts).is_file():
            records[relative] = record
    _write_records(path, records, "output")
    return records


def _load_rgb(path: str):
    import numpy as np

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32).copy()


def reconstruct(autoencoder: str, manifest: Path, data_root: Path, output: Path, count: int,
                seed: str, batch: int) -> None:
    """Encode and decode FF++ training reals; outputs mirror the source relative path."""
    if autoencoder == "movq":
        model_name, revision = KANDINSKY_DECODER
    elif autoencoder == "dcae":
        model_name, revision = SANA
    else:
        raise ValueError(f"unknown autoencoder: {autoencoder}")
    _ensure_provenance(output, {
        "schema_version": SCHEMA_VERSION, "kind": "reconstruction",
        "autoencoder": autoencoder, "model": model_name, "revision": revision,
        "seed": seed, "count": count, "manifest_sha256": _sha256(manifest),
    })
    rows = _train_reals(manifest, data_root, count, seed)
    records = _sync_recon_metadata(output, autoencoder, seed, rows, revision, model_name)
    pending = [
        row for row in rows
        if not output.joinpath(
            *PurePosixPath(row["relative_path"]).with_suffix(".png").parts
        ).is_file()
    ]
    if not pending:
        return
    import numpy as np
    import torch

    if autoencoder == "movq":
        from diffusers import VQModel

        model = VQModel.from_pretrained(
            model_name, subfolder="movq", revision=revision, torch_dtype=torch.float32
        ).to("cuda")

        def roundtrip(pixels: torch.Tensor) -> torch.Tensor:
            # The Kandinsky decoder pipeline decodes unquantized latents.
            return model.decode(model.encode(pixels).latents, force_not_quantize=True).sample
    else:
        from diffusers import AutoencoderDC

        model = AutoencoderDC.from_pretrained(
            model_name, subfolder="vae", revision=revision, variant="fp16",
            torch_dtype=torch.float32,
        ).to("cuda")

        def roundtrip(pixels: torch.Tensor) -> torch.Tensor:
            return model.decode(model.encode(pixels).latent).sample
    model.eval()
    for start in range(0, len(pending), batch):
        chunk = pending[start:start + batch]
        arrays = [_load_rgb(row["_path"]) for row in chunk]
        shapes = {array.shape for array in arrays}
        if len(shapes) != 1:
            raise ValueError(f"reconstruction batch has mixed image shapes: {sorted(shapes)}")
        pixels = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).to("cuda") / 127.5 - 1
        with torch.inference_mode():
            decoded = roundtrip(pixels).clamp(-1, 1)
        images = ((decoded + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        for row, image in zip(chunk, images, strict=True):
            relative = PurePosixPath(row["relative_path"]).with_suffix(".png")
            target = output.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f"{target.name}.part")
            Image.fromarray(image).save(temporary, format="PNG")
            temporary.replace(target)
            record = _recon_record(autoencoder, seed, row, revision, model_name)
            records[record["output"]] = record
        _write_records(output / "metadata.jsonl", records, "output")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    t2i = commands.add_parser("t2i")
    t2i.add_argument("--generator", choices=("kandinsky22", "sana600m", "stable_cascade_lite"),
                     required=True)
    t2i.add_argument("--count", type=_positive_int, required=True)
    t2i.add_argument("--batch", type=_positive_int, default=4)
    recon = commands.add_parser("recon")
    recon.add_argument("--autoencoder", choices=("movq", "dcae"), required=True)
    recon.add_argument("--manifest", type=Path, required=True)
    recon.add_argument("--data-root", type=Path, required=True)
    recon.add_argument("--count", type=_positive_int, required=True)
    recon.add_argument("--batch", type=_positive_int, default=16)
    for command in (t2i, recon):
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--seed", default="round2-20261008")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == "t2i":
        {"kandinsky22": generate_kandinsky, "sana600m": generate_sana,
         "stable_cascade_lite": generate_cascade}[args.generator](
            args.output, args.count, args.seed, args.batch)
    else:
        reconstruct(args.autoencoder, args.manifest, args.data_root, args.output, args.count,
                    args.seed, args.batch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
