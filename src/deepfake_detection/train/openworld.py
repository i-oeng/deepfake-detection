"""Open-world detector training and evaluation on the generator-disjoint protocol.

Checkpoints and settings are chosen on validation only. The selection score is
the mean, over validation domains (FF++, Celeb-DF, and each external
collection), of the domain's macro video AUROC across its held-out generators,
so no single domain or generator family dominates the choice. Test predictions
are written once, by ``export-test``, from an already selected checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageFilter
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from deepfake_detection.data.config import find_project_root
from deepfake_detection.data.openworld import (
    PROTOCOL_ROUND2,
    PROTOCOL_ROUND3,
    PROTOCOL_V1,
    method_roles,
)
from deepfake_detection.data.unseen import audit_unseen_manifest

from . import benchmark, corruptions
from .common import (
    git_commit,
    load_manifest_rows,
    read_manifest_rows,
    sampling_weights,
    sha256_file,
)
from .evaluation import report as domain_report
from .loop import _seed_worker
from .sbi import self_blend

PROTOCOLS = frozenset((PROTOCOL_V1, PROTOCOL_ROUND2, PROTOCOL_ROUND3))
CLIP_REPOSITORIES = {
    "clip_vit_b16": "openai/clip-vit-base-patch16",
    "clip_vit_l14": "openai/clip-vit-large-patch14",
}
CLIP_MEAN, CLIP_STD = benchmark.NORMALIZE["clip_vit_b16"]
SBI_METHOD = "sbi"
TRAINING_SOURCES = (
    "src/deepfake_detection/data/manifest.py",
    "src/deepfake_detection/data/openworld.py",
    "src/deepfake_detection/data/unseen.py",
    "src/deepfake_detection/train/benchmark.py",
    "src/deepfake_detection/train/common.py",
    "src/deepfake_detection/train/corruptions.py",
    "src/deepfake_detection/train/evaluation.py",
    "src/deepfake_detection/train/loop.py",
    "src/deepfake_detection/train/metrics.py",
    "src/deepfake_detection/train/openworld.py",
    "src/deepfake_detection/train/sbi.py",
)


class Detector(nn.Module):
    """CLIP vision encoder with a binary head; only LayerNorm and the head train."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        from transformers import CLIPVisionConfig, CLIPVisionModel

        repository = CLIP_REPOSITORIES[config["model"]]
        revision = str(config["clip_revision"])
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError("clip_revision must be a pinned 40-character commit SHA")
        vision_config = CLIPVisionConfig.from_pretrained(repository, revision=revision)
        self.encoder = CLIPVisionModel.from_pretrained(
            repository, revision=revision, config=vision_config
        )
        if config.get("gradient_checkpointing"):
            # Non-reentrant checkpointing propagates gradients to LayerNorm weights
            # even though the frozen embeddings never require gradients.
            self.encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        # GenD (Yermakov et al., WACV 2026): classify unit-norm features.
        self.normalize = bool(config.get("normalize_features", False))
        self.head = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(pixel_values=images).pooler_output
        if self.normalize:
            features = F.normalize(features.float(), dim=-1)
        return self.head(features).flatten(), features


def set_trainable(model: Detector) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in model.encoder.modules():
        if isinstance(module, nn.LayerNorm):
            for parameter in module.parameters():
                parameter.requires_grad = True
    for parameter in model.head.parameters():
        parameter.requires_grad = True


class Degrade:
    """Training-only capture and transmission shifts, applied after geometry and color.

    ``base`` repeats the frozen-protocol augmentation. ``strong`` adds resampling,
    sensor noise, and single-frame H.264, which cost 0.19-0.36 AUROC in P4.
    """

    def __init__(self, strength: str) -> None:
        if strength not in {"base", "strong"}:
            raise ValueError(f"unknown augmentation strength: {strength}")
        self.strength = strength
        self.base = benchmark.QualityAugment()

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.strength == "base":
            return self.base(image)
        width, height = image.size
        if random.random() < 0.25:
            factor = random.uniform(0.35, 0.9)
            small = (max(8, round(width * factor)), max(8, round(height * factor)))
            method = random.choice((Image.Resampling.BILINEAR, Image.Resampling.BICUBIC,
                                    Image.Resampling.LANCZOS))
            image = image.resize(small, method).resize((width, height), method)
        if random.random() < 0.2:
            image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.2, 1.6)))
        if random.random() < 0.15:
            pixels = np.asarray(image, dtype=np.float32)
            pixels += np.random.normal(0.0, random.uniform(1.0, 6.0), pixels.shape)
            image = Image.fromarray(np.clip(np.rint(pixels), 0, 255).astype(np.uint8))
        draw = random.random()
        if draw < 0.15:
            image = corruptions._h264(image, float(random.randint(18, 40)))
        elif draw < 0.6:
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=random.randint(30, 95))
            buffer.seek(0)
            with Image.open(buffer) as encoded:
                image = encoded.convert("RGB")
        return image


def build_transform(training: bool, strength: str = "base") -> transforms.Compose:
    operations: list = []
    if training:
        operations += [
            transforms.RandomResizedCrop(
                224, scale=(0.88, 1.0), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.10),
            Degrade(strength),
        ]
    else:
        operations.append(
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC)
        )
    operations += [transforms.ToTensor(), transforms.Normalize(CLIP_MEAN, CLIP_STD)]
    return transforms.Compose(operations)


class Images(Dataset):
    def __init__(self, rows: list[dict[str, str]], *, training: bool,
                 strength: str = "base") -> None:
        self.rows = rows
        self.transform = build_transform(training, strength)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        row = self.rows[index]
        with Image.open(row["_path"]) as image:
            image = image.convert("RGB")
        if row.get("_sbi"):
            # Blend before augmentation, so degradations cover the boundary too.
            rng = np.random.default_rng(random.getrandbits(64))
            image = Image.fromarray(self_blend(np.asarray(image), rng))
        return self.transform(image), torch.tensor(float(row["label"] == "FAKE")), index


def with_pseudo_fakes(rows: list[dict[str, str]], share: float) -> tuple[list, list[float]]:
    """Append one self-blend row per real row; give them ``share`` of the fake mass."""
    if not 0.0 <= share < 1.0:
        raise ValueError("sbi_share must be in [0, 1)")
    if share == 0.0:
        return rows, sampling_weights(rows)
    pseudo = [{**row, "label": "FAKE", "fake_method": SBI_METHOD, "_sbi": "1"}
              for row in rows if row["label"] == "REAL"]
    combined = rows + pseudo
    weights = sampling_weights(combined)
    methods = len({row["fake_method"] for row in combined if row["label"] == "FAKE"})
    # sampling_weights gives every fake method 0.5 / methods of the mass.
    for index, row in enumerate(combined):
        if row["label"] == "FAKE":
            weights[index] *= (share if row.get("_sbi") else (1 - share) / (methods - 1)) \
                * methods
    return combined, weights


def lineage_namespace(row: dict[str, str]) -> str:
    """Source collection of a sample: the prefix of its first lineage unit.

    FF++ frames and their reconstructions carry ``ff:<clip>``; FairFace photos and
    their reconstructions carry ``fairface:<id>``.
    """
    return row.get("lineage_video_ids", "").split("|")[0].split(":", 1)[0]


def balance_namespaces(rows: list[dict[str, str]], weights: list[float], select) -> list[float]:
    """Give each lineage namespace an equal share of the selected rows' sampling mass.

    The selected rows keep their total mass and, within a namespace, their relative
    weights. Without this, single-image photo groups outnumber FF++ clips and take
    almost all of the real mass, because sampling_weights splits it by video.
    """
    chosen = [index for index, row in enumerate(rows) if select(row)]
    mass: dict[str, float] = defaultdict(float)
    for index in chosen:
        mass[lineage_namespace(rows[index])] += weights[index]
    total = sum(mass.values())
    balanced = list(weights)
    for index in chosen:
        share = mass[lineage_namespace(rows[index])]
        balanced[index] = weights[index] * total / (len(mass) * share)
    return balanced


def sampling_plan(rows: list[dict[str, str]], config: dict) -> tuple[list, list[float], dict]:
    """Training rows with self-blends, their sampling weights, and the mass per source."""
    rows, weights = with_pseudo_fakes(rows, float(config.get("sbi_share", 0.0)))
    balance = config.get("real_balance")
    if balance == "lineage_namespace":
        # Self-blends derive from reals, so their namespaces are balanced the same way.
        weights = balance_namespaces(rows, weights, lambda row: row["label"] == "REAL")
        weights = balance_namespaces(rows, weights, lambda row: bool(row.get("_sbi")))
    elif balance is not None:
        raise ValueError(f"unknown real_balance: {balance}")
    total = sum(weights)
    mass: dict[str, float] = defaultdict(float)
    for row, weight in zip(rows, weights, strict=True):
        kind = "sbi" if row.get("_sbi") else row["label"].lower()
        mass[f"{kind}/{lineage_namespace(row)}"] += weight / total
    return rows, weights, dict(sorted(mass.items()))


def alignment_uniformity(features: torch.Tensor, labels: torch.Tensor) -> tuple:
    """Wang and Isola's alignment (same-class pairs) and uniformity on the hypersphere."""
    unit = F.normalize(features.float(), dim=-1)
    distance = torch.cdist(unit, unit).pow(2)
    off_diagonal = ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    same = (labels[:, None] == labels[None, :]) & off_diagonal
    alignment = distance[same].mean() if same.any() else distance.new_zeros(())
    uniformity = torch.log(torch.exp(-2.0 * distance[off_diagonal]).mean())
    return alignment, uniformity


@torch.inference_mode()
def predict(model: nn.Module, rows: list[dict[str, str]], config: dict,
            device: torch.device) -> list[dict]:
    model.eval()
    loader = DataLoader(Images(rows, training=False), batch_size=int(config["batch_size"]) * 2,
                        shuffle=False, num_workers=int(config.get("workers", 4)),
                        pin_memory=device.type == "cuda")
    records = []
    for images, _, indices in loader:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            logits, _ = model(images.to(device, non_blocking=True))
        for index, logit in zip(indices.tolist(), logits.float().cpu().tolist(), strict=True):
            row = rows[index]
            records.append({
                "sample_id": row["sample_id"], "source_id": row["source_id"],
                "split": row["split"], "video_id": row["video_id"],
                "source_domain": row["source_domain"], "fake_method": row["fake_method"],
                "manipulation_family": row["manipulation_family"],
                "label": int(row["label"] == "FAKE"), "logit": float(logit),
                "score": 1.0 / (1.0 + math.exp(-logit)),
                "lineage_video_ids": row.get("lineage_video_ids", ""),
            })
    return records


def open_world_report(records: list[dict]) -> dict:
    """Per-domain macro video AUROC and their unweighted mean (the selection score).

    A domain with one class only, such as a real-only false-positive source, has no
    AUROC; it is listed under ``single_class_domains`` and left out of the mean.
    """
    labels: dict[str, set[int]] = defaultdict(set)
    for row in records:
        labels[row["source_domain"]].add(int(row["label"]))
    domains = {domain: domain_report(records, domain=domain)
               for domain in sorted(labels) if labels[domain] == {0, 1}}
    families: dict[str, list[float]] = {}
    for result in domains.values():
        for method, values in result["by_method"].items():
            family = next(row["manipulation_family"] for row in records
                          if row["fake_method"] == method)
            families.setdefault(family, []).append(values["auroc"])
    return {
        "score": sum(result["macro"]["auroc"] for result in domains.values()) / len(domains),
        "domain_macro_auroc": {name: result["macro"]["auroc"] for name, result in
                               domains.items()},
        "family_mean_auroc": {name: sum(values) / len(values) for name, values in
                              sorted(families.items())},
        "domains": domains,
        "single_class_domains": sorted(set(labels) - set(domains)),
    }


def _load(config: dict, root: Path) -> tuple[dict[str, list], Path, dict]:
    protocol = str(config.get("protocol", ""))
    if protocol not in PROTOCOLS:
        raise ValueError(f"configuration protocol must be one of {sorted(PROTOCOLS)}")
    rows, manifest_path, _ = load_manifest_rows(config, root)
    roles = method_roles(protocol)
    audit = audit_unseen_manifest(manifest_path.parent, roles)
    if not audit["passed"]:
        raise ValueError("protocol audit failed: " + "; ".join(audit["failures"]))
    splits = {name: [row for row in rows if row["split"] == name] for name in roles}
    for split, methods in roles.items():
        actual = {row["fake_method"] for row in splits[split] if row["label"] == "FAKE"}
        if actual != methods:
            raise ValueError(f"{split} methods differ from the protocol: {sorted(actual)}")
    return splits, manifest_path, audit


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _device(config: dict) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.get("require_cuda", True) and device.type != "cuda":
        raise RuntimeError("CUDA required by this configuration")
    return device


def source_fingerprint(root: Path) -> tuple[str, dict[str, str]]:
    """Hash the exact uncommitted training sources used by a run."""
    digest = hashlib.sha256()
    files = {}
    for relative in TRAINING_SOURCES:
        path = root / relative
        payload = path.read_bytes()
        value = hashlib.sha256(payload).hexdigest()
        files[relative] = value
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return digest.hexdigest(), files


def snapshot_sources(root: Path, directory: Path, files: dict[str, str]) -> None:
    """Copy hashed source files into the artifact so a dirty-tree run is reviewable."""
    snapshot = directory / "source_snapshot"
    for relative, expected in files.items():
        source = root / relative
        target = snapshot / relative
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"training source changed while creating the run: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    (directory / "source_manifest.json").write_text(
        json.dumps(files, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def train(config_path: Path, *, seed: int | None = None) -> dict:
    root = find_project_root(config_path.resolve().parent)
    config = dict(yaml.safe_load(config_path.read_text(encoding="utf-8"))["experiment"])
    if seed is not None:
        config["seed"] = seed
    splits, manifest_path, audit = _load(config, root)
    _seed(int(config["seed"]))
    device = _device(config)
    model = Detector(config).to(device)
    set_trainable(model)
    commit = git_commit(root)
    source_sha256, source_files = source_fingerprint(root)
    config_id = hashlib.sha256(
        json.dumps(
            {"config": config, "source_sha256": source_sha256},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()[:8]
    run_id = f"{config['model']}-{config['seed']}-{config['manifest_id']}-{commit[:8]}-{config_id}"
    directory = root / str(config.get("output_root", "artifacts/openworld")) / run_id
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    (directory / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    snapshot_sources(root, directory, source_files)

    train_rows, weights, sampling_mass = sampling_plan(splits["train"], config)
    print(json.dumps({"sampling_mass": sampling_mass}), flush=True)
    dataset = Images(train_rows, training=True, strength=str(config.get("augment", "base")))
    batch_size = int(config["batch_size"])
    steps_per_epoch = int(config["samples_per_epoch"]) // batch_size
    epochs = int(config["epochs"])
    encoder_parameters = [p for p in model.encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": encoder_parameters, "lr": float(config["lr"]),
         "weight_decay": float(config.get("weight_decay", 1e-4))},
        {"params": list(model.head.parameters()), "lr": float(config["head_lr"]),
         "weight_decay": 0.0},
    ])
    total_steps = steps_per_epoch * epochs
    warmup = max(1, int(0.03 * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / warmup)
        * 0.5 * (1 + math.cos(math.pi * min(step, total_steps) / total_steps)),
    )
    align_weight = float(config.get("alignment_weight", 0.0))
    uniform_weight = float(config.get("uniformity_weight", 0.0))
    history, best_score, best_epoch = [], -1.0, 0
    for epoch in range(1, epochs + 1):
        generator = torch.Generator().manual_seed(int(config["seed"]) + epoch)
        loader = DataLoader(
            dataset, batch_size=batch_size,
            sampler=WeightedRandomSampler(weights, steps_per_epoch * batch_size,
                                          replacement=True, generator=generator),
            num_workers=int(config.get("workers", 4)), pin_memory=device.type == "cuda",
            worker_init_fn=_seed_worker, generator=generator, drop_last=True,
            persistent_workers=False,
        )
        model.train()
        totals = {"loss": 0.0, "bce": 0.0, "alignment": 0.0, "uniformity": 0.0}
        for images, labels, _ in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                logits, features = model(images)
            bce = F.binary_cross_entropy_with_logits(logits.float(), labels)
            loss = bce
            alignment = uniformity = torch.zeros((), device=device)
            if align_weight or uniform_weight:
                alignment, uniformity = alignment_uniformity(features, labels)
                loss = loss + align_weight * alignment + uniform_weight * uniformity
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            for name, value in (("loss", loss), ("bce", bce), ("alignment", alignment),
                                ("uniformity", uniformity)):
                totals[name] += float(value.detach()) / steps_per_epoch
        validation = open_world_report(predict(model, splits["validation"], config, device))
        record = {"epoch": epoch, **totals, "validation_score": validation["score"],
                  "validation_domains": validation["domain_macro_auroc"]}
        history.append(record)
        print(json.dumps(record), flush=True)
        if validation["score"] > best_score:
            best_score, best_epoch = validation["score"], epoch
            temporary = directory / "best.pt.part"
            torch.save(model.state_dict(), temporary)
            temporary.replace(directory / "best.pt")
        elif epoch - best_epoch >= int(config.get("patience", 3)):
            break
    model.load_state_dict(torch.load(directory / "best.pt", map_location=device,
                                     weights_only=True))
    records = predict(model, splits["validation"], config, device)
    _write_jsonl(directory / "validation.jsonl", records)
    report = {
        "protocol": config["protocol"], "run_id": run_id, "git_commit": commit,
        "source_sha256": source_sha256,
        "manifest_id": config["manifest_id"], "manifest_sha256": config["manifest_sha256"],
        "checkpoint_sha256": sha256_file(directory / "best.pt"),
        "best_epoch": best_epoch, "history": history, "sampling_mass": sampling_mass,
        "validation": open_world_report(records),
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }
    (directory / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return {"run_dir": str(directory), "best_epoch": best_epoch,
            "validation_score": report["validation"]["score"],
            "validation_domains": report["validation"]["domain_macro_auroc"]}


def _write_jsonl(path: Path, records: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite predictions: {path}")
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")


def _model_from_run(run_dir: Path, device: torch.device) -> tuple[nn.Module, dict, dict]:
    """Rebuild an open-world run's model, or a frozen-protocol benchmark model."""
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    if sha256_file(run_dir / "best.pt") != report["checkpoint_sha256"]:
        raise ValueError("checkpoint hash differs from the run report")
    if config.get("protocol") in PROTOCOLS:
        model: nn.Module = Detector(config)
    else:
        model = benchmark.BinaryEncoder(config["model"], config)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu",
                                     weights_only=True), strict=True)
    return model.to(device), config, report


def score_run(run_dir: Path, protocol_config: Path, split: str, output: Path) -> dict:
    """Score a selected checkpoint (open-world or frozen-protocol) on one split.

    Predictions are never overwritten, so each output directory scores a split once.
    """
    root = find_project_root(protocol_config.resolve().parent)
    protocol = dict(yaml.safe_load(protocol_config.read_text(encoding="utf-8"))["experiment"])
    splits, _, _ = _load(protocol, root)
    device = _device(protocol)
    model, config, report = _model_from_run(run_dir.resolve(), device)
    if config.get("protocol") not in PROTOCOLS and config["model"] != "clip_vit_b16":
        raise ValueError("frozen-protocol baselines are scored with CLIP preprocessing only")
    output.mkdir(parents=True, exist_ok=True)
    records = predict(model, splits[split], {**protocol, "workers": protocol.get("workers", 4)},
                      device)
    _write_jsonl(output / f"{split}.jsonl", records)
    result = {
        "protocol": protocol["protocol"], "split": split, "run_dir": str(run_dir),
        "checkpoint_sha256": report["checkpoint_sha256"],
        "manifest_id": protocol["manifest_id"], split: open_world_report(records),
    }
    (output / f"{split}_report.json").write_text(json.dumps(result, indent=2, sort_keys=True)
                                                 + "\n")
    return {"output": str(output), "score": result[split]["score"],
            "domains": result[split]["domain_macro_auroc"]}


def score_manifest(run_dir: Path, evaluation_config: Path, output: Path) -> dict:
    """Score a selected checkpoint on every row of a test-only manifest, once."""
    root = find_project_root(evaluation_config.resolve().parent)
    evaluation = dict(yaml.safe_load(evaluation_config.read_text(encoding="utf-8"))["evaluation"])
    manifest_dir = (root / str(evaluation["manifest_dir"])).resolve()
    rows, manifest_path = read_manifest_rows(manifest_dir, root / str(evaluation["data_root"]))
    if manifest_dir.name != str(evaluation["manifest_id"]) or sha256_file(
        manifest_path
    ) != str(evaluation["manifest_sha256"]):
        raise ValueError("configured manifest ID or SHA-256 differs from the artifact")
    if {row["split"] for row in rows} != {"test"}:
        raise ValueError("score-manifest expects a test-only manifest")
    device = _device(evaluation)
    model, config, report = _model_from_run(run_dir.resolve(), device)
    if config.get("protocol") not in PROTOCOLS and config["model"] != "clip_vit_b16":
        raise ValueError("frozen-protocol baselines are scored with CLIP preprocessing only")
    output.mkdir(parents=True, exist_ok=True)
    records = predict(model, rows, evaluation, device)
    _write_jsonl(output / "test.jsonl", records)
    result = {
        "evaluation": evaluation["name"], "run_dir": str(run_dir),
        "checkpoint_sha256": report["checkpoint_sha256"],
        "manifest_id": evaluation["manifest_id"], "test": open_world_report(records),
    }
    (output / "test_report.json").write_text(json.dumps(result, indent=2, sort_keys=True)
                                             + "\n")
    return {"output": str(output), "score": result["test"]["score"],
            "domains": result["test"]["domain_macro_auroc"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    fit.add_argument("--config", type=Path, required=True)
    fit.add_argument("--seed", type=int)
    score = commands.add_parser("score", help="score a selected checkpoint on one split")
    score.add_argument("--run-dir", type=Path, required=True)
    score.add_argument("--protocol-config", type=Path, required=True)
    score.add_argument("--split", choices=("validation", "test"), required=True)
    score.add_argument("--output", type=Path, required=True)
    fresh = commands.add_parser("score-manifest",
                                help="score a selected checkpoint on a test-only manifest")
    fresh.add_argument("--run-dir", type=Path, required=True)
    fresh.add_argument("--evaluation-config", type=Path, required=True)
    fresh.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "train":
        result = train(args.config, seed=args.seed)
    elif args.command == "score-manifest":
        result = score_manifest(args.run_dir, args.evaluation_config, args.output)
    else:
        result = score_run(args.run_dir, args.protocol_config, args.split, args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
