"""Manifest-pinned RGB benchmark; test predictions require a separate audit gate."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
from pathlib import Path

import numpy as np
import torch
import torchvision
import yaml
from PIL import Image, ImageFilter
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    ResNet18_Weights,
    convnext_tiny,
    resnet18,
)

from deepfake_detection.data.config import find_project_root
from deepfake_detection.data.unseen import audit_unseen_manifest

from .common import git_commit as _git_commit
from .common import load_manifest_rows as _load_rows
from .common import sampling_weights as _sampling_weights
from .common import sha256_file as _sha256
from .evaluation import report as frozen_report
from .loop import _seed_worker
from .metrics import aggregate_videos, metric_report

PREPROCESS_ID = "df40-rgb-224-compression-quality-v1"
SAMPLE_PREPROCESS_ID = "df40-face-frame-manifest-v1"
NORMALIZE = {
    "clip_vit_b16": ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}


class QualityAugment:
    """Simulate a modest re-encode and image-quality shift before normalization."""

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < 0.55:
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=random.randint(35, 95))
            buffer.seek(0)
            with Image.open(buffer) as encoded:
                image = encoded.convert("RGB")
        if random.random() < 0.2:
            image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.2, 1.2)))
        return image


class Images(Dataset):
    def __init__(self, rows: list[dict[str, str]], model_name: str, training: bool) -> None:
        self.rows = rows
        mean, std = NORMALIZE["clip_vit_b16" if model_name == "clip_vit_b16" else "imagenet"]
        operations: list = []
        if training:
            operations.extend(
                [
                    transforms.RandomResizedCrop(
                        224, scale=(0.88, 1.0), interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.10),
                    QualityAugment(),
                ]
            )
        else:
            operations.append(
                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC)
            )
        operations.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
        self.transform = transforms.Compose(operations)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        row = self.rows[index]
        with Image.open(row["_path"]) as image:
            pixels = self.transform(image.convert("RGB"))
        return pixels, torch.tensor(float(row["label"] == "FAKE")), index


class BinaryEncoder(nn.Module):
    def __init__(self, name: str, config: dict) -> None:
        super().__init__()
        self.name = name
        if name == "resnet18":
            encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            width = encoder.fc.in_features
            encoder.fc = nn.Identity()
        elif name == "convnext_tiny":
            encoder = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            width = encoder.classifier[-1].in_features
            encoder.classifier[-1] = nn.Identity()
        elif name == "clip_vit_b16":
            from transformers import CLIPVisionConfig, CLIPVisionModel

            revision = str(config["clip_revision"])
            if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
                raise ValueError("clip_revision must be a pinned 40-character commit SHA")
            # The repository stores a top-level CLIPConfig containing separate
            # text and vision sections. Supplying the nested vision config
            # avoids treating that top-level object as CLIPVisionConfig.
            vision_config = CLIPVisionConfig.from_pretrained(
                "openai/clip-vit-base-patch16", revision=revision
            )
            encoder = CLIPVisionModel.from_pretrained(
                "openai/clip-vit-base-patch16",
                revision=revision,
                config=vision_config,
            )
            width = encoder.config.hidden_size
        else:
            raise ValueError(f"unsupported RGB model: {name}")
        self.encoder = encoder
        self.head = nn.Linear(width, 1)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.name == "clip_vit_b16":
            features = self.encoder(pixel_values=images).pooler_output
        else:
            features = self.encoder(images)
        return self.head(features).flatten(), features


def set_trainable(model: BinaryEncoder, epoch: int) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.head.parameters():
        parameter.requires_grad = True
    if model.name == "convnext_tiny":
        for parameter in model.encoder.parameters():
            parameter.requires_grad = True
    elif model.name == "resnet18" and epoch > 1:
        for parameter in model.encoder.layer4.parameters():
            parameter.requires_grad = True
    elif model.name == "clip_vit_b16":
        # Normalization layers adapt image features while the large attention
        # and MLP weights remain pinned to the pretrained checkpoint.
        for module in model.encoder.modules():
            if isinstance(module, nn.LayerNorm):
                for parameter in module.parameters():
                    parameter.requires_grad = True


def _loader(
    rows: list[dict[str, str]], config: dict, *, training: bool, epoch: int = 0
) -> DataLoader:
    generator = torch.Generator().manual_seed(int(config["seed"]) + epoch)
    sampler = (
        WeightedRandomSampler(
            _sampling_weights(rows), len(rows), replacement=True, generator=generator
        )
        if training
        else None
    )
    return DataLoader(
        Images(rows, config["model"], training),
        batch_size=int(config["batch_size"]),
        sampler=sampler,
        shuffle=False,
        num_workers=int(config.get("workers", 4)),
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker,
        generator=generator,
    )


def _train_epoch(
    model: BinaryEncoder,
    rows: list[dict[str, str]],
    config: dict,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    # A frozen ResNet block must not update BatchNorm running statistics. The
    # layer4 statistics start adapting with its weights after the warm-up.
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d) and not any(
            parameter.requires_grad for parameter in module.parameters()
        ):
            module.eval()
    total = count = 0
    for images, labels, _ in _loader(rows, config, training=True, epoch=epoch):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits, _ = model(images)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach()) * len(labels)
        count += len(labels)
    return total / count


@torch.inference_mode()
def predict(
    model: BinaryEncoder, rows: list[dict[str, str]], config: dict, device: torch.device
) -> tuple[list[dict], np.ndarray]:
    model.eval()
    records: list[dict] = []
    embeddings = []
    for images, _, indices in _loader(rows, config, training=False):
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits, features = model(images.to(device))
        logits = logits.float().cpu().tolist()
        embeddings.append(features.float().cpu().numpy())
        for index, logit in zip(indices.tolist(), logits, strict=True):
            row = rows[index]
            records.append(
                {
                    "sample_id": row["sample_id"],
                    "source_id": row["source_id"],
                    "split": row["split"],
                    "video_id": row["video_id"],
                    "source_domain": row["source_domain"],
                    "fake_method": row["fake_method"],
                    "label": int(row["label"] == "FAKE"),
                    "logit": float(logit),
                    "score": float(torch.sigmoid(torch.tensor(logit))),
                    "source_video_id": row.get("source_video_id", ""),
                    "lineage_video_ids": row.get("lineage_video_ids", ""),
                }
            )
    return records, np.concatenate(embeddings)


def _export(
    directory: Path, split: str, records: list[dict], embeddings: np.ndarray, metadata: dict
) -> None:
    # The order of the NPZ arrays is exactly the JSONL order and is checked by
    # sample ID at fusion time. Split files prevent accidental test access.
    with (directory / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    np.savez_compressed(
        directory / f"{split}_embeddings.npz",
        sample_ids=np.array([row["sample_id"] for row in records]),
        embeddings=embeddings.astype(np.float16),
    )
    (directory / f"{split}_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run(config_path: Path, *, seed: int | None = None, evaluate_test: bool = False) -> dict:
    root = find_project_root(config_path.resolve().parent)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["experiment"]
    config = dict(config)
    if seed is not None:
        config["seed"] = seed
    rows, manifest_path, _ = _load_rows(config, root)
    audit = audit_unseen_manifest(manifest_path.parent)
    if evaluate_test and not audit["passed"]:
        raise ValueError(
            "held-out evaluation blocked by protocol audit: " + "; ".join(audit["failures"])
        )
    for split, methods in (
        (
            "train",
            {"simswap", "blendface", "wav2lip", "fomm", "sadtalker", "stylegan2", "sd21", "dit"},
        ),
        ("validation", {"facedancer", "mraa", "stylegan3"}),
    ):
        actual = {
            row["fake_method"] for row in rows if row["split"] == split and row["label"] == "FAKE"
        }
        if actual != methods:
            raise ValueError(f"{split} methods differ from frozen protocol: {sorted(actual)}")
    random.seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.get("require_cuda", True) and device.type != "cuda":
        raise RuntimeError("CUDA required by benchmark config")
    model = BinaryEncoder(config["model"], config).to(device)
    commit = _git_commit(root)
    config_id = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:8]
    run_id = (
        f"{config['model']}-{config['seed']}-{manifest_path.parent.name}-{commit[:8]}-{config_id}"
    )
    directory = root / str(config.get("output_root", "artifacts/benchmark")) / run_id
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    (directory / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    splits = {
        name: [row for row in rows if row["split"] == name]
        for name in ("train", "validation", "test")
    }
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    set_trainable(model, 1)
    # Keep one optimizer so AdamW moments persist across epochs. Parameters
    # frozen during the ResNet warm-up are already registered and begin
    # updating when layer4 is enabled in epoch two.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    best_auc = -1.0
    history = []
    for epoch in range(1, int(config["epochs"]) + 1):
        set_trainable(model, epoch)
        loss = _train_epoch(model, splits["train"], config, optimizer, scaler, device, epoch)
        validation, _ = predict(model, splits["validation"], config, device)
        # FF++ unseen-method transfer is the primary benchmark. CDF remains a
        # separate cross-dataset result and cannot select the checkpoint.
        auc = frozen_report(validation, domain="ff")["macro"]["auroc"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss,
                "validation_ff_macro_video_auroc": auc,
            }
        )
        print(json.dumps(history[-1]), flush=True)
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), directory / "best.pt")
            best_epoch = epoch
        elif epoch - best_epoch >= int(config.get("patience", 3)):
            break
    # Load and inspect the saved weights before exporting any predictions.
    model.load_state_dict(
        torch.load(directory / "best.pt", map_location=device, weights_only=True), strict=True
    )
    checkpoint_hash = _sha256(directory / "best.pt")
    metadata = {
        "manifest_id": config["manifest_id"],
        "manifest_sha256": config["manifest_sha256"],
        "sample_preprocessing_id": SAMPLE_PREPROCESS_ID,
        "preprocessing_id": PREPROCESS_ID,
        "checkpoint_file": "best.pt",
        "checkpoint_sha256": checkpoint_hash,
        "model": config["model"],
        "git_commit": commit,
        "seed": config["seed"],
    }
    training, embeddings = predict(model, splits["train"], config, device)
    _export(directory, "train", training, embeddings, metadata)
    validation, embeddings = predict(model, splits["validation"], config, device)
    _export(directory, "validation", validation, embeddings, metadata)
    test = None
    if evaluate_test:
        test, embeddings = predict(model, splits["test"], config, device)
        _export(directory, "test", test, embeddings, metadata)
    report = {
        **metadata,
        "best_epoch": best_epoch,
        "history": history,
        "validation_video": metric_report(aggregate_videos(validation)),
        "validation_frozen": {
            domain: frozen_report(validation, domain=domain) for domain in ("ff", "cdf")
        },
        "test_exported": evaluate_test,
        "protocol_audit_passed": audit["passed"],
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }
    if test is not None:
        report["test_frozen"] = {
            domain: frozen_report(test, domain=domain) for domain in ("ff", "cdf")
        }
    (directory / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return {
        "run_dir": str(directory),
        "best_epoch": best_epoch,
        "validation_ff_macro_video_auroc": best_auc,
        "test_exported": evaluate_test,
    }


def export_test(run_dir: Path) -> dict:
    """Export the untouched split from an already selected checkpoint exactly once."""
    directory = run_dir.resolve()
    if any((directory / name).exists() for name in ("test.jsonl", "test_embeddings.npz")):
        raise FileExistsError(f"test predictions already exist in {directory}")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    training_report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    root = find_project_root(directory)
    rows, manifest_path, _ = _load_rows(config, root)
    audit = audit_unseen_manifest(manifest_path.parent)
    if not audit["passed"]:
        raise ValueError(
            "held-out evaluation blocked by protocol audit: " + "; ".join(audit["failures"])
        )
    checkpoint = directory / "best.pt"
    if _sha256(checkpoint) != training_report["checkpoint_sha256"]:
        raise ValueError("selected checkpoint hash differs from the training report")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.get("require_cuda", True) and device.type != "cuda":
        raise RuntimeError("CUDA required by benchmark config")
    model = BinaryEncoder(config["model"], config).to(device)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True), strict=True
    )
    test_rows = [row for row in rows if row["split"] == "test"]
    predictions, embeddings = predict(model, test_rows, config, device)
    metadata = {
        key: training_report[key]
        for key in (
            "manifest_id",
            "manifest_sha256",
            "sample_preprocessing_id",
            "preprocessing_id",
            "checkpoint_file",
            "checkpoint_sha256",
            "model",
            "git_commit",
            "seed",
        )
    }
    _export(directory, "test", predictions, embeddings, metadata)
    result = {
        **metadata,
        "protocol_audit_passed": True,
        "test_frozen": {
            domain: frozen_report(predictions, domain=domain) for domain in ("ff", "cdf")
        },
    }
    path = directory / "test_report.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"test_report": str(path), "checkpoint_sha256": metadata["checkpoint_sha256"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--config", type=Path)
    inputs.add_argument("--export-test-run", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args(argv)
    if args.export_test_run:
        if args.seed is not None or args.evaluate_test:
            parser.error("--seed and --evaluate-test apply only to --config")
        result = export_test(args.export_test_run)
    else:
        result = run(args.config, seed=args.seed, evaluate_test=args.evaluate_test)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
