"""Shared PyTorch training and evaluation loop for manifest-pinned binary detectors.

Model modules supply a dataset factory, called as ``make_dataset(rows, training=...)``,
and a ``set_trainable(model, finetune=...)`` callback; everything else here is
identical across experiments so that their results stay comparable.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .common import sampling_weights
from .metrics import aggregate_videos, metric_report, validation_threshold

DatasetFactory = Callable[..., Dataset]
TrainableSetter = Callable[..., None]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(config: dict) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.get("require_cuda", True) and device.type != "cuda":
        raise RuntimeError("CUDA is required by this run configuration")
    return device


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def train_epoch(model: nn.Module, rows: list[dict[str, str]], make_dataset: DatasetFactory,
                config: dict, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler,
                device: torch.device, epoch: int) -> float:
    generator = torch.Generator().manual_seed(int(config["seed"]) + epoch)
    sampler = WeightedRandomSampler(
        sampling_weights(rows), num_samples=len(rows), replacement=True, generator=generator
    )
    loader = DataLoader(
        make_dataset(rows, training=True), batch_size=int(config["batch_size"]),
        sampler=sampler, num_workers=int(config["workers"]), pin_memory=device.type == "cuda",
        worker_init_fn=_seed_worker, generator=generator,
    )
    model.train()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d) and not any(
            parameter.requires_grad for parameter in module.parameters()
        ):
            module.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = total_count = 0
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images).flatten()
            loss = loss_fn(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach()) * len(labels)
        total_count += len(labels)
    return total_loss / total_count


@torch.inference_mode()
def predict(model: nn.Module, rows: list[dict[str, str]], make_dataset: DatasetFactory,
            config: dict, device: torch.device) -> list[dict[str, object]]:
    loader = DataLoader(
        make_dataset(rows, training=False), batch_size=int(config["batch_size"]),
        shuffle=False, num_workers=int(config["workers"]), pin_memory=device.type == "cuda",
    )
    model.eval()
    output: list[dict[str, object]] = []
    for images, _, indices in loader:
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            probabilities = model(images.to(device, non_blocking=True)).flatten().sigmoid()
        for index, score in zip(indices.tolist(), probabilities.cpu().tolist(), strict=True):
            row = rows[index]
            output.append({
                "source_id": row["source_id"], "video_id": row["video_id"],
                "source_domain": row["source_domain"], "fake_method": row["fake_method"],
                "label": int(row["label"] == "FAKE"), "score": float(score),
            })
    return output


def fit(model: nn.Module, splits: dict[str, list[dict[str, str]]],
        make_dataset: DatasetFactory, set_trainable: TrainableSetter, config: dict,
        device: torch.device, directory: Path) -> tuple[list[dict], int, float]:
    """Train head then fine-tune, keeping the best validation-video-AUROC checkpoint."""
    history = []
    best_auc = -1.0
    best_epoch = 0
    no_improvement = 0
    total_epochs = int(config["head_epochs"]) + int(config["finetune_epochs"])
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    optimizer = None
    for epoch in range(1, total_epochs + 1):
        finetune = epoch > int(config["head_epochs"])
        if epoch == 1 or epoch == int(config["head_epochs"]) + 1:
            set_trainable(model, finetune=finetune)
            learning_rate = float(config["lr_finetune"] if finetune else config["lr_head"])
            optimizer = torch.optim.AdamW(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                lr=learning_rate, weight_decay=float(config["weight_decay"]),
            )
        assert optimizer is not None
        train_loss = train_epoch(model, splits["train"], make_dataset, config, optimizer,
                                 scaler, device, epoch)
        validation_frames = predict(model, splits["validation"], make_dataset, config, device)
        validation_videos = aggregate_videos(validation_frames)
        validation_auc = float(metric_report(validation_videos)["auroc"])
        record = {"epoch": epoch, "phase": "finetune" if finetune else "head",
                  "train_loss": train_loss, "validation_video_auroc": validation_auc}
        history.append(record)
        print(json.dumps(record), flush=True)
        if validation_auc > best_auc:
            best_auc, best_epoch, no_improvement = validation_auc, epoch, 0
            temporary = directory / "best.pt.part"
            torch.save(model.state_dict(), temporary)
            temporary.replace(directory / "best.pt")
        else:
            no_improvement += 1
            if no_improvement >= int(config["patience"]) and finetune:
                break
    return history, best_epoch, best_auc


def evaluate(model: nn.Module, splits: dict[str, list[dict[str, str]]],
             make_dataset: DatasetFactory, config: dict, device: torch.device,
             directory: Path) -> dict:
    """Reload the best checkpoint, lock the threshold on validation, then score test once."""
    model.load_state_dict(torch.load(directory / "best.pt", map_location=device,
                                     weights_only=True))
    validation_frames = predict(model, splits["validation"], make_dataset, config, device)
    validation_videos = aggregate_videos(validation_frames)
    threshold = validation_threshold(
        [int(row["label"]) for row in validation_videos],
        [float(row["score"]) for row in validation_videos],
    )
    test_frames = predict(model, splits["test"], make_dataset, config, device)
    return {
        "threshold_source": "validation_video_balanced_accuracy",
        "validation_frame": metric_report(validation_frames, threshold),
        "validation_video": metric_report(validation_videos, threshold),
        "test_frame": metric_report(test_frames, threshold),
        "test_video": metric_report(aggregate_videos(test_frames), threshold),
    }
