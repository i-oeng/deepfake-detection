"""Single-GPU RGB reference run on a verified, frozen image manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torchvision
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

from deepfake_detection.data.config import find_project_root
from deepfake_detection.data.manifest import verify_manifest

from .metrics import aggregate_videos, metric_report, validation_threshold

PREPROCESS_VERSION = "df40-authors-crops-rgb-resize224-v1"
REQUIRED = {"source_id", "relative_path", "split", "label", "status", "video_id",
            "source_domain", "fake_method"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(path: Path) -> tuple[dict, Path]:
    path = path.resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("experiment"), dict):
        raise ValueError("training config needs an experiment mapping")
    return config["experiment"], find_project_root(path.parent)


def _load_rows(config: dict, root: Path) -> tuple[list[dict[str, str]], Path, Path]:
    manifest_dir = (root / str(config["manifest_dir"])).resolve()
    data_root = (root / str(config["data_root"])).resolve()
    valid, errors = verify_manifest(manifest_dir)
    if not valid:
        raise ValueError(f"invalid immutable manifest: {errors}")
    manifest_path = manifest_dir / "manifest.csv"
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not REQUIRED.issubset(reader.fieldnames or ()):
            raise ValueError("manifest lacks required training columns")
        rows = list(reader)
    if manifest_dir.name != str(config["manifest_id"]):
        raise ValueError("configured manifest ID differs from its directory")
    if _sha256(manifest_path) != str(config["manifest_sha256"]):
        raise ValueError("configured manifest SHA-256 differs from the artifact")
    if not rows:
        raise ValueError("empty training manifest")
    for row in rows:
        if row["status"] != "ok" or row["label"] not in {"REAL", "FAKE"}:
            raise ValueError(f"invalid image status or label: {row['source_id']}")
        relative = Path(row["relative_path"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "images":
            raise ValueError(f"unsafe manifest path: {relative}")
        image = (data_root / relative).resolve()
        if not image.is_relative_to(data_root) or not image.is_file():
            raise ValueError(f"manifest image is missing or escapes data root: {relative}")
        row["_path"] = str(image)
    counts = Counter(row["split"] for row in rows)
    if set(counts) != {"train", "validation", "test"}:
        raise ValueError(f"expected train/validation/test splits: {counts}")
    return rows, manifest_path, data_root


class ManifestImages(Dataset):
    def __init__(self, rows: list[dict[str, str]], *, training: bool) -> None:
        self.rows = rows
        operations: list = [transforms.Resize((224, 224))]
        if training:
            operations += [transforms.RandomHorizontalFlip(),
                           transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)]
        operations += [transforms.ToTensor(),
                       transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                            std=(0.229, 0.224, 0.225))]
        self.transform = transforms.Compose(operations)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        row = self.rows[index]
        with Image.open(row["_path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        label = torch.tensor(1.0 if row["label"] == "FAKE" else 0.0)
        return tensor, label, index


def _sampling_weights(rows: list[dict[str, str]]) -> list[float]:
    methods = sorted({row["fake_method"] for row in rows if row["label"] == "FAKE"})
    if not methods:
        raise ValueError("training split has no fake methods")
    group_counts = Counter((row["label"], row["fake_method"], row["video_id"]) for row in rows)
    groups_per_stratum = Counter((label, method) for label, method, _ in group_counts)
    weights = []
    for row in rows:
        label, method, video = row["label"], row["fake_method"], row["video_id"]
        mass = 0.5 if label == "REAL" else 0.5 / len(methods)
        weights.append(mass / groups_per_stratum[label, method]
                       / group_counts[label, method, video])
    return weights


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def _train_epoch(model: nn.Module, rows: list[dict[str, str]], config: dict,
                 optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler,
                 device: torch.device, epoch: int) -> float:
    generator = torch.Generator().manual_seed(int(config["seed"]) + epoch)
    sampler = WeightedRandomSampler(
        _sampling_weights(rows), num_samples=len(rows), replacement=True, generator=generator
    )
    loader = DataLoader(
        ManifestImages(rows, training=True), batch_size=int(config["batch_size"]),
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
def _predict(model: nn.Module, rows: list[dict[str, str]], config: dict,
             device: torch.device) -> list[dict[str, object]]:
    loader = DataLoader(
        ManifestImages(rows, training=False), batch_size=int(config["batch_size"]),
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


def _set_trainable(model: nn.Module, *, finetune: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.fc.parameters():
        parameter.requires_grad = True
    if finetune:
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True


def _git_commit(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def run(config_path: Path, *, check_only: bool = False) -> dict:
    config, root = _load_config(config_path)
    rows, manifest_path, _ = _load_rows(config, root)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.get("require_cuda", True) and device.type != "cuda":
        raise RuntimeError("CUDA is required by this run configuration")
    if config.get("model") != "resnet18" or config.get("weights") != "IMAGENET1K_V1":
        raise ValueError("this baseline requires pretrained ResNet18 IMAGENET1K_V1")
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    weights_name = ResNet18_Weights.IMAGENET1K_V1.url.rsplit("/", 1)[-1]
    weights_path = Path(torch.hub.get_dir()) / "checkpoints" / weights_name
    weights_sha256 = _sha256(weights_path)
    if weights_sha256 != str(config["weights_sha256"]):
        raise ValueError("pretrained weight checksum differs from the run configuration")
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.to(device)
    splits = {split: [row for row in rows if row["split"] == split]
              for split in ("train", "validation", "test")}
    if check_only:
        sample = [next(row for row in rows if row["label"] == label)
                  for label in ("REAL", "FAKE")]
        dataset = ManifestImages(sample, training=False)
        _set_trainable(model, finetune=False)
        batch = torch.stack([dataset[0][0], dataset[1][0]]).to(device)
        labels = torch.tensor([0.0, 1.0], device=device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            output = model(batch).flatten()
            loss = nn.BCEWithLogitsLoss()(output, labels)
        loss.backward()
        return {"check_passed": True, "device": str(device), "logit_shape": list(output.shape),
                "loss": float(loss.detach()),
                "split_counts": {split: len(part) for split, part in splits.items()}}

    commit = _git_commit(root)
    identity = json.dumps({"config": config, "commit": commit,
                           "manifest_sha256": _sha256(manifest_path)}, sort_keys=True).encode()
    run_id = hashlib.sha256(identity).hexdigest()[:20]
    directory = root / str(config.get("output_root", "artifacts/rgb")) / run_id
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
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
            _set_trainable(model, finetune=finetune)
            learning_rate = float(config["lr_finetune"] if finetune else config["lr_head"])
            optimizer = torch.optim.AdamW(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                lr=learning_rate, weight_decay=float(config["weight_decay"]),
            )
        assert optimizer is not None
        train_loss = _train_epoch(model, splits["train"], config, optimizer, scaler, device, epoch)
        validation_frames = _predict(model, splits["validation"], config, device)
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

    model.load_state_dict(torch.load(directory / "best.pt", map_location=device,
                                     weights_only=True))
    validation_frames = _predict(model, splits["validation"], config, device)
    validation_videos = aggregate_videos(validation_frames)
    threshold = validation_threshold(
        [int(row["label"]) for row in validation_videos],
        [float(row["score"]) for row in validation_videos],
    )
    test_frames = _predict(model, splits["test"], config, device)
    report = {
        "run_id": run_id, "git_commit": commit, "seed": seed,
        "manifest_id": config["manifest_id"], "manifest_sha256": _sha256(manifest_path),
        "preprocessing": PREPROCESS_VERSION, "model": "resnet18",
        "weights": "IMAGENET1K_V1", "device": str(device),
        "pretrained_weights_sha256": weights_sha256,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch_version": torch.__version__, "torchvision_version": torchvision.__version__,
        "cuda_version": torch.version.cuda, "best_epoch": best_epoch,
        "history": history, "threshold_source": "validation_video_balanced_accuracy",
        "validation_frame": metric_report(validation_frames, threshold),
        "validation_video": metric_report(validation_videos, threshold),
        "test_frame": metric_report(test_frames, threshold),
        "test_video": metric_report(aggregate_videos(test_frames), threshold),
        "checkpoint_sha256": _sha256(directory / "best.pt"),
        "limitation": (
            "Exploratory run: source identity IDs are absent; "
            "identity disjointness is unverified."
        ),
    }
    (directory / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return {"report": str(directory / "report.json"), "run_id": run_id,
            "best_epoch": best_epoch, "validation_video_auroc": best_auc,
            "test_video_auroc": report["test_video"]["auroc"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    arguments = parser.parse_args(argv)
    print(json.dumps(run(arguments.config, check_only=arguments.check_only), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
