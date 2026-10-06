"""Single-GPU RGB reference run on a verified, frozen image manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

from .common import (
    create_run_directory,
    git_commit,
    load_config,
    load_manifest_rows,
    run_id,
    sha256_file,
    split_rows,
)
from .loop import evaluate, fit, seed_everything, select_device

PREPROCESS_VERSION = "df40-authors-crops-rgb-resize224-v1"


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


def _set_trainable(model: nn.Module, *, finetune: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.fc.parameters():
        parameter.requires_grad = True
    if finetune:
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True


def run(config_path: Path, *, check_only: bool = False) -> dict:
    config, root = load_config(config_path)
    rows, manifest_path, _ = load_manifest_rows(config, root)
    seed = int(config["seed"])
    seed_everything(seed)
    device = select_device(config)
    if config.get("model") != "resnet18" or config.get("weights") != "IMAGENET1K_V1":
        raise ValueError("this baseline requires pretrained ResNet18 IMAGENET1K_V1")
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    weights_name = ResNet18_Weights.IMAGENET1K_V1.url.rsplit("/", 1)[-1]
    weights_path = Path(torch.hub.get_dir()) / "checkpoints" / weights_name
    weights_sha256 = sha256_file(weights_path)
    if weights_sha256 != str(config["weights_sha256"]):
        raise ValueError("pretrained weight checksum differs from the run configuration")
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.to(device)
    splits = split_rows(rows)
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

    commit = git_commit(root)
    manifest_sha256 = sha256_file(manifest_path)
    identifier = run_id(config, commit, manifest_sha256)
    directory = create_run_directory(config, root, identifier, default_output_root="artifacts/rgb")
    history, best_epoch, best_auc = fit(model, splits, ManifestImages, _set_trainable,
                                        config, device, directory)
    evaluation = evaluate(model, splits, ManifestImages, config, device, directory)
    report = {
        "run_id": identifier, "git_commit": commit, "seed": seed,
        "manifest_id": config["manifest_id"], "manifest_sha256": manifest_sha256,
        "preprocessing": PREPROCESS_VERSION, "model": "resnet18",
        "weights": "IMAGENET1K_V1", "device": str(device),
        "pretrained_weights_sha256": weights_sha256,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch_version": torch.__version__, "torchvision_version": torchvision.__version__,
        "cuda_version": torch.version.cuda, "best_epoch": best_epoch,
        "history": history, **evaluation,
        "checkpoint_sha256": sha256_file(directory / "best.pt"),
        "limitation": (
            "Exploratory run: source identity IDs are absent; "
            "identity disjointness is unverified."
        ),
    }
    (directory / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return {"report": str(directory / "report.json"), "run_id": identifier,
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
