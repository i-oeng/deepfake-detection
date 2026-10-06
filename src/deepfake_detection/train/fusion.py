"""Align RGB/frequency exports and select late fusion on development data."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from deepfake_detection.data.unseen import audit_unseen_manifest

from .evaluation import grouped_bootstrap_difference, report
from .fusion_contract import align, blend, load_embeddings, load_export


def _unique_correct(left: list[dict], right: list[dict]) -> dict:
    a, b = align(left, right)
    left_only = right_only = 0
    for x, y in zip(a, b, strict=True):
        label = int(x["label"])
        l_ok = (float(x["score"]) >= 0.5) == bool(label)
        r_ok = (float(y["score"]) >= 0.5) == bool(label)
        left_only += l_ok and not r_ok
        right_only += r_ok and not l_ok
    return {
        "rgb_only_correct_frames": left_only,
        "frequency_only_correct_frames": right_only,
        "complementary": left_only > 0 and right_only > 0,
    }


class GatedHead(nn.Module):
    """Small branch gate; encoders stay frozen and only this head is trained."""

    def __init__(self, rgb_width: int, frequency_width: int, hidden: int = 64) -> None:
        super().__init__()
        self.rgb = nn.Sequential(nn.LayerNorm(rgb_width), nn.Linear(rgb_width, hidden), nn.GELU())
        self.frequency = nn.Sequential(
            nn.LayerNorm(frequency_width), nn.Linear(frequency_width, hidden), nn.GELU()
        )
        self.gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, rgb: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        rgb_features = self.rgb(rgb)
        frequency_features = self.frequency(frequency)
        gate = self.gate(torch.cat((rgb_features, frequency_features), dim=1)).sigmoid()
        return self.classifier(gate * rgb_features + (1 - gate) * frequency_features).flatten()


def _gated_predictions(rows: list[dict], logits: torch.Tensor) -> list[dict]:
    probabilities = logits.float().sigmoid().cpu().tolist()
    values = logits.float().cpu().tolist()
    return [
        {**row, "logit": float(logit), "score": float(score)}
        for row, logit, score in zip(rows, values, probabilities, strict=True)
    ]


def train_gated_head(
    rgb_dir: Path,
    frequency_dir: Path,
    *,
    seed: int = 20261006,
) -> tuple[list[dict], GatedHead, dict]:
    train_rgb_rows, train_rgb, _ = load_embeddings(rgb_dir, "train")
    train_frequency_rows, train_frequency, _ = load_embeddings(frequency_dir, "train")
    train_rgb_rows, train_frequency_rows = align(train_rgb_rows, train_frequency_rows)
    validation_rgb_rows, validation_rgb, _ = load_embeddings(rgb_dir, "validation")
    validation_frequency_rows, validation_frequency, _ = load_embeddings(
        frequency_dir, "validation"
    )
    validation_rgb_rows, validation_frequency_rows = align(
        validation_rgb_rows, validation_frequency_rows
    )

    # The NPZ order was checked against JSONL in load_export. align sorts both
    # JSONL sets, so reorder arrays to the same stable sample order explicitly.
    def reorder(
        original_rows: list[dict], aligned_rows: list[dict], values: np.ndarray
    ) -> np.ndarray:
        positions = {row["sample_id"]: index for index, row in enumerate(original_rows)}
        return values[[positions[row["sample_id"]] for row in aligned_rows]]

    raw_train_rgb_rows, _, _ = load_embeddings(rgb_dir, "train")
    raw_train_frequency_rows, _, _ = load_embeddings(frequency_dir, "train")
    raw_validation_rgb_rows, _, _ = load_embeddings(rgb_dir, "validation")
    raw_validation_frequency_rows, _, _ = load_embeddings(frequency_dir, "validation")
    train_rgb = reorder(raw_train_rgb_rows, train_rgb_rows, train_rgb)
    train_frequency = reorder(raw_train_frequency_rows, train_frequency_rows, train_frequency)
    validation_rgb = reorder(raw_validation_rgb_rows, validation_rgb_rows, validation_rgb)
    validation_frequency = reorder(
        raw_validation_frequency_rows, validation_frequency_rows, validation_frequency
    )

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head = GatedHead(train_rgb.shape[1], train_frequency.shape[1]).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-3)
    rgb_tensor = torch.from_numpy(train_rgb)
    frequency_tensor = torch.from_numpy(train_frequency)
    labels = torch.tensor([float(row["label"]) for row in train_rgb_rows])
    val_rgb = torch.from_numpy(validation_rgb).to(device)
    val_frequency = torch.from_numpy(validation_frequency).to(device)
    generator = torch.Generator().manual_seed(seed)
    best_auc = -1.0
    best_state = None
    best_epoch = 0
    history = []
    for epoch in range(1, 31):
        head.train()
        permutation = torch.randperm(len(labels), generator=generator)
        total_loss = 0.0
        for indices in permutation.split(256):
            batch_rgb = rgb_tensor[indices].to(device)
            batch_frequency = frequency_tensor[indices].to(device)
            batch_labels = labels[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(batch_rgb, batch_frequency)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        head.eval()
        with torch.inference_mode():
            validation_logits = head(val_rgb, val_frequency)
        predictions = _gated_predictions(validation_rgb_rows, validation_logits)
        auc = report(predictions, domain="ff")["macro"]["auroc"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / len(labels),
                "validation_ff_macro_auroc": auc,
            }
        )
        if auc > best_auc:
            best_auc, best_epoch = auc, epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in head.state_dict().items()
            }
        elif epoch - best_epoch >= 5:
            break
    assert best_state is not None
    head.load_state_dict(best_state)
    head.to(device).eval()
    with torch.inference_mode():
        validation_logits = head(val_rgb, val_frequency)
    return (
        _gated_predictions(validation_rgb_rows, validation_logits),
        head,
        {
            "best_epoch": best_epoch,
            "validation_ff_macro_auroc": best_auc,
            "history": history,
            "training_split": "train",
            "selection_split": "validation",
            "seed": seed,
        },
    )


def predict_gated(head: GatedHead, rgb_dir: Path, frequency_dir: Path, split: str) -> list[dict]:
    rgb_rows_raw, rgb_values, _ = load_embeddings(rgb_dir, split)
    frequency_rows_raw, frequency_values, _ = load_embeddings(frequency_dir, split)
    rgb_rows, frequency_rows = align(rgb_rows_raw, frequency_rows_raw)
    rgb_position = {row["sample_id"]: index for index, row in enumerate(rgb_rows_raw)}
    frequency_position = {row["sample_id"]: index for index, row in enumerate(frequency_rows_raw)}
    rgb_values = rgb_values[[rgb_position[row["sample_id"]] for row in rgb_rows]]
    frequency_values = frequency_values[
        [frequency_position[row["sample_id"]] for row in frequency_rows]
    ]
    device = next(head.parameters()).device
    with torch.inference_mode():
        logits = head(
            torch.from_numpy(rgb_values).to(device), torch.from_numpy(frequency_values).to(device)
        )
    return _gated_predictions(rgb_rows, logits)


def run(
    rgb_dir: Path,
    frequency_dir: Path,
    manifest_dir: Path,
    output: Path,
    *,
    baseline_dir: Path | None = None,
    final_test: bool = False,
) -> dict:
    audit = audit_unseen_manifest(manifest_dir)
    if final_test and not audit["passed"]:
        raise ValueError("held-out test blocked by protocol audit: " + "; ".join(audit["failures"]))
    rgb, rgb_meta = load_export(rgb_dir, "validation")
    freq, freq_meta = load_export(frequency_dir, "validation")
    if any(
        meta["manifest_id"] != audit["manifest_id"]
        or meta["manifest_sha256"] != audit["manifest_sha256"]
        for meta in (rgb_meta, freq_meta)
    ):
        raise ValueError("prediction manifest differs from the frozen audit")
    if rgb_meta["sample_preprocessing_id"] != freq_meta["sample_preprocessing_id"]:
        raise ValueError("RGB and frequency exports use different sample preprocessing")
    aligned_rgb, aligned_freq = align(rgb, freq)
    complementarity = _unique_correct(aligned_rgb, aligned_freq)
    weights = [i / 20 for i in range(21)]
    # FF++ unseen-method transfer is the selection metric. CDF is reported
    # separately and never changes the scalar chosen on development data.
    scores = [
        (report(blend(aligned_rgb, aligned_freq, weight))["macro"]["auroc"], weight)
        for weight in weights
    ]
    best_score = max(value for value, _ in scores)
    selected_weight = min(
        (weight for value, weight in scores if value == best_score),
        key=lambda weight: abs(weight - 0.5),
    )
    variants = {
        "rgb": aligned_rgb,
        "frequency": aligned_freq,
        "equal_logits": blend(aligned_rgb, aligned_freq, 0.5),
        "tuned_logits": blend(aligned_rgb, aligned_freq, selected_weight),
    }
    gated_head = None
    gated_bytes = None
    gated_metadata = None
    if complementarity["complementary"]:
        gated_validation, gated_head, gated_metadata = train_gated_head(rgb_dir, frequency_dir)
        variants["gated_features"] = gated_validation
        buffer = io.BytesIO()
        torch.save(gated_head.state_dict(), buffer)
        gated_bytes = buffer.getvalue()
        gated_metadata["checkpoint_sha256"] = hashlib.sha256(gated_bytes).hexdigest()
    result = {
        "manifest_id": audit["manifest_id"],
        "manifest_sha256": audit["manifest_sha256"],
        "rgb": rgb_meta,
        "frequency": freq_meta,
        "selection_split": "validation",
        "selection_domain": "ff",
        "selected_rgb_weight": selected_weight,
        "development": {
            name: {domain: report(rows, domain=domain) for domain in ("ff", "cdf")}
            for name, rows in variants.items()
        },
        "complementarity": complementarity,
        "gated_features": gated_metadata,
    }
    if final_test:
        rgb_test, rgb_test_meta = load_export(rgb_dir, "test")
        freq_test, freq_test_meta = load_export(frequency_dir, "test")
        if rgb_test_meta != rgb_meta or freq_test_meta != freq_meta:
            raise ValueError("test export provenance differs from development export")
        test_a, test_b = align(rgb_test, freq_test)
        test_variants = {
            "rgb": test_a,
            "frequency": test_b,
            "equal_logits": blend(test_a, test_b, 0.5),
            "tuned_logits": blend(test_a, test_b, selected_weight),
        }
        if gated_head is not None:
            test_variants["gated_features"] = predict_gated(
                gated_head, rgb_dir, frequency_dir, "test"
            )
        result["final_test"] = {
            name: {domain: report(rows, domain=domain) for domain in ("ff", "cdf")}
            for name, rows in test_variants.items()
        }
        if baseline_dir:
            baseline, baseline_meta = load_export(baseline_dir, "test")
            if baseline_meta["manifest_id"] != audit["manifest_id"]:
                raise ValueError("baseline uses a different manifest")
            result["comparison_with_resnet18"] = {
                name: grouped_bootstrap_difference(baseline, rows)
                for name, rows in test_variants.items()
            }
    # Content-derived output prevents an adjusted blend from replacing a result.
    content = json.dumps(result, indent=2, sort_keys=True) + "\n"
    digest = hashlib.sha256(content.encode()).hexdigest()[:20]
    output.mkdir(parents=True, exist_ok=True)
    if gated_bytes is not None and gated_metadata is not None:
        gated_path = output / f"gated-{gated_metadata['checkpoint_sha256'][:20]}.pt"
        if gated_path.exists() and gated_path.read_bytes() != gated_bytes:
            raise RuntimeError(f"refusing to overwrite changed gated checkpoint: {gated_path}")
        gated_path.write_bytes(gated_bytes)
    path = output / f"fusion-{digest}.json"
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise RuntimeError(f"refusing to overwrite changed fusion result: {path}")
    path.write_text(content, encoding="utf-8")
    return {"report": str(path), "selected_rgb_weight": selected_weight, "final_test": final_test}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-run", type=Path, required=True)
    parser.add_argument("--frequency-run", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/fusion"))
    parser.add_argument("--baseline-run", type=Path)
    parser.add_argument("--final-test", action="store_true")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            run(
                args.rgb_run,
                args.frequency_run,
                args.manifest_dir,
                args.output,
                baseline_dir=args.baseline_run,
                final_test=args.final_test,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
