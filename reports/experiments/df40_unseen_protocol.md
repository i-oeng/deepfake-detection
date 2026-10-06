# Frozen DF40 unseen-method protocol

This protocol replaces the mixed-domain pilot split. It selects checkpoints on
development data and keeps UniFace, MCNet, and RDDM untouched until the final
export.

## Frozen artifact

- Manifest ID: `07f906d9ef61a537efa8`
- Manifest SHA-256:
  `07f906d9ef61a537efa81be8da1ba1a568bb8fc339380a2987ad8ac717e73dff`
- Final subset ID: `b19543f733d7d0d8b4a3`
- Rows: 11,640
- Maximum frames per video group: 8
- Split priority during collision pruning: test, validation, train

The image audit removed 24 frames from the lower-priority video group involved
in the only cross-split perceptual collision. The final lineage audit reports no
cross-split video, parent, candidate identity, pixel hash, or perceptual hash
overlap. It also reports no missing parent or identity evidence. The machine
readable result is in
[`df40_unseen_clean_v3_07f906d9ef61a537efa8.json`](../identity_audit/df40_unseen_clean_v3_07f906d9ef61a537efa8.json).

FF++ source clip IDs and Celeb-DF `idNN` tokens are the available identity
units. They prevent known source reuse across splits, but they do not establish
person-level separation for different FF++ clips. Any person-disjoint claim
still requires face-identity metadata or a separately reviewed visual audit.

## Method roles

| Role | Methods |
|---|---|
| Train | SimSwap, BlendFace, Wav2Lip, FOMM, SadTalker, StyleGAN2, SD-2.1, DiT |
| Development | FaceDancer, MRAA, StyleGAN3 |
| Final test | UniFace, MCNet, RDDM |

The FF++ development and test slices are primary. Celeb-DF results are reported
as cross-dataset measurements and never select a checkpoint or fusion weight.
The architecture and checkpoint decision made before final-test export is in
the [development-only selection record](df40_unseen_development_selection.md).

## Model selection and final export

Training writes `train` and `validation` predictions, embeddings, provenance,
and a hashed checkpoint. It does not read or export test predictions by
default. The separate export command rechecks the manifest audit and checkpoint
hash, and refuses to overwrite an existing final export.

```bash
scripts/submit-benchmark configs/training/df40_unseen_resnet18.yaml
scripts/submit-benchmark configs/training/df40_unseen_convnext_tiny.yaml
scripts/submit-benchmark configs/training/df40_unseen_clip_vit_b16.yaml

deepfake-train-benchmark --export-test-run artifacts/benchmark/<selected-run>
deepfake-summarize-seeds --run <seed-1> --run <seed-2> --run <seed-3> --split test
```

Every model uses method and video balanced sampling, mixed precision, and image
quality augmentation. ResNet18 provides the frozen-protocol baseline.
ConvNeXt-Tiny is fully fine-tuned. CLIP ViT-B/16 updates its LayerNorm parameters
and binary classifier while retaining its pinned pretrained attention and MLP
weights.

E-ConvNeXt remains an optional development candidate. Its public reference is a
Paddle implementation and does not provide a matching, pinned PyTorch
pretrained checkpoint, so adding it to the immediate comparison would change
both framework and initialization. It can enter a later comparison after a
reproducible checkpoint and preprocessing contract are frozen.

## Frequency branch export contract

Each branch run directory must contain `best.pt` (or the safe relative path in
`checkpoint_file`) and, for each exported split:

- `<split>.jsonl`, with sample ID, logit, score, label, method, domain, video,
  split, and source lineage;
- `<split>_embeddings.npz`, with `sample_ids` in the exact JSONL order and a
  two-dimensional `embeddings` array;
- `<split>_metadata.json`, with manifest ID and SHA-256, shared sample
  preprocessing ID, branch preprocessing ID, checkpoint file and SHA-256.

Fusion first reports both branches independently. It then evaluates equal-logit
averaging and a scalar RGB weight selected on FF++ development macro AUROC. A
small feature gate is trained from training embeddings and selected on FF++
development only when both branches have unique correct development examples.
