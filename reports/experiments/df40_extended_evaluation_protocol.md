# Extended evaluation and release protocol

Recorded on 2026-10-07, after the frozen unseen-method RGB, frequency, and fusion
test results were published (PRs #11 and #16), and **before** any data listed
below as new was selected, extracted, or scored. Analyses not listed here will be
labelled exploratory.

This record covers the plan phases still open after the unseen-method result:
seen-method, unseen-domain, robustness, and raw-video cross-dataset evaluation
(M6); versioned face preprocessing (M2); and a calibrated inference package with
video aggregation and a model card (M7).

## Frozen models evaluated

No architecture, hyperparameter, checkpoint, blend weight, or threshold below may
be changed after a new protocol is scored.

| Model | Seeds | Source |
|---|---|---|
| ResNet18 baseline | 20261006 | `df40_unseen_development_selection.md` |
| ConvNeXt-Tiny | 20261006, 20261007, 20261008 | same |
| CLIP ViT-B/16 | 20261006 (frozen); 20261007, 20261008 (new) | same; new seeds use the unchanged frozen config |
| Frequency branch (all inputs + spectral filter) | 20261006, 20261007, 20261008 | `df40_unseen_frequency_development_selection.md` |

The two new CLIP seeds are trained with `configs/training/df40_unseen_clip_vit_b16.yaml`
unchanged and only measure seed variance. They select nothing.

## Release model

The research finalist remains ConvNeXt-Tiny, selected by the pre-registered FF++
development rule. A deployed detector also faces domain shift, so the release
model uses a separate rule computed from **development data only**: the highest
worst-domain development macro video AUROC, `min(FF++ dev, Celeb-DF dev)`, among
the frozen single-model checkpoints.

| Checkpoint | FF++ dev | Celeb-DF dev | Worst domain |
|---|---:|---:|---:|
| CLIP ViT-B/16 / 20261006 | 0.9635 | 0.9459 | **0.9459** |
| ResNet18 / 20261006 | 0.8617 | 0.8801 | 0.8617 |
| Frequency / 20261006 | 0.9032 | 0.8305 | 0.8305 |
| ConvNeXt-Tiny / 20261006 | 0.9778 | 0.7619 | 0.7619 |
| ConvNeXt-Tiny / 20261008 | 0.9625 | 0.6995 | 0.6995 |
| ConvNeXt-Tiny / 20261007 | 0.9679 | 0.6380 | 0.6380 |

The frequency row is seed 20261006 of the frozen variant. Its three seeds reach
at most 0.9067 on FF++ and average 0.7645 on Celeb-DF, below CLIP on both.

**CLIP ViT-B/16 seed 20261006 (`a3bcb4fb…cdf1`) is the release checkpoint.**
This rule is written after the RGB test results were known, which also favoured
CLIP on Celeb-DF. The release claim therefore rests only on the protocols below,
none of which has been selected or scored.

## Release decisions fixed on FF++ development data

The release checkpoint's `validation.jsonl` frame predictions (FF++ development
methods FaceDancer, MRAA, StyleGAN3) fix three choices before any new protocol is
scored:

1. **Video aggregation.** Candidates: mean probability (the current protocol),
   mean logit, median logit, 20% trimmed-mean logit, and top-25% mean logit. The
   highest FF++ development macro video AUROC wins; candidates within 0.002 of the
   best resolve to mean probability, then mean logit.
2. **Calibration.** Platt scaling, `sigmoid(a * aggregated_logit + b)`, fitted by
   maximum likelihood on FF++ development video-level aggregated logits.
3. **Threshold.** The calibrated FF++ development video score that maximises
   balanced accuracy, using the existing `validation_threshold` helper.

Celeb-DF development data is not used for any of these choices.

## Inference abstention policy

The detector reports `ABSTAIN` instead of a label when:

- no face is detected, or the best detection confidence is below 0.80;
- the selected face's shorter bounding-box side is below 64 pixels;
- a video yields fewer than 4 frames with an accepted face.

When several faces pass, the largest face is scored and the result records the
face count. Media, crops, and embeddings are never written unless the caller asks.

## New evaluation protocols

### P1. Seen methods, FF++ (`df40_heldout_v1`)

Seven of the eight training methods: SimSwap, BlendFace, Wav2Lip, SadTalker,
StyleGAN2, SD-2.1, DiT. DF40 publishes FOMM only in its training archive, so FOMM
has no held-out videos and is excluded. Videos come from the authors' non-training
archives. A video is eligible only if none of its source clips or identities
occurs in the frozen manifest's train or validation split. The sampler takes 20
videos per method and eight frames per video, balanced 1:1 with real FF++ frames
from eligible real videos.

### P2. Seen methods, Celeb-DF domain

The same seven methods and eligibility rule on Celeb-DF, with 10 videos per method
(real Celeb-DF videos are the limiting pool) and 1:1 real frames. Training used
FF++ only, so P2 isolates domain shift from method shift. The existing frozen
Celeb-DF test (unseen methods) shows both shifts together.

### P3. Unseen domain and family: StarGANv2 face editing on CelebA

300 StarGANv2 images against 300 real CelebA images from DF40's evaluation
archive. CelebA never occurs in training or development, and face editing is a
manipulation family absent from training. Each image is its own group, so P3
reports image-level metrics.

P1–P3 form one manifest. Before scoring, an audit must report no video, source
clip, identity, pixel-hash, or perceptual-hash overlap with the frozen train or
validation split.

### P4. Robustness

Each model scores the frozen test split and P1–P2 under fixed, seeded
corruptions applied to the stored face crop before model preprocessing:

| Corruption | Severities |
|---|---|
| JPEG re-encode (quality) | 90, 70, 50, 30 |
| Downscale then upscale (scale) | 0.75, 0.50, 0.25 |
| Gaussian blur (sigma, pixels) | 0.5, 1.0, 2.0 |
| Gaussian noise (std, 0–255 scale) | 2, 5, 10 |
| Centre crop, then resize (kept fraction) | 0.90, 0.80, 0.70 |
| H.264 single-frame re-encode (CRF) | 23, 30, 38 |

The reported quantity is FF++ macro video AUROC (and Celeb-DF separately) per
corruption and severity, plus the drop from clean.

### P5. Raw-video cross-dataset: Celeb-DF v2 official test list

All 518 videos in Celeb-DF v2's published test list (178 real, 340 fake) go
through the complete inference pipeline: our face detection and alignment, the
fixed aggregation, calibration, and threshold. This is the only protocol that
uses our own preprocessing instead of DF40's crops. Abstentions are counted and
reported. AUROC is computed over non-abstained videos, and a second figure scores
abstentions as 0.5.

Before P5 is scored, a preprocessing agreement check compares our crops of
Celeb-DF real frames with DF40's crops of the same frames. The check reports the
mean landmark and pixel differences; it gates nothing.

DF40's Celeb-DF development set shares identities with this list and informed the
release-model rule. P5 is therefore also reported on the subset of videos whose
`idNN` identities do not occur in DF40 Celeb-DF development data.

## Pre-registered comparisons

Each uses the existing grouped paired bootstrap (2,000 replicates, seed
20261006, source-video lineage groups; P3 groups by image; P5 groups by Celeb-DF
identity).

- **H1 (P2).** CLIP seed 20261006 minus ConvNeXt-Tiny seed 20261006 macro AUROC
  is positive, with a 95% interval excluding zero. Development data predicts
  this; the frozen Celeb-DF test could not separate method shift from domain
  shift.
- **H2 (P5).** The same comparison through the raw-video pipeline, using
  video-level AUROC.
- **H3 (P1).** The ConvNeXt-Tiny and CLIP clear-improvement gate versus ResNet18
  (difference ≥ 0.05 and interval excluding zero) on seen methods.

All other tables, including per-method results, seed spread, frequency results
on P1–P4, and the robustness curves, are descriptive.
