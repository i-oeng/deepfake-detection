# Model card: DF40 CLIP ViT-B/16 face-manipulation detector

| | |
|---|---|
| Release bundle | `clip_vit_b16-43cfb4cf0eb7` (`artifacts/bundles/clip-20261006`) |
| Checkpoint | `clip_vit_b16-20261006-07f906d9ef61a537efa8-8f3b95e2-a77f7ac7`, SHA-256 `a3bcb4fb29af0d4d7e18c56f69b28fd48826bfbad3ce51d94f3b1e20c1d9cdf1` |
| Architecture | OpenAI CLIP ViT-B/16 vision tower (pinned revision `57c2164`), LayerNorms and a linear head trained; attention and MLP weights frozen |
| Input | One aligned 256 px face crop per frame, resized to 224 px |
| Output | Calibrated probability that the face is manipulated, a `FAKE` / `REAL` / `ABSTAIN` label, and the face decisions behind it |
| Selection record | [`df40_extended_evaluation_protocol.md`](reports/experiments/df40_extended_evaluation_protocol.md) |
| Results | [`df40_extended_evaluation.md`](reports/experiments/df40_extended_evaluation.md) |

## Intended use

Research on how face-manipulation detectors generalize, and assisted triage of
face images or short face videos by a person who makes the final call. The
probability is an input to human review. It is not evidence that a specific
person did or did not appear in, or create, a piece of media.

Out of scope:

- automated takedown, moderation, or account action without human review;
- legal, forensic, journalistic, or employment decisions;
- identity verification or liveness checks;
- media without a clearly visible face, including fully synthetic scenes;
- surveillance or the profiling of individuals.

## How a decision is made

1. **Face.** A pinned YuNet detector (`yunet-2023mar-8f2383e4dd3c`) finds faces.
   The largest face with confidence of at least 0.80 is aligned to the
   DeepfakeBench five-landmark template with a 1.3 margin, matching DF40's crops.
   If no face passes, or the face is under 64 px, the result is `ABSTAIN`.
2. **Frames.** A video is sampled at 32 evenly spaced frames and needs at least 4
   accepted faces; otherwise the result is `ABSTAIN`.
3. **Score.** Frame probabilities are averaged (mean probability), Platt-scaled
   with parameters fitted on FF++ development videos, and compared with the
   development balanced-accuracy threshold (0.2656).

Aggregation, calibration, and threshold were all fixed on FF++ development data
before any of the extended protocols were scored.

## Performance

Video-level macro AUROC unless stated. Pre-registered protocols; full tables in
[`df40_extended_evaluation.md`](reports/experiments/df40_extended_evaluation.md).

| Protocol | Release model (CLIP, seed 20261006) | CLIP, 3 seeds | ConvNeXt-Tiny, 3 seeds | ResNet18 |
|---|---:|---:|---:|---:|
| Unseen methods, FF++ (frozen test) | 0.999 | 0.998 ± 0.001* | 0.997 ± 0.001 | 0.888 |
| Seen methods, FF++ (P1) | 0.986 | 0.987 ± 0.001 | 0.992 ± 0.003 | 0.866 |
| Unseen methods, Celeb-DF (frozen test) | 0.963 | 0.933 ± 0.029* | 0.661 ± 0.035 | 0.858 |
| Seen methods, Celeb-DF (P2) | 0.819 | 0.800 ± 0.016 | 0.719 ± 0.006 | 0.758 |
| StarGANv2 on CelebA, images (P3) | 0.806 | 0.814 ± 0.013 | 0.634 ± 0.013 | 0.452 |
| Raw Celeb-DF v2 videos, full pipeline (P5) | 0.749 | — | 0.654 (one seed) | 0.607 |

\* Seeds 20261007 and 20261008 were trained after the frozen test was
published, with the frozen configuration unchanged; they measure seed spread
only.

**The decision threshold does not transfer across domains.** On raw Celeb-DF
v2 video the release bundle labels 515 of 517 decided videos `FAKE` (accuracy
0.66, equal to the fake share), although its ranking (AUROC 0.749) is useful.
Expected calibration error is 0.09 on FF++ but 0.28–0.49 on Celeb-DF. Treat the
probability as a ranking score outside FF++-like footage, and re-fit the
threshold on labeled data from the target domain before relying on labels.

**Method blind spots.** On Celeb-DF, CLIP detects identity swaps and full-face
synthesis (AUROC 0.95–1.00) but not lip-sync and talking-head reenactment
(SadTalker 0.50, Wav2Lip 0.61).

**Compression.** Strong H.264 re-encoding (CRF 38) drops FF++ AUROC to
0.66–0.73 and Celeb-DF to about 0.56. JPEG quality 30 costs 0.10 on FF++.

**Cost.** 85.8 M parameters; 3.0 ms per face at batch 1 and 658 faces/s at
batch 32 on an RTX 5060 with mixed precision; 564 MiB peak at batch 32.

## Training data

DF40 crops of FaceForensics++ source videos: 8 manipulation methods (SimSwap,
BlendFace, Wav2Lip, FOMM, SadTalker, StyleGAN2, SD-2.1, DiT) with matched real
frames, about 75 videos per method and up to 8 frames per video (4,776 fake and
4,800 real frames). Training used FF++
only. Development used FaceDancer, MRAA, and StyleGAN3. The frozen manifest
`07f906d9ef61a537efa8` passes a lineage audit with no cross-split video, source
clip, candidate identity, pixel-hash, or perceptual-hash overlap.

## Licenses and data terms

- **DF40**, **FaceForensics++**, and **Celeb-DF v2** are distributed for
  non-commercial research under their authors' terms. Checkpoints trained on
  them inherit those restrictions; do not deploy this model commercially.
- **CLIP ViT-B/16** weights: MIT license (OpenAI).
- **YuNet** face detector: MIT license (OpenCV Zoo).
- **ConvNeXt-Tiny / ResNet18** comparison weights: torchvision ImageNet weights,
  BSD-3-Clause.

## Known limitations

- **Small training set.** 4,776 fake and 4,800 real training frames from one
  source domain.
- **Domain shift.** Training saw FF++ only. Celeb-DF and CelebA results measure
  transfer; other capture pipelines, ethnicities, ages, lighting, and codecs are
  untested. No demographic slice metrics exist because the datasets publish no
  reliable demographic labels.
- **New generators.** Results cover DF40's 14 methods plus Celeb-DF v2's
  synthesis. Current text-to-image and video generators are unseen, and
  performance on them is unknown.
- **Image size.** DF40's full-face synthesis crops are 512 px while most others
  are 256 px; resizing to 224 px removes the size but may leave resampling traces
  a model could exploit.
- **Face policy.** The detector scores only the largest confident face. It can
  abstain on clear faces that YuNet scores below 0.80 (seen on stylized
  portraits), and it ignores smaller faces in group photos.
- **Identity evidence.** FF++ separation relies on source-clip IDs and Celeb-DF
  separation on `idNN` name tokens, not face recognition; person-level
  disjointness is not proven.
- **Calibration** was fitted on FF++ development videos (90 videos) and is not
  guaranteed on other domains.

## Ethical considerations

A false `FAKE` can harm a real person's reputation; a false `REAL` can lend
credibility to a manipulation. Report the probability and the abstention reasons
alongside any label, keep a human decision-maker in the loop, and do not log or
retain the faces processed. The CLI writes no media, crops, or embeddings unless
the caller explicitly builds a crop artifact.
