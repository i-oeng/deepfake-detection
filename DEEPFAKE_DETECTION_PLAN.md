# Deepfake Detection Project Plan

## 1. Goal

Build a reproducible detector that classifies face images or video frames as `REAL` or `FAKE` and measures how well the detector handles manipulation methods and source domains that were absent from training.

The planned detector has three experiment tracks:

1. An RGB model that learns spatial and semantic evidence.
2. A frequency model that learns spectral and high-frequency evidence.
3. A fused model that combines both signals.

All three tracks have frozen results, and a calibrated inference package
exists. Section 2 records where each milestone stands; sections 4-10 remain the
reference design.

## 2. Current state (2026-10-07)

| Milestone | Status | Evidence |
|---|---|---|
| M1 Dataset | Done | Frozen manifest `07f906d9ef61a537efa8`; [protocol](reports/experiments/df40_unseen_protocol.md) |
| M2 Face preprocessing | Done for raw media | DF40 ships crops; `deepfake-detect crops` builds content-addressed crops from raw images and video with a pinned detector, and rebuilds are bit-identical. Crops agree with DF40's to a median 4.6 px |
| M3 RGB baseline | Done | ResNet18, ConvNeXt-Tiny (3 seeds), CLIP ViT-B/16 (3 seeds); [RGB result](reports/experiments/df40_unseen_final_rgb.md) |
| M4 Frequency branch | Done; exit criterion not met | Ablation picks all inputs + spectral filter, but the branch improves no protocol; [frequency result](reports/experiments/df40_unseen_frequency_final.md) |
| M5 Fusion | Done; exit criterion not met | Tuned blends do not beat RGB alone (0/3 seeds per backbone). Fusion adds 1.3 ms and 11.3 M parameters per face |
| M6 Evaluation | Done | Seen-method, Celeb-DF domain, CelebA face-editing, robustness, and raw Celeb-DF video protocols; [extended result](reports/experiments/df40_extended_evaluation.md) |
| M7 Inference | Done | `deepfake-detect`, CLIP release bundle with development-fixed aggregation, calibration, and threshold; [model card](MODEL_CARD.md) |

Main findings:

- CLIP ViT-B/16 generalizes across domains best: Celeb-DF seen methods 0.80 vs
  ConvNeXt 0.72 (pre-registered H1, interval above zero), CelebA face editing
  0.81 vs 0.63, raw Celeb-DF video 0.75 vs 0.65.
- ConvNeXt-Tiny is marginally best on FF++, where every model trained.
- The frequency branch and RGB-frequency fusion add nothing measurable.
- Lip-sync and talking-head methods on Celeb-DF stay near chance, and strong
  H.264 compression (CRF 38) costs 0.19-0.36 AUROC.
- The FF++-calibrated threshold does not transfer to raw Celeb-DF video.

## 3. Target pipeline

```text
Official metadata
      |
      v
Normalized catalog with video, identity, method, family, and lineage fields
      |
      v
Deterministic pilot selection
      |
      v
Image download or video-frame extraction
      |
      v
Immutable manifest and data audit
      |
      v
Versioned face detection, alignment, and crop generation
      |
      v
RGB baseline -> frequency baseline -> fused detector
      |
      v
Seen-method, unseen-method, unseen-domain, and cross-dataset evaluation
      |
      v
Calibrated image/video inference package
```

## 4. Phase 1: Freeze a trustworthy dataset

### Tasks

1. Normalize the official DF40 metadata into one row per selected image or frame.
2. Preserve these fields when the source data provides them:
   - `image_id`;
   - `relative_path`;
   - `label`;
   - `split`;
   - `fake_method`;
   - `manipulation_family`;
   - `source_domain`;
   - `video_id`;
   - `identity_id`;
   - `frame_index`;
   - source-video and target-identity lineage.
3. Generate the pilot subset and inspect all quota shortfalls.
4. Download images or extract the selected frames.
5. Build the immutable manifest.
6. Fix every missing, ambiguous, unsafe, or corrupt sample.
7. Remove cross-split duplicate content and group leakage.
8. Set `strict_quotas: true` before the first reportable experiment.
9. Record the accepted subset ID and manifest ID in the training configuration.

### Exit criteria

- All configured quality gates pass.
- All expected methods and source domains meet their minimum sample counts.
- Train, validation, and test identities and videos are disjoint.
- The manifest checksum verifies.
- A second run produces the same subset and manifest IDs.

## 5. Phase 2: Add versioned face preprocessing

### Tasks

- Detect faces and landmarks.
- Handle zero, one, and multiple detected faces through an explicit policy.
- Produce an expanded face crop that retains some blending boundaries and context.
- Align and resize crops consistently.
- Record crop coordinates, detector version, confidence, output size, and preprocessing configuration.
- Store derived crops under a content-addressed artifact ID.
- Apply the same deterministic preprocessing to every experiment.

Training augmentation should simulate realistic distribution shifts:

- JPEG and video compression;
- resize and resampling changes;
- blur, sensor noise, and sharpening;
- moderate color and gamma changes;
- partial occlusion and crop jitter.

Apply spatial transforms consistently to the RGB and frequency views of the same sample.

### Exit criteria

- Every accepted manifest row has a reproducible crop or a documented exclusion reason.
- Preprocessing does not use label, method, or split metadata.
- Rebuilding with the same input and configuration produces the same derived artifact ID.

## 6. Phase 3: Train an RGB baseline

Start with a simple, strong reference model. Suitable candidates include a pretrained vision foundation encoder or an EfficientNet/Xception-style baseline.

### Initial experiment

- Binary real/fake classification head.
- Frozen encoder followed by partial fine-tuning as a separate experiment.
- Binary cross-entropy loss.
- Group-aware batches with controlled method and source-domain balance.
- Early stopping based on validation AUROC, with the decision threshold selected on validation data only.
- At least three random seeds for the final comparison.

Each run should store:

- Git commit;
- subset and manifest IDs;
- preprocessing artifact ID;
- full resolved configuration;
- package and CUDA versions;
- random seeds;
- metrics and per-slice metrics;
- checkpoint hash.

### Exit criteria

- The same configuration can reproduce the run.
- Training does not read files outside the frozen manifest.
- Metrics are available by manipulation method, family, domain, compression level, and video.

## 7. Phase 4: Add the frequency branch

Avoid treating a raw FFT magnitude image as the entire frequency solution. A detector can memorize generator- or codec-specific spectral patterns.

Evaluate a frequency branch that combines:

- log-amplitude and phase information;
- high-pass or residual representations;
- learnable frequency processing inside intermediate feature maps;
- the same crop and augmentation provenance as the RGB branch.

Train and report the frequency model separately before fusion. Its value should come from better unseen-method or unseen-domain performance, not only from higher training accuracy.

### Exit criteria

- RGB-only and frequency-only results use identical data and evaluation protocols.
- The frequency branch improves at least one frozen generalization protocol without a large regression on the others.
- Frequency ablations identify which input and feature components help.

## 8. Phase 5: Add RGB-frequency fusion

Begin with calibrated late fusion:

```text
RGB logit -----------\
                      > weighted calibrated sum -> fake probability
Frequency logit -----/
```

Compare:

1. Equal-weight average.
2. Learned scalar weights fitted on validation data.
3. Feature concatenation with a small fusion head.

Add gated fusion or cross-attention only if the simpler variants show that the branches contain complementary information.

### Exit criteria

- The fused model beats both individual branches on frozen unseen-method or cross-domain metrics.
- The comparison includes parameter count, latency, memory use, and calibration.
- An ablation proves that both branches contribute.

## 9. Phase 6: Evaluation protocol

Maintain separate frozen protocols:

- **Seen method:** methods represented in training, using an untouched test partition.
- **Unseen method:** manipulation methods excluded from training.
- **Unseen domain:** new source datasets, identities, capture pipelines, or codecs.
- **Cross-dataset:** train on DF40 and evaluate on a separately audited dataset.
- **Robustness:** repeated evaluation after controlled compression, resizing, blur, noise, cropping, and re-encoding.

Report:

- AUROC and average precision;
- TPR at low false-positive rates;
- equal error rate;
- F1 at a validation-locked threshold;
- Brier score or expected calibration error;
- image-level and video-level metrics;
- bootstrap confidence intervals;
- mean and spread across seeds;
- per-method, family, source, compression, and demographic slices when metadata quality permits.

Do not select a model from test-set performance. Lock all model, threshold, and fusion choices using training and validation data.

## 10. Phase 7: Video support and inference

The first release can classify individual frames and aggregate their logits at video level. Compare mean, median, trimmed mean, and top-k aggregation on validation data.

The second release can add temporal modeling for manipulations such as Wav2Lip and SadTalker. Candidate inputs include short face tracks, frame-difference features, optical-flow features, or a video encoder. Treat temporal detection as a separate experiment so its gain remains measurable.

The inference package should:

- accept an image or video;
- report face-detection and preprocessing failures;
- produce calibrated probabilities;
- expose the threshold and model version;
- aggregate frame scores for video;
- return an uncertainty or abstention result for unsupported inputs;
- log no biometric media by default.

## 11. Recommended changes to the current data layer

### Priority 0

1. **Validate every grouping field across splits.** The subset sampler currently uses the first non-empty configured group, which normally means `video_id`. Check `identity_id` independently even when `video_id` exists.
2. **Detect near duplicates.** Add pHash or dHash Hamming-distance search instead of grouping only identical hash strings.
3. **Validate unique identifiers.** Reject duplicate `image_id` values and conflicting paths before building a manifest.
4. **Add coverage gates.** Fail on insufficient per-method, per-family, per-domain, or per-split counts.
5. **Freeze the real experiment with `strict_quotas: true`.** The relaxed setting should remain limited to catalog exploration.

### Priority 1

1. Add source-video lineage and matched real/fake sampling where the dataset provides it.
2. Add optional face-embedding identity audits for datasets without reliable identity IDs.
3. Change suspicious metadata leakage from a warning to a failure for final experiments.
4. Add resolution, aspect-ratio, codec, compression, and frame-position distributions to the audit report.
5. Create separate seen-method and unseen-method test protocols.

### Priority 2

1. Add experiment tracking and machine-readable result summaries.
2. Add preprocessing and training smoke tests to CI.
3. Add calibration, latency, and memory benchmarks.
4. Document dataset licenses and approved use cases before deployment.

## 12. Suggested milestone order

| Milestone | Deliverable |
|---|---|
| M1 | Real DF40 catalog adapter and passing pilot audit |
| M2 | Content-addressed face crops and preprocessing report |
| M3 | Reproducible RGB baseline |
| M4 | Frequency baseline and ablation |
| M5 | Late-fusion model and frozen protocol comparison |
| M6 | Cross-dataset and robustness evaluation |
| M7 | Video aggregation, calibrated inference, and model card |

## 13. Next steps

1. Re-fit the decision threshold and calibration on labeled target-domain data
   before using `FAKE` / `REAL` labels outside FF++-like footage.
2. Add a temporal model for lip-sync and talking-head manipulation (Phase 7's
   second release); frame models are near chance on these methods off-domain.
3. Train with compression augmentation that includes H.264, and re-run P4.
4. Collect a labeled whole-image set (real photos and current text-to-image
   outputs) before claiming anything about fully generated images; DF40 covers
   face crops only.
5. Add demographic slice metrics once a dataset with reliable labels is available.

## References

- [DF40 official repository](https://github.com/YZY-stack/DF40)
- [DF40 paper](https://arxiv.org/abs/2406.13495)
- [DeepfakeBench](https://arxiv.org/abs/2307.01426)
- [FreqNet: Frequency-Aware Deepfake Detection](https://arxiv.org/abs/2403.07240)
- [GenD: Deepfake Detection that Generalizes Across Benchmarks](https://openaccess.thecvf.com/content/WACV2026/papers/Yermakov_Deepfake_Detection_that_Generalizes_Across_Benchmarks_WACV_2026_paper.pdf)
