# DF40 unseen-method RGB result

The final test was exported once after the development-only selection record was
committed. FF++ is the primary unseen-method benchmark. Celeb-DF is reported
separately as cross-dataset transfer.

## Primary FF++ result

All values are video-level. Calibration reports Brier score and 10-bin expected
calibration error (ECE); lower is better for those two columns.

| Model | AUROC | Average precision | TPR at 1% FPR | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| ResNet18 rerun, seed 20261006 | 0.8884 | 0.8486 | 0.6889 | 0.1054 | 0.1483 |
| CLIP ViT-B/16, seed 20261006 | **0.9990** | **0.9971** | 0.9556 | 0.0267 | 0.1144 |
| ConvNeXt-Tiny, seed 20261006 | 0.9985 | 0.9957 | 0.9333 | 0.0328 | **0.0822** |
| ConvNeXt-Tiny, three-seed mean | 0.9974 | 0.9941 | **0.9630** | **0.0205** | **0.0554** |

The frozen ConvNeXt seed 20261006 improves macro AUROC over the rerun ResNet18
by **0.1101**. Its source-video-lineage grouped paired bootstrap 95% confidence
interval is **[0.0227, 0.1913]** over 2,000 valid replicates and 37 source
groups. The improvement exceeds 0.05 and the interval excludes zero, so it
passes the predefined **clear improvement** gate. The other two ConvNeXt seeds
also pass independently, with deltas of 0.1086 and 0.1081 and positive lower
confidence bounds.

The earlier 0.515 pilot used a mixed-domain test and is not directly comparable.
The 0.8884 ResNet18 result above is the rerun on this frozen protocol.

### ConvNeXt per-method result, seed 20261006

| Held-out method | AUROC | Average precision | TPR at 1% FPR | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| MCNet | 0.9985 | 0.9958 | 0.9333 | 0.0324 | 0.0850 |
| RDDM | 1.0000 | 1.0000 | 1.0000 | 0.0304 | 0.0871 |
| UniFace | 0.9970 | 0.9914 | 0.8667 | 0.0355 | 0.0744 |
| Macro | 0.9985 | 0.9957 | 0.9333 | 0.0328 | 0.0822 |

### ConvNeXt three-seed spread

| Metric | Mean | Sample standard deviation | Minimum | Maximum |
|---|---:|---:|---:|---:|
| Macro AUROC | 0.9974 | 0.0010 | 0.9965 | 0.9985 |
| Average precision | 0.9941 | 0.0015 | 0.9929 | 0.9957 |
| TPR at 1% FPR | 0.9630 | 0.0257 | 0.9333 | 0.9778 |
| Brier | 0.0205 | 0.0107 | 0.0133 | 0.0328 |
| ECE | 0.0554 | 0.0232 | 0.0409 | 0.0822 |

## Cross-dataset Celeb-DF result

| Model | AUROC | Average precision | TPR at 1% FPR | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| ResNet18 rerun, seed 20261006 | 0.8580 | 0.6760 | 0.2222 | 0.2248 | 0.2944 |
| CLIP ViT-B/16, seed 20261006 | **0.9630** | **0.9127** | **0.6667** | **0.3127** | **0.4402** |
| ConvNeXt-Tiny, seed 20261006 | 0.6389 | 0.3786 | 0.0000 | 0.6449 | 0.6692 |
| ConvNeXt-Tiny, three-seed mean | 0.6605 | 0.3931 | 0.0185 | 0.6007 | 0.6379 |

ConvNeXt's FF++ gain does not transfer to Celeb-DF. CLIP is the strongest
cross-dataset RGB model. The CDF paired bootstrap has only five source-lineage
groups, so its interval is much less informative than the 37-group FF++ result.

### ConvNeXt CDF per-method result, seed 20261006

| Held-out method | AUROC | Average precision | TPR at 1% FPR | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| MCNet | 0.5370 | 0.2887 | 0.0000 | 0.6452 | 0.6719 |
| RDDM | 0.6111 | 0.3160 | 0.0000 | 0.6448 | 0.6677 |
| UniFace | 0.7685 | 0.5310 | 0.0000 | 0.6448 | 0.6680 |
| Macro | 0.6389 | 0.3786 | 0.0000 | 0.6449 | 0.6692 |

## Artifacts and remaining fusion work

- [`df40_unseen_final_summary.json`](df40_unseen_final_summary.json) contains
  every model's FF++ and CDF per-method metrics and paired bootstrap results.
- [`df40_convnext_three_seed_test.json`](df40_convnext_three_seed_test.json)
  contains all three ConvNeXt runs, checkpoint hashes, and metric spreads.
- The frequency branch and checkpoint are not present on the current GitHub
  origin yet. Fusion remains pending that export; no final-test fusion weight
  will be tuned.
