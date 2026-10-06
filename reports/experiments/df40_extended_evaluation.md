# DF40 extended evaluation and release result

## Outcome

All protocols were fixed in the [pre-registration](df40_extended_evaluation_protocol.md)
before any of their data was selected or scored. Every pre-registered test in
P1–P2 holds, and the release model generalizes better across domains than the
FF++ finalist. The release detector's FF++-calibrated threshold does not
transfer to raw Celeb-DF video.

- **H3 (seen methods, FF++).** ConvNeXt-Tiny (all three seeds) and CLIP beat the
  ResNet18 baseline by +0.120 to +0.129 macro AUROC, with every 95% interval
  above +0.07 over 142 source groups. On methods it trained on, ConvNeXt (0.992)
  and CLIP (0.987) are close to ceiling.
- **H1 (seen methods, Celeb-DF domain).** CLIP beats ConvNeXt by +0.098, 95% CI
  [+0.020, +0.169]. CLIP reaches 0.800 three-seed mean; ConvNeXt 0.719, below
  ResNet18 (0.758). The development-data prediction that motivated the release
  rule holds on data neither selected nor seen. ConvNeXt's earlier Celeb-DF
  collapse is mainly domain shift, not only unseen methods.
- **P3 (unseen domain and family).** On StarGANv2 face edits of CelebA, CLIP
  scores 0.814, ConvNeXt 0.634, and ResNet18 0.452, which is below chance.
- **Where CLIP still fails.** On Celeb-DF, CLIP detects identity swaps and
  full-face synthesis (0.95–1.00 AUROC) but not lip-sync and talking-head
  reenactment: SadTalker 0.50 and Wav2Lip 0.61.
- **P4 (robustness).** Every model degrades under compression. Strong H.264
  (CRF 38) drops FF++ AUROC to 0.63–0.73 and Celeb-DF to 0.50–0.56. CLIP
  degrades least under JPEG, resizing, blur, and noise; ConvNeXt degrades most.
- **P5 (raw Celeb-DF v2 video, full pipeline).** CLIP ranks the 518 official
  test videos at 0.749 AUROC (0.752 on the 384 videos with no DF40 development
  identity), ahead of ConvNeXt (0.654) and ResNet18 (0.607). **H2 is not
  supported:** CLIP minus ConvNeXt is +0.095, 95% CI [−0.013, +0.131]. One
  identity component holds 335 of the 518 videos, which leaves few effective
  groups. The calibrated threshold labels 515 of the 517 decided videos FAKE,
  so accuracy (0.66) equals the fake share. The FF++ operating point does not
  carry over to this capture and compression pipeline.
- **CLIP seed spread.** The two new CLIP seeds score 0.998 and 0.998 on the frozen
  FF++ test and 0.904 and 0.932 on the frozen Celeb-DF test. The three-seed mean
  is 0.933 ± 0.029 on Celeb-DF, so the release seed's 0.963 is its best seed, and
  that single-seed figure is optimistic. It is still far above ConvNeXt's
  0.661 ± 0.035.
- **Frequency branch.** Its checkpoints are not readable from the shared
  project directory, so it was not scored on P1–P4. Its development and
  final-test results stand as published in PR #16.

## Frozen inputs

- Protocol: [`df40_extended_evaluation_protocol.md`](df40_extended_evaluation_protocol.md)
- Held-out manifest: `956ece9d65a3deee6b95`, gated by [`df40_heldout_v1_956ece9d65a3deee6b95.json`](../identity_audit/df40_heldout_v1_956ece9d65a3deee6b95.json)
- Frozen unseen-method manifest: `07f906d9ef61a537efa8`
- Runs without readable checkpoints: frequency-20261006, frequency-20261007, frequency-20261008

## Held-out protocols P1-P3

### P1. Seen methods, FF++ (20 videos per method)

Mean ± sample SD [min, max] over seeds; one value when a family has one run.

| Model | Seeds | AUROC | AP | TPR@1% FPR | Brier | ECE-10 |
|---|---:|---:|---:|---:|---:|---:|
| ResNet18 | 1 | 0.8657 | 0.6673 | 0.5500 | 0.0941 | 0.1335 |
| ConvNeXt-Tiny | 3 | 0.9922 ± 0.0026 [0.9897, 0.9950] | 0.9721 ± 0.0122 [0.9585, 0.9820] | 0.9500 ± 0.0071 [0.9429, 0.9571] | 0.0190 ± 0.0139 [0.0107, 0.0350] | 0.0414 ± 0.0274 [0.0240, 0.0730] |
| CLIP ViT-B/16 | 3 | 0.9868 ± 0.0007 [0.9860, 0.9874] | 0.9399 ± 0.0046 [0.9356, 0.9447] | 0.7952 ± 0.0082 [0.7857, 0.8000] | 0.0323 ± 0.0015 [0.0313, 0.0341] | 0.0934 ± 0.0056 [0.0880, 0.0991] |

### P2. Seen methods, Celeb-DF domain (10 videos per method)

Mean ± sample SD [min, max] over seeds; one value when a family has one run.

| Model | Seeds | AUROC | AP | TPR@1% FPR | Brier | ECE-10 |
|---|---:|---:|---:|---:|---:|---:|
| ResNet18 | 1 | 0.7578 | 0.4046 | 0.1351 | 0.2400 | 0.3187 |
| ConvNeXt-Tiny | 3 | 0.7191 ± 0.0061 [0.7123, 0.7242] | 0.3061 ± 0.0265 [0.2757, 0.3241] | 0.0048 ± 0.0082 [0.0000, 0.0143] | 0.6843 ± 0.0760 [0.6028, 0.7533] | 0.7402 ± 0.0561 [0.6807, 0.7920] |
| CLIP ViT-B/16 | 3 | 0.8004 ± 0.0159 [0.7910, 0.8187] | 0.5895 ± 0.0196 [0.5719, 0.6107] | 0.4208 ± 0.0213 [0.4026, 0.4442] | 0.3477 ± 0.0075 [0.3414, 0.3559] | 0.4906 ± 0.0059 [0.4869, 0.4975] |

### P3. StarGANv2 face editing on CelebA (300 + 300 images)

Mean ± sample SD [min, max] over seeds; one value when a family has one run.

| Model | Seeds | AUROC | AP | TPR@1% FPR | Brier | ECE-10 |
|---|---:|---:|---:|---:|---:|---:|
| ResNet18 | 1 | 0.4515 | 0.4627 | 0.0033 | 0.4625 | 0.4401 |
| ConvNeXt-Tiny | 3 | 0.6336 ± 0.0134 [0.6181, 0.6421] | 0.5967 ± 0.0168 [0.5796, 0.6133] | 0.0122 ± 0.0038 [0.0100, 0.0167] | 0.3359 ± 0.0374 [0.3048, 0.3774] | 0.2996 ± 0.0546 [0.2578, 0.3613] |
| CLIP ViT-B/16 | 3 | 0.8140 ± 0.0125 [0.8056, 0.8284] | 0.8270 ± 0.0063 [0.8210, 0.8335] | 0.1911 ± 0.0269 [0.1667, 0.2200] | 0.2301 ± 0.0082 [0.2243, 0.2395] | 0.2141 ± 0.0239 [0.1978, 0.2415] |

### Per-method AUROC, clip-20261006

| Protocol | Method | Videos (incl. real) | AUROC | AP |
|---|---|---:|---:|---:|
| P1 | blendface | 160 | 0.9968 | 0.9803 |
| P1 | dit | 160 | 0.9818 | 0.8145 |
| P1 | sadtalker | 160 | 0.9404 | 0.8504 |
| P1 | sd21 | 160 | 1.0000 | 1.0000 |
| P1 | simswap | 160 | 0.9982 | 0.9892 |
| P1 | stylegan2 | 160 | 1.0000 | 1.0000 |
| P1 | wav2lip | 160 | 0.9850 | 0.9145 |
| P2 | blendface | 81 | 0.9535 | 0.8386 |
| P2 | dit | 81 | 0.7085 | 0.1996 |
| P2 | sadtalker | 81 | 0.5028 | 0.1592 |
| P2 | sd21 | 82 | 0.9974 | 0.9860 |
| P2 | simswap | 81 | 0.9563 | 0.7558 |
| P2 | stylegan2 | 81 | 1.0000 | 1.0000 |
| P2 | wav2lip | 81 | 0.6127 | 0.3354 |
| P3 | starganv2 | 600 | 0.8056 | 0.8210 |

## Pre-registered comparisons

Grouped paired bootstrap, 2,000 replicates, source-video lineage groups. Clear improvement requires a difference of at least 0.05 and a 95% interval above zero.

| Comparison | Difference | 95% CI | Source groups | Clear improvement |
|---|---:|---:|---:|---|
| H1 CLIP minus ConvNeXt on P2 | +0.0979 | [+0.0197, +0.1691] | 30 | yes |
| H3 CLIP 20261006 minus ResNet18 on P1 | +0.1204 | [+0.0737, +0.1670] | 142 | yes |
| H3 ConvNeXt 20261006 minus ResNet18 on P1 | +0.1241 | [+0.0807, +0.1688] | 142 | yes |
| H3 ConvNeXt 20261007 minus ResNet18 on P1 | +0.1263 | [+0.0828, +0.1705] | 142 | yes |
| H3 ConvNeXt 20261008 minus ResNet18 on P1 | +0.1293 | [+0.0841, +0.1745] | 142 | yes |

## P4. Robustness

### Frozen unseen-method test, FF++

Macro video AUROC per corruption.

| Model | clean | jpeg-90 | jpeg-70 | jpeg-50 | jpeg-30 | resize-0.75 | resize-0.5 | resize-0.25 | blur-0.5 | blur-1 | blur-2 | noise-2 | noise-5 | noise-10 | crop-0.9 | crop-0.8 | crop-0.7 | h264-23 | h264-30 | h264-38 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet18 | 0.888 | 0.878 | 0.870 | 0.866 | 0.818 | 0.873 | 0.852 | 0.779 | 0.882 | 0.853 | 0.778 | 0.844 | 0.737 | 0.671 | 0.881 | 0.891 | 0.888 | 0.845 | 0.795 | 0.701 |
| ConvNeXt-Tiny 20261006 | 0.999 | 0.916 | 0.882 | 0.846 | 0.835 | 0.998 | 0.988 | 0.760 | 0.999 | 0.978 | 0.665 | 0.839 | 0.757 | 0.682 | 1.000 | 1.000 | 0.997 | 0.900 | 0.842 | 0.686 |
| CLIP ViT-B/16 20261006 | 0.999 | 0.981 | 0.960 | 0.948 | 0.898 | 0.997 | 0.971 | 0.888 | 0.999 | 0.982 | 0.875 | 0.929 | 0.856 | 0.806 | 0.998 | 0.996 | 0.988 | 0.912 | 0.847 | 0.731 |

### P1 seen methods, FF++

Macro video AUROC per corruption.

| Model | clean | jpeg-90 | jpeg-70 | jpeg-50 | jpeg-30 | resize-0.75 | resize-0.5 | resize-0.25 | blur-0.5 | blur-1 | blur-2 | noise-2 | noise-5 | noise-10 | crop-0.9 | crop-0.8 | crop-0.7 | h264-23 | h264-30 | h264-38 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet18 | 0.866 | 0.853 | 0.842 | 0.841 | 0.828 | 0.853 | 0.838 | 0.802 | 0.862 | 0.826 | 0.767 | 0.828 | 0.771 | 0.752 | 0.854 | 0.858 | 0.822 | 0.831 | 0.808 | 0.693 |
| ConvNeXt-Tiny 20261006 | 0.990 | 0.907 | 0.855 | 0.835 | 0.801 | 0.979 | 0.960 | 0.856 | 0.989 | 0.958 | 0.822 | 0.830 | 0.749 | 0.710 | 0.989 | 0.993 | 0.988 | 0.875 | 0.790 | 0.632 |
| CLIP ViT-B/16 20261006 | 0.986 | 0.944 | 0.925 | 0.906 | 0.884 | 0.973 | 0.938 | 0.878 | 0.985 | 0.959 | 0.872 | 0.931 | 0.884 | 0.837 | 0.985 | 0.991 | 0.988 | 0.884 | 0.831 | 0.664 |

### Frozen unseen-method test, Celeb-DF

Macro video AUROC per corruption.

| Model | clean | jpeg-90 | jpeg-70 | jpeg-50 | jpeg-30 | resize-0.75 | resize-0.5 | resize-0.25 | blur-0.5 | blur-1 | blur-2 | noise-2 | noise-5 | noise-10 | crop-0.9 | crop-0.8 | crop-0.7 | h264-23 | h264-30 | h264-38 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet18 | 0.858 | 0.840 | 0.824 | 0.787 | 0.725 | 0.864 | 0.833 | 0.793 | 0.864 | 0.843 | 0.784 | 0.796 | 0.679 | 0.633 | 0.855 | 0.799 | 0.713 | 0.741 | 0.602 | 0.568 |
| ConvNeXt-Tiny 20261006 | 0.639 | 0.457 | 0.460 | 0.485 | 0.478 | 0.642 | 0.623 | 0.466 | 0.636 | 0.552 | 0.383 | 0.398 | 0.380 | 0.429 | 0.648 | 0.552 | 0.509 | 0.488 | 0.429 | 0.531 |
| CLIP ViT-B/16 20261006 | 0.963 | 0.938 | 0.914 | 0.889 | 0.836 | 0.957 | 0.944 | 0.892 | 0.966 | 0.969 | 0.929 | 0.818 | 0.735 | 0.858 | 0.901 | 0.904 | 0.864 | 0.910 | 0.778 | 0.673 |

### P2 seen methods, Celeb-DF

Macro video AUROC per corruption.

| Model | clean | jpeg-90 | jpeg-70 | jpeg-50 | jpeg-30 | resize-0.75 | resize-0.5 | resize-0.25 | blur-0.5 | blur-1 | blur-2 | noise-2 | noise-5 | noise-10 | crop-0.9 | crop-0.8 | crop-0.7 | h264-23 | h264-30 | h264-38 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet18 | 0.758 | 0.747 | 0.745 | 0.724 | 0.711 | 0.756 | 0.747 | 0.701 | 0.758 | 0.741 | 0.675 | 0.739 | 0.694 | 0.662 | 0.728 | 0.708 | 0.657 | 0.717 | 0.686 | 0.562 |
| ConvNeXt-Tiny 20261006 | 0.721 | 0.680 | 0.633 | 0.624 | 0.602 | 0.724 | 0.667 | 0.669 | 0.715 | 0.702 | 0.634 | 0.629 | 0.597 | 0.570 | 0.723 | 0.699 | 0.685 | 0.630 | 0.567 | 0.504 |
| CLIP ViT-B/16 20261006 | 0.819 | 0.819 | 0.821 | 0.815 | 0.738 | 0.797 | 0.765 | 0.764 | 0.827 | 0.812 | 0.743 | 0.763 | 0.752 | 0.767 | 0.799 | 0.789 | 0.749 | 0.783 | 0.709 | 0.560 |

## P5. Raw Celeb-DF v2 video through the inference pipeline

Videos: 518; identity-disjoint from DF40 Celeb-DF development: 384.

| Bundle | FAKE / REAL / ABSTAIN | Accuracy at threshold | AUROC (all) | AUROC (identity-disjoint) | AUROC, abstain = 0.5 | AP | TPR@1% FPR | Brier | ECE-10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clip-20261006 | 515 / 2 / 1 | 0.6615 | 0.7490 | 0.7520 | 0.7504 | 0.8130 | 0.0118 | 0.2721 | 0.2814 |
| convnext-20261006 | 511 / 6 / 1 | 0.6692 | 0.6537 | 0.6332 | 0.6555 | 0.7570 | 0.0206 | 0.2816 | 0.2672 |
| resnet18-20261006 | 201 / 316 / 1 | 0.5455 | 0.6066 | 0.5806 | 0.6060 | 0.7557 | 0.0559 | 0.2774 | 0.2051 |

**H2** (clip-20261006 minus convnext-20261006, video AUROC): +0.0952, 95% CI [-0.0127, +0.1314], 75 identity groups, 1973 valid replicates.

## Face-crop agreement with DF40

On 40 real Celeb-DF frames that DF40 also cropped, landmarks in our 1.3-margin crops lie a median 4.6 px (mean 5.3 px) from DF40's on a 256 px crop; mean absolute pixel difference is 16.1 of 255. Other margins were 1.2: 8.7 px, 1.25: 6.9 px, 1.3: 5.3 px, 1.35: 5.0 px, 1.4: 5.3 px (mean). The check gated nothing; the margin stayed at DeepfakeBench's 1.3.

## Cost

Measured on NVIDIA GeForce RTX 5060 with PyTorch 2.14.1+cu130, mixed precision.

| Model | Parameters (M) | Latency, batch 1 (ms) | Throughput, batch 32 (img/s) | Peak memory, batch 32 (MiB) |
|---|---:|---:|---:|---:|
| ResNet18 | 11.2 | 1.0 | 4572 | 210 |
| ConvNeXt-Tiny | 27.8 | 2.0 | 1357 | 379 |
| CLIP ViT-B/16 | 85.8 | 3.0 | 658 | 564 |
| Frequency | 11.3 | 1.3 | 3590 | 267 |

Late fusion runs both branches, so CLIP + frequency costs 97.1 M parameters and 4.3 ms per image at batch 1, for no accuracy gain on the frozen test.
