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
- **Exploratory: whole frames instead of face crops.** Not pre-registered. On
  the same 517 raw Celeb-DF videos and frames, feeding each frozen model the
  whole frame (stretched or letterboxed to 224 px) instead of the aligned face
  crop lowers video AUROC for every model: CLIP 0.749 to 0.610/0.641, ConvNeXt
  0.654 to 0.571/0.561, ResNet18 0.607 to 0.581/0.554. For CLIP the paired
  bootstrap intervals exclude zero (−0.139 [−0.179, −0.040] stretched). These
  models were trained on crops only, so this rules out swapping the input, not
  a model trained on whole frames. Data:
  [`df40_whole_frame_comparison.json`](df40_whole_frame_comparison.json).
