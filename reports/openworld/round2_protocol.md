# Open-world training round 2

This round is frozen before the fresh v2 test is scored. It addresses the v1
E7 model's weak transfer to MidJourney and its FF++ regression by adding modern
face synthesis and autoencoder traces to training while keeping model selection
on development data.

## Data roles

| Role | Method | Raw target | Purpose |
|---|---|---:|---|
| train | Kandinsky 2.2 | 2,000 | MoVQ based text-to-image faces |
| train | Sana 600M | 2,000 | DC-AE based text-to-image faces |
| train | MoVQ reconstructions | 2,000 | Autoencoder traces on aligned FF++ training reals |
| train | DC-AE reconstructions | 2,000 | A second, architecturally distinct autoencoder |
| validation | Stable Cascade lite | 1,000 | Unseen synthesis and compression development domain |

Every new fake is paired with a real source in the same preprocessing
collection. Both classes are face detected, aligned, scale matched, and encoded
as JPEG quality 90. Face policy exclusions are recorded when the additions are
materialized. The v1 rows remain byte for byte unchanged in the merged catalog.

Stable Diffusion 1.5, SDXL, FLUX.1, InsightFace, and Celeb-DF synthesis remain
reserved for the untouched fresh v2 test. Round 2 uses neither their weights nor
their autoencoders. A frozen audit must find no exact file, decoded pixel,
perceptual hash, or source lineage overlap between the additions and fresh v2.
The first post-canonicalization manifest is used only for this leakage check.
Any colliding addition is removed as a complete video group, after which the
catalog and immutable manifest are rebuilt. This pruning does not inspect test
labels or model predictions. When DF40 lacks an explicit person ID, copied
FF++ rows retain their documented `ff:<source clip>` lineage as the identity
unit used by the split audit.

## Selection and final evaluation

The E7 checkpoint and each Round 2 candidate are first scored on the merged
validation split. A candidate may advance only when its mean domain macro video
AUROC exceeds E7 and its FF++ validation AUROC is no more than 0.01 below E7.
Ties use Stable Cascade AUROC, then the original v1 validation mean.

After selection, the chosen candidate, E7, and the release CLIP B/16 baseline
are scored on fresh v2 once. The primary result is the unweighted mean of the
`celebdf_video`, `dff`, and `sfhq_t2i` macro video AUROCs. We call Round 2 a
clear improvement only when it exceeds E7 by at least 0.05 and the paired,
domain-stratified 95% bootstrap confidence interval for the difference excludes
zero. Per-domain AUROC, average precision, TPR at 1% FPR, calibration, seed
spread, and confidence intervals are reported even when that gate is missed.
