# Open-world DF40 result (protocol `df40_openworld_v1`)

Every generator has exactly one role. The model trains on 19 DF40 generators
(FF++ faces). Every choice was made on validation: five held-out DF40
generators on Celeb-DF and held-out FF++ clips, plus StarGAN, StyleCLIP, and
whichfaceisreal. The test split was scored once, after the recipe and seeds were
fixed: seven more held-out DF40 generators, plus DeepFaceLab, HeyGen, MidJourney,
CollabDiff, StarGANv2, and e4e.

- Manifest: `adf10611e74ccef2191d` (79,142 images). Lineage audit: no video,
  clip, identity, pixel, or perceptual-hash overlap across splits.
- Score: mean over domains of each domain's macro video AUROC.
- Baseline: the frozen-protocol release model (CLIP ViT-B/16, checkpoint
  `a3bcb4fb…`), scored without retraining.

## Recipe selection (validation, seed 20261007)

| Run | Change | Validation score |
|---|---|---:|
| Baseline | release CLIP B/16 | 0.869 |
| E1 | same recipe, 19 training generators | 0.879 |
| E2 | CLIP ViT-L/14, LayerNorm tuning | 0.908 |
| E3 | + resampling, noise, H.264 augmentation | 0.920 |
| E4 | + self-blended pseudo-fakes (30% of fake mass) | 0.922 |
| E5 | + GenD unit-norm features, alignment and uniformity losses | 0.959 |
| E6 | E4, validated every 8k samples | 0.942 |
| **E7** | **E5, validated every 8k samples** | **0.962** |

Every run without GenD losses peaked after the first validation interval and
declined while FF++ rose towards 1.0. GenD flattened that decline. E7 was
selected by the predefined rule (highest validation score); its margin over E5
is within seed noise.

## Validation, three seeds

| Domain | Baseline | E7 mean ± SD | Seeds with 95% CI > 0 |
|---|---:|---:|---:|
| Celeb-DF | 0.866 | 0.974 ± 0.001 | 3/3 |
| FF++ | 0.986 | 0.991 ± 0.000 | 0/3 |
| StarGAN | 0.874 | 0.984 ± 0.002 | 3/3 |
| StyleCLIP | 0.882 | 0.972 ± 0.001 | 3/3 |
| whichfaceisreal | 0.738 | 0.887 ± 0.002 | 3/3 |
| **Mean** | **0.869** | **0.962 ± 0.001** | |

## Test, three seeds (scored once)

| Domain | Baseline | E7 mean ± SD | Seeds with 95% CI > 0 | Note |
|---|---:|---:|---:|---|
| Celeb-DF | 0.900 | 0.925 ± 0.001 | (uninformative) | one identity component holds 7,051 of 7,651 frames |
| CollabDiff | 0.852 | 0.970 ± 0.001 | 3/3 | shortcut probe 0.79 |
| DeepFaceLab | 0.855 | 0.922 ± 0.005 | 2/3 | |
| e4e | 0.993 | 0.992 ± 0.001 | 0/3 | at ceiling for both |
| FF++ | 0.995 | 0.957 ± 0.001 | 0/3 (all CIs below 0) | regression |
| HeyGen | 0.938 | 0.981 ± 0.001 | 1/3 | shortcut probe 0.85 |
| MidJourney | 0.350 | 0.475 ± 0.010 | 3/3 | both at or below chance; shortcut probe 0.86 |
| StarGANv2 | 0.706 | 0.888 ± 0.002 | 3/3 | |
| **Mean** | **0.824** | **0.889 ± 0.002** | | |

Per family (mean of per-generator AUROC, seed 20261008): face edit 0.851 →
0.952, entire-face synthesis 0.852 → 0.896, face swap 0.889 → 0.892, face
reenactment 0.957 → 0.948.

### What the test shows

- **Gains on other pipelines are large and seed-stable:** StarGANv2 +0.18,
  CollabDiff +0.12, DeepFaceLab +0.07, HeyGen +0.04, Celeb-DF +0.03.
- **FF++ regresses by 0.038, every seed, CI below zero.** The losses are InSwap
  (0.977 → 0.857), MCNet (0.997 → 0.92–0.94), and PIRender (0.996 → 0.94–0.95)
  on FF++ faces. Validation did not predict this: its FF++ generators stayed at
  0.99 for both models. The baseline was selected on FF++ alone and is
  specialised to it.
- **InSwap is the weakest generator in both domains** (Celeb-DF 0.76–0.77 for
  E7, 0.68 for the baseline).
- **MidJourney is not detected.** Both models score its real images as fake more
  often than its fakes (72% and 59% of reals above 0.5). The set's reals are small
  JPEGs and its fakes polished 1024 px renders; the shortcut probe also separates
  them at 0.86. None of the 19 training generators resembles current commercial
  text-to-image output.

## Caveats

- The test split has now been used once. Further changes judged on it would make
  it a development set; a new held-out set is required for the next comparison.
- MidJourney, HeyGen, and CollabDiff stay partly separable by low-level image
  statistics after canonicalization (`shortcut_probe.json`).
- Celeb-DF bootstrap intervals are degenerate because swap videos link most
  Celeb-DF identities into one component.

## Artifacts

| Model | Run directory (`artifacts/openworld/`) | Checkpoint SHA-256 |
|---|---|---|
| E7 seed 20261007 | `clip_vit_l14-20261007-adf10611e74ccef2191d-b3d28c85-642ae400` | `96ff1c8b…` |
| E7 seed 20261008 | `clip_vit_l14-20261008-adf10611e74ccef2191d-b3d28c85-24359c93` | `2a9f6f88…` |
| E7 seed 20261009 | `clip_vit_l14-20261009-adf10611e74ccef2191d-b3d28c85-90a88076` | `33bcad33…` |

Test predictions are in `artifacts/openworld/test/<run>/test.jsonl`;
[`test_summary.json`](test_summary.json) and
[`validation_summary.json`](validation_summary.json) hold per-generator AUROC and
every bootstrap interval.

```bash
scripts/rebuild-openworld
python -m deepfake_detection.train.openworld train \
  --config configs/training/openworld/e7_l14_strongaug_sbi_gend_short.yaml --seed 20261008
python -m deepfake_detection.train.openworld score --run-dir <run> \
  --protocol-config configs/training/openworld/e7_l14_strongaug_sbi_gend_short.yaml \
  --split test --output artifacts/openworld/test/<run>
python scripts/summarize_openworld.py --baseline release=<baseline>/test.jsonl \
  --candidate s1=<run1>/test.jsonl --output reports/openworld/test_summary.json
```
