# Open-world training round 3

This round is frozen before any round 3 model is trained or fresh v3 is scored.
It removes two training shortcuts that the round 2 models use, and keeps the E8
recipe, budget, and seeds otherwise:

1. **Real source.** Every training real was an FF++ video frame, and E8 separates
   FF++ reals from real photos of every other source at a video AUROC of
   0.93-0.98 ([real-source bias](real_source_bias.md)).
2. **Alignment padding.** The canonical crop fills whatever lies outside the
   source photo with black (see below).

Both changes go into one candidate, E9, so a single GPU session can train and
test it. The result therefore cannot say which change helped; an ablation that
trains each change alone is the next step if E9 succeeds.

## Training data

| Collection | Fakes | Reals | Change from round 2 |
|---|---|---|---|
| v1 DF40 training (19 generators) | unchanged | FF++ frames | none |
| MoVQ and DC-AE reconstructions of FF++ | 2,000 each | their FF++ sources | none |
| Kandinsky 2.2 and Sana 600M faces | the round 2 generations, 2,000 each | **FairFace photos**, 2,000 each | reals were FF++ frames |
| MoVQ and DC-AE reconstructions of FairFace | **2,000 each, new** | their FairFace sources | new |

- FairFace: `HuggingFaceM4/FairFace` revision `54d573cd`, 1.25 padding, first
  training shard (CC BY 4.0; YFCC-100M Flickr photos). Only the adult age bins
  (20 and over) are used, matching the adult-only generation prompts; 16,257 are
  eligible and four disjoint hash-ranked pools of 2,000 are drawn with seed
  `round3-20261027`. FairFace occurs in no earlier split.
- Reconstructions run the same autoencoders as round 2 at the photos' 448 px
  source resolution, in float32 (on CPU, with any remainder on GPU).
- Each collection is canonicalized as before: YuNet detection, minimum face size
  in source pixels, ArcFace alignment, collection-wise scale matching, JPEG 90.
- **Sampling:** `real_balance: lineage_namespace` gives FF++ and FairFace reals
  equal sampling mass, and likewise for the self-blended pseudo-fakes made from
  them. Otherwise each single-photo FairFace group would weigh as much as a whole
  FF++ clip, and FairFace would take most of the real mass.
- **Alignment-padding augmentation:** half of the training images, real and fake
  alike, get a black margin along one or two edges (4-30% deep, slightly tilted),
  applied before the other augmentations (`padding_augment: 0.5`).
- **Recipe (E9):** otherwise E8 unchanged: CLIP ViT-L/14 with LayerNorm tuning,
  strong augmentation, 30% self-blends, GenD losses, 6 x 8,000 samples, seeds
  20261007-9.

## Alignment padding

A crop counts as padded when more than 10% of its outer 6-pixel band is black
(all channels at most 4). Close-up generated faces run out of frame more often
than training reals: 40% of the Kandinsky crops are padded, against 20% of their
FairFace reals (round 2: 21% of FF++ reals); Sana is balanced (22% against
21%). Tightly cropped FFHQ reals show the opposite: 66% of the reals in
`stable_cascade_lite` and `sfhq_t2i` are padded, against 9% and 19% of their
fakes. E8 learned the cue. Within FFHQ reals it scores those with a black
margin at a median of 0.65-0.66 and those without at 0.47-0.49, and on
`sfhq_t2i` images without a margin it already ranks fakes above reals (median
0.53 against 0.47). Fresh v3's `openfake` domain is balanced on margins (AUROC of
the margin share for the label 0.498), so it measures detection without this
cue.

## Leakage control

Every addition, including the round 2 rows that stay, is compared with every
image any model is scored on: round 3 validation and v1 test rows, fresh v2, and
fresh v3. A whole video group is removed on an exact file, pixel, or dHash match,
or when its 32 x 32 grayscale thumbnail correlates at 0.99 or more with an
evaluation image. FFHQ (validation `whichfaceisreal`, fresh v2 `sfhq_t2i`) and
FairFace are both Flickr photos, so a shared photo is possible.

The threshold was calibrated on unrelated images. A 64-bit dHash cannot do this
among aligned faces: for a quarter of round 2 training images, some unrelated
evaluation face lies within 6 bits. With thumbnails, 5,000 round 2 training
images against 62,278 evaluation images reached at most 0.988 (99.9th percentile
0.966), while resized JPEG 70 copies of evaluation images scored at least 0.9988.
The final manifest must pass the round 3 audit (no exact, near-duplicate, or
lineage overlap; method roles as in round 2) before E9 trains.

## Fresh v3 test set

Manifest `f8fb7343da06e7a81dd7` (3,823 images), built by
`scripts/rebuild-fresh-v3` from sources no earlier split used, with no image
matching round 2 content:

| Domain | Fakes | Reals | Best single image statistic |
|---|---:|---:|---:|
| `openfake`: OpenFakeTiny, all three published splits | 946 | 454 | 0.56 |
| `openfake_reddit`: OpenFakeTiny Reddit split | 383 | 40 | 0.72 |
| `lfw`: LFW news photos, one per person | 0 | 2,000 | (real only) |

- OpenFakeTiny (`ComplexDataLab/OpenFakeTiny` revision `9a7f5a7b`): real LAION,
  Pexels, ImageNet, and DOCCI photos against FLUX.1 and FLUX.2, SDXL and its
  fine-tunes, SD 1.5, SD 3.5, Midjourney 6 and 7, GPT Image, Imagen, Ideogram,
  Qwen-Image, Z-Image, Seedream, video models, and others. SD 2.1 and SD-Turbo
  (DF40 trains on SD 2.1) and a random-weight Sana are excluded as training
  families. Some generators share autoencoders with fresh v2's reserved models;
  none shares one with a training generator.
- Faces must be detected and at least 64 source pixels across, which removes
  most landscape and object images: 47% of OpenFake fakes and 76% of its reals.
  The uneven exclusion mirrors fresh v2's: generated images more often show one
  large face.
- `openfake_reddit` has only 40 real videos and a 0.72 low-level shortcut, so it
  is reported but not gated.

## Development selection

E9's three seeds are compared with E8's three on the unchanged validation split.
E9 advances to fresh v3 only when, averaged over seeds:

- the validation real-source bias (mean AUROC between FF++ reals and the other
  validation real sources: `cdf`, `stargan`, `styleclip`, `whichfaceisreal`,
  `stable_cascade_lite`) falls by at least 0.03;
- the mean domain macro video AUROC falls by no more than 0.01; and
- FF++ validation AUROC falls by no more than 0.01.

If E9 does not advance, fresh v3 stays unscored.

## Final evaluation

After selection, E9, E8, E7 (three seeds each, logits averaged) and the release
CLIP B/16 are scored on fresh v3 once. The two primary outcomes compare E9 with E8:

1. **Detection:** `openfake` macro video AUROC, with a paired bootstrap (2,000
   replicates, resampled within each label).
2. **Real-source bias:** the mean over `openfake` and `lfw` reals of the AUROC
   against FF++ validation reals, with a paired bootstrap over real videos of all
   three sources.

Round 3 succeeds when the bias falls by at least 0.05 with a 95% interval above
zero **and** the detection difference has a 95% lower bound above -0.02. Detection
counts as improved only if its lower bound is above zero. Also reported:
`openfake_reddit`, per-generator AUROC, false-positive rates on each real source at
each model's validation-locked threshold, and, for continuity only, fresh v2,
which is no longer untouched.

## Commands

```bash
python scripts/download_round3_sources.py --output data/raw/round3_sources \
  --cache data/raw/round3_cache          # with PYTHONPATH=.genlibs:src
python -m deepfake_detection.data.round3 select-fairface \
  --metadata data/raw/round3_sources/fairface/metadata.jsonl \
  --output data/subsets/df40/df40_openworld_round3/fairface_selection.json
scripts/rebuild-fresh-v3
scripts/gpu-tsp -L round3 bash -lc 'umask 0007; cd ~/dd-open; exec scripts/run-round3-experiment'
```

## Amendment after freezing (2026-10-27)

The first run stopped at the manifest audit, before any model was trained or
scored. Pruning compared round 2's `stable_cascade_lite` validation additions
with the preliminary build's own validation rows, which include those same
images, and removed all 2,000. Commit `884b6a1` checks validation additions
against the test-only references alone, as round 2 did; training additions are
checked as frozen. The rebuilt manifest `e4dd0240af9df4c60b0c` lost no rows and
passed the audit. The result is in [round3_result.md](round3_result.md).
