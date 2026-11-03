# Open-world training round 4 (draft)

**Status: draft for review, not frozen.** Nothing described here has been built,
trained, or scored. The protocol freezes once the open questions at the end are
settled and the fresh v4 test set is built, before any round 4 training data is
generated.

## Goal

Earlier rounds asked whether a candidate beats the previous model. Round 4 asks
whether a model is good enough to use on two kinds of media:

1. **AI-generated photos of people** from current text-to-image generators.
2. **Deepfakes of celebrities:** face swaps and identity-preserving generations of
   real, well-known people.

The bar is absolute: **balanced accuracy of at least 0.85 at a threshold locked
on validation data**, on untouched test sets for both targets. Balanced accuracy
is the mean of the share of fakes called fake and the share of reals called
real, so the test's real/fake mix cannot inflate it.

Average precision is not the gate because its floor is the fake share of the
test set. In round 3 the release model scored an AP of 0.693 on `openfake`
(68% fake) at a chance AUROC of 0.477, and E9 scored 0.906 on `openfake_reddit`
(91% fake), which random scores would also reach. AP is still reported, with
each slice's fake share beside it.

## Starting point

Round 4 starts from the E9 recipe, as the [ablation](round3_ablation_result.md)
requires. E9 on the existing sets closest to each target (macro video AUROC; a
photo counts as a one-frame video):

| Target | Set | AUROC | Note |
|---|---|---:|---|
| Celebrity deepfakes | `dff`: SD 1.5, SD inpainting, InsightFace swaps of IMDB-WIKI celebrities | 0.923 | fresh v2, second use |
| Celebrity deepfakes | `celebdf_video` | 0.857 | fresh v2, second use |
| AI photos | `openfake` | 0.674 | fresh v3 |
| AI photos | `sfhq_t2i`: FLUX.1-dev, FLUX.1-pro, SDXL | 0.659 | fresh v2, second use |

Under an equal-variance binormal approximation, 0.85 balanced accuracy needs an
AUROC of about 0.93 even at the best possible threshold. Celebrity deepfakes are
close; AI photos are not. No training generator is newer than SD 2.1,
Kandinsky 2.2, or Sana. On `openfake`, generators close to SD 2.1 already score
well (the SD 1.5 family 0.81 to 0.89, base `sdxl` 0.885), photoreal SDXL
fine-tunes are mixed (0.53 to 0.81), and FLUX models range from 0.46 to 0.67,
which points to training coverage as the main lever.

E9's validation-locked threshold already keeps false positives low on new real
sources (2% on `openfake` reals, 4% on LFW, 12% on `openfake_reddit`); the gap is
in catching fakes.

## Retired test sets

Fresh v2 was scored at the end of round 2 and again in round 3. Fresh v3 was
scored in round 3 and again in its ablation. Neither is untouched, so both are
retired as tests and become development data:

- Their domains join validation (see [Validation](#validation)).
- The generators fresh v2 reserved for testing (Celeb-DF synthesis, Stable
  Diffusion 1.5 and its inpainting model, InsightFace swapping, FLUX.1, and
  SDXL, including models sharing their autoencoders) are released for training.
- Their images stay out of training: every training addition is checked against
  them like any other evaluation image.

## Training data

All round 3 training data stays byte for byte unchanged. Additions, with raw
targets matching rounds 2 and 3 (2,000 per collection) unless stated:

### A. Celebrity deepfakes

| Collection | Fakes | Reals |
|---|---|---|
| Celeb-DF v2, training identities | swap videos whose source and target are both training identities; 8 frames per video, YuNet crops as at inference | Celeb-real and YouTube-real videos of training identities |
| DF40 fakes built on Celeb-DF, training identities | as distributed | their Celeb-DF reals |
| InsightFace swaps (`inswapper`, the swapper FaceFusion and Roop use) | 2,000 celebrity photos with a swapped face | the target photos, pixel-aligned, as with the round 2 reconstructions |
| InstantID on SDXL | 2,000 | other photos of the same celebrities |
| PuLID on SDXL | 2,000 | other photos of the same celebrities |
| OpenFake train-split images whose prompts name one of LFW's 5,728 people: 31,045 fakes naming 2,098 people, most often Donald Trump, Prince William, Prince Harry, Queen Elizabeth II, and Michael Jackson; mostly from SD 2.1, FLUX.1, SDXL fine-tunes, SD 3.5, and HiDream | up to 2,000 | the 29,015 LAION photos in OpenFake whose captions name an LFW person |

- **Identity split.** Celeb-DF swaps link most of its identities into one
  component, so it cannot be split by video. Its identities are split once: a
  held-out set supplies validation, swaps crossing the split are dropped, and
  the rest supplies training. The held-out set is chosen to keep at least 300
  identity-disjoint swap videos; the number of dropped swaps is reported.
- **Identity sources** for the generated collections: a named celebrity photo
  set (IMDB-WIKI candidates), excluding the DFF source photos and their
  identities and every identity reserved for validation or fresh v4.
- **Both labels per identity.** Every training celebrity appears as real and as
  fake, so a face alone cannot predict the label.
- **LFW undercounts.** LFW's people come from news photos of the early 2000s, so
  the name match misses many current celebrities; 31,045 is a lower bound.
- **Generated celebrity media is never published.** It stays on xixi with the
  same access as the other data and is used only to train and evaluate.

### B. AI-generated photos

Training images come from two published datasets (checked 2026-11-02), so no
training image needs to be generated and everything fits xixi's 8 GB GPU:

| Collection | Fakes | Reals |
|---|---|---|
| OpenFake `core` train split (`ComplexDataLab/OpenFake` revision `3247b942`, CC BY-NC 4.0, not gated; 2,300,034 rows from 74 fake generators) | faces only, up to 2,000 per generator family (see [OpenFake selection](#openfake-selection)) | OpenFake's own LAION and Pexels photos with faces |
| SFHQ-T2I full release (`bitmind/SyntheticFacesHighQuality-T2I` revision `c7f422bd`, `SyntheticFacesHighQuality.zip`, 22.9 GB; MIT upstream) | 1024 px faces from FLUX.1-schnell (58,034), SDXL (53,087), FLUX.1-dev (7,273), FLUX.1-pro (3,209), DALL-E 3 (1,123); up to 2,000 per model, excluding every `small-sample.zip` image, which is now `sfhq_t2i` validation | none ship; FFHQ shards no split uses |

- **Held-out families follow OpenFake's own division.** A metadata pass over all
  641 shards ([`openfake_metadata.json`](openfake_metadata.json)) finds that its
  train and test splits share no fake generator, and every validation generator
  is also in train. The test split's 20 generators never enter training; they
  supply fresh v4's unseen slice. Its reals come from other sources (ImageNet,
  DOCCI) than train's (LAION, Pexels).
- **SFHQ-T2I selection (2026-11-03).** `scripts/download_sfhq_t2i_round4.py`
  lists the 22.9 GB zip from its central directory, excludes the 1,724
  `small-sample.zip` images by name and by SHA-256, takes up to 2,000 images per
  generator by a seeded hash of the member name (seed `round4-sfhq-20261103`),
  and fetches each with one range request (10.6 GB streamed, CRC checked). The
  reals are 9 FFHQ shards of the same pinned mirror fresh v2 used
  (`gaunernst/ffhq-1024-wds` revision `d74f1f1f`), chosen by the same seed among
  the 66 no split uses: 04000, 11000, 15000, 25000, 37000, 46000, 47000, 53000
  and 67000. The face policy kept 9,063 fakes (FLUX.1-schnell 1,996, SDXL
  1,990, FLUX.1-dev 1,986, FLUX.1-pro 1,981, DALL-E 3 1,110) and all 9,000
  reals, 11.3 GB in `data/raw/sfhq_round4_det320`.
- Video-model frames and anime or illustration models are excluded; the target
  is photos.
- Each dataset's revision is pinned and its license recorded, as round 2 did.
- `sfhq_t2i` and `openfake` validation come from the same pipelines as these
  collections, so validation on them is optimistic. Fresh v4's seen slice uses
  our own generations instead.

#### OpenFake selection

The train split is 3.28 TB in 608 parquet shards, in row groups of about 69
images. The image column has no page index, so a whole row group is the
smallest download. Row groups are streamed and only the picked rows are kept.

Rows are picked by **person prompt**, a text heuristic (`PERSON` in
`scripts/openfake_metadata.py`); YuNet decides on faces when the images are
materialized. Person-prompt training rows per family:

| Family | Rows | Family | Rows |
|---|---:|---|---:|
| FLUX.1 and its fine-tunes | 179,610 | HiDream-I1 | 22,192 |
| SDXL and its fine-tunes | 174,620 | Midjourney 6 | 22,049 |
| SD 3.5 | 125,530 | Imagen 3 and 4 | 12,525 |
| SD 2.1 and SD-Turbo | 113,119 | Grok | 8,269 |
| SD 1.x and its fine-tunes | 92,120 | Qwen-Image | 6,907 |
| GPT Image 1 | 30,559 | Playground v2.5 | 6,851 |
| DALL-E 3 | 29,188 | Seedream 4 | 2,771 |
| FLUX.2 (dev, klein 4B) | 26,474 | nano-banana-2 | 1,251 |
| Ideogram 3 | 22,263 | Kolors | 1,187 |
| **Reals:** LAION | 471,934 | **Reals:** Pexels | 311,693 |

Rare families are spread thinly over every shard, so collecting them dominates
the download. Two plans, each aiming at 2,000 faces per family after detection:

| Plan | Quotas (person-prompt rows) | Row groups | Streamed |
|---|---|---:|---:|
| A | 3,000 for each family with at least 20,000 rows (the 11 from FLUX.1 to Midjourney 6); 15,000 per real source | 2,374 | 239 GB |
| B | A, plus 2,000 each for Imagen, Grok, Qwen-Image, and Playground | 3,681 | 366 GB |

Families without a quota keep only the rows the picked row groups bring along.
Under plan A that ranges from 10 rows (a small amateur-photography fine-tune) to
763 (Imagen).

**Plan B is chosen (2026-11-02).** `scripts/download_openfake_round4.py select`
fixes 77,100 rows (`selection.csv` SHA-256 `b697c8c5…`, seed
`round4-openfake-20261102`): the quotas above, the rarer families' rows that come
along, and the 3,469 fake and 3,110 real public-figure rows in the same row
groups. Duplicate images are kept once, so a few quotas end 1 to 31 rows short.

xixi cannot hold the original images (about 135 GB). `download` therefore applies
the inference face policy while streaming, detecting on a 320 px copy with the
size limit in source pixels exactly as `data/openworld.py` does, and stores each
passing image as a lossless PNG crop at source resolution: a square 2.5 times
the face box, or the original file when that square would cover 80% of the
image. The record keeps the source size, format, JPEG quality and chroma
subsampling, crop box, face box, landmarks, and score, so canonicalization and
format matching can work from the crops. Every image is checked against
OpenFake's SHA-256.

The first download (2026-11-02) detected at full resolution, where YuNet misses
faces that fill a large frame. It kept 64% of fakes but only 38% of reals, and
Pexels photos (8 to 29 MP) lost 48% to low confidence. Re-detecting a random
30 row groups at 320 px left fakes at 70% kept but raised Pexels from 29% to 53%,
and 31% of the Pexels crops it had kept were centered on a different face than
the canonical policy picks. Its crops were therefore replaced by a second
download at 320 px (2026-11-03, `data/raw/openfake_round4_det320`, `policy.json`
records the policy). It stored 43,575 faces (42.9 GB): 28,475 fakes (64% of
requested, as before) and 15,100 reals (46%, up from 38%), of which Pexels rose
from 4,565 to 7,449. Thin families stay thin: Midjourney 822 of 3,000 (stylized
images without a face), SD 2.1 1,300 of 3,591, Kolors 67, nano-banana 59 and
Seedream 155. 2,744 fake and 1,816 real public-figure rows have faces.

A face size check found no size shortcut after canonicalization. Before it,
LAION reals have small faces (median eye distance 47 px), Pexels reals large
ones (221 px) and fakes sit between (48 to 132 px by family), so eye distance
alone separates the labels at a cross-validated AUROC of 0.70. After scale
matching the face resolution separates them at only 0.54. Source format does
separate them (next list).

- Every collection is canonicalized as before: YuNet detection, minimum face size
  in source pixels, ArcFace alignment, collection-wise scale matching, JPEG 90.
  Rounds 2 and 3 showed that a fake source without matched reals teaches the
  model the source instead of the manipulation.
- **Source format matching (new).** 54% of the stored OpenFake fakes come from
  PNG or WebP files and 98.8% of the reals from JPEG files, so source format
  alone separates them at a cross-validated AUROC of 0.76 (0.82 with detector
  score and face resolution). Canonical JPEG 90 does not remove a source's
  earlier JPEG compression, and detectors trained on such data are known to learn
  "JPEG artifacts mean real". Before canonicalization, every training image whose
  source is not a JPEG is therefore JPEG-encoded once, with a quality and chroma
  subsampling pair drawn by a seeded hash of its SHA-256. JPEG sources are not
  re-encoded.
  - The JPEG sources themselves differ in quality: LAION is mostly quality 75
    and Pexels 95, while fake families are each fixed at 75 (DALL-E 3, FLUX.1,
    Grok, Midjourney), 90 or 95 (FLUX.2, Playground, Qwen, SD 1.x). Quality alone
    separates JPEG fakes from JPEG reals at 0.62.
  - In each collection, the label with more JPEG sources is the reference
    (OpenFake: the reals). A non-JPEG image of the other label draws from the
    complement: the reference distribution of (quality, subsampling) pairs,
    scaled to that label's count, minus the pairs its own JPEG sources already
    have (clipped at zero). That label as a whole then matches the reference. A
    non-JPEG image of the reference label draws from the reference distribution.
    Simulated on the OpenFake crops, the pairs then separate the labels at 0.517
    (0.542 when fakes draw from the real distribution instead).
  - SFHQ-T2I is the mirror case: every fake is a JPEG at quality 90 with 4:2:0
    subsampling and every FFHQ real a lossless WebP, so format alone separates
    that collection perfectly. Its reference is the fakes, and every FFHQ real is
    encoded at quality 90, 4:2:0.

### Leakage control

As in round 3: a whole video group is removed on an exact file, decoded pixel,
or dHash match, or when its 32 x 32 grayscale thumbnail correlates at 0.99 or
more with any validation or test image. Celebrity collections additionally pass
a face-embedding identity audit (see [Prerequisites](#prerequisites)) against
validation and fresh v4 identities.

## Validation

Round 3's validation split stays, minus any `cdf` rows whose identities move to
training. Added for development selection only:

| Domain | Source | Target |
|---|---|---|
| `cdf_heldout`, `celebdf_heldout_video` | Celeb-DF held-out identities | celebrity deepfakes |
| `dff` | retired fresh v2 | celebrity deepfakes |
| `sfhq_t2i` | retired fresh v2 | AI photos |
| `openfake` | retired fresh v3 | AI photos |
| `lfw` (real only) | retired fresh v3 | false positives on real public-figure photos |

`openfake` validation (OpenFakeTiny) keeps only generators in OpenFake's train
split. Rows from the test-only families are dropped, so model selection never
sees the families fresh v4's unseen slice is built from.

## Fresh v4 test set

Built and frozen before any round 4 training data is generated, from sources no
earlier split used. Each slice is balanced 50/50, with about 1,000 fakes and
1,000 reals, which gives a 95% interval of roughly ±0.02 on balanced accuracy
before identity grouping widens it.

| Slice | Fakes | Reals | Role |
|---|---|---|---|
| `v4_photos_seen` | generated on xixi for this test (`scripts/generate_round4.py`), with our own prompt grammar and a fresh seed namespace, from three training families that fit 8 GB with CPU offload: SDXL base 1.0, RealVisXL V5.0 and SD 3.5 Medium | faces from Unsplash Lite and Open Images, sources no split uses (see [Fresh v4 reals](#fresh-v4-reals)) | primary |
| `v4_celebrity` | swaps and identity-preserving generations, with tools used in training, of celebrities absent from training and validation | real photos of the same celebrities | primary |
| `v4_photos_unseen` | OpenFake `core` test split (68 GB, 13 shards), photo generators only: up to 300 person-prompt rows per generator (seed `round4-openfake-test-20261103`) from 16 generators, of which 1,840 have a face (Z-Image Turbo, FLUX.2 klein 9B, GPT Image 1.5 and 2, Midjourney 7, ERNIE-Image and Turbo, Recraft v2 and v3, Seedream 5, nano-banana-pro, Ideogram 2.0 and four anonymous arena models) | every ImageNet and DOCCI photo in the split was run through the face policy, and only 455 of 45,620 have a face (DOCCI 160, ImageNet 295); most show animals, objects or scenes, and the person heuristic flags captions that mention a hand or a statue | secondary: how fast the model goes stale |
| `v4_celebrity_wild` | celebrity deepfakes collected online (candidate: Deepfake-Eval-2024, if access is granted) | real media of the same people | secondary unless large enough |

- Identity-disjoint from all training and validation, checked by name and by the
  face-embedding audit.
- OpenFakeTiny (now `openfake` validation) publishes no image hashes, so its
  overlap with OpenFake's test split is unknown until the leakage check removes
  it.
- Passes the leakage control above against all training and validation data.
- A low-level shortcut probe runs per slice, as for fresh v2 and v3. A slice
  whose best single image statistic exceeds 0.70 is reported but not gated.

### Fresh v4 reals

Two downloads (2026-11-03), both through the training face policy (320 px
detection, size limit in source pixels, lossless crops), stored on xixi and never
redistributed:

- **Unsplash Lite** (dataset 1.4.0, 25,000 photos; its terms allow storing the
  photos and internal machine-learning use, not redistribution).
  `scripts/download_unsplash_round4.py` keeps the 24,638 photos submitted before
  2022-08-01, so none can come from a modern generator (seed
  `round4-unsplash-20261103`), checks a 320 px preview of each for a face that
  could pass the policy, and fetches only those at full resolution (the CDN
  re-encodes them as JPEG, about quality 89, 4:2:0). The collection is mostly
  landscapes and small figures: 510 photos have a usable face (2.1%, 2.6 GB).
- **Open Images** (validation and test images, CC BY 2.0 Flickr photos with
  human-verified boxes). `scripts/download_openimages_round4.py` keeps the 9,545
  photos whose largest "Human face" box, neither a group box nor a depiction,
  spans at least 6% of the frame on both sides, with no depicted face and no
  rotation, ordered by seed `round4-openimages-20261103`. The first 16 of its 48
  units (3,200 photos, from the 1024 px CVDF mirror) gave 2,567 faces (80%,
  0.8 GB); the rest stay unused.

Together that is 3,077 real faces, enough for `v4_photos_seen` (about 1,000) and
for topping up `v4_photos_unseen` (see open question 1). Both slices report
false-positive rates per real source.

### Fresh v4 seen-slice fakes

`scripts/generate_round4.py t2i` renders the same prompt and size for index `i`
with every generator (seed `round4-v4-photos-seen-20261103`; the noise also
depends on the generator), at each model card's settings from a pinned revision:

| Generator | OpenFake train member | Revision | Settings |
|---|---|---|---|
| SDXL base 1.0 | `stable-diffusion-xl-base-1.0` | `46216598` | Euler, 40 steps, guidance 5, no refiner |
| RealVisXL V5.0 | `sdxl-realvis-v5` | `ac93e0dd` | DPM++ 2M Karras, 50 steps, guidance 5, the card's negative prompt |
| SD 3.5 Medium | `sd-3.5` | `b940f670` | 40 steps, guidance 4.5 (gated: needs a Hugging Face token with the license accepted) |

The grammar (`round4-photo-v1`) is new, not round 2's: one adult (20s to 80s,
two genders, ten origins) in ten shot types from close-up to documentary,
twenty-five everyday scenes, ten lighting conditions and six camera
descriptions, at 1 MP in five aspect buckets from 832 x 1216 to 1216 x 832. The
longest prompt is 52 CLIP tokens. About 400 images per generator are planned,
so about 1,000 faces remain after the face policy. Generated images are
PNG; the format matching of the training data applies here too, with the reals as
the reference, so the slice cannot be solved by source format.

## Development selection

| Candidate | Change from E9 | Budget |
|---|---|---|
| E10 | round 4 training data | 6 x 8,000 samples, as E9 |
| E10-long | round 4 training data | 6 x 32,000 samples |

Round v1 found no gain from longer schedules, but on far less varied data, so
the longer budget is tested once here. Three seeds each (20261007, 20261008,
and 20261009); seed logits are averaged.

The candidate with the higher mean macro video AUROC over the target validation
domains (`cdf_heldout`, `celebdf_heldout_video`, `dff`, `sfhq_t2i`, `openfake`)
advances, provided that, against E9 on the same split:

- its mean over those domains is higher;
- FF++ validation AUROC falls by no more than 0.01; and
- the real-source bias on round 3's validation real sources rises by no more
  than 0.03.

If neither candidate qualifies, fresh v4 stays unscored.

Reported for every candidate, not gated: the **source-format gap**, the AUROC of
fakes from PNG sources minus that of fakes from JPEG sources, each against the
same reals, on `openfake` validation. A gap above 0.05 means the model reads the
source format rather than the generator, and is investigated before fresh v4 is
scored.

## Threshold

Locked on validation before fresh v4 is scored: the threshold that maximizes
mean balanced accuracy over the target validation domains (each weighted
equally), subject to a false-positive rate of at most 10% on validation reals
with real sources weighted equally. The same threshold goes into the release
bundle if round 4 succeeds.

## Final evaluation

The selected candidate, E9, and the release CLIP B/16 are scored on fresh v4
once.

- **Primary:** balanced accuracy at the locked threshold on `v4_photos_seen` and
  `v4_celebrity`, with a bootstrap of 2,000 replicates resampled within each
  label (grouped by identity for `v4_celebrity`).
- **Round 4 succeeds** when both primary slices reach a balanced accuracy of at
  least 0.85 with a 95% lower bound of at least 0.80, and no real source in them
  exceeds a 10% false-positive rate.
- **Also reported:** the secondary slices; AUROC on every slice with a paired
  bootstrap against E9; AP with the fake share; per-generator and per-tool
  results; false-positive rates per real source; calibration; the source-format
  gap on `v4_photos_unseen`.

## Prerequisites

Before the protocol freezes:

1. Move round 3's leakage check (exact, pixel, dHash, thumbnail correlation) from
   `data/round3.py` into a shared module.
2. Add a face-embedding identity audit (plan section 11, priority 1, item 2). Name
   matching alone misses the same person under different names or unnamed
   identities.
3. Report balanced accuracy at a locked threshold in the open-world comparison;
   it currently reports AUROC and false-positive rates.
4. Extend `deepfake-detect bundle` to open-world runs and seed-averaged logits,
   so a successful model can ship.
5. Write the round 4 generation script for the celebrity swap tools and the fresh
   v4 seen-slice generators: pinned revisions, CPU offload, and resumable chunks
   so one job does not hold the shared queue for days. The seen-slice part is
   written (`scripts/generate_round4.py t2i`, smoke-tested on xixi with tiny
   pipelines); the celebrity tools wait for the identity source.
6. Published data: done. The OpenFake metadata pass
   (`scripts/openfake_metadata.py`, [`openfake_metadata.json`](openfake_metadata.json)),
   the plan B training and `v4_photos_unseen` selections and downloads
   (`scripts/download_openfake_round4.py`), the SFHQ-T2I and FFHQ download
   (`scripts/download_sfhq_t2i_round4.py`) and fresh v4's reals
   (`scripts/download_unsplash_round4.py`, `scripts/download_openimages_round4.py`)
   are written and have run on xixi (`data/raw/openfake_round4_det320`,
   `openfake_round4_test_det320`, `sfhq_round4_det320`,
   `unsplash_round4_det320`, `openimages_round4_det320`).
7. Implement source format matching in the round 4 manifest build, from the
   recorded `jpeg_quality` and `subsampling`, and the source-format gap in the
   open-world comparison.
8. Run the torch tests in CI, since most of the above is torch code.

## Compute

Everything runs on xixi through the shared one-slot queue. Training time is set
by the sample budget, not the dataset size: about 28 minutes per E9-length run,
so E10's three seeds take about 1.5 hours and E10-long's about 6, plus longer
validation passes over the added domains. Offloaded generation is the slow part;
it runs in resumable chunks so aida's jobs can interleave. A rented GPU becomes
worth it only if a generator that matters is missing from published data.

Downloads, all to xixi's data directory and all done on 2026-11-03, store face
crops only: 10.6 GB of SFHQ-T2I members and 8.8 GB of FFHQ shards (11.3 GB
kept), OpenFake's whole test split (68 GB streamed, 1.6 GB kept), and plan B's
366 GB of OpenFake train row groups. Detection at full resolution held the first
download to about 20 MB/s (about 6 hours, 16 GB of crops); at 320 px the second
streamed at about 90 MB/s (about 1.1 hours) and stored 42.9 GB, 30 GB of it
Pexels crops, because the large faces of 8 to 29 MP photos are now kept at
source resolution. xixi had 7 GB free; removing round 2's generation models, a
duplicate round 3 download cache, and an unrelated project's models freed 76 GB.
Deleting the first download's crops, that project's remaining files, a duplicate
copy of fresh v2's FFHQ shards and the pip cache then left 29 GB free after all
downloads, and 25 GB after fresh v4's reals.

Generation needs the whole GPU: offloaded SDXL peaks near 7 GB. The three
models take about 7, 7 and 16 GB of disk, so they are downloaded one at a time
and deleted once their images are done. A run checks that 7 GB of GPU memory are
free before loading a model and otherwise exits, because a process outside the
queue (on 2026-11-03 an Ollama model of 6.2 GB with a 30 minute keep-alive) can
hold the GPU.

## Out of scope for round 4

- Photos without a detectable face of at least 64 source pixels; a whole-image
  model is a later round.
- Lip-synced and talking-head video; a temporal model is a later round.
  `v4_celebrity_wild` reports these separately if it contains them.
- A larger backbone.

## Open questions for review

1. **Test sources:** can we get Deepfake-Eval-2024 (or another collected set of
   real-world celebrity deepfakes)? Which source supplies the real photos for
   `v4_photos_seen`? Proposed: Unsplash Lite and Open Images (see
   [Fresh v4 reals](#fresh-v4-reals)). OpenFake's test split has only 455 real
   faces, so `v4_photos_unseen` is either balanced at about 455 per label (a 95%
   interval of roughly ±0.03 on balanced accuracy, against ±0.02 at 1,000) or its
   reals are topped up with about 545 faces from the same pool as
   `v4_photos_seen`, which then differs from its own ImageNet and DOCCI reals.
2. **Celeb-DF identity cut:** how many swap videos the split drops, and whether
   that is acceptable.
3. **Success bar:** is 0.85 with a lower bound of 0.80 the right reading of
   "usable", and is a 10% false-positive cap on real celebrity photos tight
   enough?
4. **Scope:** are faceless AI images needed soon? If so, the whole-image model
   moves into this round.
5. **Held-out families:** adopt OpenFake's train/test division as it stands, or
   hold out more? FLUX.2 klein 4B trains while klein 9B is test-only, so that
   pair measures little.
6. **Queue time:** generation and training share xixi's single slot with aida's
   work; agree on when the long jobs run.
