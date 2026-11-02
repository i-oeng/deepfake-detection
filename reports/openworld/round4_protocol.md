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
the inference face policy while streaming and stores each passing image as a
lossless PNG crop at source resolution: a square 2.5 times the face box, or the
original file when that square would cover 80% of the image. The record keeps
the source size, crop box, face box, landmarks, and score, so canonicalization
can work from the crops. Every image is checked against OpenFake's SHA-256. In
the first row groups, about 63% of fakes and 46% of reals passed the face policy,
at about 420 KB per stored image; a 3,000-row quota then gives about 1,900 faces
and a 2,000-row quota about 1,260.
- Every collection is canonicalized as before: YuNet detection, minimum face size
  in source pixels, ArcFace alignment, collection-wise scale matching, JPEG 90.
  Rounds 2 and 3 showed that a fake source without matched reals teaches the
  model the source instead of the manipulation.

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
| `v4_photos_seen` | generated on xixi for this test, with our own prompt grammar and a fresh seed namespace, from training families that fit 8 GB with CPU offload (candidates: SDXL base, a photoreal SDXL fine-tune, SD 3.5 Medium) | real photos from a source no split uses | primary |
| `v4_celebrity` | swaps and identity-preserving generations, with tools used in training, of celebrities absent from training and validation | real photos of the same celebrities | primary |
| `v4_photos_unseen` | OpenFake `core` test split (68 GB, 13 shards), photo generators only; person-prompt rows: Z-Image Turbo 10,088, FLUX.2 klein 9B 7,641, GPT Image 1.5 3,467, Midjourney 7 2,387, ERNIE-Image 703, Recraft 529, Seedream 5 302, nano-banana-pro 241, GPT Image 2 205, and smaller ones | its ImageNet (8,035) and DOCCI (2,918) person-prompt photos with faces | secondary: how fast the model goes stale |
| `v4_celebrity_wild` | celebrity deepfakes collected online (candidate: Deepfake-Eval-2024, if access is granted) | real media of the same people | secondary unless large enough |

- Identity-disjoint from all training and validation, checked by name and by the
  face-embedding audit.
- OpenFakeTiny (now `openfake` validation) publishes no image hashes, so its
  overlap with OpenFake's test split is unknown until the leakage check removes
  it.
- Passes the leakage control above against all training and validation data.
- A low-level shortcut probe runs per slice, as for fresh v2 and v3. A slice
  whose best single image statistic exceeds 0.70 is reported but not gated.

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
  results; false-positive rates per real source; calibration.

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
   so one job does not hold the shared queue for days.
6. OpenFake selection: done. The metadata pass (`scripts/openfake_metadata.py`,
   [`openfake_metadata.json`](openfake_metadata.json)) and the plan B selection
   and download (`scripts/download_openfake_round4.py`) are written; the download
   is running on xixi.
7. Run the torch tests in CI, since most of the above is torch code.

## Compute

Everything runs on xixi through the shared one-slot queue. Training time is set
by the sample budget, not the dataset size: about 28 minutes per E9-length run,
so E10's three seeds take about 1.5 hours and E10-long's about 6, plus longer
validation passes over the added domains. Offloaded generation is the slow part;
it runs in resumable chunks so aida's jobs can interleave. A rented GPU becomes
worth it only if a generator that matters is missing from published data.

Downloads, all to xixi's data directory: the SFHQ-T2I archive (22.9 GB),
OpenFake's test split (68 GB), and plan B's 366 GB of OpenFake train row groups,
streamed at about 17 MB/s (about 6 hours) with only face crops stored (about 15
to 30 GB). xixi had 7 GB free; removing round 2's generation models, a duplicate
round 3 download cache, and an unrelated project's models freed 76 GB.

## Out of scope for round 4

- Photos without a detectable face of at least 64 source pixels; a whole-image
  model is a later round.
- Lip-synced and talking-head video; a temporal model is a later round.
  `v4_celebrity_wild` reports these separately if it contains them.
- A larger backbone.

## Open questions for review

1. **Test sources:** can we get Deepfake-Eval-2024 (or another collected set of
   real-world celebrity deepfakes)? Which source supplies the real photos for
   `v4_photos_seen`?
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
