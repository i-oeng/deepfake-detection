# Fresh open-world test set v2

The v1 test split was scored once (see
[`openworld_v1_result.md`](openworld_v1_result.md)) and is spent. This test-only
set measures the next round of changes. Training and model selection stay on v1
train and validation; each finalist, the v1 E7 recipe, and the release baseline
are scored on v2 once, together, at the end of the round.

- Manifest: `b209f9bcef05d95af5c1` (27,970 images), audit passed.
- Evaluation config: [`configs/evaluation/openworld_fresh_v2.yaml`](../../configs/evaluation/openworld_fresh_v2.yaml).
- Build: `scripts/download_fresh_sources.py`, then `scripts/rebuild-fresh-v2`.

## Domains

| Domain | Fakes (images or videos) | Reals | Preprocessing |
|---|---|---|---|
| `celebdf_video` | Celeb-DF v2 synthesis: 599 videos | Celeb-real and YouTube-real: 732 videos | 16 frames per video, YuNet crops as at inference |
| `dff` | SD 1.5 text-to-image 927, SD inpainting 800, InsightFace 743 | IMDB-WIKI 708 | crop, scale match, JPEG 90 |
| `sfhq_t2i` | FLUX.1-dev 853, FLUX.1-pro 314, SDXL 548 | FFHQ 2,000 | crop, scale match, JPEG 90 |

The score is the mean over the three domains of each domain's macro video AUROC.

## Separation from v1

- None of these generators is in any v1 split, and no v1 real source is reused
  except Celeb-real videos of Celeb-DF identities.
- Celeb-DF videos showing any identity that v1 used for validation are
  excluded. Celeb-real videos of v1 test identities remain; v1 used eight
  DF40-cropped frames of some of them, and the 59 frames whose perceptual hash
  matches a v1 image are dropped.
- whichfaceisreal (v1 validation) uses FFHQ images 00000-01001, so the FFHQ reals
  come from shards 35000 and 50000. No FFHQ image matches a v1 image.
- DeepFakeFace reals and fakes are the same 1,000 IMDB-WIKI source photos before
  face-policy exclusions.

## Known limits

- **Uneven face exclusion in `dff`.** The 64 px minimum face size (applied in
  source pixels) keeps 71% of reals, 74% of InsightFace, 80% of inpainting, and
  93% of SD 1.5 fakes. Text-to-image fakes are new pictures with larger faces;
  the edited methods track the reals.
- **Low-level separability.** The best single image statistic separates real
  from fake at 0.58 (`celebdf_video`), 0.68 (`dff`), and 0.67 (`sfhq_t2i`)
  ([`fresh_v2_shortcut_probe.json`](fresh_v2_shortcut_probe.json)), below v1's
  MidJourney, HeyGen, and CollabDiff (0.79-0.86).
- **Celeb-DF grouping.** Swap videos link most Celeb-DF identities into one
  component, so lineage-grouped bootstrap intervals for `celebdf_video` are not
  informative.
- **Generators reserved for testing.** Training additions in the next round
  must avoid Celeb-DF synthesis, Stable Diffusion 1.5 and its inpainting model,
  InsightFace swapping, FLUX.1, and SDXL, including models that share the SDXL
  or FLUX autoencoder.

## Sources

| Source | Revision |
|---|---|
| `OpenRL/DeepFakeFace` (OpenRAIL) | `eb2a54fbf2567790bd3dc61d09fdaade6bb6c31c` |
| `bitmind/SyntheticFacesHighQuality-T2I`, `small-sample.zip` | `c7f422bdcaf4ffe1ced6da49198845c874221a03` |
| `gaunernst/ffhq-1024-wds`, shards 35000 and 50000 (CC BY-NC-SA 4.0) | `d74f1f1f59e3bbe975bee29872b9bef827314577` |
| Celeb-DF v2 (local archive) | SHA-256 `b22f9d74…` |
