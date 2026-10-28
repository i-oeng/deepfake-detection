# Real-source bias of the round 2 models

Post-hoc diagnosis after the round 2 result, made with
`deepfake_detection.train.source_bias` on predictions that already existed. No
model was retrained or rescored for it.

## Question

Fresh v2 put FLUX.1 and SDXL faces below chance for every model, yet the other
domains looked strong. Domain AUROC compares each domain's fakes only with that
domain's own reals, so it cannot show whether a detector simply calls every image
that does not look like its training reals fake. Every training real in v1 and
round 2 is an FF++ video frame.

## Measure

For each real source, the video-level AUROC of the detector score with FF++
validation reals as negatives and that source's reals as positives. A score that
carries no information about where a real image came from gives 0.5; a score that
identifies the source gives 1.0. FPR is the share of a source's real videos called
fake at a threshold locked on validation reals (10% false positives, every real
source weighted equally).

## Result

Means over the three seeds of E7 and E8 (release: one checkpoint, v1 validation).

| Model | Validation sources | Fresh v2 sources |
|---|---:|---:|
| Release CLIP B/16 | 0.995 | 0.990 |
| E7 | 0.946 | 0.917 |
| E8 | 0.965 | 0.944 |

### Validation real sources

| Model | `cdf` | `stable_cascade_lite` | `stargan` | `styleclip` | `whichfaceisreal` |
|---|---:|---:|---:|---:|---:|
| Release | 0.992 (1%) |  | 0.998 (21%) | 0.996 (19%) | 0.992 (8%) |
| E7 | 0.927 (1%) | 0.934 (16%) | 0.972 (14%) | 0.958 (12%) | 0.939 (16%) |
| E8 | 0.928 (1%) | 0.972 (17%) | 0.977 (12%) | 0.974 (12%) | 0.973 (18%) |

### Fresh v2 real sources

| Model | `celebdf_video` | `dff` (IMDB-WIKI) | `sfhq_t2i` (FFHQ) |
|---|---:|---:|---:|
| Release | 0.992 (3%) | 0.986 (3%) | 0.991 (7%) |
| E7 | 0.931 (1%) | 0.890 (1%) | 0.931 (16%) |
| E8 | 0.915 (0%) | 0.944 (1%) | 0.972 (18%) |

AUROC against FF++ validation reals; FPR at the locked threshold in parentheses.

## Interpretation

- Every model separates FF++ reals from other real photos about as well as it
  separates fakes from reals. The score is largely "FF++ frame or not".
- Round 2 made this worse (E7 0.946 to E8 0.965 on validation). Its text-to-image
  fakes were paired with FF++ frames, so the cheapest cue was again "not an FF++
  frame".
- E8's fresh v2 gain on `sfhq_t2i` (0.336 to 0.456) did not come from detecting
  FLUX.1 or SDXL. E7 scored those fakes below FFHQ reals (median 0.43 against
  0.55); E8 raised them to the level of FFHQ reals (0.52-0.55 against 0.56). Stable
  Cascade validation AUROC stayed at 0.49-0.51 throughout E8 training.
- Classic overfitting is absent: E8's training loss ends at 0.54 under heavy
  augmentation, validation is flat, the best checkpoint is the last, and seeds
  agree within 0.006.

Round 3 ([protocol](round3_protocol.md)) changes the real side of training and
adds this measure to model selection and to the final test. Its model E9 lowered
the bias from 0.965 to 0.663 on validation and from 0.945 to 0.498 on fresh v3
([result](round3_result.md), [tables](real_source_bias_round3.md)).

Data: [`real_source_bias_round2.json`](real_source_bias_round2.json).
