# Open-world round 3 ablation protocol

Frozen on 2026-10-28, before any ablation model is trained.

E9 combined two changes and met both round 3 outcomes
([result](round3_result.md)), so the result cannot say which change did what.
This ablation trains each change alone, attributes the effects, and fixes the
recipe round 4 starts from.

## Arms

| Arm | Manifest | Real-source balancing | Padding augmentation | Config |
|---|---|---|---|---|
| E8 (existing runs) | round 2 `8d03d7a1cc56f0a00e8c` | none | none | `e8_l14_round2.yaml` |
| E9-data (new) | round 3 `e4dd0240af9df4c60b0c` | lineage namespace | none | `e9_data_l14_round3.yaml` |
| E9-padding (new) | round 2 `8d03d7a1cc56f0a00e8c` | none | 0.5 | `e9_padding_l14_round2.yaml` |
| E9 (existing runs) | round 3 `e4dd0240af9df4c60b0c` | lineage namespace | 0.5 | `e9_l14_round3.yaml` |

The data change is the round 3 manifest (FairFace reals paired with the
text-to-image fakes, FairFace reconstructions) together with real-source
balancing, which sets how much training mass FairFace gets. Everything else is the
E8 recipe: CLIP ViT-L/14 with LayerNorm tuning, strong augmentation, 30%
self-blends, GenD losses, 6 x 8,000 samples, seeds 20261007-9. E8 and E9 are
reused, not retrained. The training code is unchanged since E9 (`884b6a1`).

## Metrics

Validation metrics average the three seeds; fresh v3 metrics average seed logits.

1. Validation real-source bias: the round 3 selection measure (FF++ reals against
   `cdf`, `stargan`, `styleclip`, `whichfaceisreal`, and `stable_cascade_lite`
   reals; 0.5 is none).
2. Validation mean domain macro video AUROC.
3. Validation FF++ AUROC.
4. Validation Stable Cascade AUROC (FFHQ reals that are often padded).
5. Fresh v3 `openfake` AUROC.
6. Fresh v3 real-source bias (`openfake` and `lfw` reals against FF++
   validation reals).

Fresh v3 was scored once in round 3; this is its second use. Its numbers
attribute effects but do not test them, and paired bootstrap intervals against E8
are reported only for scale. Round 4 needs a new untouched test set.

## Attribution

For every metric, the change from E8 of E9-data, E9-padding, and E9, and the
interaction: E9's change minus both single changes.

Expected, stated before training: most of the bias drop comes from the data
change, and most of the Stable Cascade gain from the padding augmentation.

## Round 4 recipe

Round 4 keeps both changes unless exactly one arm alone is no worse than E9 by
more than 0.01 on every metric (bias counts as worse when higher). Then round 4
keeps only that change. Nothing else follows from the ablation.

## Budget and commands

Six new runs of about 28 minutes plus scoring: about 3 hours of the whole GPU.
Arms are trained seed by seed, so an interrupted run leaves paired arms, and the
script resumes where it stopped.

```bash
scripts/gpu-tsp -L round3-ablation bash -lc 'umask 0007; cd ~/dd-open; exec scripts/run-round3-ablation'
```

Outputs: [`round3_ablation.json`](round3_ablation.json) and
[`round3_ablation_result.md`](round3_ablation_result.md).
