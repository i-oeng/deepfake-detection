# Open-world training round 3 result

- Round 3 manifest: `e4dd0240af9df4c60b0c`; audit passed
- Training real namespaces (images): `fairface` 7,976, `ff` 24,370
- Development decision: **advance E9**

## Development selection

| Recipe | Seeds | Mean domain AUROC | FF++ AUROC | Stable Cascade AUROC | Real-source bias |
|---|---:|---:|---:|---:|---:|
| E8 | 3 | 0.886 | 0.983 | 0.518 | 0.965 |
| E9 | 3 | 0.911 | 0.973 | 0.706 | 0.663 |

Real-source bias is the mean video AUROC between FF++ validation reals and each other validation real source (0.5 = the score carries no source information).

| Validation real source | E8 | E9 |
|---|---:|---:|
| `cdf` | 0.928 | 0.895 |
| `stable_cascade_lite` | 0.972 | 0.469 |
| `stargan` | 0.977 | 0.751 |
| `styleclip` | 0.974 | 0.725 |
| `whichfaceisreal` | 0.973 | 0.476 |

E9 changed the bias by -0.302 (required: at most -0.03), the mean domain AUROC by +0.025, and FF++ AUROC by -0.009 (each allowed to fall by at most 0.01). Decision: **advance E9**.

## Fresh v3

Seed logits are averaged per model. Both primary outcomes and the decision rule were fixed in the [protocol](round3_protocol.md) before scoring.

| Primary outcome | E8 | E9 | Change | 95% CI |
|---|---:|---:|---:|---:|
| Detection: `openfake` AUROC | 0.596 | 0.674 | +0.078 | [0.056, 0.098] |
| Real-source bias (`openfake`, `lfw` reals vs FF++) | 0.945 | 0.498 | -0.447 | [-0.489, -0.405] |

- Bias reduced by at least 0.05 with the interval below zero: **yes**.
- Detection non-inferior (lower bound above -0.02): **yes**; improved (lower bound above zero): **yes**.
- Round 3 succeeds: **yes**.

### All models

FPR is the share of real videos called fake at each model's threshold locked on validation reals (10% false positives, real sources weighted equally).

| Model | `openfake` AUROC | `openfake_reddit` AUROC | Bias | `openfake` real FPR | `openfake_reddit` real FPR | `lfw` FPR |
|---|---:|---:|---:|---:|---:|---:|
| Release CLIP B/16 | 0.477 | 0.353 | 0.987 | 2% | 5% | 12% |
| E7 | 0.616 | 0.563 | 0.913 | 2% | 12% | 2% |
| E8 | 0.596 | 0.329 | 0.945 | 2% | 10% | 2% |
| E9 | 0.674 | 0.617 | 0.498 | 2% | 12% | 4% |

`openfake_reddit` (40 real videos, low-level shortcut 0.72, not gated): E9 minus E8 +0.287, 95% CI [0.214, 0.364].

### `openfake` generators with at least 10 videos

| Generator | Videos | E8 | E9 |
|---|---:|---:|---:|
| `dalle-3` | 12 | 0.763 | 0.743 |
| `flux-1.1-pro` | 13 | 0.482 | 0.637 |
| `flux.1-dev` | 41 | 0.415 | 0.472 |
| `flux.1-schnell` | 21 | 0.423 | 0.460 |
| `flux.2-dev` | 19 | 0.411 | 0.527 |
| `flux.2-klein-4b` | 12 | 0.614 | 0.596 |
| `flux.2-klein-9b` | 75 | 0.508 | 0.666 |
| `flux.2-klein-base-4b` | 11 | 0.474 | 0.581 |
| `gpt-image-1` | 24 | 0.615 | 0.688 |
| `gpt-image-1.5` | 21 | 0.597 | 0.711 |
| `hidream-i1-full` | 23 | 0.468 | 0.567 |
| `ideogram-3.0` | 22 | 0.537 | 0.632 |
| `illustrious` | 42 | 0.543 | 0.720 |
| `imagen-3.0-002` | 10 | 0.505 | 0.572 |
| `midjourney-7` | 24 | 0.554 | 0.681 |
| `mystic` | 12 | 0.669 | 0.669 |
| `nano-banana` | 12 | 0.583 | 0.682 |
| `one-obsession-17-red-sdxl` | 11 | 0.549 | 0.709 |
| `playground-v2.5-1024px-aesthetic` | 10 | 0.682 | 0.747 |
| `sd-1.5` | 11 | 0.830 | 0.811 |
| `sd-1.5-dreamshaper` | 11 | 0.795 | 0.883 |
| `sd-1.5-epicdream` | 13 | 0.826 | 0.835 |
| `sd-3.5` | 47 | 0.632 | 0.674 |
| `sdxl` | 21 | 0.848 | 0.885 |
| `sdxl-epic-realism` | 24 | 0.572 | 0.527 |
| `sdxl-juggernaut` | 14 | 0.452 | 0.615 |
| `sdxl-realvis-v5` | 17 | 0.754 | 0.805 |
| `sdxl-touchofrealism` | 13 | 0.702 | 0.626 |
| `stable-diffusion-v1-5` | 10 | 0.891 | 0.885 |
| `stable-diffusion-xl-base-1.0` | 10 | 0.756 | 0.709 |
| `veo-3` | 13 | 0.529 | 0.708 |
| `z-image-turbo` | 102 | 0.564 | 0.643 |

## Fresh v2 (secondary, second use)

Fresh v2 was scored at the end of round 2 and then inspected, so it is no longer untouched. It is reported for continuity only.

| Domain | E8 | E9 |
|---|---:|---:|
| `celebdf_video` | 0.856 | 0.857 |
| `dff` | 0.942 | 0.923 |
| `sfhq_t2i` | 0.456 | 0.659 |
| **Mean** | **0.751** | **0.813** |
