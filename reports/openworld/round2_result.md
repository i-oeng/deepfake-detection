# Open-world training round 2 result

- Round 2 manifest: `8d03d7a1cc56f0a00e8c`
- Fresh v2 manifest: `b209f9bcef05d95af5c1`
- Added, audited rows: 17,925
- Development decision: **advance E8**

## Development selection

| Recipe | Seeds | Mean domain macro video AUROC | FF++ AUROC | Stable Cascade AUROC |
|---|---:|---:|---:|---:|
| E7 | 3 | 0.858 | 0.991 | 0.340 |
| E8 | 3 | 0.886 | 0.983 | 0.518 |

E8 changed the development score by 0.028 and FF++ by -0.008. The allowed FF++ drop was 0.010.

## Fresh v2

The primary score is the unweighted mean of the three domain macro video AUROCs.

| Model | Primary AUROC | Seed scores |
|---|---:|---|
| Release CLIP B/16 | 0.617 | one checkpoint |
| E7 | 0.693 | 0.688, 0.701, 0.689 |
| E8 | 0.751 | 0.745, 0.752, 0.757 |

| Domain | Model | AUROC | Average precision | TPR at 1% FPR | Brier | ECE-10 |
|---|---|---:|---:|---:|---:|---:|
| `celebdf_video` | E7 | 0.814 | 0.778 | 0.137 | 0.201 | 0.144 |
| `celebdf_video` | E8 | 0.856 | 0.844 | 0.262 | 0.183 | 0.168 |
| `dff` | E7 | 0.928 | 0.932 | 0.282 | 0.149 | 0.207 |
| `dff` | E8 | 0.942 | 0.946 | 0.348 | 0.148 | 0.218 |
| `sfhq_t2i` | E7 | 0.336 | 0.158 | 0.000 | 0.340 | 0.353 |
| `sfhq_t2i` | E8 | 0.456 | 0.184 | 0.000 | 0.348 | 0.373 |

## Paired uncertainty and decision

E8 minus E7: **0.059** AUROC; paired 95% CI [0.054, 0.063] from 2,000 bootstrap replicates (paired videos stratified by domain, label, and fake method).

The preregistered clear-improvement gate (gain at least 0.05 and CI lower bound above zero) was **met**.

## Per-method fresh-v2 AUROC

| Domain | Method | E7 | E8 |
|---|---|---:|---:|
| `celebdf_video` | `celeb_synthesis` | 0.814 | 0.856 |
| `dff` | `insightface` | 0.945 | 0.954 |
| `dff` | `sd15_text2img` | 0.906 | 0.919 |
| `dff` | `sd_inpainting` | 0.933 | 0.952 |
| `sfhq_t2i` | `flux1_dev` | 0.314 | 0.435 |
| `sfhq_t2i` | `flux1_pro` | 0.332 | 0.445 |
| `sfhq_t2i` | `sdxl` | 0.362 | 0.489 |
