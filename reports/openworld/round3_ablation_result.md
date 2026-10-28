# Open-world round 3 ablation result

The [protocol](round3_ablation_protocol.md) was fixed before training. Seeds per arm: 3. Fresh v3 is used a second time, so its rows attribute effects but do not test them.

## Metrics

| Metric | E8 | E9-data | E9-padding | E9 |
|---|---:|---:|---:|---:|
| Validation real-source bias | 0.965 | 0.709 | 0.970 | 0.663 |
| Validation mean domain AUROC | 0.886 | 0.901 | 0.899 | 0.911 |
| Validation FF++ AUROC | 0.983 | 0.986 | 0.965 | 0.973 |
| Validation Stable Cascade AUROC | 0.518 | 0.616 | 0.623 | 0.706 |
| Fresh v3 `openfake` AUROC | 0.596 | 0.660 | 0.611 | 0.674 |
| Fresh v3 real-source bias | 0.945 | 0.513 | 0.957 | 0.498 |

## Change from E8

Real-source bias is better when lower (0.5 = no source information); AUROC is better when higher. The interaction is E9's change minus both single changes.

| Metric | Data alone | Padding alone | Both (E9) | Interaction |
|---|---:|---:|---:|---:|
| Validation real-source bias | -0.256 | +0.005 | -0.302 | -0.051 |
| Validation mean domain AUROC | +0.015 | +0.013 | +0.025 | -0.003 |
| Validation FF++ AUROC | +0.003 | -0.018 | -0.009 | +0.006 |
| Validation Stable Cascade AUROC | +0.098 | +0.105 | +0.188 | -0.015 |
| Fresh v3 `openfake` AUROC | +0.064 | +0.016 | +0.078 | -0.002 |
| Fresh v3 real-source bias | -0.432 | +0.012 | -0.447 | -0.027 |

## Fresh v3 against E8 (paired bootstrap, 95% CI)

| Arm | `openfake` AUROC change | CI | Bias change | CI |
|---|---:|---:|---:|---:|
| E9-data | +0.064 | [+0.044, +0.082] | -0.432 | [-0.474, -0.390] |
| E9-padding | +0.016 | [+0.009, +0.023] | +0.012 | [+0.004, +0.022] |
| E9 | +0.078 | [+0.056, +0.098] | -0.447 | [-0.489, -0.405] |

## Round 4 recipe

Arms no worse than E9 by more than 0.01 on every metric: neither. Round 4 starts from **E9**.

## Reading

Added by hand after the run. The new runs were trained at `862eee0`, which changes
no training code since E9 (`884b6a1`).

- The data change removed the real-source shortcut. It gives 0.256 of E9's 0.302
  validation bias drop, 0.432 of 0.447 on fresh v3, and 0.064 of the 0.078
  `openfake` gain. Padding alone leaves the bias at E8's level (+0.005 on
  validation, +0.012 on fresh v3). This matches the stated expectation.
- The Stable Cascade gain splits about evenly (data +0.098, padding +0.105),
  against the expectation that padding would give most of it. Its validation
  reals are FFHQ photos, so real-source balancing plausibly helps there too; the
  ablation does not test why.
- Padding alone costs 0.018 FF++ AUROC; together with the FairFace data the cost
  is 0.009.
- The two changes are close to additive. The largest interaction is a further
  0.051 drop in validation bias when both are present.
- Neither change alone matches E9: E9-data trails it most on Stable Cascade
  (0.090) and validation bias (0.046), and E9-padding on both bias measures. By
  the frozen rule, round 4 keeps both changes.
