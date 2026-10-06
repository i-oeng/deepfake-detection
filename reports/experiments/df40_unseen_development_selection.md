# DF40 development-only model selection

Recorded on 2026-10-06 before any final-test prediction export.

The selection metric is FF++ development macro video AUROC across the unseen
methods FaceDancer, MRAA, and StyleGAN3. Celeb-DF is shown as a separate
cross-dataset result and did not affect checkpoint or architecture selection.

| Model | Seed | Best epoch | FF++ macro AUROC | CDF macro AUROC |
|---|---:|---:|---:|---:|
| ResNet18 | 20261006 | 5 | 0.8617 | 0.8801 |
| CLIP ViT-B/16 | 20261006 | 5 | 0.9635 | 0.9459 |
| ConvNeXt-Tiny | 20261006 | 4 | **0.9778** | 0.7619 |
| ConvNeXt-Tiny | 20261007 | 4 | 0.9679 | 0.6380 |
| ConvNeXt-Tiny | 20261008 | 3 | 0.9625 | 0.6995 |

ConvNeXt-Tiny is frozen as the RGB finalist. Its three-seed FF++ development
macro AUROC is 0.9694 mean, 0.0078 sample standard deviation, with a range of
0.9625 to 0.9778. Its CDF performance is materially lower than CLIP, so the CDF
numbers will remain a separate cross-dataset finding.

The one-time final export will include the ResNet18 baseline, CLIP for the
independent architecture comparison, and all three frozen ConvNeXt checkpoints.
No architecture, checkpoint, blend weight, or hyperparameter will be changed
from final-test output.

## Frozen checkpoint hashes

| Model / seed | SHA-256 |
|---|---|
| ResNet18 / 20261006 | `2c6ba714bae214ffd67ad9a6f3e6874c3d47a4033c2ac8f73627b9e0a787b848` |
| CLIP / 20261006 | `a3bcb4fb29af0d4d7e18c56f69b28fd48826bfbad3ce51d94f3b1e20c1d9cdf1` |
| ConvNeXt / 20261006 | `ad14c9a48fe78bc3fa9000872f3195f8ab30577f57af4b310494d6b1bba323b7` |
| ConvNeXt / 20261007 | `5b0f020f33ed672d89b4f35ba0ae8370471f084fa77f0b0009aac5c0c30fa257` |
| ConvNeXt / 20261008 | `cd153dc09709a2dfe8d17de9cbf4f96312cc7d52d337bbf436019623d4a43346` |
