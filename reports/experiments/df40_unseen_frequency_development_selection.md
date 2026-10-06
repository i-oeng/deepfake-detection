# DF40 frequency branch: development-only selection

Recorded on 2026-10-07 before any frequency-branch test export. No run in this
record has a `test.jsonl` or `test_embeddings.npz`.

All runs use the frozen unseen-method manifest `07f906d9ef61a537efa8`, the shared
sample preprocessing `df40-face-frame-manifest-v1`, the `benchmark.py` training
harness at commit `5c83f52`, and `frequency_resnet18` (ImageNet ResNet18 with a
rebuilt input stem). The selection metric is FF++ development macro video AUROC
over FaceDancer, MRAA, and StyleGAN3, as for the RGB models. Celeb-DF is shown
as a separate cross-dataset result and did not affect any choice.

## Input ablation, seed 20261006

| Frequency inputs | Spectral filter | Best epoch | FF++ macro AUROC | CDF macro AUROC |
|---|---|---:|---:|---:|
| log-amplitude | no | 5 | 0.8775 | 0.6902 |
| log-amplitude, phase | no | 5 | 0.8716 | 0.7222 |
| high-pass residual | no | 5 | 0.8909 | 0.5164 |
| log-amplitude, phase, high-pass | no | 6 | 0.9022 | 0.8143 |
| log-amplitude, phase, high-pass | yes | 6 | **0.9032** | **0.8305** |

Combining all three inputs is better than any single input on FF++ and much
better on CDF. The two all-input variants were within single-seed noise, so the
rule fixed before further runs was: train seeds 20261007 and 20261008 for both,
then select the higher three-seed FF++ mean.

## Three-seed comparison

| Variant | FF++ AUROC mean ± SD | Range | Average precision | TPR at 1% FPR | FaceDancer AUROC | CDF AUROC |
|---|---:|---:|---:|---:|---:|---:|
| All inputs + spectral filter | **0.9012 ± 0.0066** | 0.8938–0.9067 | 0.8057 ± 0.0070 | 0.5926 ± 0.0339 | 0.7264 ± 0.0222 | 0.7645 ± 0.0752 |
| All inputs | 0.8981 ± 0.0117 | 0.8849–0.9072 | 0.8168 ± 0.0107 | 0.6296 ± 0.0257 | 0.7072 ± 0.0328 | 0.7500 ± 0.0941 |

SD is the sample standard deviation over three seeds. For reference, the frozen
RGB development results are ResNet18 0.8617, CLIP ViT-B/16 0.9635, and
ConvNeXt-Tiny 0.9694 (three-seed mean).

## Decision

**All inputs + spectral filter** is frozen as the frequency branch by the
predefined rule. The margin (0.0031) is smaller than either variant's seed
spread, and the variant without the filter is ahead on average precision and
TPR at 1% FPR. This record therefore does not claim that the learnable spectral
filter helps; it only fixes which checkpoints are exported.

The one-time final export will include the three selected checkpoints below. No
architecture, checkpoint, blend weight, or hyperparameter will be changed from
final-test output.

## Frozen checkpoint hashes

Preprocessing ID `df40-freq-224-log_amplitude-phase-highpass-v1`, config
`configs/training/df40_unseen_frequency_all_specfilter.yaml`.

| Seed | Best epoch | Run ID | SHA-256 |
|---|---:|---|---|
| 20261006 | 6 | `frequency_resnet18-20261006-07f906d9ef61a537efa8-5c83f529-c351a5ca` | `c1866cc633d40ce7e1110f84dfd8eec799e54fef2ddf586b7e47a961558726d7` |
| 20261007 | 5 | `frequency_resnet18-20261007-07f906d9ef61a537efa8-5c83f529-0b52c27e` | `1f515fd48dffba3fb51fae0b391e7f4df0513b0776b09db256207e6e9c019662` |
| 20261008 | 4 | `frequency_resnet18-20261008-07f906d9ef61a537efa8-5c83f529-7d69d84e` | `e2ffd178f072964e05405585af6fa4a6b7f07620e93dc11aabee1a06e80d6680` |

Unselected development runs, for completeness:

| Variant / seed | SHA-256 |
|---|---|
| log-amplitude / 20261006 | `dc9b1ca7cab1201f356b9cf7eb04f6bae4b3a9a2ea5a56015566bb8d89746c9a` |
| log-amplitude, phase / 20261006 | `563f45101487636820b3639e1ef356d7f526ece4d0b67a453562d5388ef40b9d` |
| high-pass / 20261006 | `4965d1a567f1577e64a66ad1975ce95a3216049362ec20aedf3f435e20bb8485` |
| all inputs / 20261006 | `ed9cc8919d276a92b78294178666c574d4c138a7c8ba01849e6b3a4f9384bd79` |
| all inputs / 20261007 | `e97db92bc74ad2a30a61b9095a418111520932702b600a8f447150d3a8c2311e` |
| all inputs / 20261008 | `f5316acc6520c5b3d5956ba569abad796ff65c5621c7902dee84f066e3f144f5` |
