# DF40 RGB pilot: ResNet18

This is an exploratory baseline on the audited 15,870-image clean pilot. The
complete machine-readable [run report](df40_rgb_resnet18_pilot_01175dd90df8d9285619.json)
is committed beside this summary; the best checkpoint remains on xixi at
`artifacts/rgb/01175dd90df8d9285619/best.pt`.

| Item | Value |
| --- | --- |
| Run ID | `01175dd90df8d9285619` |
| Training commit | `aeeea1a3f009e9d1078bfc3c2c9b214a9bf0aa7b` |
| Clean manifest | `bbead51c5da8f9363a1c` |
| GPU | RTX 5060, PyTorch 2.14.1 + CUDA 13.0 |
| Pretrained weights | ResNet18 ImageNet V1, SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` |
| Best checkpoint | epoch 3, SHA-256 `f11f56d7bb75621c55087c093619c86f51b25501d14d28410236b0ea7e9c2f15` |

The classifier head trained for one epoch; the last residual block and head
trained for two more. Validation video AUROC selected epoch 3. The validation
video threshold was `0.199926`; the test set was evaluated only after this
selection.

| Split | Frames | Frame AUROC | Videos | Video AUROC | Video balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation | 1,550 | 0.762 | 155 | 0.798 | 0.722 |
| Test | 2,400 | 0.598 | 781 | 0.515 | 0.514 |

Test video AUROC by source domain was 0.677 on Celeb-DF, 0.668 on FF++, and
0.491 on CelebA. The 600 CelebA images count as 600 one-frame video groups,
which gives that domain substantial weight in the overall video metric. At the
validation threshold, fake video recall was 0.900 for BlendFace, 0.700 for
DiT, 0.267 for SadTalker, and 0.140 for StarGANv2.

The validation/test gap shows weak transfer to the combined unseen-method and
domain-shifted test split. The DF40 metadata has no identity IDs, so identity
overlap remains unverified. These numbers should not be presented as a
reportable identity-disjoint benchmark. The next experiment should include an
independent identity audit and compare methods within each source domain.

A subsequent [name-derived identity audit](../identity_audit/df40_pilot_clean_bbead51c5da8f9363a1c.json)
found 40 Celeb-DF `idNN` tokens shared by validation and test, touching 1,813
images. This is a candidate identity warning: manipulated names may contain
both source and target tokens, while 690 Celeb-DF images have no token. It does
not establish visual identity matches or cover FF++.
