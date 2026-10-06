# Deepfake Detection

Reproducible experiments for RGB, frequency-domain, and fused deepfake detection.
The first milestone provides a dataset audit and a content-addressed, immutable
manifest. Model training will consume a verified manifest rather than walking a
mutable data directory directly.

## Repository layout

```text
configs/datasets/       Dataset-specific audit configuration
data/raw/               Local source data (ignored by Git)
data/manifests/         Generated immutable manifests (ignored by Git)
data/subsets/           Generated pre-download selections (ignored by Git)
notebooks/              Thin exploratory notebooks
reports/data_audit/     Deterministic audit reports (ignored by Git)
src/deepfake_detection/ Reusable Python package
tests/                  Unit and integration tests
```

Large data, generated manifests, checkpoints, and reports are deliberately not
committed. The configuration, code, and checksum metadata make each run
reproducible.

## Setup

Python 3.11 or 3.12 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,notebook]"
```

## Dataset contract

Each dataset is described by a YAML file. Paths in a config are resolved from
the repository root. The audit builder supports CSV, JSON Lines, and flat JSON
metadata. A top-level JSON object must either be one record or contain a
`records` list.

The canonical manifest contains:

- stable sample identity and relative path;
- label, split, and configured metadata fields;
- decode status, dimensions, mode, and detected image format;
- file SHA-256, decoded-pixel SHA-256, and 64-bit perceptual dHash;
- a content-derived manifest ID.

The builder writes to
`data/manifests/<dataset>/<manifest-id>/manifest.csv`. Existing artifacts are
never overwritten. Rebuilding identical input resolves to the same directory;
modified input creates a new manifest ID.

## Prepare the Kaggle 2026 dataset

Place the metadata and downloaded images under:

```text
data/raw/kaggle_2026/
├── FINAL_DATASET.csv
└── images/
    └── .../<image_id>.<extension>
```

The supplied config searches recursively by `image_id`, so image extensions and
intermediate label/split folders may vary. Each ID must resolve to exactly one
file. Ambiguous and missing paths remain in the manifest and fail the audit.

Build and audit it:

```powershell
deepfake-data build --config configs/datasets/kaggle_2026.yaml
deepfake-data audit --config configs/datasets/kaggle_2026.yaml --manifest <manifest.csv>
deepfake-data verify --manifest-dir <manifest-directory>
```

The `build` command also creates an audit report automatically. Commands exit
non-zero when a configured quality gate fails, which makes the same checks
usable in CI.

## DF40

DF40's official `dataset_json` files are hierarchical. Download the method
JSON files from the [DF40 project](https://github.com/YZY-stack/DF40) into
`data/raw/df40/official_json/`. The pilot adapter accepts the FF and CDF
variants of SimSwap, Wav2Lip, StyleGAN2, SD-2.1, BlendFace, SadTalker, and DiT,
plus `starganv2.json`. Do not include aggregate files such as `DF40_all.json`.

```powershell
deepfake-data normalize-df40 --input-dir data/raw/df40/official_json --output data/raw/df40/metadata.csv --eval-split-seed 20260925
```

The adapter retains each official frame path in `source_path` and assigns a
portable `relative_path` under `images/`. This is a destination for downloaded
or extracted frames; normalization does not copy image bytes. It deduplicates
real frames repeated in several method JSONs and rejects any frame assigned to
conflicting labels or train/evaluation pools. The published evaluation frames
appear under both `val` and `test`; the explicit seed assigns whole videos to
one pilot split across methods. `official_splits` preserves those source labels.
Omitting `--eval-split-seed` keeps strict conflict detection. The official JSONs
do not consistently identify people, so absent `identity_id` values stay empty.
A passing manifest still requires a separate identity audit before reportable
training.

### Select a DF40 pilot before downloading images

The deterministic subset sampler operates on the normalized metadata catalog,
so it does not require the full image payload. The catalog must contain:

```text
image_id, relative_path, label, split, fake_method,
manipulation_family, source_domain, video_id, identity_id, frame_index
```

Canonical fake-method slugs used by the pilot are `simswap`, `wav2lip`,
`stylegan2`, `sd21`, `blendface`, `sadtalker`, `dit`, and `starganv2`.

```powershell
deepfake-data subset --config configs/subsets/df40_pilot.yaml
deepfake-data verify-subset --subset-dir <subset-directory>
```

Selection is independent of source row order. Groups are ranked using the
configured seed; videos are kept in their official split; and frames are chosen
at evenly spaced positions. Real samples are balanced per source domain. The
result is stored at
`data/subsets/df40/df40_pilot_v1/<subset-id>/subset.csv` with checksums and a
summary in `subset.json`.

The supplied pilot selects 1,500 images per training method, 200 per validation
method, and 300 per unseen test method. With 1:1 real/fake balancing, this is
16,000 images. Strict quotas reject shortfalls; the pilot remains exploratory
until source identity and image payload audits pass.

### Materialize and audit the selected images

The official DF40 method ZIPs contain processed fake images. The authors publish
the processed [FF++ and Celeb-DF real images](https://github.com/YZY-stack/DF40#-df40-dataset)
separately. Put them under `data/raw/df40/downloads/real/` with the names
`FaceForensics++_real_data_for_DF40.zip` and
`Celeb-DF-v2_real_data_for_DF40.zip`. The selected StarGANv2 real images are
already in `data/raw/df40/downloads/test/starganv2.zip`.

```powershell
deepfake-data materialize-df40 --subset-dir data/subsets/df40/df40_pilot_v1/743545ebfd6972c3a5fe --downloads-root data/raw/df40/downloads --data-root data/raw/df40 --dry-run
deepfake-data materialize-df40 --subset-dir data/subsets/df40/df40_pilot_v1/743545ebfd6972c3a5fe --downloads-root data/raw/df40/downloads --data-root data/raw/df40
deepfake-data build --config configs/datasets/df40_pilot.yaml
```

Materialization verifies the subset checksums and resolves all 16,000 ZIP members
before writing images. It extracts only selected members, checks ZIP CRCs, and
refuses to replace a changed file. The pilot dataset config pins the subset ID,
so the manifest audits the selected images rather than the full catalog.

The first image audit found identical and visually matching frames across
splits, despite disjoint video names. For the baseline, derive a separate
immutable subset that retains test before validation before train when image
fingerprints collide. It removes each affected video group in the lower
priority split, including nearby frames. The original 16,000-image selection
and failed audit remain as provenance.

```powershell
deepfake-data prune-cross-split --subset-dir data/subsets/df40/df40_pilot_v1/743545ebfd6972c3a5fe --manifest-dir data/manifests/df40_pilot/d1907365d1be9a1e20b1 --output-root data/subsets/df40/df40_pilot_clean_v1
deepfake-data verify-subset --subset-dir data/subsets/df40/df40_pilot_clean_v1/4cc6eda3de869c8b7846
deepfake-data build --config configs/datasets/df40_pilot_clean.yaml
```

This removed 130 images in 13 video groups. The clean pilot contains 15,870
images. Its `subset.json` records the input hashes, fingerprint collisions,
removed groups, and per-split counts. The final audit still checks all images
and split boundaries; this filtering does not establish identity disjointness
because the official metadata lacks identity IDs.

### Exploratory RGB baseline

The authors' processed images are already RGB face crops (256 or 512 pixels).
The baseline applies a recorded 224-pixel resize and ImageNet normalization,
then trains a pretrained ResNet18 classifier on the clean manifest. It freezes
the encoder for one epoch and fine-tunes its last block for two more. Weighted
sampling balances real/fake classes, fake methods, and video groups. Validation
video AUROC selects the checkpoint; a threshold chosen from validation videos
is applied to the test set after selection. Test methods are unseen in training.

```bash
python -m pip install -e '.[train]'
deepfake-train-rgb --config configs/training/df40_rgb_resnet18_pilot.yaml --check-only
deepfake-train-rgb --config configs/training/df40_rgb_resnet18_pilot.yaml
```

The run writes its resolved config, best checkpoint, SHA-256 checksum, epoch
history, and frame/video metrics to `artifacts/rgb/<run-id>/`. It reads only
manifest-listed images. These first results are exploratory because the DF40
catalog has no identity IDs, so identity overlap has not been ruled out.
The first completed run is summarized in
[the RGB pilot report](reports/experiments/df40_rgb_resnet18_pilot.md).

### Candidate identity overlap

Video names from the Celeb-DF source often contain `idNN` tokens. The following
audit checks those tokens across split boundaries, including both tokens in a
swap name. It is a metadata warning, not a face-recognition result:

```bash
deepfake-data identity-tokens --manifest-dir data/manifests/df40_pilot_clean/bbead51c5da8f9363a1c --output reports/identity_audit/df40_pilot_clean_bbead51c5da8f9363a1c.json
```

The [pilot identity-token report](reports/identity_audit/df40_pilot_clean_bbead51c5da8f9363a1c.json)
finds 40 candidate IDs in both validation and test, affecting 1,813 images.
Some Celeb-DF names have no such token, and FF++ identities remain unverified.
Identity-disjoint claims require a separate visual/lineage audit and a revised
split protocol.

## Notebook

Open `notebooks/01_dataset_audit.ipynb` for a compact visual review. The
notebook calls the package functions and contains no independent audit logic,
so CLI and notebook results stay consistent.

## Frozen unseen-method benchmark

The current protocol trains on eight DF40 methods, develops on FaceDancer,
MRAA, and StyleGAN3, and reserves UniFace, MCNet, and RDDM for one final test
export. The frozen manifest, split counts, audit limits, model-selection rules,
and frequency export contract are recorded in the
[DF40 unseen-method protocol](reports/experiments/df40_unseen_protocol.md).
The completed RGB comparison and confidence intervals are in the
[final RGB report](reports/experiments/df40_unseen_final_rgb.md).

Build and verify the lineage-aware catalog with:

```bash
deepfake-data extend-df40-archives \
  --catalog data/raw/df40/metadata.csv \
  --downloads-root data/raw/df40/downloads \
  --output data/raw/df40/metadata_unseen.csv \
  --eval-split-seed 20261006
deepfake-data partition-df40-unseen \
  --catalog data/raw/df40/metadata_unseen.csv \
  --output data/raw/df40/metadata_unseen_partitioned_v3.csv \
  --seed 20261006
deepfake-data subset --config configs/subsets/df40_unseen_v1.yaml
deepfake-data audit-unseen \
  --manifest-dir data/manifests/df40_unseen_clean_v3/07f906d9ef61a537efa8
```

On xixi, both researchers submit GPU work through the same one-slot
task-spooler queue:

```bash
scripts/setup-gpu-queue
scripts/gpu-tsp -l
scripts/submit-benchmark configs/training/df40_unseen_resnet18.yaml
```

The wrapper fixes `TS_SOCKET` under `runtime/gpu-queue`, where the
`deepfake-editors` setgid group gives xixi and aida access to one queue. It
also stores task logs under `runtime/gpu-queue/tmp` and changes task-spooler's
owner-only `0600` output mode to group-readable and group-writable `0660`.
Task-spooler serializes submitted jobs; it cannot account for GPU processes
started outside that queue.

## Quality gates

The default configuration fails when it finds missing/corrupt images, duplicate
content crossing splits, configured groups crossing splits, unexpected labels
or splits, or a suspicious metadata column that nearly determines the label.
Thresholds are explicit in each dataset config.

Run tests with:

```powershell
python -m pytest
```
