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
deepfake-data normalize-df40 --input-dir data/raw/df40/official_json --output data/raw/df40/metadata.csv
```

The adapter retains each official frame path in `source_path` and assigns a
portable `relative_path` under `images/`. This is a destination for downloaded
or extracted frames; normalization does not copy image bytes. It deduplicates
real frames repeated in several method JSONs and rejects any frame assigned to
conflicting splits or labels. The official JSONs do not consistently identify
people, so absent `identity_id` values stay empty. A passing manifest still
requires a separate identity audit before reportable training.

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

The supplied pilot targets up to 2,500 images per training method, 500 per
validation method, and 1,000 per unseen test method. Quota shortfalls are
recorded rather than hidden. Set `strict_quotas: true` before freezing final
report experiments. With complete quotas and 1:1 real/fake balancing, the
selection contains at most 32,000 images.

## Notebook

Open `notebooks/01_dataset_audit.ipynb` for a compact visual review. The
notebook calls the package functions and contains no independent audit logic,
so CLI and notebook results stay consistent.

## Quality gates

The default configuration fails when it finds missing/corrupt images, duplicate
content crossing splits, configured groups crossing splits, unexpected labels
or splits, or a suspicious metadata column that nearly determines the label.
Thresholds are explicit in each dataset config.

Run tests with:

```powershell
python -m pytest
```
