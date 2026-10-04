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

DF40's official metadata JSON is hierarchical and varies by protocol. The
initial `df40.yaml` config audits a normalized flat metadata file named
`metadata.csv`. Create one row per image while preserving the official
`split`, source domain, video/identity group, manipulation family, and method.
Do not randomly split extracted frames. The normalization adapter will be added
after the exact downloaded DF40 layout is fixed.

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
