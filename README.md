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
scripts/                 DF40 normalization, selective download, and verification tools
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
`scripts/normalize_df40.py` adapter creates the flat metadata catalog consumed
by the audit and subset commands. It preserves the official fake splits and
deterministically partitions unique real-video groups, preventing frames from
one real video from leaking between splits.

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

### Reproduce the available 31,000-image pilot

Obtain the `dataset_json` directory from the
[official DF40 repository](https://github.com/YZY-stack/DF40), then place it at
`data/raw/df40/official_json/dataset_json`. Normalize the catalog while excluding
two zero-byte members observed in the official StarGANv2 archive:

```bash
python scripts/normalize_df40.py \
  --output data/raw/df40/metadata_available.csv \
  --exclude-paths configs/datasets/df40_unavailable_paths.txt

deepfake-data subset --config configs/subsets/df40_pilot_available_v2.yaml
```

The final configuration selects 20,000 training, 3,000 validation, and 8,000
test images with equal real/fake counts. SD-2.1 validation is intentionally
omitted because the official archive is rate-limited. Test manipulation methods
remain disjoint from training and validation methods.

Install the optional downloader, then pass the generated `subset.csv` path to
the selective downloader. Archives are deleted after successful extraction, so
the full DF40 payload is never stored at once:

```bash
python -m pip install -e ".[download]"
scripts/download_df40_pilot.sh \
  data/subsets/df40/df40_pilot_available_v2/<subset-id>/subset.csv

deepfake-data verify-subset \
  --subset-dir data/subsets/df40/df40_pilot_available_v2/<subset-id>
python scripts/verify_df40_images.py \
  --subset data/subsets/df40/df40_pilot_available_v2/<subset-id>/subset.csv \
  --data-root data/raw/df40
```

Generated catalogs, subsets, archives, and image payloads remain ignored by
Git. Only the code and deterministic selection configuration belong in the
repository.

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
