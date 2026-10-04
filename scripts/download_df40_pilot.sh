#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/download_df40_pilot.sh SUBSET_CSV

Selectively downloads and extracts only the images listed in SUBSET_CSV.
Environment overrides:
  DF40_OUTPUT_ROOT  extraction root (default: data/raw/df40)
  DF40_STAGING      temporary archive directory (default: .tmp/df40-pilot)
  GDOWN             gdown executable (default: gdown)
  PYTHON            Python executable (default: python3)
EOF
}

if [[ $# -ne 1 ]]; then
    usage >&2
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd -- "$script_dir/.." && pwd)"
subset="$1"
output="${DF40_OUTPUT_ROOT:-$project/data/raw/df40}"
staging="${DF40_STAGING:-$project/.tmp/df40-pilot}"
gdown="${GDOWN:-gdown}"
python="${PYTHON:-python3}"
extractor="$script_dir/extract_df40_subset.py"

if [[ ! -f "$subset" ]]; then
    printf 'Subset not found: %s\n' "$subset" >&2
    exit 2
fi
command -v "$gdown" >/dev/null
command -v "$python" >/dev/null
mkdir -p "$staging" "$output"

download_extract() {
    local stage="$1"
    local kind="$2"
    local archive_name="$3"
    local url="$4"
    local missing_mode="${5:-strict}"
    local archive="$staging/$archive_name"
    local marker="$staging/.done-$stage"

    if [[ -f "$marker" ]]; then
        printf '[%s] already completed %s\n' "$(date --iso-8601=seconds)" "$stage"
        return
    fi

    printf '\n[%s] downloading %s\n' "$(date --iso-8601=seconds)" "$archive_name"
    if ! "$gdown" --continue --retries 5 --timeout 120 "$url" -O "$archive"; then
        printf '[%s] download failed for %s\n' "$(date --iso-8601=seconds)" "$stage" >&2
        return 1
    fi
    printf '[%s] selectively extracting %s\n' "$(date --iso-8601=seconds)" "$kind"
    local extract_args=(
        --subset "$subset"
        --archive "$archive"
        --kind "$kind"
        --output-root "$output"
    )
    if [[ "$missing_mode" == allow ]]; then
        extract_args+=(--allow-missing)
    fi
    if ! "$python" "$extractor" "${extract_args[@]}"; then
        printf '[%s] extraction incomplete for %s\n' "$(date --iso-8601=seconds)" "$stage" >&2
        return 1
    fi
    rm -f -- "$archive"
    touch "$marker"
    printf '[%s] completed %s\n' "$(date --iso-8601=seconds)" "$stage"
    df -h "$project"
}

download_extract simswap-train simswap train-simswap.zip \
    'https://drive.google.com/uc?id=1vnEXjxgSxmiNY-RkLQdsbhayTvAAoOIc' allow

download_extract wav2lip-train wav2lip train-wav2lip.zip \
    'https://drive.google.com/uc?id=12X6MJ9--rCuptabYPXZ74ux-haV2h7cc' allow
download_extract wav2lip-validation wav2lip test-wav2lip.zip \
    'https://drive.google.com/uc?id=1vm1GnDl07BUxH15gqiBA9yIaY_-69Shr'

download_extract stylegan2-train stylegan2 train-stylegan2.zip \
    'https://drive.google.com/uc?id=12LQnIp9gTtem9Wo4GMr6Q7MNVgxfp5Rg' allow
download_extract stylegan2-validation stylegan2 test-stylegan2.zip \
    'https://drive.google.com/uc?id=1AF9Le3doLoWyaBXc26zR-FUZzugMuiVr'

download_extract sd21-train sd21 train-sd21.zip \
    'https://drive.google.com/uc?id=1rRbjGij6Zznkj5PV7vAL1c3r_pWIGQa6' allow

download_extract blendface-test blendface test-blendface.zip \
    'https://drive.google.com/uc?id=1prCL6Rh1kvreBQJcQ409JR5XvD1Tj3xU'
download_extract sadtalker-test sadtalker test-sadtalker.zip \
    'https://drive.google.com/uc?id=1noHwPXw9cX_UzKpPA_Iq-wm6D2E6nqc0'
download_extract dit-test dit test-dit.zip \
    'https://drive.google.com/uc?id=1Y15vpgltFD1amMEABCRkOiDEK-aQpRJ8'
download_extract starganv2-test starganv2 test-starganv2.zip \
    'https://drive.google.com/uc?id=1NC8akz76RQneJIrLL8S6JObDn5VzjyC-'

download_extract real-ff real:ff real-ff.zip \
    'https://drive.google.com/uc?id=1dHJdS0NZ6wpewbGA5B0PdIBS9gz28pdb'
download_extract real-celeba real:celeba real-starganv2.zip \
    'https://drive.google.com/uc?id=1NC8akz76RQneJIrLL8S6JObDn5VzjyC-'

printf '\n[%s] all selected DF40 image archives completed\n' "$(date --iso-8601=seconds)"
