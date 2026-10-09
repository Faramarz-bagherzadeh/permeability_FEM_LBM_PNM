#!/bin/bash
set -euo pipefail

INPUT_DIR="/albedo/work/projects/p_ICECTai/B51_processed/samples"
BATCH_SIZE=50
MAX_CONCURRENT_JOBS=6
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Input directory does not exist: $INPUT_DIR" >&2
    exit 1
fi

TOTAL_TIFS=$(find "$INPUT_DIR" -maxdepth 1 -type f \
    \( -iname '*.tif' -o -iname '*.tiff' \) -printf '.' | wc -c)
if (( TOTAL_TIFS == 0 )); then
    echo "No TIFF files found in $INPUT_DIR" >&2
    exit 1
fi

TASK_COUNT=$(( (TOTAL_TIFS + BATCH_SIZE - 1) / BATCH_SIZE ))
ARRAY_SPEC="0-$((TASK_COUNT - 1))%$MAX_CONCURRENT_JOBS"
echo "Found $TOTAL_TIFS TIFF files; submitting $TASK_COUNT jobs, up to $BATCH_SIZE files per job."
cd "$SCRIPT_DIR"
sbatch --array="$ARRAY_SPEC" "$SCRIPT_DIR/slurm_chfem.slurm"
