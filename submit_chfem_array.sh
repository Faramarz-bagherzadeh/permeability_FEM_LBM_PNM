#!/bin/bash
set -euo pipefail

INPUT_DIR="/albedo/work/projects/p_ICECTai/B51_processed/samples"
BATCH_SIZE=50
MAX_CONCURRENT_JOBS=6
MAX_FILES="${MAX_FILES:-4}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! "$MAX_FILES" =~ ^[0-9]+$ ]]; then
    echo "MAX_FILES must be a non-negative integer (0 means no limit)." >&2
    exit 1
fi
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

FILES_TO_PROCESS="$TOTAL_TIFS"
if (( MAX_FILES > 0 && FILES_TO_PROCESS > MAX_FILES )); then
    FILES_TO_PROCESS="$MAX_FILES"
fi
TASK_COUNT=$(( (FILES_TO_PROCESS + BATCH_SIZE - 1) / BATCH_SIZE ))
ARRAY_SPEC="0-$((TASK_COUNT - 1))%$MAX_CONCURRENT_JOBS"
echo "Found $TOTAL_TIFS TIFF files; this submission will consider $FILES_TO_PROCESS files across $TASK_COUNT jobs."
cd "$SCRIPT_DIR"
export INPUT_DIR BATCH_SIZE MAX_FILES
sbatch --array="$ARRAY_SPEC" "$SCRIPT_DIR/slurm_chfem.slurm"
