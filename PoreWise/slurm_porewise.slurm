#!/bin/bash -l
#SBATCH --account=glaz.glaz
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --qos=48h
#SBATCH --cpus-per-task=28
#SBATCH --time=47:00:00
#SBATCH --job-name=PoreWise
#SBATCH --output=logs/porewise_%A_%a.out
## how to run the code  MAX_FILES=4 bash submit_porewise_array.sh
set -euo pipefail

module load mesa
module load conda
module load cuda
conda activate "${CONDA_ENV:-lbm_env}"

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
INPUT_DIR="${INPUT_DIR:-/albedo/work/projects/p_ICECTai/B51_processed/samples}"
OUTPUT_DIR="${OUTPUT_DIR:-/albedo/scratch/user/fabagh001/lbm}"
METADATA_FILE="${METADATA_FILE:-$REPO_DIR/B51.xlsx}"
if [[ ! -f "$METADATA_FILE" ]]; then
    echo "Metadata workbook not found: $METADATA_FILE" >&2
    exit 1
fi
BATCH_SIZE="${BATCH_SIZE:-50}"
MAX_FILES="${MAX_FILES:-0}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
TASK_OUTPUT_DIR="$OUTPUT_DIR/task_${TASK_ID}"
FILE_OFFSET=$((TASK_ID * BATCH_SIZE))
FILE_LIMIT="$BATCH_SIZE"
if (( MAX_FILES > 0 && FILE_OFFSET + FILE_LIMIT > MAX_FILES )); then
    FILE_LIMIT=$((MAX_FILES - FILE_OFFSET))
fi
if (( FILE_LIMIT <= 0 )); then
    echo "Array task $TASK_ID is beyond the requested file limit; skipping."
    exit 0
fi

python -u "$SCRIPT_DIR/run_gpu.py" \
    --input_dir "$INPUT_DIR" \
    --input_pattern '*' \
    --output_dir "$TASK_OUTPUT_DIR" \
    --metadata_file "$METADATA_FILE" \
    --porosity_threshold 0.02 \
    --file_offset "$FILE_OFFSET" \
    --file_limit "$FILE_LIMIT"