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
## how to run the code  for all MAX_FILES=0 BATCH_SIZE=10 bash submit_porewise_array.sh
set -euo pipefail

module load mesa
module load cuda/12.2.2
CUDA_LIB_DIR="$CUDA_HOME/targets/x86_64-linux/lib"
if [[ ! -f "$CUDA_LIB_DIR/libnvrtc-builtins.so.12.2" ]]; then
    echo "NVRTC built-ins library not found: $CUDA_LIB_DIR/libnvrtc-builtins.so.12.2" >&2
    exit 1
fi
export LD_LIBRARY_PATH="$CUDA_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
VENV_ACTIVATE="$SCRIPT_DIR/.venv/bin/activate"
if [[ ! -f "$VENV_ACTIVATE" ]]; then
    echo "PoreWise virtual environment not found: $VENV_ACTIVATE" >&2
    exit 1
fi
source "$VENV_ACTIVATE"

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

python - <<'PY'
import cupy as cp

print(f"CuPy {cp.__version__}: {cp.__file__}", flush=True)
print("CUDA kernel preflight:", (cp.arange(4) ** 2).get(), flush=True)
PY

python -u "$SCRIPT_DIR/run_gpu.py" \
    --input_dir "$INPUT_DIR" \
    --input_pattern '*' \
    --output_dir "$TASK_OUTPUT_DIR" \
    --metadata_file "$METADATA_FILE" \
    --porosity_threshold 0.02 \
    --file_offset "$FILE_OFFSET" \
    --file_limit "$FILE_LIMIT"