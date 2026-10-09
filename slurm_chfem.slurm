#!/bin/bash -l
#SBATCH --account=glaz.glaz
#SBATCH --partition=gpu
#SBATCH --gpus=a40:1
#SBATCH --nodes=1
#SBATCH --tasks-per-node 1
#SBATCH --qos=48h
#SBATCH --cpus-per-task 32
#SBATCH --time=47:00:00
#SBATCH --job-name=Permeability
#SBATCH --output=/albedo/work/user/fabagh001/chfem/logs/log_%A_%a.out

set -euo pipefail

module load mesa
module load conda
module load cuda
conda activate chfem39

INPUT_DIR="${INPUT_DIR:-/albedo/work/projects/p_ICECTai/B51_processed/samples}"
OUTPUT_DIR="/albedo/work/user/fabagh001/chfem/results"
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
METADATA_FILE="$SCRIPT_DIR/B51.xlsx"
BATCH_SIZE="${BATCH_SIZE:-50}"
TASK_OUTPUT_DIR="$OUTPUT_DIR/task_${SLURM_ARRAY_TASK_ID}"
FILE_OFFSET=$((SLURM_ARRAY_TASK_ID * BATCH_SIZE))

python -u "$SCRIPT_DIR/chfem_permeability.py" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$TASK_OUTPUT_DIR" \
    --metadata_file "$METADATA_FILE" \
    --porosity_threshold 0.02 \
    --file_offset "$FILE_OFFSET" \
    --file_limit "$BATCH_SIZE"
