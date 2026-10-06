#!/bin/bash -l
#SBATCH --account=glaz.glaz
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --nodes=1
#SBATCH --tasks-per-node 1
#SBATCH --qos=48h
#SBATCH --cpus-per-task 12
#SBATCH --time=47:00:00
#SBATCH --job-name=Permeability
#SBATCH --output=/albedo/work/user/fabagh001/chfem/logs/log_%A_%a.out


module load mesa
module load conda
module load cuda
conda activate chfem39

INPUT_DIR="/albedo/work/projects/p_ICECTai/B51_processed/samples"
OUTPUT_DIR="/albedo/work/user/fabagh001/chfem/results"

python -u "chfem_permeability.py" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
