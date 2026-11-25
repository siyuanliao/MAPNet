#!/bin/bash
#SBATCH --job-name=MAPNetA5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
set -euo pipefail

module load miniforge3/24.11 cuda/11.8
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate py310
cd "${SLURM_SUBMIT_DIR:-.}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

torchrun --standalone --nproc_per_node=2 train.py \
  --epochs 150 \
  --batch-size 4 \
  --grad-accum 1 \
  --workers 5 \
  --amp bf16
