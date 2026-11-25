#!/usr/bin/env bash
#SBATCH --job-name=DD-PUNet
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
set -euo pipefail
# Set up your site-specific CUDA/PyTorch environment before submitting.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" 
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
torchrun --standalone --nproc_per_node=2 train.py \
  --epochs 150 --batch-size 4 --grad-accum 1 \
  --angle-chunk 1 --workers 4 --amp bf16 "$@"
