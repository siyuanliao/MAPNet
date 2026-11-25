#!/bin/bash
#SBATCH --job-name=AGLC-Net
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit from the AGLC-Net project directory}"

DATA_DIR="${1:-../dataset}"
OUT_DIR="${2:-runs/aglc_net}"

export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1

COMMON=(
  --data-dir "$DATA_DIR"
  --output-dir "$OUT_DIR"
  --batch-size 4
  --grad-accum 1
  --angle-chunk 3
  --workers 5
  --amp bf16
)

torchrun --standalone --nproc_per_node=2 "$PWD/train.py" \
  --stage pretrain --epochs 50 "${COMMON[@]}"

torchrun --standalone --nproc_per_node=2 "$PWD/train.py" \
  --stage correct --epochs 150 \
  --decoder-freeze-epochs 20 "${COMMON[@]}"