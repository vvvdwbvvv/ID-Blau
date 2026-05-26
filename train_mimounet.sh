#!/bin/bash
#SBATCH --account=MST114139
#SBATCH --job-name=idb-mimounet
#SBATCH --partition=normal2
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=2-00:00:00
#SBATCH --output=jobs/mimounet/job-%j.out
#SBATCH --error=jobs/mimounet/job-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=110405193@g.nccu.edu.tw

set -euo pipefail

cd /work/u7692101/ID-Blau
ml load miniconda3
ml load cuda/12.4

conda activate IDBlau

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"

mkdir -p jobs/mimounet

CUDA_VISIBLE_DEVICES=0 torchrun \
  --nproc_per_node 1 \
  --master_port 29617 \
  MIMO_UNet/deblur_train_pretrained.py \
  --only_use_generate_data \
  --generate_path ./dataset/GOPRO_Large_Reblur

CUDA_VISIBLE_DEVICES=0 torchrun \
  --nproc_per_node 1 \
  --master_port 29617 \
  MIMO_UNet/deblur_train.py \
  --resume ./experiments/MIMO_UNetPlus_pretrained/epoch_500_MIMO_UNetPlus_pretrained.pth
