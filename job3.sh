#!/bin/bash
#SBATCH --account=MST114139
#SBATCH --job-name=idb-rectified-flow
#SBATCH --partition=normal2
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=2-00:00:00
#SBATCH --output=jobs/train/job-%j.out
#SBATCH --error=jobs/train/job-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=110405193@g.nccu.edu.tw

set -euo pipefail

cd /work/u7692101/ID-Blau
ml load miniconda3
ml load cuda/12.4

conda activate IDBlau

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# Stage 1: synthetic trajectory FM pretraining
CUDA_VISIBLE_DEVICES=0 python trajectory_flow_reblur.py \
  --mode train \
  --data_path ./dataset/GOPRO_Large \
  --output_dir ./experiments/trajectory_flow_reblur_stage1 \
  --epochs 500 \
  --batch_size 8 \
  --crop_size 256 \
  --trajectory_supervision synthetic \
  --fm_loss_weight 1.0 \
  --reblur_loss_weight 0.0

# Stage 2: real-pair reblur fine-tuning
CUDA_VISIBLE_DEVICES=0 python trajectory_flow_reblur.py \
  --mode train \
  --data_path ./dataset/GOPRO_Large \
  --output_dir ./experiments/trajectory_flow_reblur_stage2 \
  --resume ./experiments/trajectory_flow_reblur_stage1/last_trajectory_flow_reblur.pth \
  --epochs 1000 \
  --batch_size 8 \
  --crop_size 256 \
  --trajectory_supervision synthetic \
  --fm_loss_weight 0.1 \
  --reblur_loss_weight 1.0 \
  --train_sample_steps 4
