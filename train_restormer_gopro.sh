#!/bin/bash
#SBATCH --account=MST114139
#SBATCH --job-name=idb-restormer-gopro-2gpu
#SBATCH --partition=normal2
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=2-00:00:00
#SBATCH --output=jobs/restormer/job-gopro-2gpu-%j.out
#SBATCH --error=jobs/restormer/job-gopro-2gpu-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=110405193@g.nccu.edu.tw

set -euo pipefail

cd /work/u7692101/ID-Blau
ml load miniconda3
ml load cuda/12.4

conda activate IDBlau

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"

mkdir -p jobs/restormer

CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nproc_per_node 2 \
  --master_port 32623 \
  Restormer/deblur_train.py \
  --data_path ./dataset/GOPRO_Large \
  --dir_path ./experiments/Restormer_GoPro_original_2gpu \
  --model_name Restormer_GoPro_original_2gpu
