#!/bin/bash
#SBATCH --account=MST114139
#SBATCH --job-name=idb-stripformer-gopro-2gpu
#SBATCH --partition=normal2
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=2-00:00:00
#SBATCH --output=jobs/stripformer/job-gopro-2gpu-%j.out
#SBATCH --error=jobs/stripformer/job-gopro-2gpu-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=110405193@g.nccu.edu.tw

set -euo pipefail

cd /work/u7692101/ID-Blau
ml load miniconda3
ml load cuda/12.4

conda activate IDBlau

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"

mkdir -p jobs/stripformer

CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nproc_per_node 2 \
  --master_port 32629 \
  Stripformer/deblur_train_first.py \
  --batch_size 4 \
  --data_path ./dataset/GOPRO_Large \
  --dir_path ./experiments/Stripformer_GoPro_original_first_stage_2gpu \
  --model_name Stripformer_GoPro_original_first_stage_2gpu

CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nproc_per_node 2 \
  --master_port 32629 \
  Stripformer/deblur_train_second.py \
  --batch_size 4 \
  --data_path ./dataset/GOPRO_Large \
  --dir_path ./experiments/Stripformer_GoPro_original_2gpu \
  --model_name Stripformer_GoPro_original_2gpu \
  --resume ./experiments/Stripformer_GoPro_original_first_stage_2gpu/final_Stripformer_GoPro_original_first_stage_2gpu.pth
