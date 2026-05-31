#!/bin/bash
#SBATCH --account=MST114139
#SBATCH --job-name=idb-mimounet-4gpu
#SBATCH --partition=normal2
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=2-00:00:00
#SBATCH --output=jobs/mimounet/job-4gpu-%j.out
#SBATCH --error=jobs/mimounet/job-4gpu-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=110405193@g.nccu.edu.tw

set -euo pipefail

cd /work/u7692101/ID-Blau
ml load miniconda3
ml load cuda/12.4

conda activate IDBlau

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"

mkdir -p jobs/mimounet

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nproc_per_node 4 \
  --master_port 30617 \
  MIMO_UNet/deblur_train_pretrained.py \
  --batch_size 32 \
  --only_use_generate_data \
  --generate_path ./dataset/GOPRO_Large_Reblur

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nproc_per_node 4 \
  --master_port 30617 \
  MIMO_UNet/deblur_train.py \
  --batch_size 32 \
  --resume ./experiments/MIMO_UNetPlus_pretrained/epoch_500_MIMO_UNetPlus_pretrained.pth
