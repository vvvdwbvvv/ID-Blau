#!/bin/bash
#SBATCH --account=MST114139
#SBATCH --job-name=idb-fftformer-4gpu
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=2-00:00:00
#SBATCH --output=jobs/fftformer/job-4gpu-%j.out
#SBATCH --error=jobs/fftformer/job-4gpu-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=110405193@g.nccu.edu.tw

set -euo pipefail

cd /work/u7692101/ID-Blau
ml load miniconda3
ml load cuda/12.6

conda activate IDBlau

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"

mkdir -p jobs/fftformer

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --nproc_per_node 8 \
  --master_port 30611 \
  FFTformer/deblur_train_pretrained.py \
  --batch_size 16 \
  --only_use_generate_data \
  --generate_path ./dataset/GOPRO_Large_Reblur

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --nproc_per_node 8 \
  --master_port 30611 \
  FFTformer/deblur_train.py \
  --batch_size 16 \
  --resume ./experiments/FFTformer_pretrained/epoch_500_FFTformer_pretrained.pth
