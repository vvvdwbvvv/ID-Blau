#!/bin/bash
#SBATCH --account=MST114139
#SBATCH --job-name=idb-rf-ft256
#SBATCH --partition=normal2
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=2-00:00:00
#SBATCH --output=jobs/finetune_256/job-%j.out
#SBATCH --error=jobs/finetune_256/job-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=110405193@g.nccu.edu.tw

set -euo pipefail

cd /work/u7692101/ID-Blau
ml load miniconda3
ml load cuda/12.4

conda activate IDBlau

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

mkdir -p jobs/finetune_256 ./experiments/ID_Blau_RF_ft256

CUDA_VISIBLE_DEVICES=0 python rectified_flow_train.py \
  --data_path ./dataset/GOPRO_Large \
  --flow_data_path ./dataset/GOPRO_flow \
  --dir_path ./experiments/ID_Blau_RF_ft256 \
  --model_name ID_Blau_RF_ft256 \
  --resume ./weights/epoch_1000_ID_Blau_RF \
  --batch_size 4 \
  --crop_size 256 \
  --rf_image_size 128 \
  --end_epoch 1000 \
  --optimizer adamw \
  --sample_timesteps 1000 \
  --val_sample_timesteps 50 \
  --scheduler cosine \
  --init_lr 5e-5 \
  --min_lr 1e-6 \
  --validation_epoch 25 \
  --val_save_epochs 25 \
  --check_point_epoch 100 \
  --save_last_epoch 1
