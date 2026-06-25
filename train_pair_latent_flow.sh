#!/bin/bash
#SBATCH --account=MST114139
#SBATCH --job-name=idb-pair-latent-rf
#SBATCH --partition=normal2
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=2-00:00:00
#SBATCH --output=jobs/pair_latent_flow/job-%j.out
#SBATCH --error=jobs/pair_latent_flow/job-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=110405193@g.nccu.edu.tw

set -euo pipefail

cd /work/u7692101/ID-Blau
ml load miniconda3
ml load cuda/12.4

conda activate IDBlau

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

mkdir -p jobs/pair_latent_flow ./experiments/PairLatent_RF_quad

CUDA_VISIBLE_DEVICES=0 python pair_latent_flow_train.py \
  --data_path ./dataset/GOPRO_Large \
  --dir_path ./experiments/PairLatent_RF_quad \
  --model_name PairLatent_RF_quad \
  --batch_size 4 \
  --crop_size 256 \
  --rf_image_size 128 \
  --end_epoch 1000 \
  --optimizer adamw \
  --sample_timesteps 1000 \
  --val_sample_timesteps 50 \
  --init_lr 2e-4 \
  --min_lr 1e-6 \
  --path_type quadratic \
  --path_gamma 0.25 \
  --latent_dim 64 \
  --latent_map_channels 8 \
  --latent_encoder_channels 32 \
  --train_latent_perturb_std 0.05 \
  --path_code_dim 4 \
  --train_path_code_std 1.0 \
  --path_bend_channels 32 \
  --path_bend_scale 1.0 \
  --validation_epoch 25 \
  --val_save_epochs 25 \
  --check_point_epoch 100 \
  --save_last_epoch 1
