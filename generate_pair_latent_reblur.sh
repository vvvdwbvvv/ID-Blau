#!/bin/bash
#SBATCH --account=MST114139
#SBATCH --job-name=idb-pair-latent-gen
#SBATCH --partition=normal2
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=2-00:00:00
#SBATCH --output=jobs/pair_latent_generate/job-%j.out
#SBATCH --error=jobs/pair_latent_generate/job-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=110405193@g.nccu.edu.tw

set -euo pipefail

cd /work/u7692101/ID-Blau
ml load miniconda3
ml load cuda/12.4

conda activate IDBlau

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

mkdir -p jobs/pair_latent_generate ./dataset/GOPRO_Large_PairLatent_Reblur

CUDA_VISIBLE_DEVICES=0 python pair_latent_flow_inference.py \
  --model_path ./experiments/PairLatent_RF_quad/best_PairLatent_RF_quad.pth \
  --dir_path ./dataset/GOPRO_Large_PairLatent_Reblur \
  --data_path ./dataset/GOPRO_Large \
  --type generate_dataset \
  --dataset train \
  --sample_timesteps 50 \
  --generate_num 5 \
  --latent_perturb_std 0.10 \
  --path_code_std 1.0 \
  --rf_sampler heun
