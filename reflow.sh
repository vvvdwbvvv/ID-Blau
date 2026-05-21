#!/bin/bash
#SBATCH --account=MST114139
#SBATCH --job-name=idb-reflow
#SBATCH --partition=normal2
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=2-00:00:00
#SBATCH --output=jobs/reflow/job-%j.out
#SBATCH --error=jobs/reflow/job-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=110405193@g.nccu.edu.tw

cd /work/u7692101/ID-Blau
ml load miniconda3
ml load cuda/12.4

conda activate IDBlau

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

CUDA_VISIBLE_DEVICES=0 python diffusion_inference.py \
  --model_path ./weights/epoch_1000_ID_Blau_RF \
  --dir_path ./dataset/GOPRO_Large_Reblur \
  --strategy M10 O TURN
