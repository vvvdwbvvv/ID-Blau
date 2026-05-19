#!/bin/bash
#SBATCH --job-name=idblau-fm
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00

set -e

cd "${SLURM_SUBMIT_DIR:-$HOME/ID-Blau}"

mkdir -p logs

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

source ~/.bashrc

if command -v conda >/dev/null 2>&1; then
  conda activate IDBlau
fi

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

python diffusion_train.py \
  --data_path ./dataset/GOPRO_Large \
  --flow_data_path ./dataset/GOPRO_flow \
  --dir_path ./experiments/ID_Blau \
  --model_name ID_Blau_FM \
  --batch_size 256 \
  --num_workers 8 \
  --prefetch_factor 4 \
  --save_last_epoch 10
