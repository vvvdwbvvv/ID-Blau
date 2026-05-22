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

RESUME_PATH="${RESUME_PATH:-./weights/epoch_1000_ID_Blau_RF}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-./experiments/ID_Blau_RF_ft256}"
MODEL_NAME="${MODEL_NAME:-ID_Blau_RF_ft256}"
BATCH_SIZE="${BATCH_SIZE:-4}"
END_EPOCH="${END_EPOCH:-1000}"
INIT_LR="${INIT_LR:-5e-5}"
MIN_LR="${MIN_LR:-1e-6}"

if [ ! -f "$RESUME_PATH" ]; then
  echo "Missing checkpoint: $RESUME_PATH"
  echo "Set RESUME_PATH=/path/to/checkpoint when submitting this script."
  exit 1
fi

mkdir -p jobs/finetune_256 "$EXPERIMENT_DIR"

CUDA_VISIBLE_DEVICES=0 python rectified_flow_train.py \
  --data_path ./dataset/GOPRO_Large \
  --flow_data_path ./dataset/GOPRO_flow \
  --dir_path "$EXPERIMENT_DIR" \
  --model_name "$MODEL_NAME" \
  --resume "$RESUME_PATH" \
  --batch_size "$BATCH_SIZE" \
  --crop_size 256 \
  --val_crop_size 256 \
  --end_epoch "$END_EPOCH" \
  --optimizer adamw \
  --sample_timesteps 1000 \
  --val_sample_timesteps 50 \
  --scheduler cosine \
  --init_lr "$INIT_LR" \
  --min_lr "$MIN_LR" \
  --validation_epoch 25 \
  --val_save_epochs 25 \
  --check_point_epoch 100 \
  --save_last_epoch 1
