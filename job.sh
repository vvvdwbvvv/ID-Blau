#!/bin/bash
#SBATCH --account=MST114139
#SBATCH --job-name=idb-rectified-flow
#SBATCH --partition=normal2
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=2-00:00:00
#SBATCH --output=jobs/train/job-%j.out
#SBATCH --error=jobs/train/job-%j.err
#SBATCH --main-type=BEGIN, END, FAIL
#SBATCH --main-user=110405193@g.nccu.edu.tw


ml load miniconda3
ml load cuda/12.4

cd /work/u7692101/ID-Blau/

conda activate ID-Blau

mkdir -p /jobs/train

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"


if [ ! -d dataset/GOPRO_flow/train ] || [ ! -d dataset/GOPRO_flow/test ]; then
  echo "Generating GOPRO_flow"
  (
    cd PrepareCondition
    python generate_condition.py \
    --mode all \
    --model=weights/raft-things.pth \
    --dir_path=../dataset/GOPRO_flow \
  )
else
  echo "Found dataset/GOPRO_flow, skip condition generation"
fi

cd /work/u7692101/ID-Blau/

CUDA_VISIBLE_DEVICES=0 python rectified_flow_train.py
    --data_path ./dataset/GOPRO_Large \
    --flow_data_path ./dataset/GOPRO_flow \
    --dir_path ./experiments/ID_Blau_RF \
    --model_name ID_Blau_RF \
    --batch_size 8 \
    --crop_size 128 \
    --end_epoch 5000 \
    --optimizer adamw \
    --sample_timesteps 1000 \
    --scheduler cosine \
    --init_lr 2e-4 \
    --min_lr 1e-5
