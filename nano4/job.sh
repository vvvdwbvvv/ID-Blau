#!/bin/bash
#SBATCH --account=MST114139
#SBATCH --job-name=idb-rectified-flow
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=2-00:00:00
#SBATCH --output=jobs/train/job-%j.out
#SBATCH --error=jobs/train/job-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=110405193@g.nccu.edu.tw



cd /work/u7692101/ID-Blau
ml load miniconda3
ml load cuda/12.6

conda activate IDBlau

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"


if [ ! -d dataset/GOPRO_flow/train ] || [ ! -d dataset/GOPRO_flow/test ]; then
  echo "Generating GOPRO_flow"
  (
    cd PrepareCondition
    python generate_condition.py \
      --mode all \
      --model=weights/raft-things.pth \
      --dir_path=../dataset/GOPRO_flow
  )
else
  echo "Found dataset/GOPRO_flow, skip condition generation"
fi

cd /work/u7692101/ID-Blau

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python rectified_flow_train.py \
  --data_path ./dataset/GOPRO_Large \
  --flow_data_path ./dataset/GOPRO_flow \
  --dir_path ./experiments/ID_Blau_RF_256_pretrain \
  --model_name ID_Blau_RF_256_pretrain \
  --batch_size 32 \
  --crop_size 256 \
  --end_epoch 1000 \
  --optimizer adamw \
  --sample_timesteps 1000 \
  --val_sample_timesteps 100 \
  --scheduler cosine \
  --init_lr 1e-4 \
  --min_lr 1e-6 \
  --pad_multiple 128 \
  --pad_mode reflect 

