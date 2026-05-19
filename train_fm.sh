#!/bin/bash
#SBATCH --job-name=idblau-fm
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --account=MST114139

set -eo pipefail

module purge
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate IDBlau

cd "${SLURM_SUBMIT_DIR:-$HOME/ID-Blau}"

mkdir -p logs

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

mkdir -p dataset

# download_if_missing() {
#   local url="$1"
#   local output="$2"
#   if [ ! -f "$output" ]; then
#     echo "Downloading $output"
#     curl -L --fail --retry 5 --retry-delay 10 -o "$output" "$url"
#   else
#     echo "Found $output, skip download"
#   fi
# }

# unzip_if_missing() {
#   local zip_file="$1"
#   local output_dir="$2"
#   if [ ! -d "$output_dir" ]; then
#     echo "Unzipping $zip_file"
#     if command -v unzip >/dev/null 2>&1; then
#       unzip -q "$zip_file" -d dataset
#     else
#       python -m zipfile -e "$zip_file" dataset
#     fi
#   else
#     echo "Found $output_dir, skip unzip"
#   fi
# }

# download_if_missing \
#   "https://huggingface.co/datasets/snah/GOPRO_Large/resolve/main/GOPRO_Large.zip" \
#   "dataset/GOPRO_Large.zip"

# download_if_missing \
#   "https://huggingface.co/datasets/snah/GOPRO_Large/resolve/main/GOPRO_Large_all.zip" \
#   "dataset/GOPRO_Large_all.zip"

# unzip_if_missing "dataset/GOPRO_Large.zip" "dataset/GOPRO_Large"
# unzip_if_missing "dataset/GOPRO_Large_all.zip" "dataset/GOPRO_Large_all"

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

find dataset -maxdepth 3 -type d | sort | head -80

python diffusion_train.py \
  --data_path ./dataset/GOPRO_Large \
  --flow_data_path ./dataset/GOPRO_flow \
  --dir_path ./experiments/ID_Blau \
  --model_name ID_Blau_FM \
  --batch_size 256 \
  --num_workers 8 \
  --prefetch_factor 4 \
  --save_last_epoch 10
