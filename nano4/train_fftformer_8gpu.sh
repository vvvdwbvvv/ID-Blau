#!/bin/bash
#SBATCH --account=MST114139
#SBATCH --job-name=idb-fftformer-4gpu
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
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

conda activate IDBlau

PYTHON=/home/u7692101/.conda/envs/IDBlau/bin/python
TORCHRUN=/home/u7692101/.conda/envs/IDBlau/bin/torchrun
GPU_COUNT=4

run_step() {
  if [ -n "${SLURM_JOB_ID:-}" ] && command -v srun >/dev/null 2>&1; then
    srun --ntasks=1 --gpus="$GPU_COUNT" --gpu-bind=none "$@"
  else
    "$@"
  fi
}

if command -v nvidia-smi >/dev/null 2>&1; then
  run_step nvidia-smi
  run_step nvidia-smi -L
fi
run_step bash -lc 'ls -l /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia[0-9]* 2>&1 || true'

export EXPECTED_CUDA_DEVICES="$GPU_COUNT"
run_step "$PYTHON" - <<'PY'
import os
import torch

expected = int(os.environ["EXPECTED_CUDA_DEVICES"])
print("torch:", torch.__version__)
print("torch_cuda:", torch.version.cuda)
print("SLURM_JOB_GPUS:", os.environ.get("SLURM_JOB_GPUS", "<unset>"))
print("SLURM_GPUS_ON_NODE:", os.environ.get("SLURM_GPUS_ON_NODE", "<unset>"))
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))
print("LD_LIBRARY_PATH_HEAD:", ":".join(os.environ.get("LD_LIBRARY_PATH", "").split(":")[:5]))
device_count = torch.cuda.device_count()
print("cuda_device_count:", device_count)
if device_count < expected:
    raise SystemExit(
        f"Expected at least {expected} CUDA devices from Slurm, "
        f"got {device_count}."
    )
try:
    print("cuda_device_0:", torch.cuda.get_device_name(0))
    tensor = torch.empty(1, device="cuda")
    torch.cuda.synchronize()
    del tensor
except Exception as exc:
    raise SystemExit(
        "CUDA runtime initialization failed even though Slurm exposed "
        f"{device_count} device(s): {exc}"
    )
print("cuda_runtime_check: ok")
PY

mkdir -p jobs/fftformer

run_step "$TORCHRUN" \
  --nproc_per_node "$GPU_COUNT" \
  --master_port 30611 \
  FFTformer/deblur_train_pretrained.py \
  --batch_size 16 \
  --only_use_generate_data \
  --generate_path ./dataset/GOPRO_Large_Reblur

run_step "$TORCHRUN" \
  --nproc_per_node "$GPU_COUNT" \
  --master_port 30611 \
  FFTformer/deblur_train.py \
  --batch_size 16 \
  --resume ./experiments/FFTformer_pretrained/epoch_500_FFTformer_pretrained.pth
