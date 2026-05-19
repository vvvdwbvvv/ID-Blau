#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/PrepareCondition"

python generate_condition.py \
  --mode all \
  --model=weights/raft-things.pth \
  --dir_path=../dataset/GOPRO_flow
