#!/bin/bash
# Setup conda environment for selective generation project
# Usage: bash setup_env.sh
set -euo pipefail

ENV_PREFIX="/ocean/projects/cis250290p/tzhou6/envs/gen-eval"
CUDA_MODULE="cuda/12.4.0"

echo "=== Step 1: Create conda env ==="
conda create -p "${ENV_PREFIX}" python=3.10 -y

echo "=== Step 2: Install PyTorch (cu124, self-contained CUDA runtime) ==="
conda run -p "${ENV_PREFIX}" \
  pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124

echo "=== Step 3: Install core dependencies (pin transformers<=4.44) ==="
conda run -p "${ENV_PREFIX}" \
  pip install transformers==4.44.2 diffusers==0.30.3 accelerate==0.34.2 \
              datasets safetensors pillow tqdm

echo "=== Step 4: Install image-reward (load cuda module for potential fairscale build) ==="
module load "${CUDA_MODULE}"
conda run -p "${ENV_PREFIX}" \
  pip install image-reward
# image-reward implicitly depends on OpenAI CLIP but doesn't declare it
conda run -p "${ENV_PREFIX}" \
  pip install git+https://github.com/openai/CLIP.git

echo "=== Step 5: Verify imports ==="
conda run -p "${ENV_PREFIX}" \
  python -c "
import torch
print(f'torch={torch.__version__}, CUDA available={torch.cuda.is_available()}')
import transformers; print(f'transformers={transformers.__version__}')
import diffusers; print(f'diffusers={diffusers.__version__}')
import ImageReward; print('ImageReward OK')
import datasets; print('datasets OK')
print('All imports OK')
"

echo "=== Done ==="
echo "Activate with: conda activate ${ENV_PREFIX}"
