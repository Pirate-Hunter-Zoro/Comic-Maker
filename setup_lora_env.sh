#!/usr/bin/env bash
# Env for TRAINING LoRAs (kohya-trainer)
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR" && pwd)"
ENV_PATH="${PROJECT_ROOT}/conda_envs/lora_env"
KOHYA_DIR="${PROJECT_ROOT}/kohya-trainer"

echo "Repo root: ${PROJECT_ROOT}"
echo "Env path:  ${ENV_PATH}"
echo "Kohya dir: ${KOHYA_DIR}"

conda deactivate &>/dev/null || true
if [[ -d "${ENV_PATH}" ]]; then
  echo "--- Removing old lora_env ---"
  conda env remove --prefix "${ENV_PATH}" -y
fi

echo "--- Creating conda env (py310) ---"
conda create --prefix "${ENV_PATH}" -y python=3.10

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PATH}"

echo "--- Installing PyTorch CUDA 12.1 ---"
conda install -y -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.1

echo "--- Installing kohya dependencies (minimal/safe) ---"
python -m pip install --upgrade pip
python -m pip install \
  "numpy<2" \
  safetensors==0.5.3 \
  accelerate==1.7.0 \
  transformers==4.52.4 \
  bitsandbytes==0.42.0 \
  Pillow==10.4.0 \
  peft==0.13.2 \
  tqdm==4.67.1 \
  datasets==3.1.0 \
  toml==0.10.2

# Install local kohya repo in editable mode
if [[ -d "${KOHYA_DIR}" ]]; then
  python -m pip install -e "${KOHYA_DIR}"
else
  echo "WARNING: ${KOHYA_DIR} not found. Skip kohya install."
fi

echo "--- Verify core imports ---"
python - <<'PY'
import torch, safetensors, accelerate, transformers, PIL, bitsandbytes, peft
print("torch", torch.__version__, "CUDA", torch.version.cuda, "cuda?", torch.cuda.is_available())
print("safetensors/accelerate/transformers OK")
PY

conda deactivate
echo "DONE. Activate with: conda activate ${ENV_PATH}"
