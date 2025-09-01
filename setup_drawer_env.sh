#!/usr/bin/env bash
# Minimal env for USING LoRAs (diffusers img2img, synth generation, previews)
set -euo pipefail

# Resolve repo root (directory containing this script)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR" && pwd)"
ENV_PATH="${PROJECT_ROOT}/conda_envs/drawer_env"

echo "Repo root: ${PROJECT_ROOT}"
echo "Env path:  ${ENV_PATH}"

# Fresh env
conda deactivate &>/dev/null || true
if [[ -d "${ENV_PATH}" ]]; then
  echo "--- Removing old drawer_env ---"
  conda env remove --prefix "${ENV_PATH}" -y
fi

echo "--- Creating conda env (py311) ---"
conda create --prefix "${ENV_PATH}" -y python=3.11

# Activate
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PATH}"

echo "--- Installing PyTorch CUDA 12.1 ---"
conda install -y -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.1

echo "--- Installing Python deps ---"
python -m pip install --upgrade pip
python -m pip install \
  diffusers==0.31.0 \
  transformers==4.52.4 \
  accelerate==1.7.0 \
  safetensors==0.5.3 \
  pillow==11.2.1

# Optional: xformers (ignore if wheel not available for your CUDA/PyTorch combo)
python - <<'PY'
try:
  import subprocess, sys
  subprocess.check_call([sys.executable, "-m", "pip", "install", "xformers==0.0.28.post3"])
  print("xformers installed")
except Exception as e:
  print("xformers skipped:", e)
PY

echo "--- Verify core imports ---"
python - <<'PY'
import torch, PIL, diffusers, transformers, accelerate, safetensors
print("torch", torch.__version__, "CUDA", torch.version.cuda, "cuda?", torch.cuda.is_available())
print("PIL OK:", hasattr(PIL, "__version__"))
print("diffusers:", diffusers.__version__, "transformers:", transformers.__version__)
PY

conda deactivate
echo "DONE. Activate with: conda activate ${ENV_PATH}"
