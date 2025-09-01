#!/usr/bin/env bash
# Minimal env for USING LoRAs (diffusers img2img, synth generation, previews)
# OPTION B: keep xformers by satisfying Triton + setuptools inside the env
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

echo "--- Installing core Python deps (diffusers stack) ---"
python -m pip install --upgrade pip
python -m pip install \
  diffusers==0.31.0 \
  transformers==4.52.4 \
  accelerate==1.7.0 \
  safetensors==0.5.3 \
  pillow==11.2.1

echo "--- OPTION B: satisfy Triton/xformers toolchain INSIDE env ---"
# Keep build tooling internal to env (avoid ~/.local interference)
python -m pip install "setuptools<75" wheel jaraco.functools

# Pin a Triton that works with CUDA 12.x + recent PyTorch
python -m pip install triton==2.2.0

# Install xformers after Triton is present
python -m pip install xformers==0.0.28.post3

echo "--- Verify core imports (avoid user-site during test) ---"
export PYTHONNOUSERSITE=1
python - <<'PY'
import sys, pkgutil
import torch, PIL, diffusers, transformers, accelerate, safetensors
print("torch", torch.__version__, "CUDA", torch.version.cuda, "cuda?", torch.cuda.is_available())
print("PIL OK:", hasattr(PIL, "__version__"))
print("diffusers:", diffusers.__version__, "transformers:", transformers.__version__)
# Verify Triton/xformers
try:
    import triton, xformers
    print("triton:", getattr(triton, "__version__", "unknown"))
    print("xformers:", getattr(xformers, "__version__", "unknown"))
    # Sanity: ensure they came from the env, not ~/.local
    print("triton path:", triton.__file__)
    print("xformers path:", xformers.__file__)
except Exception as e:
    print("Triton/xformers import issue:", e)
PY

conda deactivate
echo "DONE. Activate with: conda activate ${ENV_PATH}"
