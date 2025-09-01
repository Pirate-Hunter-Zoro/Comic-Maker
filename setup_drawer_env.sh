#!/bin/bash
# Minimal environment for USING LoRA models with Diffusers + PyTorch + xformers
# (no mediapipe, no google junk—just image gen)
set -euo pipefail

# --- Static Definitions ---
PROJECT_PATH="/home/librad.laureateinstitute.org/mferguson/Comic-Maker"
VENV_NAME="drawer_env"
VENV_PATH="${PROJECT_PATH}/conda_envs/${VENV_NAME}"

echo "--- Nuking any old drawer env ---"
conda deactivate &>/dev/null || true
if [[ -d "$VENV_PATH" ]]; then
  echo "Removing old env at '$VENV_PATH'..."
  conda env remove --prefix "$VENV_PATH" -y
fi

echo "Creating new env at: '$VENV_PATH'..."
conda create --prefix "$VENV_PATH" python=3.11 -y
echo "Base env created."

# Activate & block user-site contamination
eval "$(conda shell.bash hook)"
conda activate "$VENV_PATH"
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing PyTorch (CUDA 12.1) via conda..."
conda install -y -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.1

echo "Installing minimal runtime packages (diffusers, transformers, etc.)..."
pip install -q \
  "diffusers>=0.28" \
  "transformers>=4.44" \
  safetensors \
  accelerate \
  pillow

echo "Attempting to install xformers (optional speedup)..."
if pip install -q xformers; then
  echo "xformers installed."
else
  echo "xformers wheel not available for this setup; skipping gracefully."
fi

echo "Verifying core stack..."
python - <<'PY'
import torch, pkgutil
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "CUDA available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
for p in ["diffusers","transformers","safetensors","accelerate","PIL","xformers"]:
    print(f"{p}:","OK" if pkgutil.find_loader(p if p!="PIL" else "PIL.Image") else "MISSING")
PY

conda deactivate
echo
echo "DRAWER ENV COMPLETE."
echo "Activate with:"
echo "  conda activate $VENV_PATH"
