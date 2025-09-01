#!/bin/bash
# Minimal environment for USING LoRA models with Diffusers + PyTorch
set -e

# --- Static Definitions ---
PROJECT_PATH="/home/librad.laureateinstitute.org/mferguson/Visual-Novel-Writer"
VENV_NAME="drawer_env"
VENV_PATH="${PROJECT_PATH}/conda_envs/${VENV_NAME}"

# --- Stage 0: The Purge ---
echo "--- Nuking any old drawer env ---"
conda deactivate &> /dev/null || true
if [ -d "$VENV_PATH" ]; then
    echo "Removing old env at '$VENV_PATH'..."
    conda env remove --prefix "$VENV_PATH" -y
fi

# --- Stage 1: Create fresh env ---
echo "Creating new env at: '$VENV_PATH'..."
conda create --prefix "$VENV_PATH" python=3.11 -y
echo "Base env created."

# --- Stage 2: Activate & block user site contamination ---
eval "$(conda shell.bash hook)"
conda activate "$VENV_PATH"
export PYTHONNOUSERSITE=1

# --- Stage 3: Core installs ---
echo "Upgrading pip..."
pip install --upgrade pip --no-user

echo "Installing PyTorch with CUDA 12.1..."
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

echo "Installing minimal runtime packages..."
pip install --no-user -q \
  "diffusers>=0.28" \
  "transformers>=4.44" \
  safetensors \
  accelerate \
  pillow

# optional: speedups (comment out if not needed)
# pip install --no-user xformers

# --- Wrap up ---
conda deactivate
echo ""
echo "DRAWER ENV COMPLETE."
echo "Activate with:"
echo "  conda activate $VENV_PATH"
