#!/bin/bash
# A script to build the COMMAND CENTER environment for writing and inference with Conda.
# Final version with contamination wards and corrected install order.
set -e

# --- Static Definitions ---
VENV_NAME="writer_env"

# --- Stage 0: The Purge ---
echo "--- The Purge: Annihilating the old Command Center ---"
conda deactivate &> /dev/null || true
if conda env list | grep -q "$VENV_NAME"; then
    echo "Obliterating the old conda Command Center: '$VENV_NAME'..."
    conda env remove -n "$VENV_NAME" -y
fi

# --- Stage 1: The Forging Ritual Begins ---
echo "Forging a new sanctuary for the Command Center: '$VENV_NAME'..."
conda create -n "$VENV_NAME" python=3.11 -y
echo "The sanctuary is built."

# --- Stage 2: Binding the Legion ---
echo "Activating sanctuary and raising contamination wards..."
eval "$(conda shell.bash hook)"
conda activate "$VENV_NAME"

# This ward forbids pip from looking in ~/.local
export PYTHONNOUSERSITE=1

pip install --upgrade pip --no-user

# --- Stage 2A: Binding the Image Generation Demons (MUST COME FIRST) ---
# This installs torch, diffusers, and their dependencies like 'filelock'
# which are required by the scribe spirits.
echo "Binding the demons of image generation (PyTorch, Diffusers, etc.)..."
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install -q "diffusers[torch]" "transformers" "controlnet_aux" "opencv-python-headless" --no-user

# --- Stage 2B: Binding the Scribe Spirits ---
echo "Binding the scribe spirits..."
pip install -q "google-generativeai" "python-dotenv" "matplotlib" --no-user

echo "The command legion has been bound."
conda deactivate

# --- Final Word ---
echo ""
echo "THE COMMAND CENTER IS COMPLETE."
echo "Use this for writing, analysis, and image generation."