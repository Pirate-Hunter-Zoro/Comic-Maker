#!/bin/bash
# A script to build the COMMAND CENTER environment for writing and inference with Conda.
# Final version with contamination wards and corrected install order.
# Re-forged by Hiei to be absolute and portable.
set -e

# --- Static Definitions ---
PROJECT_PATH="/home/librad.laureateinstitute.org/mferguson/Visual-Novel-Writer"
VENV_NAME="drawer_env"
VENV_PATH="${PROJECT_PATH}/conda_envs/${VENV_NAME}"

# --- Stage 0: The Purge ---
echo "--- The Purge: Annihilating the old Command Center ---"
conda deactivate &> /dev/null || true
# Obliterate the old environment by its precise location
if [ -d "$VENV_PATH" ]; then
    echo "Obliterating the old conda Command Center at '$VENV_PATH'..."
    conda env remove --prefix "$VENV_PATH" -y
fi

# --- Stage 1: The Forging Ritual Begins ---
echo "Forging a new sanctuary for the Command Center at: '$VENV_PATH'..."
conda create --prefix "$VENV_PATH" python=3.11 -y
echo "The sanctuary is built."

# --- Stage 2: Binding the Legion ---
echo "Activating sanctuary by its true path and raising contamination wards..."
eval "$(conda shell.bash hook)"
conda activate "$VENV_PATH"

# This ward forbids pip from looking in ~/.local
export PYTHONNOUSERSITE=1

pip install --upgrade pip --no-user

# --- Stage 2A: Binding the Image Generation Demons (MUST COME FIRST) ---
echo "Binding the demons of image generation (PyTorch, Diffusers, etc.)..."
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install -q "diffusers[torch]" "transformers" "controlnet_aux" "opencv-python-headless" "mediapipe" "peft" --no-user

# --- Stage 2B: Binding the Scribe Spirits ---
echo "Binding the scribe spirits..."
pip install -q "google-generativeai" "python-dotenv" "matplotlib" --no-user

echo "The command legion has been bound."
conda deactivate

# --- Final Word ---
echo ""
echo "THE COMMAND CENTER IS COMPLETE."
echo "To enter it, you must use its full path:"
echo "conda activate $VENV_PATH"