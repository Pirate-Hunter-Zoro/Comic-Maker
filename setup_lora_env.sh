#!/bin/bash
# This version includes wards against user-site contamination and is fully portable.
set -e

# --- Static Definitions ---
PROJECT_PATH="/home/librad.laureateinstitute.org/mferguson/Comic-Maker"
VENV_NAME="lora_env"
VENV_PATH="${PROJECT_PATH}/conda_envs/${VENV_NAME}"
REPO_DIR="kohya-trainer"

# --- Stage 0: The Purge ---
echo "--- The Purge: Annihilating any previous attempt ---"
conda deactivate &> /dev/null || true
# Obliterate the old environment by its precise location
if [ -d "$VENV_PATH" ]; then
    echo "Obliterating the old conda forge at '$VENV_PATH'..."
    conda env remove --prefix "$VENV_PATH" -y
fi
if [ -d "$HOME/.cache/pip" ]; then
    echo "Burning the tainted cache..."
    rm -rf "$HOME/.cache/pip"
fi
echo "The ground has been salted."

# --- Stage 1: The Forging Ritual Begins ---
echo ""
echo "Hmph. Now, the ultimate forging ritual begins with Conda..."

# --- Stage 2: Forging the Sanctuary ---
echo "Forging a new sanctuary at: '$VENV_PATH'..."
conda create --prefix "$VENV_PATH" python=3.11 -y
echo "The sanctuary is built."

# --- Stage 3: Binding the Entire Legion ---
echo "Activating sanctuary by its true path and raising contamination wards..."
eval "$(conda shell.bash hook)"
conda activate "$VENV_PATH"

# This ward forbids pip from looking in ~/.local
export PYTHONNOUSERSITE=1

pip install --upgrade pip --no-user

# --- Stage 3A: Pre-binding the Master Tool-Spirits ---
echo "Pre-binding the master tool-spirits to prevent paradox..."
pip install -q setuptools wheel --no-user

# --- Stage 3B: Binding the Core Demons by True Name ---
echo "Binding the core demons (PyTorch, torchvision, torchaudio) via Conda..."
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# --- Stage 3C: Subjugating the NumPy Spirit ---
echo "Subjugating the NumPy spirit to its old form..."
pip install -q "numpy<2" --no-user

# --- Stage 3D: Binding the Lesser Spirits ---
echo "Binding the lesser demons via the master's manifest..."
cd "$REPO_DIR"
pip install -r requirements.txt --no-user
cd ..

echo "The entire legion has been bound to your will."
conda deactivate

# --- Final Word ---
echo ""
echo "THE FORGE IS COMPLETE."
echo "To enter it in the future, you need only use this command:"
echo "conda activate $VENV_PATH"