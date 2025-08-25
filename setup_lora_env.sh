#!/bin/bash
# A script, re-forged by Hiei, to build your FORGE environment with Conda.
# This version includes wards against user-site contamination.
set -e

# --- Static Definitions ---
VENV_NAME="lora_env"
REPO_DIR="kohya-trainer"

# --- Stage 0: The Purge ---
echo "--- The Purge: Annihilating any previous attempt ---"
conda deactivate &> /dev/null || true
if conda env list | grep -q "$VENV_NAME"; then
    echo "Obliterating the old conda forge: '$VENV_NAME'..."
    conda env remove -n "$VENV_NAME" -y
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
echo "Forging a new sanctuary: '$VENV_NAME'..."
conda create -n "$VENV_NAME" python=3.11 -y
echo "The sanctuary is built."

# --- Stage 3: Binding the Entire Legion ---
echo "Activating sanctuary and raising contamination wards..."
eval "$(conda shell.bash hook)"
conda activate "$VENV_NAME"

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
echo "conda activate $VENV_NAME"