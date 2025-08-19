#!/bin/bash
# A script, re-forged by Hiei, to build your environment from scorched earth.
# This is the final, ultimate form. All conflicts have been resolved.
set -e

# --- Static Definitions ---
VENV_DIR="lora_env"
REPO_DIR="kohya-trainer"

# --- Stage 0: The Purge ---
echo "--- The Purge: Annihilating any previous attempt ---"
if [ -d "$VENV_DIR" ]; then
    echo "Obliterating the old forge at '$VENV_DIR'..."
    rm -rf "$VENV_DIR"
fi
if [ -d "$HOME/.cache/pip" ]; then
    echo "Burning the tainted cache..."
    rm -rf "$HOME/.cache/pip"
fi
echo "The ground has been salted. Nothing remains."


# --- Stage 1: The Forging Ritual Begins ---
echo ""
echo "Hmph. Now, the ultimate forging ritual begins..."

# --- Stage 2: Summoning the System's Power ---
echo "Summoning the spirits of Python and CUDA..."
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.3.0
echo "Spirits have been summoned."


# --- Stage 3: Forging the Sanctuary ---
echo "Forging a new sanctuary: '$VENV_DIR'..."
virtualenv "$VENV_DIR"
echo "The sanctuary is built."


# --- Stage 4: Binding the Entire Legion ---
echo "Activating sanctuary to bind the full legion of tools..."
source "$VENV_DIR/bin/activate"
pip install --upgrade pip

# --- Stage 4A: Binding the Core Demons by True Name ---
echo "Binding the core demons by their true names to ensure harmony..."
pip install -q "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1" --extra-index-url https://download.pytorch.org/whl/cu121

# --- Stage 4B: Subjugating the NumPy Spirit ---
# The edict is restored. This MUST come before the manifest is read.
echo "Subjugating the NumPy spirit to its old form..."
pip install -q "numpy<2"

# --- Stage 4C: Binding the Lesser Spirits ---
echo "Binding the lesser demons via the master's manifest..."
cd "$REPO_DIR"
# This automatically performs the edit on requirements.txt
sed -i.bak 's/diffusers\[torch\]/diffusers/' requirements.txt
pip install -r requirements.txt
cd ..

echo "The entire legion has been bound to your will."
deactivate


# --- Final Word ---
echo ""
echo "THE FORGE IS COMPLETE."
echo "To enter it in the future, you need only use this command:"
echo "source $VENV_DIR/bin/activate"