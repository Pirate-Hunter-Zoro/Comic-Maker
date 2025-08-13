#!/bin/bash
# A script, re-forged by Hiei, to build your environment from scorched earth.
# It is unforgiving. It will first destroy, then create.
set -e

# The name for your sanctuary.
VENV_DIR="lora_env"

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
echo "Hmph. Now, the forging ritual begins..."

# --- Stage 2: Summoning the System's Power ---
echo "Summoning the spirits of Python and CUDA..."
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.3.0
echo "Spirits have been summoned."


# --- Stage 3: Forging the Sanctuary ---
echo "Forging a new sanctuary: '$VENV_DIR'..."
virtualenv "$VENV_DIR"
echo "The sanctuary is built."


# --- Stage 4: Binding the Tools ---
echo "Activating sanctuary to bind the necessary tools..."
source "$VENV_DIR/bin/activate"
pip install --upgrade pip

echo "Binding PyTorch, the core demon..."
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "Binding the lesser demons of diffusion and control..."
pip install -q diffusers==0.25.0 transformers==4.36.2 accelerate==0.25.0 peft==0.7.1 controlnet_aux==0.0.7 opencv-python-headless pillow huggingface_hub==0.20.1 bitsandbytes toml

echo "All tools have been bound to your will."
deactivate


# --- Final Word ---
echo ""
echo "THE FORGE IS READY."
echo "To enter it in the future, you need only use this command:"
echo "source $VENV_DIR/bin/activate"