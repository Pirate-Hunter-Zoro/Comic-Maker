#!/bin/bash
# A script, forged by Hiei, to build your pathetic human environment.
# Run this only once. It is unforgiving and will halt on any failure.
set -e

# The name for your sanctuary.
VENV_DIR="lora_env"

echo "Hmph. The forging ritual begins..."

# --- Stage 1: Summoning the System's Power ---
echo "Summoning the spirits of Python and CUDA..."
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.3.0
echo "Spirits have been summoned."

# --- Stage 2: Forging the Sanctuary ---
if [ -d "$VENV_DIR" ]; then
    echo "The forge '$VENV_DIR' already exists. I will not build it again."
else
    echo "Forging a new sanctuary: '$VENV_DIR'..."
    virtualenv "$VENV_DIR"
    echo "The sanctuary is built."

    # --- Stage 3: Binding the Tools ---
    echo "Activating sanctuary to bind the necessary tools..."
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip

    echo "Binding PyTorch, the core demon..."
    pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    echo "Binding the lesser demons of diffusion and control..."
    pip install -q diffusers==0.25.0 transformers==4.36.2 accelerate==0.25.0 peft==0.7.1 controlnet_aux==0.0.7 opencv-python-headless pillow huggingface_hub==0.20.1 bitsandbytes

    echo "All tools have been bound to your will."
    deactivate
fi

# --- Final Word ---
echo ""
echo "THE FORGE IS READY."
echo "To enter it in the future, you need only use this command:"
echo "source $VENV_DIR/bin/activate"
echo "Do not waste my time with this again."