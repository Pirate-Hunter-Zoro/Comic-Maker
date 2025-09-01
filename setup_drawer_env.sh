#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="/home/librad.laureateinstitute.org/mferguson/Visual-Novel-Writer/conda_envs/drawer_env"
PY_VER="3.11"

# Always prefer the env's site over ~/.local
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ ! -d "$ENV_DIR" ]]; then
  conda create -y -p "$ENV_DIR" python=${PY_VER}
fi
conda activate "$ENV_DIR"

echo "Python $(python -V 2>&1)"
echo "$(pip --version)"

echo "--- Installing PyTorch + CUDA 12.1 toolchain ---"
conda install -y -c pytorch -c nvidia -c defaults \
  pytorch torchvision torchaudio pytorch-cuda=12.1

echo "--- Installing base Python build tooling ---"
# Conservative pins to avoid setuptools regressions in build isolation
pip install --upgrade "pip<25.3" "setuptools<75" wheel build

# Constraint to keep mediapipe compatible
CONSTRAINTS_FILE="$(mktemp)"
cat > "$CONSTRAINTS_FILE" <<'TXT'
protobuf<5,>=4.25.3
TXT

REQ_FILE="./requirements.txt"

# Try to satisfy bitmath via conda first (avoids sdist builds)
echo "--- Ensuring bitmath is available ---"
if ! python -c "import bitmath" >/dev/null 2>&1; then
  if conda install -y -c conda-forge bitmath=1.3.3.1; then
    echo "bitmath installed via conda-forge."
  else
    echo "conda-forge bitmath not available; falling back to pip build."
    pip install "jaraco.functools"  # setuptools sometimes expects it
  fi
fi

echo "--- Installing project requirements ---"
if [[ -f "$REQ_FILE" ]]; then
  pip install -r "$REQ_FILE" -c "$CONSTRAINTS_FILE"
else
  pip install -c "$CONSTRAINTS_FILE" \
    diffusers==0.31.* transformers>=4.43 accelerate>=0.33 safetensors \
    einops pillow pyyaml
fi

# If you use mediapipe during drawing, make the combo explicit.
if grep -qi '^ *mediapipe' "${REQ_FILE:-/dev/null}"; then
  pip install "mediapipe==0.10.14" --upgrade --no-deps
  pip install "protobuf>=4.25.3,<5" --upgrade --force-reinstall
fi

echo "--- Verifying ---"
pip check || true
python - <<'PY'
import torch, platform, sys
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "CUDA available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
print("python:", platform.python_version())
print("no_user_site:", bool(getattr(sys, 'flags', None)) or "PYTHONNOUSERSITE" in __import__('os').environ)
PY

echo
echo "THE COMMAND CENTER IS COMPLETE."
echo "To enter it, use:"
echo "  conda activate ${ENV_DIR}"
