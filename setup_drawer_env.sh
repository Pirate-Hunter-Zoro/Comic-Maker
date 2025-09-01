#!/usr/bin/env bash
# Writer env: diffusers/transformers/accelerate WITH xformers (pins Triton>=3.0.0)
set -euo pipefail

# ---------- config ----------
PY_VER=3.11
TORCH_CHANNELS=(-c pytorch -c nvidia)
TORCH_PKGS=(pytorch torchvision torchaudio pytorch-cuda=12.1)
PIP_CORE_DEPS=(
  diffusers==0.31.0
  transformers==4.52.4
  accelerate==1.7.0
  safetensors==0.5.3
  pillow==11.2.1
)
TRITON_VER=3.0.0
XFORMERS_VER=0.0.28.post3   # upgrade to 0.0.29.post1 if needed for your combo

# ---------- paths ----------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR" && pwd)"
ENV_PATH="${PROJECT_ROOT}/conda_envs/drawer_env"
PKG_CACHE="${PROJECT_ROOT}/.conda_pkgs_cache"
mkdir -p "${PKG_CACHE}" "$(dirname "${ENV_PATH}")"
export CONDA_PKGS_DIRS="${PKG_CACHE}"

# ---------- pick solver ----------
if command -v micromamba >/dev/null 2>&1; then
  SOLVER="micromamba"
  eval "$(micromamba shell hook -s bash)"
  CREATE(){ micromamba create -y -p "$ENV_PATH" "python=${PY_VER}"; }
  INSTALL_TORCH(){ micromamba install -y -p "$ENV_PATH" "${TORCH_CHANNELS[@]}" "${TORCH_PKGS[@]}"; }
elif command -v mamba >/dev/null 2>&1; then
  SOLVER="mamba"
  source "$(conda info --base)/etc/profile.d/conda.sh"
  CREATE(){ mamba create --prefix "$ENV_PATH" -y "python=${PY_VER}"; }
  INSTALL_TORCH(){ mamba install -y -p "$ENV_PATH" "${TORCH_CHANNELS[@]}" "${TORCH_PKGS[@]}"; }
else
  SOLVER="conda"
  source "$(conda info --base)/etc/profile.d/conda.sh"
  CREATE(){ conda create --prefix "$ENV_PATH" -y "python=${PY_VER}"; }
  INSTALL_TORCH(){ conda install -y -p "$ENV_PATH" "${TORCH_CHANNELS[@]}" "${TORCH_PKGS[@]}"; }
fi
echo "[i] Using solver: ${SOLVER}"

# ---------- create or reuse env ----------
if [[ -d "${ENV_PATH}" ]]; then
  echo "[i] Reusing env: ${ENV_PATH}"
else
  echo "[+] Creating env at: ${ENV_PATH}"
  CREATE
fi

# activate
if [[ "${SOLVER}" == "micromamba" ]]; then
  micromamba activate "${ENV_PATH}"
else
  conda activate "${ENV_PATH}"
fi

# ---------- installs ----------
echo "[=] Ensuring PyTorch/CUDA stack..."
INSTALL_TORCH

echo "[=] Ensuring Python deps via pip..."
python -m pip install --upgrade pip
python -m pip install "${PIP_CORE_DEPS[@]}"

echo "[=] Triton + xformers (compatible pair)..."
# Keep build tooling inside the env
python -m pip install "setuptools<75" wheel jaraco.functools
# Make sure any previous xformers/triton are cleaned to avoid ABI/API mismatch
python -m pip uninstall -y xformers triton || true
# Triton with use_cuda_graph support
python -m pip install "triton==${TRITON_VER}"
# xformers; if wheel isn’t available for your exact PyTorch/CUDA/Python combo, we keep going without it
python -m pip install "xformers==${XFORMERS_VER}" || {
  echo "[!] xformers wheel unavailable for this combo; continuing without it."
}

# ---------- verify ----------
echo "[=] Verifying (PYTHONNOUSERSITE=1; try to import xformers)..."
export PYTHONNOUSERSITE=1
python - <<'PY'
import torch, diffusers, transformers, accelerate, PIL, safetensors
print("torch", torch.__version__, "CUDA", torch.version.cuda, "cuda?", torch.cuda.is_available())
print("diffusers", diffusers.__version__, "transformers", transformers.__version__)
try:
    import triton, xformers
    print("triton", getattr(triton, "__version__", "n/a"))
    print("xformers", getattr(xformers, "__version__", "n/a"))
except Exception as e:
    print("xformers/triton not active:", e)
PY

# deactivate
if [[ "${SOLVER}" == "micromamba" ]]; then
  micromamba deactivate
else
  conda deactivate
fi
echo "[✓] drawer_env ready (with Triton/xformers if available). Activate with: ${SOLVER} activate ${ENV_PATH}"
