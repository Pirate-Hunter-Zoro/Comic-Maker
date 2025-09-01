#!/usr/bin/env bash
# LoRA training env (kohya)
set -euo pipefail

# ---------- config ----------
PY_VER=3.10
TORCH_CHANNELS=(-c pytorch -c nvidia)
TORCH_PKGS=(pytorch torchvision torchaudio pytorch-cuda=12.1)
PIP_DEPS=(
  "numpy<2"
  safetensors==0.5.3
  accelerate==1.7.0
  transformers==4.52.4
  bitsandbytes==0.42.0
  Pillow==10.4.0
  peft==0.13.2
  tqdm==4.67.1
  datasets==3.1.0
  toml==0.10.2
)

# ---------- paths ----------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR" && pwd)"
ENV_PATH="${PROJECT_ROOT}/conda_envs/lora_env"
KOHYA_DIR="${PROJECT_ROOT}/kohya-trainer"
PKG_CACHE="${PROJECT_ROOT}/.conda_pkgs_cache"
mkdir -p "${PKG_CACHE}" "$(dirname "${ENV_PATH}")"
export CONDA_PKGS_DIRS="${PKG_CACHE}"

# ---------- pick solver ----------
if command -v micromamba >/dev/null 2>&1; then
  SOLVER="micromamba"
  eval "$(micromamba shell hook -s bash)"
  CREATE(){ micromamba create -y -p "$ENV_PATH" "python=${PY_VER}"; }
  INSTALL_CONDA(){ micromamba install -y -p "$ENV_PATH" "${TORCH_CHANNELS[@]}" "${TORCH_PKGS[@]}"; }
elif command -v mamba >/dev/null 2>&1; then
  SOLVER="mamba"
  source "$(conda info --base)/etc/profile.d/conda.sh"
  CREATE(){ mamba create --prefix "$ENV_PATH" -y "python=${PY_VER}"; }
  INSTALL_CONDA(){ mamba install -y -p "$ENV_PATH" "${TORCH_CHANNELS[@]}" "${TORCH_PKGS[@]}"; }
else
  SOLVER="conda"
  source "$(conda info --base)/etc/profile.d/conda.sh"
  CREATE(){ conda create --prefix "$ENV_PATH" -y "python=${PY_VER}"; }
  INSTALL_CONDA(){ conda install -y -p "$ENV_PATH" "${TORCH_CHANNELS[@]}" "${TORCH_PKGS[@]}"; }
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
INSTALL_CONDA

echo "[=] Ensuring kohya deps via pip..."
python -m pip install --upgrade pip
python -m pip install "${PIP_DEPS[@]}"

if [[ -d "${KOHYA_DIR}" ]]; then
  echo "[=] Installing local kohya-trainer (editable)..."
  python -m pip install -e "${KOHYA_DIR}"
else
  echo "[!] kohya-trainer not found at ${KOHYA_DIR}; skipping."
fi

# ---------- verify ----------
python - <<'PY'
import torch, transformers, accelerate, safetensors, PIL, bitsandbytes, peft
print("torch", torch.__version__, "CUDA", torch.version.cuda, "cuda?", torch.cuda.is_available())
print("transformers", transformers.__version__, "accelerate", accelerate.__version__)
PY

# deactivate
if [[ "${SOLVER}" == "micromamba" ]]; then
  micromamba deactivate
else
  conda deactivate
fi
echo "[✓] lora_env ready. Activate with: ${SOLVER} activate ${ENV_PATH}"
