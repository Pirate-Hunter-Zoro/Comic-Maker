#!/usr/bin/env bash
# Prep dataset for LoRA training: synthesize + config
set -euo pipefail

# ====== EDIT DEFAULTS OR PASS AS ENVs ======
: "${PROJECT_ROOT:=/home/librad.laureateinstitute.org/mferguson/Comic-Maker}"
: "${UNIVERSE:=rwby_post_ever_after}"
: "${CHAR:=blake}"

# Anchor and output dataset folders (relative to project root)
: "${ANCHOR:=projects/${UNIVERSE}/data/datasets/${CHAR}_anchor/${CHAR}_001.png}"
: "${SYNTH_DIR:=projects/${UNIVERSE}/data/datasets/${CHAR}_synth}"

# Synthesis params
: "${BASE:=runwayml/stable-diffusion-v1-5}"
: "${N:=300}"; : "${W:=768}"; : "${H:=1024}"
: "${STEPS:=28}"; : "${CFG:=6.0}"; : "${SEED:=123}"

# ====== ENV PATHS ======
DRAWER_ENV="${PROJECT_ROOT}/conda_envs/drawer_env"
LORA_ENV="${PROJECT_ROOT}/conda_envs/lora_env"

# ====== 1) Synth images + captions ======
echo "[1/2] Synthesizing dataset for $CHAR in $DRAWER_ENV"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$DRAWER_ENV"

PROMPTS_FILE=""
JSON_CAND="${PROJECT_ROOT}/projects/${UNIVERSE}/prompts/${CHAR}.json"
TXT_CAND="${PROJECT_ROOT}/projects/${UNIVERSE}/prompts/${CHAR}.txt"
if [[ -f "$JSON_CAND" ]]; then
  PROMPTS_FILE="$JSON_CAND"
elif [[ -f "$TXT_CAND" ]]; then
  PROMPTS_FILE="$TXT_CAND"
fi

python "${PROJECT_ROOT}/scripts/make_synth.py" \
  --character "$CHAR" \
  --base "$BASE" \
  --anchor "${PROJECT_ROOT}/${ANCHOR}" \
  --outdir "${PROJECT_ROOT}/${SYNTH_DIR}" \
  --n "$N" --w "$W" --h "$H" --steps "$STEPS" --cfg "$CFG" --seed "$SEED" \
  ${PROMPTS_FILE:+--prompts-file "$PROMPTS_FILE"}

conda deactivate

# ====== 2) Generate dataset config ======
echo "[2/2] Generating kohya dataset config in $LORA_ENV"
conda activate "$LORA_ENV"

python "${PROJECT_ROOT}/scripts/make_dataset_config.py" \
  --project-root "$PROJECT_ROOT" \
  --universe "$UNIVERSE" \
  --character "$CHAR" \
  --image-dir "$SYNTH_DIR"

conda deactivate

echo "Dataset ready: ${PROJECT_ROOT}/${SYNTH_DIR}"
echo "Config: ${PROJECT_ROOT}/projects/${UNIVERSE}/configs/${CHAR}_dataset.config"
