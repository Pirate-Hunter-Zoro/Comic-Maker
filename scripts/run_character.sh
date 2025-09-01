#!/usr/bin/env bash
# Run end-to-end for any character: synth -> config -> sbatch training
set -euo pipefail

# ====== EDIT THESE DEFAULTS OR PASS AS ENVs ======
: "${PROJECT_ROOT:=/home/librad.laureateinstitute.org/mferguson/Comic-Maker}"
: "${DRAWER_ENV:=${PROJECT_ROOT}/conda_envs/drawer_env}"
: "${LORA_ENV:=${PROJECT_ROOT}/conda_envs/lora_env}"
: "${UNIVERSE:=rwby_post_ever_after}"
: "${CHAR:=blake}"
: "${BASE:=runwayml/stable-diffusion-v1-5}"

# Anchor and output dataset folders (relative to project root)
: "${ANCHOR:=projects/${UNIVERSE}/data/datasets/${CHAR}_anchor/${CHAR}_001.png}"
: "${SYNTH_DIR:=projects/${UNIVERSE}/data/datasets/${CHAR}_synth}"

# Optional prompts file auto-detected unless PROMPTS_FILE is set
: "${PROMPTS_FILE:=auto}"

# Synthesis params
: "${N:=300}"; : "${W:=768}"; : "${H:=1024}"
: "${STEPS:=28}"; : "${CFG:=6.0}"; : "${SEED:=123}"

# Training outputs (cluster storage)
: "${OUT_ROOT:=/media/studies/ehr_study/data-EHR-prepped/Comic-Maker/loras}"
# LoRA hparams
: "${NETWORK_DIM:=16}"; : "${NETWORK_ALPHA:=16}"
: "${MAX_STEPS:=8000}"; : "${BS:=2}"; : "${LR:=1e-4}"

echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "UNIVERSE=$UNIVERSE  CHAR=$CHAR  BASE=$BASE"

# ====== 0) Sanity ======
test -f "${PROJECT_ROOT}/${ANCHOR}" || { echo "Missing anchor: ${PROJECT_ROOT}/${ANCHOR}"; exit 1; }

# Optional prompts file auto-detected unless PROMPTS_FILE is set
: "${PROMPTS_FILE:=auto}"

# Resolve prompts file
if [[ "${PROMPTS_FILE}" == "auto" ]]; then
  JSON_CAND="${PROJECT_ROOT}/projects/${UNIVERSE}/prompts/${CHAR}.json"
  TXT_CAND="${PROJECT_ROOT}/projects/${UNIVERSE}/prompts/${CHAR}.txt"
  if [[ -f "$JSON_CAND" ]]; then
    PROMPTS_ARG=(--prompts-file "$JSON_CAND"); echo "Using prompts: $JSON_CAND"
  elif [[ -f "$TXT_CAND" ]]; then
    PROMPTS_ARG=(--prompts-file "$TXT_CAND"); echo "Using prompts: $TXT_CAND"
  else
    PROMPTS_ARG=(); echo "No prompts file found (using built-in defaults)."
  fi
elif [[ -n "${PROMPTS_FILE}" ]]; then
  PROMPTS_ARG=(--prompts-file "${PROMPTS_FILE}"); echo "Using prompts: ${PROMPTS_FILE}"
else
  PROMPTS_ARG=(); echo "Prompts disabled (using built-in defaults)."
fi

# ====== 1) Synthesize dataset ======
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$DRAWER_ENV"

python "${PROJECT_ROOT}/scripts/make_synth.py" \
  --character "$CHAR" \
  --base "$BASE" \
  --anchor "${PROJECT_ROOT}/${ANCHOR}" \
  --outdir "${PROJECT_ROOT}/${SYNTH_DIR}" \
  --n "$N" --w "$W" --h "$H" --steps "$STEPS" --cfg "$CFG" --seed "$SEED" \
  "${PROMPTS_ARG[@]}"

# ====== 2) Generate kohya dataset config ======
python "${PROJECT_ROOT}/scripts/make_dataset_config.py" \
  --project-root "$PROJECT_ROOT" \
  --universe "$UNIVERSE" \
  --character "$CHAR" \
  --image-dir "$SYNTH_DIR"

# ====== 3) Submit training to Slurm ======
conda deactivate

export CHAR UNIVERSE BASE PROJECT_ROOT LORA_ENV OUT_ROOT \
       NETWORK_DIM NETWORK_ALPHA MAX_STEPS BS LR

cd "$PROJECT_ROOT"
sbatch submit_training.ssub

echo "Submitted training for ${CHAR} in ${UNIVERSE}."
