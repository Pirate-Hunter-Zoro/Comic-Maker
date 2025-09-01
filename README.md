# Comic-Maker

A pipeline for training **LoRA models** per character and using them to generate comic panels.

---

## Environments

Two Conda envs:

- **drawer_env** → inference, image synthesis, and preview  
  Built by [`setup_drawer_env.sh`](setup_drawer_env.sh)

- **lora_env** → LoRA training with kohya-trainer  
  Built by [`setup_lora_env.sh`](setup_lora_env.sh)

### Build envs

```bash
bash setup_drawer_env.sh
bash setup_lora_env.sh
```

---

## Workflow

### 0. Project layout

```
projects/
  rwby_post_ever_after/
    data/
      datasets/
        blake_anchor/       # anchor image(s) for Blake
        blake_synth/        # synthetic dataset output
      plots/                # previews / contact sheets
    configs/
      blake_dataset.config  # kohya dataset config (auto-generated)
    prompts/
      blake.json            # prompt recipe (JSON, preferred) or .txt
    loras/
      # trained weights will be saved in cluster OUT_ROOT
```

---

### 1. Place anchor

Put your clean base image in:

```
projects/<UNIVERSE>/data/datasets/<CHAR>_anchor/<CHAR>_001.png
```

Examples:
```
projects/rwby_post_ever_after/data/datasets/blake_anchor/blake_001.png
projects/rwby_post_ever_after/data/datasets/weiss_anchor/weiss_001.png
projects/rwby_post_ever_after/data/datasets/ruby_anchor/ruby_001.png
```

---

### 2. Prepare dataset (synthesize + config)

Run the prep script:

```bash
bash scripts/prep_dataset.sh
```

This:
1. Activates **drawer_env**, runs `make_synth.py` to generate images+captions.
   - Uses prompts from `projects/<UNIVERSE>/prompts/<CHAR>.json` if present.  
   - Falls back to `<CHAR>.txt`.  
   - Else uses built-in defaults.
2. Activates **lora_env**, runs `make_dataset_config.py` to generate a kohya dataset config.

Outputs:
- Synthetic set: `projects/<UNIVERSE>/data/datasets/<CHAR>_synth/`
- Config: `projects/<UNIVERSE>/configs/<CHAR>_dataset.config`

#### Examples

**Blake**
```bash
CHAR=blake bash scripts/prep_dataset.sh
```

**Weiss**
```bash
CHAR=weiss ANCHOR=projects/rwby_post_ever_after/data/datasets/weiss_anchor/weiss_001.png SYNTH_DIR=projects/rwby_post_ever_after/data/datasets/weiss_synth bash scripts/prep_dataset.sh
```

**Ruby**
```bash
CHAR=ruby ANCHOR=projects/rwby_post_ever_after/data/datasets/ruby_anchor/ruby_001.png SYNTH_DIR=projects/rwby_post_ever_after/data/datasets/ruby_synth bash scripts/prep_dataset.sh
```

---

### 3. Train LoRA

Submit to Slurm:

```bash
# export env vars
export CHAR=blake UNIVERSE=rwby_post_ever_after
export BASE=runwayml/stable-diffusion-v1-5
export PROJECT_ROOT=/home/librad.laureateinstitute.org/mferguson/Comic-Maker
export LORA_ENV=$PROJECT_ROOT/conda_envs/lora_env
export OUT_ROOT=/media/studies/ehr_study/data-EHR-prepped/Comic-Maker/loras

sbatch submit_training.ssub
```

Weights will appear in:

```
${OUT_ROOT}/${UNIVERSE}/${CHAR}/${CHAR}_lora_v1.safetensors
```

---

### 4. Preview LoRA

Submit preview job:

```bash
export CHAR=blake UNIVERSE=rwby_post_ever_after
export BASE=runwayml/stable-diffusion-v1-5
export PROJECT_ROOT=/home/.../Comic-Maker
export DRAWER_ENV=$PROJECT_ROOT/conda_envs/drawer_env
export OUT_ROOT=/media/studies/ehr_study/data-EHR-prepped/Comic-Maker/loras

sbatch preview_lora.ssub
```

Preview images land in:

```
projects/<UNIVERSE>/data/plots/<CHAR>_preview/
```

---

## Prompt recipes

- **TXT** → simple, one prompt per line
- **JSON** → structured, supports:
  - `global`: defaults (`count`, `steps`, `cfg`, `negatives`, `denoise_strengths`)
  - `buckets`: groups like `portrait`, `action`, `wide`, each with `weight` or `count`, `prompts`, optional `size`, negatives, etc.
  - `{char}` is substituted with character key

Example `projects/rwby_post_ever_after/prompts/blake.json`:

```json
{
  "global": {
    "count": 320,
    "steps": 28,
    "cfg": 6.0,
    "negatives": ["lowres", "blurry", "watermark"],
    "denoise_strengths": [0.35, 0.45, 0.55]
  },
  "buckets": {
    "portrait": {
      "weight": 0.5,
      "prompts": [
        "close-up, {char} with cat ears, crisp inks, halftone shading",
        "portrait, {char}, rim light, high fidelity anime"
      ]
    },
    "action": {
      "weight": 0.35,
      "prompts": [
        "dynamic action panel, {char}, dramatic lighting, motion lines",
        "full body, {char} mid-leap, smoky background, comic shading"
      ]
    },
    "wide": {
      "weight": 0.15,
      "prompts": [
        "wide panel, {char} in ruined city, cinematic composition"
      ],
      "size": [1216, 832]
    }
  }
}
```

---

## Quick sanity check

After dataset prep:

```bash
# count files
ls projects/rwby_post_ever_after/data/datasets/blake_synth | wc -l

# contact sheet preview
conda activate $PROJECT_ROOT/conda_envs/drawer_env
python scripts/make_contact_sheet.py   --src projects/rwby_post_ever_after/data/datasets/blake_synth   --out projects/rwby_post_ever_after/data/plots/blake_synth_grid.png   --max 30 --cols 6 --thumb 224
```

---

## Roadmap

- [x] Per-character LoRA pipeline  
- [x] Prompt buckets with weights for comic-style variety  
- [ ] Wrapper to chain dataset prep + training in one step  
- [ ] Page layout generator for assembling panels into comic pages  
- [ ] Shot list → prompts linkage from plot scripts

---
