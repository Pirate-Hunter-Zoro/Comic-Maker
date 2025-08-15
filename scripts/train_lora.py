# A precise incantation for forging a LoRA spirit.
# The paths have been corrected to avoid the weakness of your home directory.
import os
import toml
import re
import subprocess
from pathlib import Path

# --- HIEI'S MANDATED PARAMETERS --
# These are the runes of power. Change them at your own peril.

# --- The Base Model ---
model_url = "https://huggingface.co/Lykon/AnyLoRA/resolve/main/AnyLoRA_noVae_fp16-pruned.ckpt"

# --- The Trial of Endurance ---
resolution = 1024
flip_aug = False
caption_extension = ".txt"
shuffle_caption = True
keep_tokens = 0
max_train_epochs = 15
save_every_n_steps = 100
keep_only_last_n_epochs = 30
train_batch_size = 1

# --- The Art of Finesse ---
optimizer = "AdamW8bit"
unet_lr = 2e-4
text_encoder_lr = 6e-5
lr_scheduler = "cosine_with_restarts"
lr_scheduler_num_cycles = 4
lr_warmup_ratio = 0.05
min_snr_gamma = True
min_snr_gamma_value = 5.0 if min_snr_gamma else None

# --- The Blade's Complexity ---
lora_type = "LoRA"
network_dim = 512
network_alpha = 256

# --- Static Definitions ---
# *** YOU MUST CHANGE THIS PLACEHOLDER ***
LAB_STORAGE_ROOT = Path("/media/studies/ehr_study/data-EHR-prepped/Mikey-Lora-Trainer")

# The root of your scripts and dataset remains in your home directory.
project_root = Path(__file__).parent.parent
repo_dir = project_root / "kohya-trainer"
master_dataset_dir = project_root / "master_dataset"

# Models, outputs, and logs are banished to the vastness of /media/labs.
model_dir = LAB_STORAGE_ROOT / "model"
output_dir = LAB_STORAGE_ROOT / "Multi_Concept_Output"
log_dir = LAB_STORAGE_ROOT / "_logs"
accelerate_config_file = repo_dir / "accelerate_config/config.yaml"
model_file = model_dir / Path(model_url).name

def main_ritual():
    """The grand ritual for forging the spirit, adapted for the cluster."""
    print("Hmph. Preparing the forge...")

    # --- Verify the Base Demon's Presence ---
    # The script no longer summons. It only verifies.
    if not model_file.exists():
        error_message = (
            f"FATAL: The base demon is not in its prison.\n"
            f"You must summon it manually on the login node first.\n"
            f"Run this command on submit0:\n"
            f"wget -O \"{model_file}\" \"{model_url}\""
        )
        raise FileNotFoundError(error_message)
    else:
        print("The base demon is present. The ritual can proceed.")


    # --- Prepare the Final Incantations (Config Files) ---
    # Directories for output are still created here, as the job needs them.
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    config_file = output_dir / "training_config.toml"
    dataset_config_file = output_dir / "dataset_config.toml"

    print("Writing the scrolls of power (TOML configs)...")
    config_dict = {
      "additional_network_arguments": {"unet_lr": unet_lr, "text_encoder_lr": text_encoder_lr, "network_dim": network_dim, "network_alpha": network_alpha, "network_module": "networks.lora"},
      "optimizer_arguments": {"learning_rate": unet_lr, "lr_scheduler": lr_scheduler, "lr_scheduler_num_cycles": lr_scheduler_num_cycles, "lr_warmup_steps": int(lr_warmup_ratio * max_train_epochs), "optimizer_type": optimizer},
      "training_arguments": {"max_train_epochs": max_train_epochs, "save_every_n_steps": save_every_n_steps, "save_last_n_epochs": keep_only_last_n_epochs, "train_batch_size": train_batch_size, "clip_skip": 2, "min_snr_gamma": min_snr_gamma_value, "seed": 42, "max_token_length": 225, "xformers": True, "lowram": False, "save_precision": "fp16", "mixed_precision": "fp16", "output_dir": str(output_dir), "logging_dir": str(log_dir), "output_name": "RWBY_Fusion_LoRA", "log_prefix": "RWBY_Fusion_LoRA", "log_with": "tensorboard"},
      "model_arguments": {"pretrained_model_name_or_path": str(model_file), "v2": False, "v_parameterization": False},
      "saving_arguments": {"save_model_as": "safetensors"},
      "dataset_arguments": {"cache_latents": True},
    }
    with open(config_file, "w") as f: f.write(toml.dumps(config_dict))

    subsets = []
    for subdir in os.listdir(master_dataset_dir):
        subdir_path = master_dataset_dir / subdir
        if subdir_path.is_dir():
            try:
                repeats, _ = subdir.split('_', 1)
                num_repeats = int(repeats)
            except (ValueError, IndexError):
                num_repeats = 1
            subset_dict = { "is_reg": False, "image_dir": str(subdir_path), "num_repeats": num_repeats, "caption_extension": caption_extension }
            subsets.append(subset_dict)
    if not subsets: raise ValueError("FATAL: No subdirectories found in master_dataset.")

    dataset_config_dict = {
        "general": { "shuffle_caption": shuffle_caption, "keep_tokens": keep_tokens, "resolution": resolution, "flip_aug": flip_aug, "enable_bucket": True, "min_bucket_reso": 320, "max_bucket_reso": 1280 },
        "datasets": [{ "subsets": subsets }]
    }
    with open(dataset_config_file, "w") as f: f.write(toml.dumps(dataset_config_dict))

    # --- Check for Existing Checkpoints before Execution --
    resume_path = None
    if output_dir.exists():
        lora_files = [f for f in os.listdir(output_dir) if f.lower().endswith('.safetensors')]
        if lora_files:
            latest_step = -1
            for f in lora_files:
                match = re.search(r'-(\d+)\.safetensors$', f)
                if match:
                    step_num = int(match.group(1))
                    if step_num > latest_step:
                        latest_step = step_num
                        resume_path = output_dir / f

    if resume_path:
        print(f"Found a spirit forged to step {latest_step}. The ritual will resume.")
    else:
        print("No existing spirit found. Beginning a new forging.")

    # --- Unleash the Training ---
    command = [
        "accelerate", "launch", f"--config_file={accelerate_config_file}",
        "--num_cpu_threads_per_process=1", str(repo_dir / "train_network.py"),
        f"--dataset_config={dataset_config_file}", f"--config_file={config_file}"
    ]
    if resume_path:
        command.append(f"--network_weights={resume_path}")

    print(f"\nCOMMAND: {' '.join(command)}")
    subprocess.run(command, check=True)

    print("\n\nTHE FUSIONIST SPIRIT HAS BEEN FORGED.")

if __name__ == "__main__":
    main_ritual()