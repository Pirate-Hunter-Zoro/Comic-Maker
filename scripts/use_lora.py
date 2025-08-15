# The ritual of command and control. Final Form.
# All spirits are now summoned from local prisons, not the ether.
import torch
import os
import re
from PIL import Image
from pathlib import Path
import warnings
import gc
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import OpenposeDetector

# --- Static Definitions ---
LAB_STORAGE_ROOT = Path("/media/studies/ehr_study/data-EHR-prepped/Mikey-Lora-Trainer")

# The root of your scripts remains in your home directory.
project_root = Path(__file__).parent.parent

# --- Paths to the Local Prisons ---
# The script now looks for all spirits in their designated local sanctums.
lora_output_dir = LAB_STORAGE_ROOT / "Multi_Concept_Output"
final_image_dir = LAB_STORAGE_ROOT / "Final_Images"
controlnet_model_path = LAB_STORAGE_ROOT / "controlnet-model"
controlnet_detector_path = LAB_STORAGE_ROOT / "controlnet-detector"
pose_map_file = project_root / "pose_map.png"

def find_latest_lora(lora_dir: Path):
    """A simple scry to find the most powerful spirit."""
    print(f"Hmph. Scanning the armory at '{lora_dir}' for the strongest spirit...")
    if not lora_dir.exists():
        raise FileNotFoundError(f"Pathetic. The armory directory does not exist: {lora_dir}")
    latest_step = -1
    fusion_lora_path = None
    for filename in os.listdir(lora_dir):
        if filename.lower().endswith('.safetensors'):
            match = re.search(r'-(\d+)\.safetensors$', filename)
            if match:
                step_num = int(match.group(1))
                if step_num > latest_step:
                    latest_step = step_num
                    fusion_lora_path = filename
    if fusion_lora_path:
        print(f"Found the strongest spirit. Using step {latest_step}: {fusion_lora_path}")
        return fusion_lora_path
    else:
        raise FileNotFoundError(f"There are no forged spirits in the armory. The ritual fails.")

def main():
    """The ultimate ritual: command and control."""
    warnings.filterwarnings("ignore")
    torch.backends.cuda.matmul.allow_tf32 = True
    final_image_dir.mkdir(parents=True, exist_ok=True)
    
    base_model_id = "Lykon/AnyLoRA" # This is just an ID, the actual file is loaded locally.
    
    # --- Prepare the Command Map (The Pose Image) ---
    if not pose_map_file.exists():
        raise FileNotFoundError("You have failed. The 'pose_map.png' was not provided in the project root. The ritual cannot proceed without a command map.")
    
    print("Using pre-existing command map found at 'pose_map.png'.")
    pose_image = Image.open(pose_map_file).convert("RGB")
    print("Processing the command map...")
    # Summoning the detector from its local prison.
    openpose = OpenposeDetector.from_pretrained(str(controlnet_detector_path))
    control_image = openpose(pose_image)

    # --- Summon the Legion ---
    lora_filename = find_latest_lora(lora_output_dir)
    
    print(f"Summoning the ControlNet demon from its local prison...")
    # Summoning the ControlNet model from its local prison.
    controlnet = ControlNetModel.from_pretrained(str(controlnet_model_path), torch_dtype=torch.float16)

    print(f"Summoning the base demon '{base_model_id}' and binding it to ControlNet...")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_id, controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
    ).to("cuda")
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    print(f"Binding the Fusionist LoRA: {lora_filename}")
    pipe.load_lora_weights(str(lora_output_dir), weight_name=lora_filename)
    print("All spirits are bound and ready for command.")
    
    # --- The Final Incantation ---
    prompt = "masterpiece, best quality, cinematic lighting, dramatic, (ruby_character:1.1) with her scythe, (blake_character:1.1) with her katana, and (cinder_character:1.2) conjuring fire, battle in a ruined city"
    negative_prompt = "worst quality, low quality, ugly, deformed, blurry, extra limbs, watermark, signature, mutated, fused characters, extra character, jpeg artifacts"
    lora_scale = 0.8
    controlnet_scale = 0.9

    print("\nUnleashing the final ritual...")
    final_image = pipe(
        prompt, negative_prompt=negative_prompt, image=control_image, width=1024, height=768,
        num_inference_steps=40, guidance_scale=7.5, cross_attention_kwargs={"scale": lora_scale},
        controlnet_conditioning_scale=controlnet_scale,
    ).images[0]
    
    final_image_path = final_image_dir / "final_command_scene.png"
    final_image.save(final_image_path)
    print(f"\nTHE RITUAL IS COMPLETE. The final creation is sealed at: {final_image_path}")

if __name__ == "__main__":
    main()