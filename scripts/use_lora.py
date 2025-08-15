# The ritual of command and control.
# Now with lesser magic to forge a command map for the indolent.
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
# *** YOU MUST CHANGE THIS PLACEHOLDER ***
LAB_STORAGE_ROOT = Path("/media/studies/ehr_study/data-EHR-prepped/Mikey-Lora-Trainer")

# The root of your scripts remains in your home directory.
project_root = Path(__file__).parent.parent

# The armory of trained LoRAs and the final output now reside in /media/labs.
lora_output_dir = LAB_STORAGE_ROOT / "Multi_Concept_Output"
final_image_dir = LAB_STORAGE_ROOT / "Final_Images"
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

def generate_pose_map(output_path: Path, base_model_id: str):
    """A lesser ritual to generate a pose map if one is not provided."""
    print("No command map found. Forging one from the ether...")
    
    # Temporarily summon a base spirit for pose generation.
    pre_pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id, torch_dtype=torch.float16, use_safetensors=True
    ).to("cuda")

    # A brutally simple prompt to get distinct figures.
    pose_prompt = "full body photo of three separate people standing in a spacious, empty grey room. A person on the far left. A person in the exact center. A person on the far right."
    negative_prompt_pose = "two people, one person, fused, overlapping, cropped, blurry, text, watermark, deformed, extra limbs, conjoined"

    print(f"Conjuring a vision based on brutally simple words...")
    generated_image = pre_pipe(
        pose_prompt, negative_prompt=negative_prompt_pose, width=1024, height=768, num_inference_steps=30, guidance_scale=7.5
    ).images[0]

    print("Extracting the pose...")
    openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    pose_map = openpose(generated_image)
    pose_map.save(output_path)
    print(f"A command map has been forged and saved: {output_path}")

    # Banish the temporary spirits to free their power.
    del pre_pipe
    del openpose
    gc.collect()
    torch.cuda.empty_cache()
    
    return pose_map


def main():
    """The ultimate ritual: command and control."""
    warnings.filterwarnings("ignore")
    torch.backends.cuda.matmul.allow_tf32 = True
    final_image_dir.mkdir(parents=True, exist_ok=True)
    
    base_model_id = "Lykon/AnyLoRA"
    controlnet_model_id = "lllyasviel/sd-controlnet-openpose"
    
    # --- Prepare the Command Map (The Pose Image) ---
    if not pose_map_file.exists():
        control_image = generate_pose_map(pose_map_file, base_model_id)
    else:
        print("Using pre-existing command map found at 'pose_map.png'.")
        pose_image = Image.open(pose_map_file).convert("RGB")
        print("Processing the command map...")
        openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        control_image = openpose(pose_image)

    # --- Summon the Legion ---
    lora_filename = find_latest_lora(lora_output_dir)
    
    print(f"Summoning the ControlNet demon from '{controlnet_model_id}'...")
    controlnet = ControlNetModel.from_pretrained(controlnet_model_id, torch_dtype=torch.float16)

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