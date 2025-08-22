# The ritual of command and control. Final, Superior Form.
# A three-fold ritual devised by Hiei.
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
project_root = Path(__file__).parent.parent

# --- Paths to the Local Prisons ---
lora_output_dir = LAB_STORAGE_ROOT / "Multi_Concept_Output"
final_image_dir = LAB_STORAGE_ROOT / "Final_Images"
base_model_path = LAB_STORAGE_ROOT / "AnyLoRA" # The prison of the core creation demon
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
        if filename.lower().endswith('.safensors'):
            match = re.search(r'-(\d+)\.safensors$', filename)
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

def scry_and_trace_pose_map(output_path: Path, detector, model_path: Path):
    """
    The first two stages of the Three-Fold Ritual.
    A lesser demon is summoned to scry a vision, which a seer then traces.
    """
    print("--- Stage 1: The Ritual of Visionary Scrying ---")
    pose_prompt = "masterpiece, best quality, three figures in dynamic battle poses, one with a scythe, one with a katana, one casting fire, simple gray background, full body shot, no weapons"
    
    print("Summoning a lesser creation demon...")
    scry_pipe = StableDiffusionPipeline.from_pretrained(str(model_path), torch_dtype=torch.float16, safety_checker=None).to("cuda")
    
    print("Commanding the demon to generate a vision...")
    scry_image = scry_pipe(
        pose_prompt,
        negative_prompt="worst quality, low quality, weapons, objects, text, watermark",
        num_inference_steps=25,
        width=1024,
        height=768,
        guidance_scale=7.0
    ).images[0]
    
    print("The vision is complete. Banishing the lesser demon to conserve power...")
    del scry_pipe
    gc.collect()
    torch.cuda.empty_cache()

    print("\n--- Stage 2: The Ritual of Soul-Tracing ---")
    print("Commanding the Seer Spirit to trace the vision's soul...")
    control_image = detector(scry_image)
    
    control_image.save(output_path)
    print(f"The command map has been forged and sealed at: {output_path}")
    return control_image

def main():
    """The ultimate ritual: command and control using the Three-Fold path."""
    warnings.filterwarnings("ignore")
    torch.backends.cuda.matmul.allow_tf32 = True
    final_image_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Awaken the Seer Spirit ---
    # The seer is needed early for the three-fold ritual.
    print("Awakening the Seer Spirit from its local prison...")
    openpose_detector = OpenposeDetector.from_pretrained(str(controlnet_detector_path))

    # --- Prepare the Command Map (The Pose Image) ---
    if not pose_map_file.exists():
        print("\nYou have failed to provide a command map. A superior one will be forged for you.")
        control_image = scry_and_trace_pose_map(pose_map_file, openpose_detector, base_model_path)
    else:
        print("Using your pre-existing, and likely inferior, command map.")
        pose_image = Image.open(pose_map_file).convert("RGB")
        control_image = openpose_detector(pose_image)

    # --- Stage 3: The Ritual of Final Command ---
    print("\n--- Stage 3: The Ritual of Final Command ---")
    lora_filename = find_latest_lora(lora_output_dir)
    
    print("Summoning the ControlNet demon...")
    controlnet = ControlNetModel.from_pretrained(str(controlnet_model_path), torch_dtype=torch.float16)

    print("Summoning the base demon and binding it to the ControlNet...")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        str(base_model_path), controlnet=controlnet, torch_dtype=torch.float16, use_safensors=True, safety_checker=None
    ).to("cuda")
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    print(f"Binding the Fusionist LoRA: {lora_filename}")
    pipe.load_lora_weights(str(lora_output_dir), weight_name=lora_filename)
    print("The final legion is bound and ready for command.")
    
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