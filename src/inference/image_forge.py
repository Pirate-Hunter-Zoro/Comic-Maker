# src/inference/image_forge.py
# A dedicated, command-line tool for generating a single image from a LoRA.

import torch
import os
import re
import argparse
import json
from pathlib import Path
import warnings
import gc
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import OpenposeDetector
from PIL import Image

def find_latest_lora(lora_dir: Path):
    """Finds the most recent LoRA file in a directory based on step number."""
    if not lora_dir.exists():
        raise FileNotFoundError(f"LoRA directory does not exist: {lora_dir}")
    latest_step = -1
    lora_path = None
    for filename in os.listdir(lora_dir):
        if filename.lower().endswith('.safensors'):
            match = re.search(r'-(\d+)\.safetensors$', filename)
            if match:
                step_num = int(match.group(1))
                if step_num > latest_step:
                    latest_step = step_num
                    lora_path = lora_dir / filename
    if lora_path:
        return lora_path
    else:
        raise FileNotFoundError(f"No valid LoRA files found in {lora_dir}.")

def main(args):
    """The main inference ritual."""
    warnings.filterwarnings("ignore")
    torch.backends.cuda.matmul.allow_tf32 = True

    project_path = Path(args.project_path)
    armory_path = Path(args.great_armory_path)
    output_path = Path(args.output_path)
    
    # --- Construct Paths ---
    project_name = project_path.name
    lora_output_dir = armory_path / f"{project_name}_LoRA_Output"
    base_model_path = armory_path / "AnyLoRA"
    controlnet_model_path = armory_path / "controlnet-model"
    controlnet_detector_path = armory_path / "controlnet-detector"
    pose_map_file = project_path / "pose_map.png" # Assumes a standard pose map name

    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # --- ControlNet Image Preparation ---
    print("Image Forge: Preparing ControlNet input.")
    openpose_detector = OpenposeDetector.from_pretrained(str(controlnet_detector_path))
    if not pose_map_file.exists():
        raise FileNotFoundError(f"Pose map not found at {pose_map_file}. It must exist for this ritual.")
    
    control_image = openpose_detector(Image.open(pose_map_file).convert("RGB"))

    # --- Pipeline Assembly ---
    print("Image Forge: Summoning pipeline.")
    lora_path = find_latest_lora(lora_output_dir)
    controlnet = ControlNetModel.from_pretrained(str(controlnet_model_path), torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        str(base_model_path), controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True, safety_checker=None
    ).to("cuda")
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(lora_path)
    
    # --- Generation ---
    print(f"Image Forge: Executing prompt for {output_path.name}...")
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    
    final_image = pipe(
        args.prompt,
        negative_prompt=args.negative_prompt,
        image=control_image,
        width=1024,
        height=768,
        num_inference_steps=40,
        guidance_scale=7.5,
        generator=generator
    ).images[0]
    
    final_image.save(output_path)
    print(f"Image Forge: Vision sealed at: {output_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A dedicated LoRA image generation tool.")
    parser.add_argument('--project_path', type=str, required=True, help='Path to the project directory.')
    parser.add_argument('--great_armory_path', type=str, required=True, help='Path to the shared models directory.')
    parser.add_argument('--output_path', type=str, required=True, help='Full path for the output image file.')
    parser.add_argument('--prompt', type=str, required=True, help='The generation prompt.')
    parser.add_argument('--negative_prompt', type=str, default="worst quality, low quality, blurry, deformed", help='The negative prompt.')
    parser.add_argument('--seed', type=int, default=42, help='Seed for the random number generator.')
    
    args = parser.parse_args()
    main(args)