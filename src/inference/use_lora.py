# src/inference/use_lora.py
# A generalized inference tool to generate an image using a trained LoRA.

import torch
import os
import re
import argparse
import json
from pathlib import Path
import warnings
import gc
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import OpenposeDetector
from PIL import Image

def load_config(project_path: Path) -> dict:
    config_path = project_path / "project_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    return json.load(config_path.open('r', encoding='utf-8'))

def find_latest_lora(lora_dir: Path):
    """Finds the most recent LoRA file in a directory based on step number."""
    print(f"Inference: Scanning for latest LoRA in '{lora_dir}'...")
    if not lora_dir.exists():
        raise FileNotFoundError(f"LoRA directory does not exist: {lora_dir}")
    latest_step = -1
    lora_path = None
    for filename in os.listdir(lora_dir):
        if filename.lower().endswith('.safensors'):
            match = re.search(r'-step(\d+)\.safetensors$', filename)
            if match:
                step_num = int(match.group(1))
                if step_num > latest_step:
                    latest_step = step_num
                    lora_path = lora_dir / filename
    if lora_path:
        print(f"Inference: Found LoRA at step {latest_step}: {lora_path.name}")
        return lora_path
    else:
        raise FileNotFoundError(f"No valid LoRA files found in {lora_dir}.")

def scry_and_trace_pose_map(output_path: Path, detector, model_path: Path, pose_prompt: str):
    """Generates a pose map from a text prompt."""
    print("Inference: Pose map not found. Generating from prompt...")
    print("Inference: Summoning scrying pipeline...")
    scry_pipe = StableDiffusionPipeline.from_pretrained(str(model_path), torch_dtype=torch.float16, safety_checker=None).to("cuda")
    
    scry_image = scry_pipe(
        pose_prompt,
        negative_prompt="worst quality, low quality, weapons, objects, text, watermark, blurry",
        num_inference_steps=25, width=1024, height=768, guidance_scale=7.0
    ).images[0]
    
    print("Inference: Scrying complete. Banishing pipeline.")
    del scry_pipe
    gc.collect()
    torch.cuda.empty_cache()

    print("Inference: Tracing pose from scryed image...")
    control_image = detector(scry_image)
    control_image.save(output_path)
    print(f"Inference: Pose map saved to {output_path}")
    return control_image

def main(args):
    """The main inference ritual."""
    warnings.filterwarnings("ignore")
    torch.backends.cuda.matmul.allow_tf32 = True

    project_path = Path(args.project_path)
    armory_path = Path(args.great_armory_path)
    
    config = load_config(project_path)
    project_name = config.get("project_name", project_path.name)
    
    # Construct all paths dynamically
    lora_output_dir = armory_path / f"{project_name}_LoRA_Output"
    final_image_dir = project_path / "test_images" / f"{project_name}_Final_Images"
    base_model_path = armory_path / "AnyLoRA"
    controlnet_model_path = armory_path / "controlnet-model"
    controlnet_detector_path = armory_path / "controlnet-detector"
    pose_map_file = project_path / "test_images" / f"{project_name}_Posemap_Images" / "pose_map.png"
    
    final_image_dir.mkdir(parents=True, exist_ok=True)
    pose_map_file.parent.mkdir(parents=True, exist_ok=True)

    print("Inference: Awakening OpenPose detector...")
    openpose_detector = OpenposeDetector.from_pretrained(str(controlnet_detector_path))

    if not pose_map_file.exists():
        if not args.pose_prompt:
            raise FileNotFoundError(f"Pose map not found at {pose_map_file} and no --pose_prompt was provided.")
        control_image = scry_and_trace_pose_map(pose_map_file, openpose_detector, base_model_path, args.pose_prompt)
    else:
        print(f"Inference: Using existing pose map from {pose_map_file}")
        control_image = openpose_detector(Image.open(pose_map_file).convert("RGB"))

    lora_path = find_latest_lora(lora_output_dir)
    
    print("Inference: Summoning ControlNet pipeline...")
    controlnet = ControlNetModel.from_pretrained(str(controlnet_model_path), torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        str(base_model_path), controlnet=controlnet, torch_dtype=torch.float16, use_safensors=True, safety_checker=None
    ).to("cuda")
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    print(f"Inference: Loading LoRA weights from {lora_path.name}")
    pipe.load_lora_weights(lora_path)
    
    print("Inference: Executing final prompt...")
    final_image = pipe(
        args.prompt, negative_prompt=args.negative_prompt, image=control_image,
        width=1024, height=768, num_inference_steps=40, guidance_scale=7.5
    ).images[0]
    
    timestamp = Path(lora_path.name).stem
    final_image_path = final_image_dir / f"image_{timestamp}.png"
    final_image.save(final_image_path)
    print(f"\nRitual Complete. Image saved to: {final_image_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A generalized LoRA inference tool.")
    parser.add_argument('--project_path', type=str, required=True, help='Path to the project directory.')
    parser.add_argument('--great_armory_path', type=str, required=True, help='Path to the shared models and output directory.')
    parser.add_argument('--prompt', type=str, required=True, help='The final generation prompt.')
    parser.add_argument('--negative_prompt', type=str, default="worst quality, low quality, blurry, deformed", help='The negative prompt.')
    parser.add_argument('--pose_prompt', type=str, help='A prompt to generate a pose map if one is not found.')
    
    args = parser.parse_args()
    main(args)