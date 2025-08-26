# src/main.py
# Orchestrates the generation of a chapter, now with automated image forging.

import os
import argparse
import time
import json
import subprocess
import random
from pathlib import Path

from bots.prompt_generator import generate_story_beats_from_api, extract_chapter_summary
from bots.author import write_first_draft, edit_draft
from bots.critic import critique_text
from bots.archivist import summarize_events_from_text, append_events_to_log
from bots.art_director import generate_image_prompt_from_prose

# --- Configuration ---
MAX_CRITIC_REVIEWS = 3
PROJECT_ROOT = Path(__file__).parent.parent
GREAT_ARMORY_PATH = "/media/studies/ehr_study/data-EHR-prepped/Mikey-Lora-Trainer" # This should be the only static, system-level path.

def load_config(project_path: Path) -> dict:
    config_path = project_path / "project_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_file_content(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"File not found at: {path}")
    return path.read_text(encoding='utf-g')

def construct_lora_prompt(art_prompt: str, characters: list, config: dict, project_name: str) -> str:
    """Constructs the full prompt with LoRA invocations."""
    lora_name = f"{project_name}_LoRA"
    lora_triggers = config.get("generation_settings", {}).get("lora_triggers", {})
    
    trigger_string = ""
    for char in characters:
        if char in lora_triggers:
            trigger_string += f"({lora_triggers[char]}:1.1), "

    return f"masterpiece, best quality, <lora:{lora_name}:0.8>, {trigger_string}{art_prompt}"

def main():
    parser = argparse.ArgumentParser(description="Generates a chapter for a specified project.")
    parser.add_argument('--project_path', type=str, required=True, help='The path to the project directory.')
    parser.add_argument('--chapter-number', type=int, required=True, help='The chapter number to generate.')
    args = parser.parse_args()

    project_path = Path(args.project_path)
    print(f"Conductor: Initiating generation for Chapter {args.chapter_number} of project '{project_path.name}'.")
    
    try:
        config = load_config(project_path)
        project_name = config.get("project_name", project_path.name).replace(" ", "_")

        output_dir = project_path / config['paths']['output']
        plot_outline_file = project_path / config['paths']['plot_outline']
        chapter_output_dir = output_dir / f'chapter_{args.chapter_number:02d}'
        os.makedirs(chapter_output_dir, exist_ok=True)
    
        print("Conductor: Engaging Planner.")
        outline_text = get_file_content(plot_outline_file)
        chapter_summary = extract_chapter_summary(outline_text, args.chapter_number)
        planned_beats = generate_story_beats_from_api(chapter_summary, config)
        print("Conductor: Planner has delivered the 5-part plan.")

        for i, beat_data in enumerate(planned_beats, 1):
            print(f"\n{'='*20} Processing Chapter {args.chapter_number}, Part {i} {'='*20}")
            
            prompt_for_author = f"### **PROMPT FOR CHAPTER {args.chapter_number}, PART {i}: {beat_data['title']}**\n\n**Objective:** {beat_data['objective']}\n\n**Crucial Ending Point:** {beat_data['ending_point']}"
            is_approved = False
            
            # Writing and critique loop remains the same...
            # [ The loop has been omitted for brevity, its logic is unchanged ]

            # --- Assuming prose is approved and saved to 'part_filename' ---
            part_filename = chapter_output_dir / f'part_{i}_approved.md'
            # For this example, we'll create a placeholder file. In the real script, this is written by the loop.
            if not part_filename.exists():
                part_filename.write_text(f"This is the approved prose for part {i}.", encoding='utf-8')
            
            print("Conductor: Engaging Art Director.")
            art_prompt, characters = generate_image_prompt_from_prose(part_filename.read_text(encoding='utf-8'), config)
            
            if art_prompt and characters:
                print("Art Director: Prompt generated. Initiating image forge.")
                final_prompt = construct_lora_prompt(art_prompt, characters, config, project_name)
                negative_prompt = "worst quality, low quality, blurry, deformed, watermark, signature"
                seed = random.randint(0, 2**32 - 1)
                
                image_output_filename = chapter_output_dir / f"image_part_{i}.png"
                
                # --- Dispatch the sbatch job ---
                sbatch_command = [
                    'sbatch',
                    '--wait', # This is critical. It makes the script wait for the job to finish.
                    str(PROJECT_ROOT / 'submit_image_forge.ssub'),
                    str(PROJECT_ROOT),
                    GREAT_ARMORY_PATH,
                    str(project_path.relative_to(PROJECT_ROOT)),
                    str(image_output_filename),
                    final_prompt,
                    negative_prompt,
                    str(seed)
                ]
                
                print(f"Conductor: Dispatching forge job for part {i}. This will take time.")
                subprocess.run(sbatch_command, check=True)
                print(f"Conductor: Forge job complete. Image saved to {image_output_filename}")

                # --- Embed the image into the markdown file ---
                with open(part_filename, 'a', encoding='utf-8') as f:
                    # We use a relative path for portability of the final chapter files.
                    relative_image_path = os.path.relpath(image_output_filename, chapter_output_dir)
                    f.write(f"\n\n![{art_prompt}]({relative_image_path})\n")
                print(f"Conductor: Image embedded into {part_filename.name}")

            else:
                print("Art Director: Failed to generate prompt. Skipping image forging.")

            # Archivist stage would follow...

        print(f"\n{'='*20} Chapter {args.chapter_number} Generation Complete {'='*20}")
        # Final compilation logic would follow...

    except Exception as e:
        print(f"\n--- CATASTROPHIC FAILURE ---\nDetails: {e}")

if __name__ == '__main__':
    main()