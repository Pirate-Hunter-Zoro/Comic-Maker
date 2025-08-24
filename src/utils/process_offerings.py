# src/utils/process_offerings.py
# A single, ruthless script to brand and scribe training data for a project.

import os
import argparse
import google.generativeai as genai
from pathlib import Path
from PIL import Image
import time
import json

def load_config(project_path: Path) -> dict:
    """Loads the project_config.json file."""
    config_path = project_path / "project_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def rename_images(folder_path: Path, character_name: str):
    """Sorts and renames all images in a folder sequentially."""
    image_files = sorted([p for p in folder_path.iterdir() if p.suffix.lower() in ['.png', '.jpg', '.jpeg']])
    
    for i, old_path in enumerate(image_files, 1):
        # The new name format is simple and clean.
        new_name = f"{character_name.lower()}_{i:03d}{old_path.suffix}"
        new_path = folder_path / new_name
        
        if old_path != new_path:
            try:
                old_path.rename(new_path)
                print(f"    - Branded: '{old_path.name}' -> '{new_path.name}'")
            except OSError as e:
                print(f"    ! ERROR: Could not brand '{old_path.name}'. File may be in use. Details: {e}")

def caption_images(folder_path: Path, character_name: str, config: dict):
    """Generates captions for all images in a folder that do not already have one."""
    image_files = sorted([p for p in folder_path.iterdir() if p.suffix.lower() in ['.png', '.jpg', '.jpeg']])

    model_name = "gemini-1.5-flash"
    prompt_template = (
        "You are an expert LoRA image tagger. Generate a comma-separated list of keywords "
        "for an image of a character named {character_name}.\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. The caption MUST start with the unique trigger word: '{trigger_word}'.\n"
        "2. List only objective, visual features like hair style and color, eye color, clothing, weapon, and unique physical traits."
    )
    
    # The trigger word is now constructed from a template in the config file.
    trigger_template = config.get("training_settings", {}).get("trigger_word_template", "{character_name}_style")
    trigger_word = trigger_template.format(character_name=character_name.lower())
    
    prompt_text = prompt_template.format(character_name=character_name, trigger_word=trigger_word)

    model = genai.GenerativeModel(model_name)

    for image_path in image_files:
        caption_path = image_path.with_suffix(".txt")
        if caption_path.exists():
            continue

        print(f"    > Scribing the soul of {image_path.name}...")
        try:
            img = Image.open(image_path)
            response = model.generate_content([prompt_text, img])
            caption_text = response.text.strip()
            
            caption_path.write_text(caption_text, encoding='utf-8')
            print(f"      + Soul-scroll saved: {caption_path.name}")
            
            time.sleep(1) # A necessary delay.
        except Exception as e:
            print(f"    ! ERROR: The oracle failed for {image_path.name}. Details: {e}")

def main():
    """The main ritual to process a project's training data."""
    parser = argparse.ArgumentParser(description="Brands and scribes character images for LoRA training based on a project config.")
    parser.add_argument('--project_path', type=str, required=True, help='Path to the project directory.')
    args = parser.parse_args()
    
    project_path = Path(args.project_path)
    print(f"Hmph. The ritual of branding and scribing begins for project: '{project_path.name}'")

    config = load_config(project_path)
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("The GOOGLE_API_KEY, the secret word to command the oracle, was not found.")
    genai.configure(api_key=api_key)
    print("The oracle is listening.")

    try:
        input_dir = project_path / config['paths']['training_data']
    except KeyError:
        raise ValueError("'training_data' path not defined in project_config.json.")

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Pathetic. The training data directory does not exist: {input_dir}")

    # Assumes character folders are named in the format 'XX_CharacterName'.
    character_folders = [d for d in input_dir.iterdir() if d.is_dir() and '_' in d.name]
    if not character_folders:
        print("No valid character folders found in the training data directory. A useless command.")
        return

    print(f"Found {len(character_folders)} character folders to process.")
    for char_folder in character_folders:
        # Extract the name after the 'XX_' prefix.
        character_name = '_'.join(char_folder.name.split('_')[1:])
        print(f"\n--- Processing Chamber for: {character_name} ---")

        print("  > Commencing the branding ritual (renaming)...")
        rename_images(char_folder, character_name)
        print("  > Branding complete.")

        print("  > Awakening the scribe (captioning)...")
        caption_images(char_folder, character_name, config)
        print("  > Scribing complete.")

    print("\n\nTHE RITUAL IS COMPLETE. Your offerings are prepared for the true forge.")

if __name__ == "__main__":
    main()