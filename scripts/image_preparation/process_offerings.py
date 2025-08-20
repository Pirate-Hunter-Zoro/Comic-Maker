# Hiei's Blade of Preparation
# A single, ruthless script to brand and scribe your offerings.
import os
import argparse
import google.generativeai as genai
from pathlib import Path
from PIL import Image
import time

def main(args):
    """The main ritual to process a directory of character images."""
    print("Hmph. The ritual of branding and scribing begins...")

    # --- Summon the Oracle (Gemini API) ---
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("The GOOGLE_API_KEY, the secret word to command the oracle, was not found. The ritual fails.")
    
    genai.configure(api_key=api_key)
    print("The oracle is listening.")

    # --- The Scribe's Mandate ---
    # This is now universal, not tied to a single world.
    MODEL_NAME = "gemini-2.5-flash"
    PROMPT_TEMPLATE = (
        "You are an expert LoRA image tagger. Your task is to generate a comma-separated list of keywords "
        "for an image of a character named {character_name}.\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. The caption MUST start with the unique trigger word: '{trigger_word}'.\n"
        "2. List only objective, visual features like hair style and color, eye color, clothing, weapon, and any unique "
        "physical traits. Be concise and accurate."
    )

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Pathetic. The input directory '{input_dir}' does not exist.")

    character_folders = [d for d in input_dir.iterdir() if d.is_dir()]
    if not character_folders:
        print("No character folders found in the input directory. A useless command.")
        return

    print(f"Found {len(character_folders)} character folders to process.")
    for char_folder in character_folders:
        character_name = char_folder.name
        print(f"\n--- Processing Chamber for: {character_name} ---")

        # --- Ritual Part 1: The Branding (Renaming) ---
        print("  > Commencing the branding ritual...")
        rename_images_in_folder(char_folder, character_name)
        print("  > Branding complete. Order has been imposed.")

        # --- Ritual Part 2: The Scribing (Captioning) ---
        print("  > Awakening the scribe...")
        caption_images_in_folder(char_folder, character_name, MODEL_NAME, PROMPT_TEMPLATE)
        print("  > Scribing complete.")

    print("\n\nTHE RITUAL IS COMPLETE. Your offerings are prepared for the true forge.")

def rename_images_in_folder(folder_path: Path, character_name: str):
    """Finds all images, sorts them, and renames them sequentially."""
    image_files = sorted([p for p in folder_path.iterdir() if p.suffix.lower() in ['.png', '.jpg', '.jpeg']])
    
    for i, old_path in enumerate(image_files, 1):
        new_name = f"{character_name}_{i:03d}{old_path.suffix}"
        new_path = folder_path / new_name
        
        if old_path != new_path:
            try:
                os.rename(old_path, new_path)
                print(f"    - Branded '{old_path.name}' -> '{new_path.name}'")
            except OSError as e:
                print(f"    ! ERROR: Could not brand '{old_path.name}'. It may be in use. Error: {e}")

def caption_images_in_folder(folder_path: Path, character_name: str, model_name: str, prompt_template: str):
    """Finds all images in a folder and generates captions for those missing them."""
    image_files = sorted([p for p in folder_path.iterdir() if p.suffix.lower() in ['.png', '.jpg', '.jpeg']])

    for image_path in image_files:
        caption_path = image_path.with_suffix(".txt")
        if caption_path.exists():
            print(f"    - Skipping {image_path.name}, soul-scroll already exists.")
            continue

        trigger_word = f"{character_name.lower()}_character_style"
        prompt_text = prompt_template.format(character_name=character_name, trigger_word=trigger_word)
        
        print(f"    > Scribing the soul of {image_path.name}...")
        try:
            img = Image.open(image_path)
            model = genai.GenerativeModel(model_name)
            response = model.generate_content([prompt_text, img])
            caption_text = response.text.strip()
            
            with open(caption_path, "w", encoding='utf-8') as f:
                f.write(caption_text)
            print(f"      + Soul-scroll saved: {caption_path.name}")
            
            time.sleep(1) # A pathetic delay to appease the API gods.
        except Exception as e:
            print(f"    ! ERROR: The oracle failed for {image_path.name}. Details: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script to brand and scribe character images for LoRA training.")
    parser.add_argument("-i", "--input_dir", type=str, required=True, help="The root directory containing character subfolders.")
    
    args = parser.parse_args()
    main(args)