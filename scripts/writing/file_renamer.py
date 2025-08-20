# file_renamer.py (v2.0 - The AUTONOMOUS Janitor Bot!)
# This bot now automatically scans the entire 'training_images' directory
# and cleans up every character subfolder it finds! No arguments needed!

import os
import re

# --- CONFIGURATION & PATHING ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
TRAINING_IMAGES_DIR = os.path.join(PROJECT_ROOT, "training_images")

def rename_files_in_folder(character_name: str):
    """Scans a character's folder and renames poorly named files."""
    
    character_folder = os.path.join(TRAINING_IMAGES_DIR, character_name)
    
    print(f"\n--- ✨ Engaging Janitor Bot for folder: {character_name} ✨ ---")
    
    # --- Step 1: Find all files and identify a starting number ---
    files = os.listdir(character_folder)
    start_num = 1
    
    for f in files:
        match = re.search(r'(\d+)', f)
        if match:
            num = int(match.group(1))
            if num >= start_num:
                start_num = num + 1
    
    print(f"Found existing numbered files. New files will start counting from: {start_num}")

    # --- Step 2: Find and rename the messy files! ---
    files_to_rename = [f for f in files if f.lower().endswith(('.txt'))]
    image_files_to_rename = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files_to_rename:
        print("✅ No messy files found! This folder is already clean! Hooray!")
        return
        
    print(f"Found {len(image_files_to_rename)} messy image files to clean up...")

    # --- Step 3: Loop and rename! ---
    for i, old_image_name in enumerate(image_files_to_rename, start=1):
        file_ext = os.path.splitext(old_image_name)[1]
        new_name_base = f"{character_name}_{i:03d}"
        new_image_name = f"{new_name_base}{file_ext}"
        new_txt_name = f"{new_name_base}.txt"
        
        old_image_path = os.path.join(character_folder, old_image_name)
        new_image_path = os.path.join(character_folder, new_image_name)
        
        old_txt_name = os.path.splitext(old_image_name)[0] + ".txt"
        old_txt_path = os.path.join(character_folder, old_txt_name)
        new_txt_path = os.path.join(character_folder, new_txt_name)

        print(f"Renaming '{old_image_name}' -> '{new_image_name}'")
        os.rename(old_image_path, new_image_path)
        
        if os.path.exists(old_txt_path):
            print(f"Renaming '{old_txt_name}' -> '{new_txt_name}'")
            os.rename(old_txt_path, new_txt_path)
        else:
            print(f"--- ⚠️ WARNING: No matching .txt file found for {old_image_name}")

    print(f"--- ✅ Janitor Bot finished cleaning the {character_name} folder! So shiny! ---")


def main():
    """The main function that finds all character folders and cleans them."""
    print("--- 🤖 DEPLOYING AUTONOMOUS JANITOR BOT v2.0 🤖 ---")
    print("Scanning for all character directories...")

    if not os.path.isdir(TRAINING_IMAGES_DIR):
        print(f"--- 💥 CRITICAL ERROR: 'training_images' directory not found at {TRAINING_IMAGES_DIR}! ---")
        return

    # This is the new brain! It gets a list of all items in the directory
    # and then keeps only the ones that are actual folders! So smart!
    all_folders = [f for f in os.listdir(TRAINING_IMAGES_DIR) if os.path.isdir(os.path.join(TRAINING_IMAGES_DIR, f))]

    if not all_folders:
        print("--- 🤷 No character folders found to clean. All done! ---")
        return
        
    print(f"Found {len(all_folders)} character folders to process!")
    
    for character_folder_name in all_folders:
        rename_files_in_folder(character_folder_name)
        
    print("\n\n--- 🎉 ALL FOLDERS PROCESSED! The entire lab is sparkling clean! 🎉 ---")


if __name__ == '__main__':
    # No more arguments needed! We just run the main function!
    main()