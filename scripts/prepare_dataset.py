# scripts/prepare_dataset.py

import os
import json

def create_dataset_manifest():
    """
    Scans the training_images directory and creates a JSONL manifest file
    required for Vertex AI tuning jobs.
    """
    print("🤖 Data Processor Bot Activated! Preparing the menu for the big brain...")
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    training_dir = os.path.join(project_root, 'training_images')
    output_file = os.path.join(training_dir, 'metadata.jsonl')

    if not os.path.isdir(training_dir):
        print(f"❌ ERROR: The directory '{training_dir}' does not exist! I can't find the pictures!")
        return

    manifest_entries = []
    print(f"Scanning subfolders in: {training_dir}")
    
    # Walk through all the character folders
    for character_folder in os.listdir(training_dir):
        char_path = os.path.join(training_dir, character_folder)
        if os.path.isdir(char_path):
            print(f"  - Processing folder for '{character_folder}'...")
            for filename in os.listdir(char_path):
                # We're looking for the image files, not the text files
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_name_without_ext = os.path.splitext(filename)[0]
                    text_filename = image_name_without_ext + '.txt'
                    text_path = os.path.join(char_path, text_filename)
                    
                    if os.path.exists(text_path):
                        with open(text_path, 'r', encoding='utf-8') as f:
                            caption = f.read().strip()
                        
                        # The menu needs the CHARACTER'S FOLDER and the FILENAME
                        gcs_image_path = f"{character_folder}/{filename}"
                        
                        entry = {
                            # This is the path INSIDE the cloud bucket!
                            "imageGcsUri": gcs_image_path,
                            "caption": caption
                        }
                        manifest_entries.append(entry)
                    else:
                        print(f"    - WARNING: No matching .txt file found for {filename}. Skipping.")

    # Now we write the menu!
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry) + '\n')
            
    print(f"\n✅ SUCCESS! The menu has been created at: {output_file}")
    print(f"You should now have a 'metadata.jsonl' file inside your 'training_images' folder!")

if __name__ == "__main__":
    create_dataset_manifest()