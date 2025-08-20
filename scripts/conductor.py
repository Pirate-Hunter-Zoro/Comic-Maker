# scripts/conductor.py

import os
import argparse
import time
import subprocess
import sys
import shutil

# All our brilliant little bots!
from prompt_generator import generate_story_beats_from_api, extract_chapter_summary, load_file_content, CORE_DIRECTIVES
from author import write_first_draft, edit_draft
from critic import critique_text
from archivist import summarize_events_from_text, append_events_to_log
from art_director import generate_image_prompt_from_prose

# --- CONFIGURATION ---
MAX_CRITIC_REVIEWS = 3

# --- PATHING GPS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
KNOWLEDGE_DB_DIR = os.path.join(PROJECT_ROOT, "knowledge_db")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "generated_chapters")
PLOT_OUTLINE_FILE = os.path.join(KNOWLEDGE_DB_DIR, "rwby_novel_plot_outline.md")
# Path to our new, unified art generator bot!
# Make sure your file is named 'art_generator.py' or change the name here!
ART_GENERATOR_SCRIPT = os.path.join(SCRIPT_DIR, "art_generator.py")

# --- MAIN ORCHESTRATION ENGINE ---

def main():
    """The main function that orchestrates the entire chapter generation."""
    parser = argparse.ArgumentParser(description="The Grand Orchestrator: Generates a full chapter from start to finish.")
    parser.add_argument('--chapter-number', type=int, required=True, help='The number of the chapter to generate.')
    args = parser.parse_args()

    print(f"--- CONDUCTOR ONLINE: INITIATING GENERATION FOR CHAPTER {args.chapter_number} ---")
    
    chapter_output_dir = os.path.join(OUTPUT_DIR, f'chapter_{args.chapter_number:02d}')
    os.makedirs(chapter_output_dir, exist_ok=True)
    
    try:
        print("\n--- [CONDUCTOR] Engaging Planner Bot ---")
        outline_text = load_file_content(PLOT_OUTLINE_FILE)
        chapter_summary = extract_chapter_summary(outline_text, args.chapter_number)
        planned_beats = generate_story_beats_from_api(chapter_summary)
        print("--- [CONDUCTOR] Planner Bot has delivered the 5-part plan! ---")

        for i, beat_data in enumerate(planned_beats, 1):
            print(f"\n\n{'='*20} STARTING CHAPTER {args.chapter_number}, PART {i} OF 5 {'='*20}")
            
            prompt_for_author = f"{CORE_DIRECTIVES}\n\n---\n\n### **PROMPT FOR CHAPTER {args.chapter_number}, PART {i}: {beat_data['title']}**\n\n**Objective:** {beat_data['objective']}\n\n**Crucial Ending Point:** {beat_data['ending_point']}"
            current_prose = ""
            is_approved = False
            critique_result = ""
            
            for attempt in range(MAX_CRITIC_REVIEWS + 1):
                if attempt == 0:
                    current_prose = write_first_draft(prompt_for_author)
                else:
                    current_prose = edit_draft(current_prose, critique_result, prompt_for_author)

                print(f"--- [CONDUCTOR] Text generated. Sending to Critic for review (Attempt {attempt + 1}/{MAX_CRITIC_REVIEWS + 1}) ---")
                time.sleep(1)
                critique_result = critique_text(current_prose, prompt_for_author, beat_data['key_characters'], beat_data['key_locations'])

                if critique_result == "SUCCESS":
                    print(f"--- [CRITIC] SUCCESS! Chapter Part {i} has been approved! ---")
                    is_approved = True
                    break
                else:
                    print(f"--- [CRITIC] Edits required for Chapter Part {i}. Feedback: ---\n{critique_result}\n----------------------------------------------------------")

            if is_approved:
                part_filename = os.path.join(chapter_output_dir, f'chapter_{args.chapter_number:02d}_part_{i}_approved.md')
                with open(part_filename, 'w', encoding='utf-8') as f:
                    f.write(current_prose)
                print(f"\n--- [CONDUCTOR] Approved text saved to: {os.path.basename(part_filename)}")

                # --- ART GENERATION MODIFICATION ---
                print("\n--- [CONDUCTOR] Engaging Art Director Bot ---")
                art_prompt, characters = generate_image_prompt_from_prose(current_prose)

                if art_prompt and characters:
                    print("\n--- [CONDUCTOR] Art prompt generated successfully.")
                    print("--- Use the following prompt in your Colab notebook for image generation: ---")
                    # Construct the final prompt with the LoRA invocation
                    lora_name = f"{characters[0].split()[0]}.safetensors" # Assumes LoRA is named after the first character
                    final_art_prompt = f"<lora:{lora_name}:1> {art_prompt}"
                    print(f"\nPROMPT: {final_art_prompt}\n")
                    
                else:
                    print("--- [CONDUCTOR] Art Director failed to provide a prompt or characters. Skipping art generation. ---")
                
                print("\n--- [CONDUCTOR] Engaging Archivist Bot ---")
                events_summary = summarize_events_from_text(current_prose, f"Chapter {args.chapter_number}, Part {i}")
                append_events_to_log(events_summary, f"Chapter {args.chapter_number}, Part {i}")
                print("--- [CONDUCTOR] Archivist has updated the official canon! ---")

            else:
                raise Exception(f"Failed to produce approved text for Chapter {args.chapter_number}, Part {i} after {MAX_CRITIC_REVIEWS + 1} attempts.")

        print(f"\n\n{'='*20} CHAPTER {args.chapter_number} GENERATION COMPLETE! {'='*20}")
        final_chapter_filename = os.path.join(chapter_output_dir, f'chapter_{args.chapter_number:02d}_complete.md')
        
        final_chapter_parts = []
        for i in range(1, len(planned_beats) + 1):
            part_path = os.path.join(chapter_output_dir, f'chapter_{args.chapter_number:02d}_part_{i}_approved.md')
            with open(part_path, 'r', encoding='utf-8') as f:
                final_chapter_parts.append(f.read())
        final_text = "\n\n---\n\n".join(final_chapter_parts)

        with open(final_chapter_filename, 'w', encoding='utf-8') as f:
            f.write(final_text)
        
        print(f"Successfully generated and saved the complete Chapter {args.chapter_number} to:\n{final_chapter_filename}")
        print("\n--- CONDUCTOR OFFLINE ---")

    except Exception as e:
        print(f"\n--- A CATASTROPHIC ERROR OCCURRED IN THE CONDUCTOR ---\nProcess halted. Details: {e}\n--- CONDUCTOR EMERGENCY SHUTDOWN ---")

if __name__ == '__main__':
    main()