# main.py
# Orchestrates the generation of a chapter based on a project configuration.

import os
import argparse
import time
import json
from pathlib import Path

# Updated imports for the new structure
from bots.prompt_generator import generate_story_beats_from_api, extract_chapter_summary
from bots.author import write_first_draft, edit_draft
from bots.critic import critique_text
from bots.archivist import summarize_events_from_text, append_events_to_log
from bots.art_director import generate_image_prompt_from_prose

# --- Configuration ---
MAX_CRITIC_REVIEWS = 3

def load_config(project_path: Path) -> dict:
    """Loads the project_config.json file from the specified project path."""
    config_path = project_path / "project_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_file_content(path: Path) -> str:
    """A direct file content loader."""
    if not path.exists():
        raise FileNotFoundError(f"File not found at: {path}")
    return path.read_text(encoding='utf-8')

# --- Main Orchestration ---

def main():
    """Orchestrates the chapter generation process."""
    parser = argparse.ArgumentParser(description="Generates a chapter for a specified project.")
    parser.add_argument('--project_path', type=str, required=True, help='The path to the project directory.')
    parser.add_argument('--chapter-number', type=int, required=True, help='The chapter number to generate.')
    args = parser.parse_args()

    project_path = Path(args.project_path)
    print(f"Conductor: Initiating generation for Chapter {args.chapter_number} of project '{project_path.name}'.")
    
    try:
        config = load_config(project_path)
        
        # Construct paths from config
        output_dir = project_path / config['paths']['output']
        plot_outline_file = project_path / config['paths']['plot_outline']
        chapter_output_dir = output_dir / f'chapter_{args.chapter_number:02d}'
        os.makedirs(chapter_output_dir, exist_ok=True)
    
        # --- Planner Stage ---
        print("Conductor: Engaging Planner.")
        outline_text = get_file_content(plot_outline_file)
        
        # NOTE: These bot functions will need to be refactored next to accept the config.
        # The calls are updated here in anticipation of that change.
        chapter_summary = extract_chapter_summary(outline_text, args.chapter_number)
        planned_beats = generate_story_beats_from_api(chapter_summary, config)
        print("Conductor: Planner has delivered the 5-part plan.")

        # --- Writing Loop ---
        for i, beat_data in enumerate(planned_beats, 1):
            print(f"\n{'='*20} Processing Chapter {args.chapter_number}, Part {i} {'='*20}")
            
            # This prompt construction will be moved into the prompt_generator bot later.
            prompt_for_author = f"### **PROMPT FOR CHAPTER {args.chapter_number}, PART {i}: {beat_data['title']}**\n\n**Objective:** {beat_data['objective']}\n\n**Crucial Ending Point:** {beat_data['ending_point']}"
            current_prose = ""
            is_approved = False
            critique_result = ""
            
            for attempt in range(MAX_CRITIC_REVIEWS + 1):
                if attempt == 0:
                    current_prose = write_first_draft(prompt_for_author, config)
                else:
                    current_prose = edit_draft(current_prose, critique_result, prompt_for_author, config)

                print(f"Conductor: Text generated. Submitting for review (Attempt {attempt + 1}/{MAX_CRITIC_REVIEWS + 1}).")
                time.sleep(1)
                critique_result = critique_text(current_prose, prompt_for_author, beat_data['key_characters'], beat_data['key_locations'], project_path, config)

                if critique_result == "SUCCESS":
                    print(f"Critic: SUCCESS. Chapter Part {i} approved.")
                    is_approved = True
                    break
                else:
                    print(f"Critic: Revisions required for Chapter Part {i}.\nFeedback:\n{critique_result}")

            if is_approved:
                part_filename = chapter_output_dir / f'part_{i}_approved.md'
                part_filename.write_text(current_prose, encoding='utf-8')
                print(f"Conductor: Approved text saved to: {part_filename.name}")

                # --- Art Director Stage ---
                print("Conductor: Engaging Art Director.")
                art_prompt, characters = generate_image_prompt_from_prose(current_prose, config)
                
                # Art generation logic will be updated in Phase 3. For now, it's a placeholder.
                if art_prompt and characters:
                    print("Art Director: Prompt generated.")
                else:
                    print("Art Director: Failed to generate prompt.")
                
                # --- Archivist Stage ---
                print("Conductor: Engaging Archivist.")
                chapter_part_info = f"Chapter {args.chapter_number}, Part {i}"
                events_summary = summarize_events_from_text(current_prose, chapter_part_info, config)
                append_events_to_log(events_summary, chapter_part_info, project_path, config)
                print("Conductor: Archivist has updated the canon.")

            else:
                raise Exception(f"Failed to produce approved text for Chapter {args.chapter_number}, Part {i} after {MAX_CRITIC_REVIEWS + 1} attempts.")

        # --- Final Compilation ---
        print(f"\n{'='*20} Chapter {args.chapter_number} Generation Complete {'='*20}")
        final_chapter_filename = chapter_output_dir / f'chapter_{args.chapter_number:02d}_complete.md'
        
        final_chapter_parts = []
        for i in range(1, len(planned_beats) + 1):
            part_path = chapter_output_dir / f'part_{i}_approved.md'
            final_chapter_parts.append(part_path.read_text(encoding='utf-8'))
        
        final_text = "\n\n---\n\n".join(final_chapter_parts)
        final_chapter_filename.write_text(final_text, encoding='utf-8')
        
        print(f"Conductor: Chapter {args.chapter_number} compiled and saved to: {final_chapter_filename}")
        print("Conductor: Mission Complete.")

    except Exception as e:
        print(f"\n--- CATASTROPHIC FAILURE ---")
        print(f"Process halted. Details: {e}")
        print("--- Conductor: Emergency Shutdown. ---")

if __name__ == '__main__':
    main()