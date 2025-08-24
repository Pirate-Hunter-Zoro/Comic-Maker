# src/bots/archivist.py
# Summarizes key plot events and appends them to the project's canon log.

import os
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

def summarize_events_from_text(prose_text: str, chapter_part_info: str, config: dict) -> str:
    """
    Uses a generative model to summarize key plot events from prose.
    """
    print("Archivist: Summarizing key events.")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment.")
    genai.configure(api_key=api_key)
    
    model_name = config.get("model_names", {}).get("archivist", "gemini-1.5-pro")
    model = genai.GenerativeModel(model_name)

    meta_prompt = f"""
You are a meticulous archivist. Your task is to read the following text from a novel and summarize the key, canon plot events that occurred within it.

**Instructions:**
- List only concrete events, discoveries, and major character decisions.
- Ignore internal monologues and descriptive prose.
- Present the events as a concise, bulleted list.
- If no significant plot events occurred, return the single phrase "No major plot events."

**Source:** {chapter_part_info}
---
{prose_text}
---
Now, provide the summary of events.
"""
    
    response = model.generate_content(meta_prompt)
    print("Archivist: Summary complete.")

    return response.text.strip()

def append_events_to_log(events_summary: str, chapter_part_info: str, project_path: Path, config: dict):
    """

    Appends the summarized events to the project's plot events log.
    """
    if "No major plot events." in events_summary:
        print("Archivist: No plot events to log.")
        return

    # The path is now constructed dynamically from the project config.
    try:
        plot_events_file = project_path / config['paths']['plot_events']
        log_entry = f"\n\n## {chapter_part_info}\n{events_summary}"
        
        with open(plot_events_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        print(f"Archivist: Appended new events to {plot_events_file.name}.")
    except KeyError:
        print(f"Archivist ERROR: 'plot_events' path not defined in project_config.json.")
    except Exception as e:
        print(f"Archivist ERROR: Failed to write to the plot events file. Details: {e}")
        raise