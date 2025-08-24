# src/bots/prompt_generator.py
# Generates a 5-part story plan from a chapter summary.

import json
import re
import os
import google.generativeai as genai

def extract_chapter_summary(full_outline_text: str, chapter_number: int) -> str:
    """Finds and returns the summary for a specific chapter from the outline."""
    print("Planner: Extracting chapter summary.")
    pattern = re.compile(
        rf"### Chapter {chapter_number}: .*?\n(.*?)(?=\n### Chapter|\Z)",
        re.DOTALL
    )
    match = pattern.search(full_outline_text)
    if match:
        return match.group(1).strip()
    else:
        raise ValueError(f"Chapter {chapter_number} could not be found in the outline.")

def generate_story_beats_from_api(chapter_summary: str, config: dict) -> list:
    """
    Calls the Gemini API to break down a chapter summary into five story beats.
    """
    print("Planner: Deconstructing chapter summary into 5 story beats.")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-pro')
    
    universe_name = config.get("universe_name", "the story")

    # The meta-prompt is now generalized using the universe_name from the config.
    meta_prompt = f"""
You are a master storyteller and a narrative deconstruction expert. Your task is to take a high-level chapter summary for a {universe_name} novel and break it down into exactly five sequential, logical, and compelling story beats.

**Output Format Directives:**
* You MUST return your response as a valid JSON object.
* The root of the object must be a single key "beats" which contains a list of the five beat objects.
* Do not include any other text, explanation, or markdown formatting like ```json outside of the JSON object itself.
* For each beat, you must provide: `title`, `objective`, `ending_point`, `key_characters`, and `key_locations`.

**Creative & Stylistic Directives:**
1.  **Adhere to the Source:** The five beats, when combined, must faithfully represent all key events and plot points from the provided chapter summary. Do not invent major events not implied by the summary.
2.  **Logical Progression:** Ensure the five beats represent a logical and well-paced narrative arc. Each beat must flow directly from the previous one.
3.  **Grounded Tone:** The tone must be serious and consistent with the {universe_name} universe. Avoid clichés and tired tropes.
4.  **Actionable Objectives:** The `objective` and `ending_point` for each beat should be concrete and actionable for a writer, focusing on character actions, emotional shifts, and key discoveries.

Here is the chapter summary to deconstruct:
---
{chapter_summary}
---
"""

    print("Planner: Sending deconstruction request to generative model.")
    response = model.generate_content(meta_prompt)

    # Simplified cleanup and robust parsing.
    try:
        clean_response = re.search(r'\{.*\}', response.text, re.DOTALL).group(0)
        story_beats = json.loads(clean_response)['beats']
        if len(story_beats) != 5:
            raise ValueError(f"Model returned {len(story_beats)} beats instead of 5.")
        print("Planner: Successfully parsed 5 story beats.")
        return story_beats
    except (json.JSONDecodeError, KeyError, ValueError, AttributeError) as e:
        print(f"--- PLANNER ERROR: Failed to parse model response. ---")
        print(f"Details: {e}")
        print("--- RAW MODEL RESPONSE ---")
        print(response.text)
        raise