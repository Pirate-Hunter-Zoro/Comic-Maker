# archivist.py (v1.0 - The Keeper of the Canon)
# This bot summarizes the key events from a chapter part and appends them
# to the official plot events log.

import os
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai

# --- INITIALIZATION & ENVIRONMENT SETUP ---
load_dotenv()

# --- CONFIGURATION & PATHING ---
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
KNOWLEDGE_DB_DIR = os.path.join(PROJECT_ROOT, "knowledge_db")
PLOT_EVENTS_FILE = os.path.join(KNOWLEDGE_DB_DIR, "rwby_plot_events.md")

# --- BOT FUNCTIONS ---

def summarize_events_from_text(prose_text: str, chapter_part_info: str) -> str:
    """
    Uses the Gemini API to read a block of prose and summarize the key plot events.
    """
    print("--- Archivist Bot: Engaging SUMMARY Mode ---")

    if not GEMINI_API_KEY:
        raise ValueError("ERROR: GOOGLE_API_KEY not found. Please set it in your .env file.")

    genai.configure(api_key=GEMINI_API_KEY)
    
    # A cost-effective model is perfect for this summarization task.
    model = genai.GenerativeModel('gemini-2.5-pro')

    meta_prompt = f"""
You are a meticulous archivist. Your task is to read the following text from a novel and summarize the key, canon plot events that occurred within it.

**Instructions:**
- List only concrete events, discoveries, and major character decisions.
- Ignore internal monologues, descriptive prose, and general emotional states.
- Present the events as a concise, bulleted list.
- If no significant plot events occurred, return the single phrase "No major plot events."

**Source:** {chapter_part_info}

--- PROSE TO SUMMARIZE ---
{prose_text}
---

Now, provide the summary of events.
"""
    
    print("Archivist Bot: Summarizing key events...")
    response = model.generate_content(meta_prompt)
    print("Archivist Bot: Summary complete!")

    return response.text.strip()

def append_events_to_log(events_summary: str, chapter_part_info: str):
    """
    Appends the summarized events to the rwby_plot_events.md file.
    """
    print(f"--- Archivist Bot: Updating {os.path.basename(PLOT_EVENTS_FILE)} ---")
    
    # Don't do anything if there are no events to log!
    if "No major plot events." in events_summary:
        print("No plot events to log. The timeline remains unchanged.")
        return
    
    # Format the entry for the markdown file
    log_entry = f"""
## {chapter_part_info}
{events_summary}
"""
    
    try:
        # Use 'a' mode to append to the end of the file! So simple!
        with open(PLOT_EVENTS_FILE, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        print("Successfully appended new events to the plot log!")
    except Exception as e:
        print(f"--- CRITICAL ARCHIVIST ERROR ---")
        print(f"Failed to write to the plot events file. Details: {e}")
        raise

# --- TESTING BLOCK ---
def main():
    """A simple function to test the archivist bot's capabilities."""
    print("--- Running Archivist Bot Test Sequence ---")

    # 1. Create a sample piece of prose with clear events
    sample_prose = """
The team stumbled through the sand until Blake's sharp ears picked up a faint, rhythmic clang. Following the sound, they crested a dune and saw it: the half-buried wreckage of a Shade Academy transport skiff. Weiss analyzed the claw marks on the hull and confirmed it was attacked by a large Grimm, likely a Manticore. As they approached, Jaune found a datapad in the sand, its screen still faintly glowing.
"""
    chapter_info = "Chapter 1, Part 3"

    print("\n--- ARCHIVING THE FOLLOWING PROSE ---")
    print(sample_prose)
    print("------------------------------------")

    # 2. Summarize the events
    event_summary = summarize_events_from_text(sample_prose, chapter_info)

    print("\n--- GENERATED SUMMARY ---")
    print(event_summary)

    # 3. Append the events to the log
    # WRAPPER TO PREVENT ACCIDENTALLY WRITING TO YOUR REAL FILE DURING A TEST
    try:
        print("\n--- Attempting to append to log... ---")
        append_events_to_log(event_summary, chapter_info)
        print("\nNOTE: This was a successful simulation. Check your rwby_plot_events.md file to see the new entry!")
    except Exception as e:
        print(f"Test failed: {e}")
    
    print("\n--- Archivist Bot Test Sequence Complete! ---")


if __name__ == '__main__':
    main()