# critic.py (v2.0 - The Precision Lore Master & Style Guardian)
# This bot uses targeted knowledge retrieval to be both efficient and thorough.

import os
import argparse
import re
from dotenv import load_dotenv
import google.generativeai as genai

# --- INITIALIZATION & ENVIRONMENT SETUP ---
load_dotenv()

# --- CONFIGURATION & PATHING ---
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
KNOWLEDGE_DB_DIR = os.path.join(PROJECT_ROOT, "knowledge_db")

# A mapping of our knowledge files for easy access!
# We now categorize them for smarter retrieval!
KNOWLEDGE_FILES = {
    "characters": os.path.join(KNOWLEDGE_DB_DIR, "rwby_characters.md"),
    "locations": os.path.join(KNOWLEDGE_DB_DIR, "rwby_locations.md"),
    # These are our "Always Include" files for general context!
    "lore_magic": os.path.join(KNOWLEDGE_DB_DIR, "rwby_lore_magic.md"),
    "plot_outline": os.path.join(KNOWLEDGE_DB_DIR, "rwby_novel_plot_outline.md"),
    "plot_events": os.path.join(KNOWLEDGE_DB_DIR, "rwby_plot_events.md")
}

# --- HELPER FUNCTIONS ---

def load_file_content(file_path):
    """A trusty function to load content from any text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"CRITIC WARNING: Knowledge file not found at {file_path}")
        return ""

def get_relevant_knowledge(key_characters: list, key_locations: list) -> str:
    """
    THIS IS THE NEW, SMARTER BRAIN!
    It performs targeted data retrieval for characters and locations,
    but ALWAYS includes general lore, plot, and events.
    """
    print("Critic Bot: Accessing Knowledge Database with precision...")
    knowledge_text = ""
    
    # --- 1. Targeted Retrieval for Characters ---
    print(f"Searching for data on characters: {key_characters}")
    char_content = load_file_content(KNOWLEDGE_FILES["characters"])
    relevant_char_text = ""
    for char_name in key_characters:
        # This regex finds the character's section from an '##' heading
        # to the next '##' heading. So clever!
        pattern = re.compile(rf"## {re.escape(char_name)}(.*?)(?=\n## |\Z)", re.DOTALL)
        match = pattern.search(char_content)
        if match:
            print(f"Found entry for {char_name}.")
            relevant_char_text += match.group(0) + "\n"
    if relevant_char_text:
        knowledge_text += f"\n\n--- RELEVANT KNOWLEDGE: CHARACTERS ---\n\n{relevant_char_text}"

    # --- 2. Targeted Retrieval for Locations ---
    print(f"Searching for data on locations: {key_locations}")
    loc_content = load_file_content(KNOWLEDGE_FILES["locations"])
    relevant_loc_text = ""
    for loc_name in key_locations:
        pattern = re.compile(rf"## {re.escape(loc_name)}(.*?)(?=\n## |\Z)", re.DOTALL)
        match = pattern.search(loc_content)
        if match:
            print(f"Found entry for {loc_name}.")
            relevant_loc_text += match.group(0) + "\n"
    if relevant_loc_text:
        knowledge_text += f"\n\n--- RELEVANT KNOWLEDGE: LOCATIONS ---\n\n{relevant_loc_text}"

    # --- 3. Always-Include General Lore ---
    print("Loading general lore, plot, and events files for full context...")
    knowledge_text += "\n\n--- GENERAL KNOWLEDGE: LORE & MAGIC ---\n\n"
    knowledge_text += load_file_content(KNOWLEDGE_FILES["lore_magic"])
    knowledge_text += "\n\n--- GENERAL KNOWLEDGE: PLOT OUTLINE ---\n\n"
    knowledge_text += load_file_content(KNOWLEDGE_FILES["plot_outline"])
    knowledge_text += "\n\n--- GENERAL KNOWLEDGE: PLOT EVENTS (SO FAR) ---\n\n"
    knowledge_text += load_file_content(KNOWLEDGE_FILES["plot_events"])

    print("Critic Bot: All necessary knowledge loaded.")
    return knowledge_text

# --- BOT FUNCTION ---

def critique_text(
    prose_text: str,
    original_prompt: str,
    key_characters: list,
    key_locations: list
) -> str:
    """
    Analyzes the prose against the prompt, directives, and our knowledge base.
    """
    print("--- Critic Bot: Engaging ANALYSIS Mode v2.0 ---")

    if not GEMINI_API_KEY:
        raise ValueError("ERROR: GOOGLE_API_KEY not found. Please set it in your .env file.")
    
    genai.configure(api_key=GEMINI_API_KEY)
    
    model = genai.GenerativeModel('gemini-2.5-pro')  # Using the powerful critic model!

    # 1. Gather all the knowledge the Critic needs using our new smart function!
    knowledge_base = get_relevant_knowledge(key_characters, key_locations)
    
    # 2. The meta-prompt is the same, but now it receives a smarter knowledge base!
    meta_prompt = f"""
You are an expert literary critic and a RWBY lore master. Your job is to analyze the provided 'PROSE TO REVIEW' based on a strict set of rules and a comprehensive knowledge base.

Your analysis must cover three areas:
1.  **Writing Quality:** Does the prose adhere to all directives in the 'ORIGINAL PROMPT' (e.g., word count, "show don't tell," sensory details)? Is the writing style compelling?
2.  **Prompt Adherence:** Does the prose successfully achieve the `Objective` and `Crucial Ending Point` outlined in the 'ORIGINAL PROMPT'?
3.  **Lore & Continuity:** Is the prose consistent with the information provided in the 'KNOWLEDGE BASE'? Check for character voice, location accuracy, and correct use of lore.

**Output Rules:**
- If the prose is perfect and passes all checks, your ONLY response must be the single word: SUCCESS
- If there are any issues, provide a bulleted list of specific, actionable points of feedback for the author. Do not be conversational.
- Be strict. Do not approve text with even minor issues.

--- KNOWLEDGE BASE ---
{knowledge_base}
---

--- ORIGINAL PROMPT ---
{original_prompt}
---

--- PROSE TO REVIEW ---
{prose_text}
---

Now, provide your critique.
"""

    print("Critic Bot: Analyzing prose with precision knowledge...")
    response = model.generate_content(meta_prompt)
    print("Critic Bot: Analysis complete!")

    return response.text.strip()

# --- TESTING BLOCK ---
def main():
    """A simple function to test the critic bot's new capabilities."""
    print("--- Running Critic Bot v2.0 Test Sequence ---")

    sample_prompt = "Doesn't matter for this test."
    # This text has a lore mistake! Ruby's Semblance is Petal Burst, not super speed.
    sample_prose = "Ruby zipped across the sand, moving so fast she was just a blur. Her super speed was a handy trick for covering ground quickly."

    print("\n--- CRITIQUING THE FOLLOWING PROSE ---")
    print(sample_prose)
    print("--- (EXPECTING A LORE CORRECTION) ---")

    critique_result = critique_text(
        prose_text=sample_prose,
        original_prompt=sample_prompt,
        key_characters=["Ruby"],
        key_locations=["Vacuo Desert"]
    )

    print("\n--- CRITIQUE RESULT ---")
    print(critique_result)
    
    print("\n--- Critic Bot Test Sequence Complete! ---")


if __name__ == '__main__':
    main()