# src/bots/critic.py
# Analyzes prose against a dynamic, project-specific knowledge base.

import os
import re
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

def get_relevant_knowledge(project_path: Path, config: dict, key_characters: list, key_locations: list) -> str:
    """
    Dynamically retrieves knowledge from files specified in the project config.
    """
    print("Critic: Loading relevant knowledge.")
    knowledge_text = ""
    knowledge_dir = project_path / config['paths']['knowledge']
    knowledge_map = config.get("knowledge_files", {})

    # Targeted retrieval for characters
    if key_characters and "characters" in knowledge_map:
        char_file = knowledge_dir / knowledge_map["characters"]
        if char_file.exists():
            char_content = char_file.read_text(encoding='utf-8')
            relevant_char_text = ""
            for char_name in key_characters:
                pattern = re.compile(rf"## {re.escape(char_name)}(.*?)(?=\n## |\Z)", re.DOTALL)
                match = pattern.search(char_content)
                if match:
                    relevant_char_text += match.group(0) + "\n"
            if relevant_char_text:
                knowledge_text += f"\n\n--- RELEVANT KNOWLEDGE: CHARACTERS ---\n\n{relevant_char_text}"

    # Targeted retrieval for locations
    if key_locations and "locations" in knowledge_map:
        loc_file = knowledge_dir / knowledge_map["locations"]
        if loc_file.exists():
            loc_content = loc_file.read_text(encoding='utf-8')
            relevant_loc_text = ""
            for loc_name in key_locations:
                pattern = re.compile(rf"## {re.escape(loc_name)}(.*?)(?=\n## |\Z)", re.DOTALL)
                match = pattern.search(loc_content)
                if match:
                    relevant_loc_text += match.group(0) + "\n"
            if relevant_loc_text:
                knowledge_text += f"\n\n--- RELEVANT KNOWLEDGE: LOCATIONS ---\n\n{relevant_loc_text}"

    # Always-include general lore files
    if "general_lore" in knowledge_map:
        knowledge_text += "\n\n--- GENERAL KNOWLEDGE ---\n"
        for lore_filename in knowledge_map["general_lore"]:
            lore_file = knowledge_dir / lore_filename
            if lore_file.exists():
                knowledge_text += f"\n--- From {lore_filename} ---\n"
                knowledge_text += lore_file.read_text(encoding='utf-8')

    print("Critic: Knowledge loaded.")
    return knowledge_text

def critique_text(
    prose_text: str,
    original_prompt: str,
    key_characters: list,
    key_locations: list,
    project_path: Path,
    config: dict
) -> str:
    """
    Analyzes prose against the prompt, directives, and a dynamic knowledge base.
    """
    print("Critic: Engaging analysis.")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment.")
    
    genai.configure(api_key=api_key)
    
    model_name = config.get("model_names", {}).get("critic", "gemini-1.5-pro")
    model = genai.GenerativeModel(model_name)
    universe_name = config.get("universe_name", "the story's")

    knowledge_base = get_relevant_knowledge(project_path, config, key_characters, key_locations)
    
    meta_prompt = f"""
You are an expert literary critic and a {universe_name} lore master. Your job is to analyze the provided 'PROSE TO REVIEW' based on a strict set of rules and a comprehensive knowledge base.

Your analysis must cover three areas:
1.  **Writing Quality:** Does the prose adhere to all directives in the 'ORIGINAL PROMPT'? Is the writing style compelling?
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

    print("Critic: Analyzing prose.")
    response = model.generate_content(meta_prompt)
    print("Critic: Analysis complete.")

    return response.text.strip()