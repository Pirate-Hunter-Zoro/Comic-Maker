# src/bots/author.py
# Writes and edits narrative prose based on provided prompts and critiques.

import os
from dotenv import load_dotenv
import google.generativeai as genai

# Environment setup is loaded once, centrally.
load_dotenv()

def configure_model(config: dict):
    """Configures and returns a Gemini model instance based on the project config."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment.")
    
    genai.configure(api_key=api_key)
    
    # The model name is now drawn from the project configuration.
    model_name = config.get("model_names", {}).get("author", "gemini-1.5-pro")
    print(f"Author: Using model '{model_name}'.")
    return genai.GenerativeModel(model_name)

def write_first_draft(prompt_string: str, config: dict) -> str:
    """Takes a prompt and writes the first draft of a narrative section."""
    print("Author: Writing first draft.")
    
    model = configure_model(config)

    meta_prompt = f"""
You are a master storyteller and a brilliant author. Your task is to take the following prompt, which outlines a specific part of a chapter, and write a compelling, immersive, and high-quality narrative section based on it.

Your writing should be vivid, emotional, and deeply engaging.

Here is your prompt:
---
{prompt_string}
---

Now, write the chapter part.
"""
    response = model.generate_content(meta_prompt)
    print("Author: First draft complete.")
    
    return response.text

def edit_draft(original_text: str, critique_feedback: str, original_prompt: str, config: dict) -> str:
    """Takes original text, a critique, and the original prompt to produce a revised version."""
    print("Author: Revising draft based on critique.")
    
    model = configure_model(config)

    meta_prompt = f"""
You are an expert editor. Your task is to intelligently revise the 'ORIGINAL TEXT' based on the specific points provided in the 'CRITIQUE FEEDBACK'.

**Your Core Mission:** Do NOT rewrite the entire text from scratch. Your goal is to act like a human editor, preserving the original prose as much as possible while surgically implementing the required changes. Make the edits feel seamless and natural.

For context, here is the 'ORIGINAL PROMPT' the text was based on. Use it to ensure your edits remain true to the initial objective.

--- ORIGINAL PROMPT ---
{original_prompt}
---

--- ORIGINAL TEXT TO EDIT ---
{original_text}
---

--- CRITIQUE FEEDBACK (Points to address) ---
{critique_feedback}
---

Now, provide the full, edited version of the text with the feedback incorporated.
"""
    response = model.generate_content(meta_prompt)
    print("Author: Revisions complete.")
    
    return response.text