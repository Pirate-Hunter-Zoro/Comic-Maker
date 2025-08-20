# author.py (v2.0 - The Dual-Core Prose Engine!)
# This bot has two functions:
# 1. write_first_draft: Takes a prompt and writes a new piece of text.
# 2. edit_draft: Takes existing text and a critique, and edits it.

import os
from dotenv import load_dotenv
import google.generativeai as genai

# --- INITIALIZATION & ENVIRONMENT SETUP ---
load_dotenv()

# --- CONFIGURATION ---
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

def configure_model(model_name='gemini-2.5-pro'):
    """Configures and returns a Gemini model instance."""
    if not GEMINI_API_KEY:
        raise ValueError("ERROR: GOOGLE_API_KEY not found. Please set it in your .env file.")
    
    genai.configure(api_key=GEMINI_API_KEY)
    
    # We use the most powerful model for creative writing tasks!
    return genai.GenerativeModel(model_name)

# --- BOT FUNCTIONS ---

def write_first_draft(prompt_string: str) -> str:
    """
    Takes a detailed prompt from the Planner and writes the first draft.
    """
    print("--- Author Bot: Engaging WRITE Mode ---")
    
    model = configure_model() # Using the powerful author model!

    # This meta-prompt tells the AI its role as a creative writer.
    meta_prompt = f"""
You are a master storyteller and a brilliant author. Your task is to take the following prompt, which outlines a specific part of a chapter, and write a compelling, immersive, and high-quality narrative section based on it.

Adhere strictly to all 'CORE DIRECTIVES FOR SUPERIOR WRITING QUALITY' and the specific objectives laid out in the prompt. Your writing should be vivid, emotional, and deeply engaging.

Here is your prompt:
---
{prompt_string}
---

Now, write the chapter part.
"""

    print("Generating first draft...")
    response = model.generate_content(meta_prompt)
    print("First draft complete!")
    
    return response.text

def edit_draft(original_text: str, critique_feedback: str, original_prompt: str) -> str:
    """
    Takes the original text, a critique, and the original prompt, and produces an edited version.
    """
    print("--- Author Bot: Engaging EDIT Mode ---")
    
    model = configure_model() # Also uses the powerful model for smart editing.

    # This meta-prompt is for the complex task of editing, not rewriting!
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
    
    print("Revising draft based on critique...")
    response = model.generate_content(meta_prompt)
    print("Edits complete!")
    
    return response.text

# --- TESTING BLOCK ---
def main():
    """A simple function to test the author bot's capabilities."""
    print("--- Running Author Bot Test Sequence ---")
    
    # 1. Create a sample prompt (like one from our Planner)
    sample_prompt = """
### **CORE DIRECTIVES FOR SUPERIOR WRITING QUALITY**

* **Word Count Goal:** This section should be approximately 100-200 words for this test.
* **PROFOUND Individual Character Resonance & Internal Monologue:** Focus on Ruby's feelings.
* **"Show, Don't Tell" - In EXCRUCIATING Detail:** Use her actions to show her emotions.

---

### **PROMPT FOR CHAPTER 1, PART 1: Crash Landing**

**Objective:** Ruby must survive her chaotic arrival in Vacuo, grappling with the disorientation of her return.

**Crucial Ending Point:** The section must end with Ruby getting to her feet and looking at the sky.
"""

    # 2. Test the WRITE function
    first_draft = write_first_draft(sample_prompt)
    print("\n--- FIRST DRAFT ---")
    print(first_draft)
    
    # 3. Create a sample critique
    sample_critique = """
- The description is good, but it 'tells' us Ruby is confused. Please 'show' it more through her actions. For example, have her stumble or check her weapon reflexively.
- The ending is a bit weak. Make her observation of the sky more meaningful. Does it look different from the Ever After's sky?
"""
    
    print("\n--- SENDING CRITIQUE FOR REVISION ---")
    print(sample_critique)
    
    # 4. Test the EDIT function
    edited_draft = edit_draft(
        original_text=first_draft,
        critique_feedback=sample_critique,
        original_prompt=sample_prompt
    )
    
    print("\n--- EDITED DRAFT ---")
    print(edited_draft)
    print("\n--- Author Bot Test Sequence Complete! ---")


if __name__ == '__main__':
    main()