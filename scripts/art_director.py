# scripts/art_director.py

import os
import google.generativeai as genai
from dotenv import load_dotenv
import json

# --- Configuration ---
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Using a powerful model for this creative task
MODEL_NAME = "gemini-2.5-pro"

def generate_image_prompt_from_prose(prose_text):
    """
    Analyzes prose to generate a detailed image prompt AND a list of key characters.
    Returns a tuple: (prompt_string, character_list)
    """
    print("🎨 Art Director Bot v2.0 (Diplomatic Edition) activated! Analyzing prose for visual potential...")
    
    model = genai.GenerativeModel(MODEL_NAME)
    
    prompt_for_director = f"""
    You are an expert Art Director for the RWBY series. Your task is to read a passage of prose and generate a single, detailed, and vivid image prompt that captures the most visually striking moment. Additionally, you must identify the key characters present.

    **CRITICAL SAFETY DIRECTIVE:** The image generation model you are writing a prompt for has sensitive safety filters. To avoid tripping them, do NOT use words directly associated with violence, gore, self-harm, or overt aggression (e.g., avoid words like 'battle', 'brutal', 'blood', 'kill', 'fight', 'war-torn', 'attack'). Instead, use creative, evocative language to imply the scene's intensity and mood. Describe the aftermath of a conflict, not the conflict itself. Focus on expressions, environment, and atmosphere to convey the drama.

    Analyze the following prose:
    ---
    {prose_text}
    ---

    Based on the prose, provide your output in a single JSON object with two keys:
    1. "prompt": A string containing the detailed, evocative, and SAFETY-COMPLIANT art prompt. The prompt should be a single paragraph describing the scene, characters' actions and expressions, the environment, and the overall mood. Describe the visual style as "beautiful anime-style" and "vibrant colors".
    2. "characters": A JSON list of strings containing the full names of the key characters in the scene (e.g., ["Ruby Rose", "Weiss Schnee"]). If no specific characters are mentioned, provide an empty list [].

    Do not include any other text or formatting in your response besides the single JSON object.
    """

    try_count = 0
    while try_count < 3:
        try:
            response = model.generate_content(prompt_for_director)
            
            clean_response_text = response.text.strip().replace("```json", "").replace("```", "").strip()
            
            data = json.loads(clean_response_text)
            
            image_prompt = data.get("prompt", "")
            characters = data.get("characters", [])
            
            if image_prompt and isinstance(characters, list):
                print(f"✅ Art Director analysis complete.")
                print(f"   - Identified Characters: {characters}")
                print(f"   - Generated (Safety-Compliant) Prompt: '{image_prompt}'")
                return image_prompt, characters
            else:
                raise ValueError("JSON output was missing 'prompt' or 'characters' key.")

        except (json.JSONDecodeError, ValueError, Exception) as e:
            try_count += 1
            print(f"⚠️ Anomaly in Art Director's brainwave! Retrying... (Attempt {try_count}/3). Error: {e}")
            if try_count >= 3:
                print("❌ ERROR: Art Director failed to generate a valid response after 3 attempts.")
                return "A beautiful anime-style portrait of a main character from RWBY.", []

    return "A beautiful anime-style portrait of a main character from RWBY.", []

# --- NEW DE-EDGIFIER RAY FUNCTION! ---
def refine_prompt_for_safety(original_prompt):
    """
    Takes a rejected prompt and rephrases it to be safer for generation models.
    """
    print("   > Engaged De-Edgifier Ray! Rephrasing rejected prompt...")
    model = genai.GenerativeModel(MODEL_NAME)
    
    refinement_prompt = f"""
    The following image prompt was rejected by an AI's safety filter, likely due to words implying violence, aggression, or other sensitive topics.
    Your task is to rephrase this prompt to be more evocative and atmospheric, focusing on emotion and environment rather than direct action or potentially triggering words. Maintain the core characters, setting, and mood, but express it in a safer, more poetic way.

    ORIGINAL REJECTED PROMPT:
    "{original_prompt}"

    Return only the new, safer prompt as a single string.
    """
    
    try:
        response = model.generate_content(refinement_prompt)
        new_prompt = response.text.strip()
        print(f"   > De-Edgifier successful! New prompt: '{new_prompt}'")
        return new_prompt
    except Exception as e:
        print(f"   > 💥 De-Edgifier Ray malfunctioned! Error: {e}")
        return original_prompt # Failsafe, return the original prompt