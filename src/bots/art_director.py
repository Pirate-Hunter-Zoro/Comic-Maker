# src/bots/art_director.py
# Analyzes prose to generate a visually descriptive and safe image prompt.

import os
import google.generativeai as genai
from dotenv import load_dotenv
import json
import re

load_dotenv()

def generate_image_prompt_from_prose(prose_text: str, config: dict) -> tuple[str, list]:
    """
    Analyzes prose to generate an image prompt and identify key characters.
    Returns a tuple: (prompt_string, character_list).
    """
    print("Art Director: Analyzing prose for visual potential.")
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment.")
    genai.configure(api_key=api_key)

    model_name = config.get("model_names", {}).get("art_director", "gemini-1.5-pro")
    model = genai.GenerativeModel(model_name)
    universe_name = config.get("universe_name", "the story's")

    prompt_for_director = f"""
    You are an expert Art Director for the {universe_name} series. Your task is to read a passage of prose and generate a single, detailed, and vivid image prompt that captures the most visually striking moment. Additionally, you must identify the key characters present.

    **CRITICAL SAFETY DIRECTIVE:** The image generation model has sensitive safety filters. Do NOT use words directly associated with violence or gore (e.g., 'battle', 'blood', 'kill', 'fight'). Instead, use evocative language to imply the scene's intensity and mood. Focus on expressions, environment, and atmosphere.

    Analyze the following prose:
    ---
    {prose_text}
    ---

    Provide your output in a single JSON object with two keys:
    1. "prompt": A string containing the detailed, evocative, and SAFETY-COMPLIANT art prompt. Describe the visual style as "beautiful anime-style" and "vibrant colors".
    2. "characters": A JSON list of strings containing the full names of the key characters in the scene (e.g., ["Ruby Rose", "Weiss Schnee"]). If no characters, provide an empty list [].

    Do not include any other text or formatting besides the single JSON object.
    """

    for attempt in range(3):
        try:
            response = model.generate_content(prompt_for_director)
            clean_response_text = re.search(r'\{.*\}', response.text, re.DOTALL).group(0)
            data = json.loads(clean_response_text)
            
            image_prompt = data.get("prompt")
            characters = data.get("characters")
            
            if image_prompt and isinstance(characters, list):
                print("Art Director: Analysis complete.")
                return image_prompt, characters
            else:
                raise ValueError("JSON output was missing required keys.")
        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            print(f"Art Director: Retrying due to model response error (Attempt {attempt + 1}/3). Details: {e}")
            if attempt >= 2:
                print("Art Director: Failed to generate a valid response after 3 attempts.")
                fallback_prompt = f"A beautiful anime-style portrait of a main character from {universe_name}."
                return fallback_prompt, []

    fallback_prompt = f"A beautiful anime-style portrait of a main character from {universe_name}."
    return fallback_prompt, []