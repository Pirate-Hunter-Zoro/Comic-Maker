# model_lister.py (v1.0 - The Oracle)
# This script has one purpose: to connect to the Google AI API and list
# all available models and their capabilities, so we can finally learn
# the TRUE NAME of the image generation model.

import os
from dotenv import load_dotenv
import google.generativeai as genai

# --- INITIALIZATION & ENVIRONMENT SETUP ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

def main():
    """Connects to the API and prints a list of available models."""
    print("--- Oracle Bot Online ---")
    print("Asking the great eye in the server for its secrets...")

    if not GEMINI_API_KEY:
        raise ValueError("ERROR: GOOGLE_API_KEY not found. Please set it in your .env file.")

    genai.configure(api_key=GEMINI_API_KEY)

    print("\n--- AVAILABLE MODELS & SUPPORTED METHODS ---")
    for m in genai.list_models():
        print(f"MODEL NAME: {m.name}")
        print(f"  - Supported Methods: {m.supported_generation_methods}\n")
    
    print("--- Oracle has spoken. These are the true names. ---")


if __name__ == '__main__':
    main()