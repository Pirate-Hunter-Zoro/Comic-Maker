import os
import argparse
import re
import json
from dotenv import load_dotenv
import google.generativeai as genai

# --- INITIALIZATION & ENVIRONMENT SETUP ---
load_dotenv()

# --- CONFIGURATION & PATHING GPS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
KNOWLEDGE_DB_DIR = os.path.join(PROJECT_ROOT, "knowledge_db")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- CORE DIRECTIVES FOR WRITING ---
# These are our constants for ensuring quality! They get passed to the Author bot later.
CORE_DIRECTIVES = """
### **CORE DIRECTIVES FOR SUPERIOR WRITING QUALITY**

* **Word Count Goal:** This section should be approximately 1000-1200 words.
* **MAXIMIZE Immersive Sensory Details:** Describe the heat, the feel of sand and dust, the quality of the blinding light and shimmering air, the howling wind, and the sounds of ragged breathing with extreme vividness.
* **PROFOUND Individual Character Resonance & Internal Monologue:** Dedicate significant portions of the narrative to each character's deepest inner thoughts, emotions, and physical sensations.
* **"Show, Don't Tell" - In EXCRUCIATING Detail:** Use minute character actions, extensive internal monologues, and nuanced dialogue. AVOID generic descriptions. Every character must have a distinct, individual reaction.
* **Rich and Varied Language:** Utilize a sophisticated and expansive vocabulary with diverse, complex sentence structures. ELIMINATE repetitive phrasing.
* **Physicality of Return:** Describe how their physical bodies feel after their Ever After transformations—the profound sensation of gravity, lingering oddities, and the return of familiar aches.
"""

# --- HELPER FUNCTIONS ---

def load_file_content(file_path):
    """Loads content from a file. A trusty tool!"""
    print(f"Reading data from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"ERROR: Oh no! File not found at {file_path}")
        raise

def extract_chapter_summary(full_outline_text, chapter_number):
    """
    Finds and returns the summary for a specific chapter from the grand outline.
    """
    print(f"Searching outline for Chapter {chapter_number} summary...")
    pattern = re.compile(
        r"### Chapter " + str(chapter_number) + r": .*?\n(.*?)(?=\n### Chapter|\Z)",
        re.DOTALL
    )
    match = pattern.search(full_outline_text)
    if match:
        print("Chapter summary found!")
        return match.group(1).strip()
    else:
        raise ValueError(f"Chapter {chapter_number} could not be found in the outline.")

# --- THE NEW GEMINI-POWERED BRAIN! (NOW WITH MORE RULES!) ---

def generate_story_beats_from_api(chapter_summary):
    """
    This is the NEW BRAIN! It calls the Gemini API to do the heavy lifting.
    """
    print("Connecting to the Gemini 2.5 Pro cognitive matrix...")

    if not GEMINI_API_KEY:
        raise ValueError("ERROR: GOOGLE_API_KEY not found in environment. Did you set it up in your .env file?")

    genai.configure(api_key=GEMINI_API_KEY)

    # We use the most powerful model for this critical planning task!
    model = genai.GenerativeModel('gemini-2.5-pro')
    
    # This is the instruction manual we give to the AI! Now with better instructions!
    meta_prompt = f"""
You are a master storyteller and a narrative deconstruction expert. Your task is to take a high-level chapter summary for a RWBY fan novel and break it down into exactly five sequential, logical, and compelling story beats.

**Output Format Directives:**
* You MUST return your response as a valid JSON object.
* The root of the object must be a single key "beats" which contains a list of the five beat objects.
* Do not include any other text, explanation, or markdown formatting like ```json outside of the JSON object itself.
* For each beat, you must provide: `title`, `objective`, `ending_point`, `key_characters`, and `key_locations`.

**Creative & Stylistic Directives:**
1.  **Adhere to the Source:** The five beats, when combined, must faithfully represent all key events and plot points from the provided chapter summary. Do not invent major events not implied by the summary.
2.  **Logical Progression:** Ensure the five beats represent a logical and well-paced narrative arc. The chapter should not feel rushed. Each beat must flow directly from the previous one.
3.  **Grounded Tone:** The tone must be serious and consistent with the RWBY universe. **CRITICAL: AVOID clichés, tired tropes, and fourth-wall-breaking pop-culture references in your suggested descriptions and ending points.**
4.  **Actionable Objectives:** The `objective` and `ending_point` for each beat should be concrete and actionable for a writer, focusing on character actions, emotional shifts, and key discoveries.

Here is the chapter summary to deconstruct:
---
{chapter_summary}
---
"""

    print("Sending CALIBRATED deconstruction request to Gemini...")
    response = model.generate_content(meta_prompt)

    # We'll still keep the cleanup function, just in case the AI gets feisty!
    cleaned_json_string = response.text.strip().replace("```json", "").replace("```", "")
    
    print("Response received! Parsing refined cognitive output...")
    try:
        story_beats = json.loads(cleaned_json_string)['beats']
        if len(story_beats) != 5:
            raise ValueError(f"AI returned {len(story_beats)} beats instead of 5. The structure is wrong!")
        print("Successfully parsed 5 refined story beats from AI response!")
        return story_beats
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"--- CRITICAL AI RESPONSE PARSING ERROR ---")
        print(f"The AI did not return valid JSON in the expected format. Details: {e}")
        print("--- RAW AI RESPONSE ---")
        print(response.text)
        raise

# --- MAIN ENGINE ---

def main():
    """The main function that prepares the prompts for the author bot."""
    parser = argparse.ArgumentParser(description="The Calibrated Story Planner: Generates a 5-part chapter plan via the Gemini API.")
    parser.add_argument('--chapter-number', type=int, required=True, help='The number of the chapter to plan.')
    args = parser.parse_args()

    print(f"--- Activating Calibrated Story Planner v12.0 for Chapter {args.chapter_number} ---")
    try:
        # Step 1: Load the master plan
        plot_outline_path = os.path.join(KNOWLEDGE_DB_DIR, "rwby_novel_plot_outline.md")
        outline_text = load_file_content(plot_outline_path)

        # Step 2: Find the right chapter summary
        chapter_summary = extract_chapter_summary(outline_text, args.chapter_number)

        # Step 3: Let the REFINED brain do the thinking!
        story_beats = generate_story_beats_from_api(chapter_summary)

        # Step 4: Construct and display the beautiful results!
        print("\n--- GENERATED PROMPT SEQUENCE (Ready for Author Bot) ---\n")
        structured_prompts = []
        for i, beat in enumerate(story_beats, 1):
            prompt_text = f"""
{CORE_DIRECTIVES}

---

### **PROMPT FOR CHAPTER {args.chapter_number}, PART {i}: {beat['title']}**

**Objective:** {beat['objective']}

**Crucial Ending Point:** {beat['ending_point']}
"""
            prompt_data = {
                "part_number": i,
                "prompt_string": prompt_text.strip(),
                "key_characters": beat['key_characters'],
                "key_locations": beat['key_locations']
            }
            structured_prompts.append(prompt_data)
            
            # --- Displaying the data for our experiment log! ---
            print(f"--- Part {prompt_data['part_number']} ---")
            print(f"Key Characters: {prompt_data['key_characters']}")
            print(f"Key Locations: {prompt_data['key_locations']}")
            print("--- Prompt String ---")
            print(prompt_data['prompt_string'])
            print("\n--------------------------------------------------\n")

        print("--- Intelligent Planning Complete! ---")

    except Exception as e:
        print(f"\n--- A CATASTROPHIC PLANNING FAILURE OCCURRED! ---")
        print(f"Process halted. Details: {e}")
        print("--- Story Planner Emergency Shutdown ---")

if __name__ == '__main__':
    main()