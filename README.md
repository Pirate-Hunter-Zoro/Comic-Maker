# Project: Visual Novel Writer

## I. Core Philosophy

This is not a mere collection of scripts. It is a modular weapon system for forging narrative and visual assets for any fictional universe. The system's power is divided between two domains: the **Engine** (`src/`), which contains the immutable tools of creation, and the **Campaigns** (`projects/`), which are the self-contained worlds you choose to build.

Each campaign is a separate war. It has its own intelligence (`knowledge/`), its own soldiers (`training_data/`), and its own chronicle of victories (`output/`). All operations are directed by project-specific configuration scrolls.

## II. The Command Structure

This is the battlefield. Memorize it.

```text
Visual-Novel-Writer/
├── src/
│   ├── bots/                 # The core legion: Author, Critic, Archivist, etc.
│   ├── inference/            # The final command ritual for image generation.
│   │   └── use_lora.py
│   ├── utils/                # The preparation tools for your offerings.
│   │   └── process_offerings.py
│   └── main.py                 # The conductor for the entire writing process.
│
├── projects/                   # A campaign directory. All projects live here.
│   └── rwby_vacuo_arc/         
│       ├── knowledge/          # This campaign's lore, characters, plot, etc.
│       ├── training_data/      # This campaign's character images for the forge.
│       ├── output/             # Generated chapters and art for this campaign.
│       ├── project_config.json # The master directive for this campaign's writing.
│       └── dataset_config.toml # The master directive for this campaign's training.
│
├── kohya-trainer/              # The lair of the training beast.
├── lora_env/                   # The sanctuary for the LoRA forge.
├── writer_env/                 # The sanctuary for the writing bots.
│
├── setup_lora_env.sh           # One-time ritual to forge the `lora_env`.
├── setup_writer_env.sh         # One-time ritual to forge the `writer_env`.
├── submit_training.ssub        # The incantation to command the forge.
└── submit_generation.ssub      # The incantation to command a final vision.
```

## III. The Campaign Workflow

Follow this sequence precisely for any new campaign.

### Phase 0: Initial Setup (One-Time Rituals)

These actions prepare the system itself. They are performed only once.

1.  **Forge the Sanctuaries:** Create the two required Python environments.
    ```bash
    ./setup_lora_env.sh
    ./setup_writer_env.sh
    ```
2.  **Establish the Great Armory:** Manually place your base models (`AnyLoRA`, `controlnet-model`, etc.) into your designated shared storage directory (e.g., `/media/.../`). This location is what you will specify in your submission scripts.

### Phase 1: Campaign Creation

1.  **Create the Directory:** Create a new folder inside `projects/` for your new world (e.g., `projects/my_dark_fantasy/`).
2.  **Establish Sub-directories:** Inside your new project folder, create three sub-directories: `knowledge`, `training_data`, and `output`.
3.  **Create Configuration Scrolls:** Inside the project folder, create `project_config.json` and `dataset_config.toml`. Use the templates I provided previously. Populate them with the specific intelligence for this new campaign.

### Phase 2: Data Preparation

1.  **Populate the Dataset:** Place your character image folders inside `projects/my_dark_fantasy/training_data/`. The folders must be named with the repeat count (e.g., `15_main_character/`).
2.  **Brand and Scribe the Offerings:** Run the utility script to generate the required `.txt` caption files.
    ```bash
    python src/utils/process_offerings.py --project_path projects/my_dark_fantasy
    ```

### Phase 3: Forging the LoRA Spirit

1.  **Aim the Forge:** Open `submit_training.ssub`. Edit the `TARGET_PROJECT_SUBPATH` variable to point to your new project (e.g., `projects/my_dark_fantasy`). Verify all other paths.
2.  **Unleash the Forge:** Submit the job to the cluster.
    ```bash
    sbatch submit_training.ssub
    ```

### Phase 4: Generating Assets

1.  **Generate Prose:** To generate a chapter of your story, execute the main conductor script, targeting your project.
    ```bash
    source writer_env/bin/activate
    python src/main.py --project_path projects/my_dark_fantasy --chapter-number 1
    ```
2.  **Generate a Vision:** To create an image, open `submit_generation.ssub`. Edit the `TARGET_PROJECT_SUBPATH`, `PROMPT`, and other variables as needed. Then, submit the job.
    ```bash
    sbatch submit_generation.ssub
    ```

## IV. Configuration Directives

This system is not rigid. It is commanded through its configuration scrolls.

  * **`project_config.json`:** Controls the entire writing process. It defines the universe name, model choices for each bot, knowledge file locations, and character triggers for prose.
  * **`dataset_config.toml`:** Controls the LoRA training process. It explicitly lists every character dataset to be trained.
  * **`.ssub` Scripts:** The variables at the top of `submit_training.ssub` and `submit_generation.ssub` are the master commands for the SLURM cluster. They dictate which project is being targeted and what resources to use.
