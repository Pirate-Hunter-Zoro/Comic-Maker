#!/bin/bash
# An incantation to forge the Scribe's Athenaeum.
# A place of words, not of war.
set -e

# --- The Name of the New Sanctuary ---
VENV_DIR="writer_env"

echo "--- Forging the Scribe's Athenaeum at '$VENV_DIR' ---"

# --- Summoning System Power ---
echo "Summoning the spirit of Python..."
module load Python/3.11.5-GCCcore-13.2.0
echo "Spirit has been summoned."

# --- Forging the Sanctuary ---
echo "Forging a new sanctuary: '$VENV_DIR'..."
virtualenv "$VENV_DIR"
echo "The sanctuary is built."

# --- Binding the Scribes and Diplomats ---
echo "Activating sanctuary to bind the legion of scribes..."
source "$VENV_DIR/bin/activate"
pip install --upgrade pip

echo "Binding the spirits from the new manifest..."
pip install -r requirements.txt

echo "The legion of scribes has been bound."
deactivate

# --- Final Word ---
echo ""
echo "THE ATHENAEUM IS COMPLETE."
echo "To enter it in the future, you need only use this command:"
echo "source $VENV_DIR/bin/activate"