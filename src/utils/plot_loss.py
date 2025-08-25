# src/utils/plot_loss.py
# A script to visualize the training loss from a SLURM log file.

import re
from pathlib import Path
import argparse
import matplotlib.pyplot as plt

def plot_loss(log_file: Path, output_location: str):
    """
    Parses a log file to extract and plot training loss over steps.
    """
    if not log_file.is_file():
        print(f"Error: Log file not found at '{log_file}'. A pathetic failure.")
        return

    print(f"Analyzing log file: {log_file.name}")

    # A precise pattern to capture the step and the average loss.
    pattern = re.compile(r"steps:\s+.*?(\d+)/\d+.*avr_loss=([\d.]+)")

    steps = []
    losses = []

    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                step = int(match.group(1))
                loss = float(match.group(2))
                steps.append(step)
                losses.append(loss)

    if not steps:
        print("No training steps found in the log file. The ritual may have failed early.")
        return
    
    print(f"Found {len(steps)} data points. Generating plot...")

    # --- The Visualization Ritual ---
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(steps, losses, color='cyan')
    
    ax.set_title("Training Loss Analysis", fontsize=16)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Average Loss", fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    
    # A simple moving average to see the trend.
    if len(steps) > 50:
        window_size = 50
        moving_avg = [sum(losses[i-window_size:i]) / window_size for i in range(window_size, len(losses))]
        ax.plot(steps[window_size:], moving_avg, color='magenta', linestyle='--', linewidth=2, label=f'{window_size}-Step Moving Average')
        ax.legend()

    plt.tight_layout()

    # --- Saving the Scryed Vision ---
    output_dir = Path(f"{output_location}/training_loss")
    output_dir.mkdir(exist_ok=True)
    output_filename = output_dir / f"loss_plot_{log_file.stem}.png"
    
    plt.savefig(output_filename, dpi=150)
    print(f"Plot saved successfully to: {output_filename}")

TARGET_PROJECT_SUBPATH="projects/rwby_vacuo_arc"

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Generate a loss plot from a training log file.")
    parser.add_argument(
        '--log_file',
        type=str,
        help="Optional: Path to a specific log file. If not provided, the latest training log is used."
    )
    parser.add_argument(
        '--project_name',
        type=str,
        required=True,
        help="Name of the project within the 'projects' directory - it is the caller's responsibility to ensure that specified or default log file pertains to the LORA training for this project"
    )
    args = parser.parse_args()
    output_graph_location=f"projects/{args.project_name}/"

    if args.log_file:
        log_file_path = Path(args.log_file)
    else:
        log_dir = Path("logs")
        if not log_dir.is_dir():
            print("Error: 'logs' directory not found. Have you run any training jobs?")
            return
        
        # Find the most recent training log. It is a superior method.
        training_logs = sorted(log_dir.glob("training-*.stderr"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not training_logs:
            print("No training log files found in the 'logs' directory.")
            return
        log_file_path = training_logs[0]

    plot_loss(log_file_path, output_graph_location)


if __name__ == "__main__":
    main()