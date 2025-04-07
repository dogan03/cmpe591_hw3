import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


def plot_rewards(csv_file, output_dir="plots", window_size=20):
    """
    Plot reward data from a CSV file with columns: Wall time, Step, Value

    Args:
        csv_file: Path to the CSV file containing reward data
        output_dir: Directory to save the generated plots
        window_size: Window size for the smoothed plot
    """

    os.makedirs(output_dir, exist_ok=True)

    algo_name = os.path.basename(csv_file).split("_")[0].upper()

    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file, skiprows=1, names=["Wall_time", "Step", "Value"])

    plt.figure(figsize=(12, 6))

    plt.plot(df["Step"], df["Value"], alpha=0.3, color="blue", label="Distances")

    if len(df) > window_size:
        smoothed = df["Value"].rolling(window=window_size, min_periods=1).mean()
        plt.plot(
            df["Step"],
            smoothed,
            linewidth=2,
            color="darkblue",
            label=f"Smoothed (window={window_size})",
        )

    plt.axhline(y=0, color="r", linestyle="-", alpha=0.3)

    plt.xlabel("Training Steps")
    plt.ylabel("Distances")
    plt.title(f"{algo_name} Training Distances")

    plt.grid(True, alpha=0.3)
    plt.legend()

    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    min_reward = df["Value"].min()
    max_reward = df["Value"].max()
    avg_reward = df["Value"].mean()

    stats_text = f"Min: {min_reward:.2f}\nMax: {max_reward:.2f}\nAvg: {avg_reward:.2f}"
    plt.annotate(
        stats_text,
        xy=(0.02, 0.95),
        xycoords="axes fraction",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    filename = os.path.basename(csv_file).replace(".csv", ".png")
    output_path = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot rewards from CSV file")
    parser.add_argument(
        "csv_file", help="CSV file containing Wall time, Step, Value columns"
    )
    parser.add_argument("--output_dir", default="plots", help="Directory to save plots")
    parser.add_argument(
        "--window", type=int, default=20, help="Window size for smoothing"
    )

    args = parser.parse_args()
    plot_rewards(args.csv_file, args.output_dir, args.window)
