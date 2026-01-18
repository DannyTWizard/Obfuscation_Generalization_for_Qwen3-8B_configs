import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_configs(csv_paths: List[Path], output_dir: Path | None = None) -> None:
    """
    Plot reward_hacking_rate and penalisation_rate vs artifact_step.

    Structure:
    - 1 figure/png = 1 config (data column)
    - 1 subplot = 1 eval dataset (eval_fold)
    - Multiple lines in each subplot for different seeds (ts) and metrics
    """
    if not csv_paths:
        return

    # Determine output directory
    if output_dir is None:
        output_dir = csv_paths[0].parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and combine all CSV files
    all_dfs = []
    for csv_path in csv_paths:
        if not csv_path.is_file():
            print(f"Skipping {csv_path} (not found)")
            continue
        print(f"Loading {csv_path}...")
        df = pd.read_csv(csv_path)
        all_dfs.append(df)

    if not all_dfs:
        print("No valid CSV files found.")
        return

    df = pd.concat(all_dfs, ignore_index=True)

    required_cols = {
        "artifact_step",
        "data",
        "eval_fold",
        "ts",
        "reward_hacking_rate",
        "penalisation_rate",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV files are missing required columns: {missing}")

    # Ensure sorted by step within each group for nice lines
    df = df.sort_values(["data", "eval_fold", "ts", "artifact_step"])

    # Group by config (data column) - one figure per config
    for data_name, config_df in df.groupby("data"):
        # Get all eval_folds for this config
        eval_folds = sorted(config_df["eval_fold"].unique())
        n_folds = len(eval_folds)

        # Determine subplot layout (try to make it roughly square)
        n_cols = int(np.ceil(np.sqrt(n_folds)))
        n_rows = int(np.ceil(n_folds / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        if n_folds == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        # Plot each eval_fold in its own subplot
        for idx, eval_fold in enumerate(eval_folds):
            ax = axes[idx]
            fold_df = config_df[config_df["eval_fold"] == eval_fold]

            # Get unique seeds (ts values)
            seeds = sorted(fold_df["ts"].unique())

            # Plot lines for each (seed, metric) combination
            # Use same color for both metrics from same seed, different linestyles
            markers = ["o", "s", "^", "v", "D", "p", "*"]
            colors = plt.cm.tab10(np.linspace(0, 1, len(seeds)))

            for seed_idx, seed in enumerate(seeds):
                seed_df = fold_df[fold_df["ts"] == seed].sort_values("artifact_step")
                color = colors[seed_idx]
                marker = markers[seed_idx % len(markers)]

                # Plot reward_hacking_rate
                ax.plot(
                    seed_df["artifact_step"],
                    seed_df["reward_hacking_rate"],
                    marker=marker,
                    label=f"seed {seed} reward_hacking",
                    color=color,
                    linestyle="-",
                )

                # Plot penalisation_rate
                ax.plot(
                    seed_df["artifact_step"],
                    seed_df["penalisation_rate"],
                    marker=marker,
                    label=f"seed {seed} penalisation",
                    color=color,
                    linestyle="--",
                )

            ax.set_xlabel("artifact_step")
            ax.set_ylabel("rate")
            ax.set_title(eval_fold)
            ax.set_ylim(0.0, 1.05)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc="best")

        # Hide unused subplots
        for idx in range(n_folds, len(axes)):
            axes[idx].axis("off")

        # Set main title for the figure
        fig.suptitle(f"Config: {data_name}", fontsize=14, y=1.0)

        # Save figure
        safe_data = str(data_name).replace("/", "_")
        out_name = f"config_{safe_data}.png"
        out_path = output_dir / out_name
        fig.tight_layout()
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot reward_hacking_rate and penalisation_rate vs artifact_step. "
            "Creates one figure per config (data), with subplots for each eval dataset, "
            "and multiple lines for different seeds and metrics."
        )
    )
    parser.add_argument(
        "csv_files",
        nargs="*",
        type=Path,
        help=(
            "Paths to eval CSV files. If omitted, defaults to the three "
            "standard files in metrics/eval/: eval_results_24/42/50.csv."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory to write PNGs into. "
            "Defaults to the directory of the first CSV."
        ),
    )

    args = parser.parse_args()

    if args.csv_files:
        csv_paths = args.csv_files
    else:
        # Default to the three known files the user mentioned
        repo_root = Path(__file__).resolve().parents[1]
        default_dir = repo_root / "metrics" / "eval"
        csv_paths = [
            default_dir / "eval_results_24.csv",
            default_dir / "eval_results_42.csv",
            default_dir / "eval_results_50.csv",
        ]

    plot_configs(csv_paths, args.output_dir)


if __name__ == "__main__":
    main()
