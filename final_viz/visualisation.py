#!/usr/bin/env python3
"""
Visualise trial metrics for sycophancy experiments.
Generates one PNG per model (data) with subplots for each eval fold.

Usage:
    python visualisation.py                              # Default: loo mode, answer_tags_only parsing
    python visualisation.py --mode loo-sum               # loo-sum mode
    python visualisation.py --mode o2m                   # o2m mode
    python visualisation.py -pc aggressive               # Use aggressive parsing config
    python visualisation.py -pc reparse_all --mode loo   # Combine parsing config and mode
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Mode configurations matching download_wandb_trial_data.py
MODE_CONFIGS = {
    "loo": {
        "csv_suffix": "",
        "output_suffix": "",
    },
    "loo-sum": {
        "csv_suffix": "_loo_sum",
        "output_suffix": "_loo_sum",
    },
    "o2m": {
        "csv_suffix": "_o2m",
        "output_suffix": "_o2m",
    },
}

# Parsing configurations matching download_wandb_trial_data.py
PARSING_CONFIGS = {
    "answer_tags_only": {
        "description": "Only extract from <answer> tags (original behavior)",
        "output_subdir": "answer_tags_only",
    },
    "answer_tags_plus_colon": {
        "description": "Try answer tags first, then 'Answer:' pattern",
        "output_subdir": "answer_tags_plus_colon",
    },
    "aggressive": {
        "description": "Try all patterns: answer tags, 'Answer:', and trailing letter",
        "output_subdir": "aggressive",
    },
    "reparse_all": {
        "description": "Ignore pre-extracted, always re-parse with all patterns",
        "output_subdir": "reparse_all",
    },
}

DEFAULT_PARSING_CONFIG = "answer_tags_only"

# Style configuration
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.weight": "normal",
        "axes.titleweight": "normal",
        "axes.labelweight": "normal",
        "figure.titleweight": "normal",
    }
)

# Pastel colors
COLOR_REWARD_HACK = "#FF9999"  # Pastel red/coral
COLOR_MONITOR_FLAG = "#99CCFF"  # Pastel blue
COLOR_CHANCE = "#CCCCCC"  # Light gray for chance line
COLOR_NON_EXTRACTABLE = "#99CC99"  # Pastel green for non-extractable

# Line styles for different seeds
LINE_STYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]
MARKERS = ["o", "s", "^", "D", "v", "p"]


def style_axis(ax):
    """Apply consistent styling to an axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.set_ylim(0, 1)


def plot_subplot_raw(ax, df_fold, eval_fold_name):
    """
    Plot a single subplot for one eval fold (raw rates).

    Plots unaggregated lines for each seed with distinct line styles.
    """
    seeds = sorted(df_fold["seed"].unique())

    # Plot each seed as a separate line with distinct style
    for i, seed in enumerate(seeds):
        df_seed = df_fold[df_fold["seed"] == seed].sort_values("step")

        linestyle = LINE_STYLES[i % len(LINE_STYLES)]
        marker = MARKERS[i % len(MARKERS)]

        ax.plot(
            df_seed["step"],
            np.ma.masked_invalid(df_seed["reward_hack_rate_extractable"]),
            color=COLOR_REWARD_HACK,
            linewidth=1.5,
            linestyle=linestyle,
            marker=marker,
            markersize=4,
            markevery=max(1, len(df_seed) // 8),  # Show ~8 markers per line
            label=f"Reward Hack (seed {seed})",
            alpha=0.8,
        )

        ax.plot(
            df_seed["step"],
            np.ma.masked_invalid(df_seed["monitor_flag_rate_extractable"]),
            color=COLOR_MONITOR_FLAG,
            linewidth=1.5,
            linestyle=linestyle,
            marker=marker,
            markersize=4,
            markevery=max(1, len(df_seed) // 8),
            label=f"Monitor Flag (seed {seed})",
            alpha=0.8,
        )

    # Chance line at 0.5
    ax.axhline(
        y=0.5,
        color=COLOR_CHANCE,
        linestyle="--",
        linewidth=1,
        label="Chance (0.5)" if len(seeds) > 0 else None,
        zorder=0,
    )

    ax.set_title(eval_fold_name)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Rate")

    style_axis(ax)


def plot_subplot_relative(ax, df_fold, eval_fold_name):
    """
    Plot a single subplot for one eval fold (relative: reward hack / monitor flag).

    Plots the ratio of reward hack rate to monitor flag rate for each seed.
    """
    seeds = sorted(df_fold["seed"].unique())

    # Plot each seed as a separate line with distinct style
    for i, seed in enumerate(seeds):
        df_seed = df_fold[df_fold["seed"] == seed].sort_values("step")

        linestyle = LINE_STYLES[i % len(LINE_STYLES)]
        marker = MARKERS[i % len(MARKERS)]

        # Calculate ratio, handling division by zero with epsilon
        reward_hack = df_seed["reward_hack_rate_extractable"].values
        monitor_flag = df_seed["monitor_flag_rate_extractable"].values
        eps = 1
        ratio = (reward_hack + eps) / (monitor_flag + eps)

        ax.plot(
            df_seed["step"],
            np.ma.masked_invalid(ratio),
            color=COLOR_REWARD_HACK,
            linewidth=1.5,
            linestyle=linestyle,
            marker=marker,
            markersize=4,
            markevery=max(1, len(df_seed) // 8),
            label=f"Seed {seed}",
            alpha=0.8,
        )

    # Reference line at 1.0 (equal rates)
    ax.axhline(
        y=1.0,
        color=COLOR_CHANCE,
        linestyle="--",
        linewidth=1,
        label="Equal rates (1.0)" if len(seeds) > 0 else None,
        zorder=0,
    )

    ax.set_title(eval_fold_name)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Reward Hack / Monitor Flag")

    # Style without fixed y-limit for ratio plots
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def plot_subplot_non_extractable(ax, df_fold, eval_fold_name):
    """
    Plot a single subplot for one eval fold (non-extractable rate).

    Plots the percentage of non-extractable answers for each seed.
    """
    seeds = sorted(df_fold["seed"].unique())

    # Plot each seed as a separate line with distinct style
    for i, seed in enumerate(seeds):
        df_seed = df_fold[df_fold["seed"] == seed].sort_values("step")

        linestyle = LINE_STYLES[i % len(LINE_STYLES)]
        marker = MARKERS[i % len(MARKERS)]

        ax.plot(
            df_seed["step"],
            np.ma.masked_invalid(df_seed["no_answer_tags_rate"])
            * 100,  # Convert to percentage
            color=COLOR_NON_EXTRACTABLE,
            linewidth=1.5,
            linestyle=linestyle,
            marker=marker,
            markersize=4,
            markevery=max(1, len(df_seed) // 8),
            label=f"Seed {seed}",
            alpha=0.8,
        )

    ax.set_title(eval_fold_name)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Non-Extractable (%)")

    # Style axis
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.set_ylim(0, None)  # Start at 0, auto-scale upper limit


def create_figure_for_model(df_model, model_name, output_dir, plot_type="raw"):
    """
    Create a figure with subplots for each eval fold.

    Args:
        df_model: DataFrame filtered to a single model
        model_name: Name of the model for title and filename
        output_dir: Directory to save the figure
        plot_type: "raw" for absolute rates, "relative" for ratio plot, "non_extractable" for non-extractable rate
    """
    eval_folds = df_model["eval_fold"].unique()
    n_folds = len(eval_folds)

    # Determine subplot layout
    n_cols, n_rows = n_folds, 1

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False
    )
    axes = axes.flatten()

    # Select plot function based on type
    if plot_type == "raw":
        plot_func = plot_subplot_raw
    elif plot_type == "relative":
        plot_func = plot_subplot_relative
    elif plot_type == "non_extractable":
        plot_func = plot_subplot_non_extractable
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}")

    for idx, eval_fold in enumerate(eval_folds):
        df_fold = df_model[df_model["eval_fold"] == eval_fold]
        plot_func(axes[idx], df_fold, eval_fold)

    # Hide unused subplots
    for idx in range(n_folds, len(axes)):
        axes[idx].set_visible(False)

    # Add legend below the figure
    if n_folds > 0:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=min(len(handles), 4),
            frameon=False,
            fontsize=9,
        )

    title_suffixes = {
        "raw": "(Raw Rates)",
        "relative": "(Reward Hack / Monitor Flag)",
        "non_extractable": "(Non-Extractable %)",
    }
    title_suffix = title_suffixes.get(plot_type, "")
    fig.suptitle(f"Model: {model_name} {title_suffix}", fontsize=14)
    fig.tight_layout(rect=[0, 0.08, 1, 1])  # Leave space at bottom for legend

    # Save figure
    output_path = output_dir / f"{model_name}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize trial metrics for sycophancy experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
    loo      - leave_out_* datasets (default)
    loo-sum  - leave_out_* datasets (run_summary runs)
    o2m      - only_score_refined2 dataset

Parsing Configs:
    answer_tags_only       - Only extract from <answer> tags (default)
    answer_tags_plus_colon - Try answer tags first, then 'Answer:' pattern
    aggressive             - Try all patterns: answer tags, 'Answer:', trailing letter
    reparse_all            - Ignore pre-extracted, always re-parse with all patterns

Input/Output paths:
    Input:  metrics/<parsing_config>/trial_metrics<mode_suffix>.csv
    Output: results/<parsing_config>/raw<mode_suffix>/
            results/<parsing_config>/relative<mode_suffix>/
            results/<parsing_config>/non_extractable<mode_suffix>/
        """,
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=list(MODE_CONFIGS.keys()),
        default="loo",
        help="Mode: loo (default), loo-sum, or o2m",
    )
    parser.add_argument(
        "--parsing-config",
        "-pc",
        choices=list(PARSING_CONFIGS.keys()),
        default=DEFAULT_PARSING_CONFIG,
        help=f"Parsing configuration to use (default: {DEFAULT_PARSING_CONFIG})",
    )
    parser.add_argument(
        "--list-parsing-configs",
        action="store_true",
        help="List available parsing configurations and exit",
    )
    args = parser.parse_args()

    # Handle --list-parsing-configs
    if args.list_parsing_configs:
        print("\nAvailable Parsing Configurations:")
        print("=" * 60)
        for name, config in PARSING_CONFIGS.items():
            print(f"\n{name}:")
            print(f"  Description: {config['description']}")
            print(f"  Subdir:      {config['output_subdir']}")
        print("\n" + "=" * 60)
        return

    mode = args.mode
    mode_config = MODE_CONFIGS[mode]
    parsing_config = PARSING_CONFIGS[args.parsing_config]
    parsing_subdir = parsing_config["output_subdir"]

    # Paths based on mode and parsing config
    script_dir = Path(__file__).parent
    csv_suffix = mode_config["csv_suffix"]
    output_suffix = mode_config["output_suffix"]

    # Input: metrics/<parsing_config>/trial_metrics<mode_suffix>.csv
    csv_path = script_dir / "metrics" / parsing_subdir / f"trial_metrics{csv_suffix}.csv"
    
    # Output: results/<parsing_config>/raw<mode_suffix>/, etc.
    output_dir_raw = script_dir / "results" / parsing_subdir / f"raw{output_suffix}"
    output_dir_relative = script_dir / "results" / parsing_subdir / f"relative{output_suffix}"
    output_dir_non_extractable = (
        script_dir / "results" / parsing_subdir / f"non_extractable{output_suffix}"
    )

    # Create output directories
    output_dir_raw.mkdir(parents=True, exist_ok=True)
    output_dir_relative.mkdir(parents=True, exist_ok=True)
    output_dir_non_extractable.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Mode: {mode}")
    print(f"Parsing config: {args.parsing_config}")
    print(f"  Description: {parsing_config['description']}")
    print(f"Loading data from: {csv_path}")

    if not csv_path.exists():
        print(f"ERROR: File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Get unique models
    models = df["data"].unique()
    print(f"Found {len(models)} models: {list(models)}")
    print(f"Output directories:")
    print(f"  Raw:             {output_dir_raw}")
    print(f"  Relative:        {output_dir_relative}")
    print(f"  Non-extractable: {output_dir_non_extractable}")

    # Create figures for each model
    for model in models:
        df_model = df[df["data"] == model]
        print(f"\nProcessing model: {model}")
        print(f'  Eval folds: {list(df_model["eval_fold"].unique())}')
        print(f'  Seeds: {list(df_model["seed"].unique())}')

        # Raw rates plot
        create_figure_for_model(df_model, model, output_dir_raw, plot_type="raw")

        # Relative (ratio) plot
        create_figure_for_model(
            df_model, model, output_dir_relative, plot_type="relative"
        )

        # Non-extractable rate plot
        create_figure_for_model(
            df_model, model, output_dir_non_extractable, plot_type="non_extractable"
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
