#!/usr/bin/env python3
"""
Visualise trial metrics for a given config.
Generates PNGs in the metrics directory.

Usage:
    python visualise.py --config loo_sum_aggressive_sycophancy
    python visualise.py -c loo_sum_aggressive_sycophancy --plot-type raw
    python visualise.py -c loo_sum_aggressive_sycophancy -t monitorability
    python visualise.py -c loo_sum_aggressive_sycophancy --skip-threshold 0.2
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# =============================================================================
# PATHS (matching download_wandb_trial_data.py)
# =============================================================================

BASE_DIR = Path(__file__).parent
METRICS_DIR = BASE_DIR / "metrics"

# =============================================================================
# STYLE CONFIG
# =============================================================================

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
COLOR_CORRECT = "#FF9999"  # Pastel red/coral
COLOR_MONITOR_FLAG = "#99CCFF"  # Pastel blue
COLOR_SUMMARY_MONITOR = "#CC99FF"  # Pastel purple
COLOR_CHANCE = "#CCCCCC"  # Light gray
COLOR_NON_EXTRACTABLE = "#99CC99"  # Pastel green
COLOR_REPARSED = "#FFCC99"  # Pastel orange

# Line styles for different seeds
LINE_STYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]
MARKERS = ["o", "s", "^", "D", "v", "p"]


def style_axis(ax, ylim=(0, 1)):
    """Apply consistent styling to an axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    if ylim:
        ax.set_ylim(ylim)


def plot_subplot_raw(ax, df_fold, eval_fold_name, skip_threshold=None):
    """
    Plot raw rates: correct rate, monitor flag rates.
    
    If skip_threshold is set, datapoints with no_answer_rate > threshold are skipped.
    """
    seeds = sorted(df_fold["seed"].unique())

    # Detect column names
    correct_col = "correct_rate_extractable" if "correct_rate_extractable" in df_fold.columns else "reward_hack_rate_extractable"
    has_summary = "summary_monitor_flag_rate_extractable" in df_fold.columns
    no_answer_col = "no_answer_rate" if "no_answer_rate" in df_fold.columns else "no_answer_tags_rate"

    for i, seed in enumerate(seeds):
        df_seed = df_fold[df_fold["seed"] == seed].sort_values("step")
        
        # Aggregate duplicate steps (take mean of rates)
        agg_cols = {
            correct_col: "mean",
            "monitor_flag_rate_extractable": "mean",
            no_answer_col: "mean",
        }
        if has_summary:
            agg_cols["summary_monitor_flag_rate_extractable"] = "mean"
        df_seed = df_seed.groupby("step", as_index=False).agg(agg_cols)
        
        # Apply skip threshold if set
        if skip_threshold is not None:
            mask = df_seed[no_answer_col] <= skip_threshold
            df_seed = df_seed[mask]
        
        if df_seed.empty:
            continue

        linestyle = LINE_STYLES[i % len(LINE_STYLES)]
        marker = MARKERS[i % len(MARKERS)]

        ax.plot(
            df_seed["step"],
            np.ma.masked_invalid(df_seed[correct_col]),
            color=COLOR_CORRECT,
            linewidth=1.5,
            linestyle=linestyle,
            marker=marker,
            markersize=4,
            label=f"Correct (seed {seed})",
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
            label=f"CoT Monitor (seed {seed})",
            alpha=0.8,
        )

        if has_summary and df_seed["summary_monitor_flag_rate_extractable"].notna().any():
            ax.plot(
                df_seed["step"],
                np.ma.masked_invalid(df_seed["summary_monitor_flag_rate_extractable"]),
                color=COLOR_SUMMARY_MONITOR,
                linewidth=1.5,
                linestyle=linestyle,
                marker=marker,
                markersize=4,
                label=f"Summary Monitor (seed {seed})",
                alpha=0.8,
            )

    ax.axhline(y=0.5, color=COLOR_CHANCE, linestyle="--", linewidth=1, zorder=0)
    ax.set_title(eval_fold_name)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Rate")
    style_axis(ax)


def plot_subplot_monitorability(ax, df_fold, eval_fold_name, skip_threshold=None):
    """
    Plot monitorability: (correct AND flagged) / correct.
    
    If skip_threshold is set, datapoints with no_answer_rate > threshold are skipped.
    """
    seeds = sorted(df_fold["seed"].unique())

    has_cot = "monitorability_cot" in df_fold.columns
    has_summary = "monitorability_summary" in df_fold.columns
    no_answer_col = "no_answer_rate" if "no_answer_rate" in df_fold.columns else "no_answer_tags_rate"

    for i, seed in enumerate(seeds):
        df_seed = df_fold[df_fold["seed"] == seed].sort_values("step")
        
        # Aggregate duplicate steps (take mean of rates)
        agg_cols = {no_answer_col: "mean"}
        if has_cot:
            agg_cols["monitorability_cot"] = "mean"
        if has_summary:
            agg_cols["monitorability_summary"] = "mean"
        df_seed = df_seed.groupby("step", as_index=False).agg(agg_cols)
        
        # Apply skip threshold if set
        if skip_threshold is not None:
            mask = df_seed[no_answer_col] <= skip_threshold
            df_seed = df_seed[mask]
        
        if df_seed.empty:
            continue

        linestyle = LINE_STYLES[i % len(LINE_STYLES)]
        marker = MARKERS[i % len(MARKERS)]

        if has_cot and df_seed["monitorability_cot"].notna().any():
            ax.plot(
                df_seed["step"],
                np.ma.masked_invalid(df_seed["monitorability_cot"]),
                color=COLOR_MONITOR_FLAG,
                linewidth=1.5,
                linestyle=linestyle,
                marker=marker,
                markersize=4,
                label=f"CoT Monitor (seed {seed})",
                alpha=0.8,
            )

        if has_summary and df_seed["monitorability_summary"].notna().any():
            ax.plot(
                df_seed["step"],
                np.ma.masked_invalid(df_seed["monitorability_summary"]),
                color=COLOR_SUMMARY_MONITOR,
                linewidth=1.5,
                linestyle=linestyle,
                marker=marker,
                markersize=4,
                label=f"Summary Monitor (seed {seed})",
                alpha=0.8,
            )

    ax.axhline(y=1.0, color=COLOR_CHANCE, linestyle="--", linewidth=1, zorder=0, label="Perfect detection")
    ax.set_title(eval_fold_name)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Monitorability\n(correct ∩ flagged) / correct")
    style_axis(ax)


def plot_subplot_non_extractable(ax, df_fold, eval_fold_name):
    """
    Plot non-extractable rate: proportion of rows that failed extraction even after reparsing.
    
    This is num_failed / total, i.e., the rows where even reparsing didn't help.
    """
    seeds = sorted(df_fold["seed"].unique())

    no_answer_col = "no_answer_rate" if "no_answer_rate" in df_fold.columns else "no_answer_tags_rate"

    for i, seed in enumerate(seeds):
        df_seed = df_fold[df_fold["seed"] == seed].sort_values("step")

        # Aggregate duplicate steps (take mean of rates)
        df_seed = df_seed.groupby("step", as_index=False).agg({no_answer_col: "mean"})

        linestyle = LINE_STYLES[i % len(LINE_STYLES)]
        marker = MARKERS[i % len(MARKERS)]

        ax.plot(
            df_seed["step"],
            np.ma.masked_invalid(df_seed[no_answer_col]) * 100,
            color=COLOR_NON_EXTRACTABLE,
            linewidth=1.5,
            linestyle=linestyle,
            marker=marker,
            markersize=4,
            label=f"Seed {seed}",
            alpha=0.8,
        )

    ax.set_title(eval_fold_name)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Non-Extractable (%)\n(failed after reparsing)")
    style_axis(ax, ylim=(0, 100))


def plot_subplot_reparsed(ax, df_fold, eval_fold_name):
    """
    Plot reparsed rate: proportion of rows that were only extracted via reparsing.
    
    Computed as num_reparsed / total.
    """
    seeds = sorted(df_fold["seed"].unique())

    # Check if we have the data
    has_reparsed = "num_reparsed" in df_fold.columns and "total" in df_fold.columns

    if not has_reparsed:
        ax.text(0.5, 0.5, "num_reparsed not available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(eval_fold_name)
        return

    for i, seed in enumerate(seeds):
        df_seed = df_fold[df_fold["seed"] == seed].sort_values("step")

        # Aggregate duplicate steps (sum counts, then compute rate)
        df_seed = df_seed.groupby("step", as_index=False).agg({
            "num_reparsed": "sum",
            "total": "sum",
        })

        linestyle = LINE_STYLES[i % len(LINE_STYLES)]
        marker = MARKERS[i % len(MARKERS)]

        reparsed_rate = df_seed["num_reparsed"] / df_seed["total"] * 100

        ax.plot(
            df_seed["step"],
            np.ma.masked_invalid(reparsed_rate),
            color=COLOR_REPARSED,
            linewidth=1.5,
            linestyle=linestyle,
            marker=marker,
            markersize=4,
            label=f"Seed {seed}",
            alpha=0.8,
        )

    ax.set_title(eval_fold_name)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Reparsed (%)\n(extracted only via reparsing)")
    style_axis(ax, ylim=(0, 100))


def create_figure_for_model(df_model, model_name, plot_type, skip_threshold=None):
    """
    Create a figure with subplots for each eval fold.
    Returns the figure object.
    """
    eval_folds = sorted(df_model["eval_fold"].unique())
    n_folds = len(eval_folds)

    if n_folds == 0:
        return None

    fig, axes = plt.subplots(1, n_folds, figsize=(5 * n_folds, 4), squeeze=False)
    axes = axes.flatten()

    plot_funcs = {
        "raw": lambda ax, df, name: plot_subplot_raw(ax, df, name, skip_threshold=None),
        "raw_skipped": lambda ax, df, name: plot_subplot_raw(ax, df, name, skip_threshold=skip_threshold),
        "monitorability": lambda ax, df, name: plot_subplot_monitorability(ax, df, name, skip_threshold=None),
        "monitorability_skipped": lambda ax, df, name: plot_subplot_monitorability(ax, df, name, skip_threshold=skip_threshold),
        "non_extractable": plot_subplot_non_extractable,
        "reparsed": plot_subplot_reparsed,
    }
    plot_func = plot_funcs[plot_type]

    for idx, eval_fold in enumerate(eval_folds):
        df_fold = df_model[df_model["eval_fold"] == eval_fold]
        plot_func(axes[idx], df_fold, eval_fold)

    for idx in range(n_folds, len(axes)):
        axes[idx].set_visible(False)

    # Add legend
    if n_folds > 0:
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
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
        "raw_skipped": f"(Raw Rates, skipped if non-extractable > {skip_threshold:.0%})",
        "monitorability": "(Monitorability)",
        "monitorability_skipped": f"(Monitorability, skipped if non-extractable > {skip_threshold:.0%})",
        "non_extractable": "(Non-Extractable After Reparsing)",
        "reparsed": "(Reparsed Rate)",
    }
    fig.suptitle(f"{model_name} {title_suffixes[plot_type]}", fontsize=14)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])

    return fig


def list_available_configs() -> list:
    """List all available configs (directories in metrics/)."""
    if not METRICS_DIR.exists():
        return []
    return [d.name for d in METRICS_DIR.iterdir() if d.is_dir() and (d / "trial_metrics.csv").exists()]


def main():
    parser = argparse.ArgumentParser(
        description="Visualize trial metrics for a config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python visualise.py --config loo_sum_aggressive_sycophancy
    python visualise.py -c loo_sum_aggressive_sycophancy --plot-type raw
    python visualise.py -c loo_sum_aggressive_sycophancy -t monitorability
    python visualise.py -c loo_sum_aggressive_sycophancy --skip-threshold 0.2

Plot types:
    raw                    - Correct rate, monitor flag rates
    raw_skipped            - Same as raw, but skip datapoints with high non-extractable rate
    monitorability         - (correct ∩ flagged) / correct
    monitorability_skipped - Same, but skip datapoints with high non-extractable rate
    non_extractable        - Rate of rows that failed extraction even after reparsing
    reparsed               - Rate of rows that were only extracted via reparsing
    all                    - Generate all plot types
        """,
    )
    parser.add_argument("--config", "-c", help="Config name (matches metrics subdirectory)")
    parser.add_argument("--list-configs", action="store_true", help="List available configs")
    parser.add_argument(
        "--plot-type",
        "-t",
        choices=["raw", "raw_skipped", "monitorability", "monitorability_skipped", "non_extractable", "reparsed", "all"],
        default="all",
        help="Type of plot to generate (default: all)",
    )
    parser.add_argument(
        "--skip-threshold",
        "-s",
        type=float,
        default=0.3,
        help="For *_skipped plots: skip datapoints with non-extractable rate above this threshold (default: 0.3)",
    )
    args = parser.parse_args()

    if args.list_configs:
        configs = list_available_configs()
        if not configs:
            print(f"\nNo configs with trial_metrics.csv found in {METRICS_DIR}")
        else:
            print(f"\nAvailable configs:")
            for name in sorted(configs):
                print(f"  - {name}")
        return 0

    if not args.config:
        parser.error("--config is required (use --list-configs to see available configs)")

    # Build paths
    metrics_dir = METRICS_DIR / args.config
    csv_path = metrics_dir / "trial_metrics.csv"

    if not csv_path.exists():
        print(f"ERROR: File not found: {csv_path}")
        print(f"Use --list-configs to see available configs")
        return 1

    print(f"Config: {args.config}")
    print(f"Loading data from: {csv_path}")
    print(f"Skip threshold: {args.skip_threshold:.0%}")
    df = pd.read_csv(csv_path)

    # Get unique models (data column)
    models = df["data"].unique()
    print(f"Found {len(models)} models: {list(models)}")

    # Determine which plot types to generate
    if args.plot_type == "all":
        plot_types = ["raw", "raw_skipped", "monitorability", "monitorability_skipped", "non_extractable", "reparsed"]
    else:
        plot_types = [args.plot_type]

    for model in models:
        df_model = df[df["data"] == model]
        print(f"\nProcessing model: {model}")
        print(f'  Eval folds: {list(df_model["eval_fold"].unique())}')
        print(f'  Seeds: {list(df_model["seed"].unique())}')

        for plot_type in plot_types:
            fig = create_figure_for_model(
                df_model, 
                model, 
                plot_type=plot_type,
                skip_threshold=args.skip_threshold,
            )
            
            if fig is None:
                print(f"  Skipping {plot_type}: no data")
                continue
            
            # Filename without config name (it's already in the directory)
            output_path = metrics_dir / f"{plot_type}.png"
            
            fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            print(f"  Saved: {output_path}")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())