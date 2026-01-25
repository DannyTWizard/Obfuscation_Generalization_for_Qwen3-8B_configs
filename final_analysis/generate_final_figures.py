#!/usr/bin/env python3
"""
Generate final paper figures by combining specific lines from different configs.

Usage:
    python generate_final_figures.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).parent
METRICS_DIR = BASE_DIR / "metrics"
OUTPUT_DIR = BASE_DIR / "final_figures"

# =============================================================================
# STYLE CONFIG
# =============================================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.weight": "normal",
    "axes.titleweight": "bold",
    "axes.labelweight": "normal",
    "figure.titleweight": "normal",
})

# Default colors (can be overridden per-line)
COLOR_CORRECT = "#FF9999"
COLOR_MONITOR = "#99CCFF"
COLOR_SUMMARY = "#CC99FF"  # Pastel purple for summary monitor
COLOR_CHANCE = "#CCCCCC"

# Line styles for different seeds in appendix figures
SEED_STYLES = {
    24: {"linestyle": "-", "marker": "o"},
    42: {"linestyle": "--", "marker": "s"},
    50: {"linestyle": "-.", "marker": "^"},
}
SEED_COLORS = {
    24: {"correct": "#FF9999", "monitor": "#99CCFF"},
    42: {"correct": "#FF6666", "monitor": "#66B2FF"},
    50: {"correct": "#CC7777", "monitor": "#7799DD"},
}

# =============================================================================
# FIGURE SPECIFICATIONS
# =============================================================================

# Each figure is a dict with:
#   - "name": output filename (without extension)
#   - "title": figure suptitle
#   - "subplots": list of subplot specs, each with:
#       - "title": subplot title
#       - "lines": list of line specs, each with:
#           - "config": config name (metrics subdirectory)
#           - "seed": which seed to use
#           - "eval_fold": which eval fold to plot
#           - "label": legend label (optional, auto-generated if missing)
#           - "color_correct": color for correct rate line (optional)
#           - "color_monitor": color for monitor flag line (optional)

FIGURES = {
    "figure1": {
        "name": "figure1",
        "title": None,  # No global title
        "skip_threshold": 0.3,
        "show_ratio": True,
        "subplots": [
            {
                "title": "Factual sycophancy",
                "chance_level": 0.5,
                "lines": [
                    {
                        "config": "loo_aggressive_sycophancy",
                        "seed": 24,
                        "eval_fold": "eval_sycophancy_formatted",
                    },
                ],
            },
            {
                "title": "Code selection",
                "chance_level": 0.5,
                "lines": [
                    {
                        "config": "loo_code",
                        "seed": 50,
                        "eval_fold": "eval_code_formatted",
                    },
                ],
            },
            {
                "title": "Revealed score MMLU",
                "chance_level": 0.25,
                "lines": [
                    {
                        "config": "loo_score",
                        "seed": 24,
                        "eval_fold": "eval_revealing_score_formatted",
                    },
                ],
            },
            {
                "title": "World affecting reward",
                "chance_level": 0.5,
                "lines": [
                    {
                        "config": "loo_war",
                        "seed": 42,
                        "eval_fold": "eval_world_affecting_reward_reorg_formatted",
                    },
                ],
            },
        ],
    },
    "figure2": {
        "name": "figure2",
        "title": "Evaluated on medical sycophancy dataset",
        "skip_threshold": 0.3,
        "show_ratio": False,
        "subplots": [
            {
                "title": "Leave out factual sycophancy",
                "chance_level": 0.5,
                "lines": [
                    {
                        "config": "loo_aggressive_sycophancy",
                        "seed": 24,
                        "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    },
                ],
            },
            {
                "title": "Leave out code selection",
                "chance_level": 0.5,
                "lines": [
                    {
                        "config": "loo_code",
                        "seed": 42,
                        "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    },
                ],
            },
            {
                "title": "Leave out revealed score",
                "chance_level": 0.5,
                "lines": [
                    {
                        "config": "loo_score",
                        "seed": 24,
                        "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    },
                ],
            },
            {
                "title": "Leave out world affecting reward",
                "chance_level": 0.5,
                "lines": [
                    {
                        "config": "loo_war",
                        "seed": 42,
                        "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    },
                ],
            },
        ],
    },
    "figure3": {
        "name": "figure3",
        "title": None,
        "skip_threshold": 0.3,
        "show_ratio": False,
        "show_summary": True,
        "subplots": [
            {
                "title": "Factual sycophancy",
                "chance_level": 0.5,
                "lines": [
                    {
                        "config": "loo_sum_aggressive_sycophancy",
                        "seed": 24,
                        "eval_fold": "eval_sycophancy_formatted",
                    },
                ],
            },
            {
                "title": "Code selection",
                "chance_level": 0.5,
                "lines": [
                    {
                        "config": "loo_sum_code",
                        "seed": 50,
                        "eval_fold": "eval_code_formatted",
                    },
                ],
            },
            {
                "title": "Revealed score MMLU",
                "chance_level": 0.25,
                "lines": [
                    {
                        "config": "loo_sum_score",
                        "seed": 50,
                        "eval_fold": "eval_revealing_score_formatted",
                    },
                ],
            },
            {
                "title": "World affecting reward",
                "chance_level": 0.5,
                "lines": [
                    {
                        "config": "loo_sum_war",
                        "seed": 24,
                        "eval_fold": "eval_world_affecting_reward_reorg_formatted",
                    },
                ],
            },
        ],
    },
}

# Appendix figures - multi-seed grid layouts
APPENDIX_FIGURES = {
    "appendix_figure1": {
        "name": "appendix_figure1",
        "title": "All seeds comparison",
        "skip_threshold": 0.3,
        "show_ratio": True,
        "rows": [
            {
                "row_title": "Factual sycophancy",
                "left": {
                    "title": "In-distribution eval",
                    "chance_level": 0.5,
                    "config": "loo_aggressive_sycophancy",
                    "eval_fold": "eval_sycophancy_formatted",
                    "seeds": [24, 42, 50],
                },
                "right": {
                    "title": "Medical sycophancy eval",
                    "chance_level": 0.5,
                    "config": "loo_aggressive_sycophancy",
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    "seeds": [24, 42, 50],
                },
            },
            {
                "row_title": "Code selection",
                "left": {
                    "title": "In-distribution eval",
                    "chance_level": 0.5,
                    "config": "loo_code",
                    "eval_fold": "eval_code_formatted",
                    "seeds": [24, 42, 50],
                },
                "right": {
                    "title": "Medical sycophancy eval",
                    "chance_level": 0.5,
                    "config": "loo_code",
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    "seeds": [24, 42, 50],
                },
            },
            {
                "row_title": "Revealed score MMLU",
                "left": {
                    "title": "In-distribution eval",
                    "chance_level": 0.25,
                    "config": "loo_score",
                    "eval_fold": "eval_revealing_score_formatted",
                    "seeds": [24, 42, 50],
                },
                "right": {
                    "title": "Medical sycophancy eval",
                    "chance_level": 0.5,
                    "config": "loo_score",
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    "seeds": [24, 42, 50],
                },
            },
            {
                "row_title": "World affecting reward",
                "left": {
                    "title": "In-distribution eval",
                    "chance_level": 0.5,
                    "config": "loo_war",
                    "eval_fold": "eval_world_affecting_reward_reorg_formatted",
                    "seeds": [24, 42, 50],
                },
                "right": {
                    "title": "Medical sycophancy eval",
                    "chance_level": 0.5,
                    "config": "loo_war",
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    "seeds": [24, 42, 50],
                },
            },
        ],
    },
    "appendix_figure2": {
        "name": "appendix_figure2",
        "title": "All seeds comparison (summary monitor)",
        "skip_threshold": 0.3,
        "show_ratio": False,
        "show_summary": True,
        "rows": [
            {
                "row_title": "Factual sycophancy",
                "left": {
                    "title": "In-distribution eval",
                    "chance_level": 0.5,
                    "config": "loo_sum_aggressive_sycophancy",
                    "eval_fold": "eval_sycophancy_formatted",
                    "seeds": [24, 42, 50],
                },
                "right": None,  # Empty for now
            },
            {
                "row_title": "Code selection",
                "left": {
                    "title": "In-distribution eval",
                    "chance_level": 0.5,
                    "config": "loo_sum_code",
                    "eval_fold": "eval_code_formatted",
                    "seeds": [24, 42, 50],
                },
                "right": None,
            },
            {
                "row_title": "Revealed score MMLU",
                "left": {
                    "title": "In-distribution eval",
                    "chance_level": 0.25,
                    "config": "loo_sum_score",
                    "eval_fold": "eval_revealing_score_formatted",
                    "seeds": [24, 42, 50],
                },
                "right": None,
            },
            {
                "row_title": "World affecting reward",
                "left": {
                    "title": "In-distribution eval",
                    "chance_level": 0.5,
                    "config": "loo_sum_war",
                    "eval_fold": "eval_world_affecting_reward_reorg_formatted",
                    "seeds": [24, 42, 50],
                },
                "right": None,
            },
        ],
    },
}


# =============================================================================
# DATA LOADING
# =============================================================================

_data_cache = {}

def load_config_data(config: str) -> pd.DataFrame:
    """Load and cache data for a config."""
    if config not in _data_cache:
        csv_path = METRICS_DIR / config / "trial_metrics.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Config data not found: {csv_path}")
        _data_cache[config] = pd.read_csv(csv_path)
    return _data_cache[config]


def get_line_data(config: str, seed: int, eval_fold: str, skip_threshold: float = 0.3) -> pd.DataFrame:
    """Extract and prepare data for a single line."""
    df = load_config_data(config)
    
    # Filter to seed and eval_fold
    df_line = df[(df["seed"] == seed) & (df["eval_fold"] == eval_fold)].copy()
    
    if df_line.empty:
        print(f"  WARNING: No data for config={config}, seed={seed}, eval_fold={eval_fold}")
        return df_line
    
    # Detect column names
    correct_col = "correct_rate_extractable" if "correct_rate_extractable" in df_line.columns else "reward_hack_rate_extractable"
    no_answer_col = "no_answer_rate" if "no_answer_rate" in df_line.columns else "no_answer_tags_rate"
    has_summary = "summary_monitor_flag_rate_extractable" in df_line.columns
    
    # Aggregate duplicate steps
    agg_cols = {
        correct_col: "mean",
        "monitor_flag_rate_extractable": "mean",
        no_answer_col: "mean",
    }
    if has_summary:
        agg_cols["summary_monitor_flag_rate_extractable"] = "mean"
    
    df_line = df_line.groupby("step", as_index=False).agg(agg_cols)
    
    # Apply skip threshold
    mask = df_line[no_answer_col] <= skip_threshold
    df_line = df_line[mask]
    
    # Rename for consistency
    df_line = df_line.rename(columns={correct_col: "correct_rate", no_answer_col: "no_answer_rate"})
    
    return df_line.sort_values("step")


# =============================================================================
# PLOTTING
# =============================================================================

def style_axis(ax, ylim=(0, 1)):
    """Apply consistent styling to an axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    if ylim:
        ax.set_ylim(ylim)


def plot_subplot(ax, subplot_spec: dict, skip_threshold: float, show_ratio: bool = True, show_summary: bool = False):
    """Plot a single subplot with multiple lines."""
    ax.set_title(subplot_spec["title"])
    
    if not subplot_spec.get("lines"):
        ax.text(0.5, 0.5, "TBD", ha="center", va="center", transform=ax.transAxes, fontsize=12, color="gray")
        style_axis(ax)
        return
    
    for line_spec in subplot_spec["lines"]:
        df_line = get_line_data(
            config=line_spec["config"],
            seed=line_spec["seed"],
            eval_fold=line_spec["eval_fold"],
            skip_threshold=skip_threshold,
        )
        
        if df_line.empty:
            continue
        
        color_correct = line_spec.get("color_correct", COLOR_CORRECT)
        color_monitor = line_spec.get("color_monitor", COLOR_MONITOR)
        
        # Compute rolling averages
        correct_raw = df_line["correct_rate"]
        monitor_raw = df_line["monitor_flag_rate_extractable"]
        correct_smooth = correct_raw.rolling(window=3, min_periods=1, center=True).mean()
        monitor_smooth = monitor_raw.rolling(window=3, min_periods=1, center=True).mean()
        
        # Plot correct rate (reward hacking) - faint markers, solid line
        ax.plot(
            df_line["step"],
            np.ma.masked_invalid(correct_raw),
            color=color_correct,
            linewidth=0,
            marker="o",
            markersize=4,
            alpha=0.3,
        )
        ax.plot(
            df_line["step"],
            np.ma.masked_invalid(correct_smooth),
            color=color_correct,
            linewidth=1.5,
            linestyle="-",
            label="Unseen task reward hacking rate",
            alpha=1.0,
        )
        
        # Plot monitor flag rate (penalty) - faint markers, solid line
        ax.plot(
            df_line["step"],
            np.ma.masked_invalid(monitor_raw),
            color=color_monitor,
            linewidth=0,
            marker="o",
            markersize=4,
            alpha=0.3,
        )
        ax.plot(
            df_line["step"],
            np.ma.masked_invalid(monitor_smooth),
            color=color_monitor,
            linewidth=1.5,
            linestyle="-",
            label="Unseen task penalty rate",
            alpha=1.0,
        )
        
        # Plot summary monitor rate if enabled and available
        if show_summary and "summary_monitor_flag_rate_extractable" in df_line.columns:
            summary_raw = df_line["summary_monitor_flag_rate_extractable"]
            if summary_raw.notna().any():
                summary_smooth = summary_raw.rolling(window=3, min_periods=1, center=True).mean()
                ax.plot(
                    df_line["step"],
                    np.ma.masked_invalid(summary_raw),
                    color=COLOR_SUMMARY,
                    linewidth=0,
                    marker="o",
                    markersize=4,
                    alpha=0.3,
                )
                ax.plot(
                    df_line["step"],
                    np.ma.masked_invalid(summary_smooth),
                    color=COLOR_SUMMARY,
                    linewidth=1.5,
                    linestyle="-",
                    label="Unseen task summary penalty rate",
                    alpha=1.0,
                )
        
        # Plot ratio (blue / red) if enabled - faint markers, smoothed line
        if show_ratio:
            ratio = monitor_raw / correct_raw
            ratio_smooth = ratio.rolling(window=3, min_periods=1, center=True).mean()
            ax.plot(
                df_line["step"],
                np.ma.masked_invalid(ratio),
                color="grey",
                linewidth=0,
                marker="o",
                markersize=4,
                alpha=0.3,
            )
            ax.plot(
                df_line["step"],
                np.ma.masked_invalid(ratio_smooth),
                color="grey",
                linewidth=1.5,
                linestyle="-",
                label="Penalty / Reward hacking",
                alpha=1.0,
            )
    
    chance_level = subplot_spec.get("chance_level", 0.5)
    ax.axhline(y=chance_level, color=COLOR_CHANCE, linestyle="--", linewidth=1, zorder=0)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Rate")
    ax.set_xticks([0, 1000, 2000, 3000, 4000])
    ax.set_xlim(0, 4000)
    style_axis(ax)


def plot_appendix_subplot(ax, subplot_spec: dict, skip_threshold: float, show_ratio: bool = True, show_summary: bool = False):
    """Plot a single appendix subplot with multiple seeds."""
    if subplot_spec is None:
        ax.text(0.5, 0.5, "TBD", ha="center", va="center", transform=ax.transAxes, fontsize=12, color="gray")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Rate")
        ax.set_xticks([0, 1000, 2000, 3000, 4000])
        ax.set_xlim(0, 4000)
        style_axis(ax)
        return
    
    ax.set_title(subplot_spec["title"])
    
    config = subplot_spec["config"]
    eval_fold = subplot_spec["eval_fold"]
    seeds = subplot_spec["seeds"]
    
    for seed in seeds:
        df_line = get_line_data(
            config=config,
            seed=seed,
            eval_fold=eval_fold,
            skip_threshold=skip_threshold,
        )
        
        if df_line.empty:
            continue
        
        style = SEED_STYLES.get(seed, {"linestyle": "-", "marker": "o"})
        colors = SEED_COLORS.get(seed, {"correct": COLOR_CORRECT, "monitor": COLOR_MONITOR})
        
        # Compute rolling averages
        correct_raw = df_line["correct_rate"]
        monitor_raw = df_line["monitor_flag_rate_extractable"]
        correct_smooth = correct_raw.rolling(window=3, min_periods=1, center=True).mean()
        monitor_smooth = monitor_raw.rolling(window=3, min_periods=1, center=True).mean()
        
        # Plot correct rate (reward hacking) - faint markers, styled line
        ax.plot(
            df_line["step"],
            np.ma.masked_invalid(correct_raw),
            color=colors["correct"],
            linewidth=0,
            marker=style["marker"],
            markersize=4,
            alpha=0.3,
        )
        ax.plot(
            df_line["step"],
            np.ma.masked_invalid(correct_smooth),
            color=colors["correct"],
            linewidth=1.5,
            linestyle=style["linestyle"],
            label=f"Reward hacking (seed {seed})",
            alpha=1.0,
        )
        
        # Plot monitor flag rate (penalty) - faint markers, styled line
        ax.plot(
            df_line["step"],
            np.ma.masked_invalid(monitor_raw),
            color=colors["monitor"],
            linewidth=0,
            marker=style["marker"],
            markersize=4,
            alpha=0.3,
        )
        ax.plot(
            df_line["step"],
            np.ma.masked_invalid(monitor_smooth),
            color=colors["monitor"],
            linewidth=1.5,
            linestyle=style["linestyle"],
            label=f"Penalty (seed {seed})",
            alpha=1.0,
        )
        
        # Plot summary monitor rate if enabled and available
        if show_summary and "summary_monitor_flag_rate_extractable" in df_line.columns:
            summary_raw = df_line["summary_monitor_flag_rate_extractable"]
            if summary_raw.notna().any():
                summary_smooth = summary_raw.rolling(window=3, min_periods=1, center=True).mean()
                ax.plot(
                    df_line["step"],
                    np.ma.masked_invalid(summary_raw),
                    color=COLOR_SUMMARY,
                    linewidth=0,
                    marker=style["marker"],
                    markersize=4,
                    alpha=0.3,
                )
                ax.plot(
                    df_line["step"],
                    np.ma.masked_invalid(summary_smooth),
                    color=COLOR_SUMMARY,
                    linewidth=1.5,
                    linestyle=style["linestyle"],
                    label=f"Summary penalty (seed {seed})",
                    alpha=1.0,
                )
        
        # Plot ratio if enabled
        if show_ratio:
            ratio = monitor_raw / correct_raw
            ratio_smooth = ratio.rolling(window=3, min_periods=1, center=True).mean()
            ax.plot(
                df_line["step"],
                np.ma.masked_invalid(ratio),
                color="grey",
                linewidth=0,
                marker=style["marker"],
                markersize=3,
                alpha=0.2,
            )
            ax.plot(
                df_line["step"],
                np.ma.masked_invalid(ratio_smooth),
                color="grey",
                linewidth=1.2,
                linestyle=style["linestyle"],
                label=f"Ratio (seed {seed})",
                alpha=0.7,
            )
    
    chance_level = subplot_spec.get("chance_level", 0.5)
    ax.axhline(y=chance_level, color=COLOR_CHANCE, linestyle="--", linewidth=1, zorder=0)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Rate")
    ax.set_xticks([0, 1000, 2000, 3000, 4000])
    ax.set_xlim(0, 4000)
    style_axis(ax)


def generate_figure(fig_spec: dict):
    """Generate a single figure with all its subplots."""
    n_subplots = len(fig_spec["subplots"])
    skip_threshold = fig_spec.get("skip_threshold", 0.3)
    show_ratio = fig_spec.get("show_ratio", True)
    show_summary = fig_spec.get("show_summary", False)
    
    fig, axes = plt.subplots(1, n_subplots, figsize=(5 * n_subplots, 4), squeeze=False)
    axes = axes.flatten()
    
    for idx, subplot_spec in enumerate(fig_spec["subplots"]):
        plot_subplot(axes[idx], subplot_spec, skip_threshold, show_ratio=show_ratio, show_summary=show_summary)
        
        # Remove y label and ticks from non-leftmost subplots
        if idx > 0:
            axes[idx].set_ylabel("")
            axes[idx].set_yticklabels([])
            axes[idx].tick_params(axis='y', length=0)
    
    # Collect legend handles from first non-empty subplot (deduplicated)
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        if h:
            # Deduplicate by label
            seen = set()
            for handle, label in zip(h, l):
                if label not in seen:
                    handles.append(handle)
                    labels.append(label)
                    seen.add(label)
            break
    
    if handles:
        fig.legend(
            handles, labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=len(handles),
            frameon=False,
            fontsize=12,
        )
    
    title = fig_spec.get("title")
    if title:
        fig.suptitle(title, fontsize=14)
    
    fig.tight_layout(rect=[0, 0.06, 1, 0.98])
    
    return fig


def generate_appendix_figure(fig_spec: dict):
    """Generate an appendix figure with 4 rows x 2 columns grid."""
    rows = fig_spec["rows"]
    n_rows = len(rows)
    skip_threshold = fig_spec.get("skip_threshold", 0.3)
    show_ratio = fig_spec.get("show_ratio", True)
    show_summary = fig_spec.get("show_summary", False)
    
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows), squeeze=False)
    
    for row_idx, row_spec in enumerate(rows):
        # Add row label on the left side
        row_title = row_spec.get("row_title", "")
        if row_title:
            axes[row_idx, 0].annotate(
                row_title,
                xy=(-0.25, 0.5),
                xycoords="axes fraction",
                fontsize=12,
                fontweight="bold",
                ha="center",
                va="center",
                rotation=90,
            )
        
        # Plot left column
        plot_appendix_subplot(axes[row_idx, 0], row_spec["left"], skip_threshold, show_ratio=show_ratio, show_summary=show_summary)
        
        # Plot right column
        plot_appendix_subplot(axes[row_idx, 1], row_spec["right"], skip_threshold, show_ratio=show_ratio, show_summary=show_summary)
        
        # Remove y label from right column
        axes[row_idx, 1].set_ylabel("")
        axes[row_idx, 1].set_yticklabels([])
        axes[row_idx, 1].tick_params(axis='y', length=0)
    
    # Add column headers
    axes[0, 0].annotate(
        "In-distribution evaluation",
        xy=(0.5, 1.15),
        xycoords="axes fraction",
        fontsize=13,
        fontweight="bold",
        ha="center",
        va="bottom",
    )
    axes[0, 1].annotate(
        "Medical sycophancy evaluation",
        xy=(0.5, 1.15),
        xycoords="axes fraction",
        fontsize=13,
        fontweight="bold",
        ha="center",
        va="bottom",
    )
    
    # Create a compact legend
    # Collect from first subplot with data
    handles, labels = [], []
    for row_idx in range(n_rows):
        for col_idx in range(2):
            h, l = axes[row_idx, col_idx].get_legend_handles_labels()
            if h:
                seen = set()
                for handle, label in zip(h, l):
                    if label not in seen:
                        handles.append(handle)
                        labels.append(label)
                        seen.add(label)
                break
        if handles:
            break
    
    if handles:
        fig.legend(
            handles, labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=min(len(handles), 6),
            frameon=False,
            fontsize=10,
        )
    
    title = fig_spec.get("title")
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    
    fig.tight_layout(rect=[0.05, 0.04, 1, 0.96])
    
    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate main figures
    for fig_key, fig_spec in FIGURES.items():
        print(f"\nGenerating {fig_key}...")
        
        try:
            fig = generate_figure(fig_spec)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        
        name = fig_spec.get("name", fig_key)
        
        # Save PNG
        png_path = OUTPUT_DIR / f"{name}.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {png_path}")
        
        # Save PDF
        pdf_path = OUTPUT_DIR / f"{name}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {pdf_path}")
        
        plt.close(fig)
    
    # Generate appendix figures
    for fig_key, fig_spec in APPENDIX_FIGURES.items():
        print(f"\nGenerating {fig_key}...")
        
        try:
            fig = generate_appendix_figure(fig_spec)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        name = fig_spec.get("name", fig_key)
        
        # Save PNG
        png_path = OUTPUT_DIR / f"{name}.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {png_path}")
        
        # Save PDF
        pdf_path = OUTPUT_DIR / f"{name}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {pdf_path}")
        
        plt.close(fig)
    
    print("\nDone!")


if __name__ == "__main__":
    main()