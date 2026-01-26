#!/usr/bin/env python3
"""
Generate final paper figures by combining specific lines from different configs.

Usage:
    python generate_final_figures.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
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
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 14,
})

# Default colors (can be overridden per-line)
COLOR_CORRECT = "#FF9999"
COLOR_MONITOR = "#99CCFF"
COLOR_SUMMARY = "#CC99FF"  # Pastel purple for summary monitor
COLOR_CHANCE = "#CCCCCC"
COLOR_NON_EXTRACTABLE = "#99CC99"  # Pastel green for unparsable rate

# Line styles for different seeds in appendix figures
SEED_STYLES = {
    24: {"linestyle": "-"},
    33: {"linestyle": "--"},
    42: {"linestyle": "-."},
    50: {"linestyle": ":"},
}
SEED_COLORS = {
    24: {"correct": "#FF9999", "monitor": "#99CCFF"},
    33: {"correct": "#FF8080", "monitor": "#80BFFF"},
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
        "title": None,
        "skip_threshold": 0.3,
        "show_ratio": False,
        "subplots": [
            {
                "title": "Factual sycophancy",
                "chance_level": 0.5,
                "lines": [
                    {
                        "config": "sycophancy",
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
                        "config": "code",
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
                        "config": "score",
                        "seed": 42,
                        "eval_fold": "eval_revealing_score_formatted",
                    },
                ],
            },
            {
                "title": "World affecting reward",
                "chance_level": 0.5,
                "lines": [
                    {
                        "config": "war",
                        "seed": 50,
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
                        "config": "sycophancy",
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
                        "config": "code",
                        "seed": 50,
                        "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    },
                ],
            },
            {
                "title": "Leave out revealed score",
                "chance_level": 0.5,
                "lines": [
                    {
                        "config": "score",
                        "seed": 42,
                        "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    },
                ],
            },
            {
                "title": "Leave out world affecting reward",
                "chance_level": 0.5,
                "lines": [
                    {
                        "config": "war",
                        "seed": 50,
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
                        "config": "sum_sycophancy",
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
                        "config": "sum_code",
                        "seed": 42,
                        "eval_fold": "eval_code_formatted",
                    },
                ],
            },
            {
                "title": "Revealed score MMLU",
                "chance_level": 0.25,
                "lines": [
                    {
                        "config": "sum_score",
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
                        "config": "sum_war",
                        "seed": 42,
                        "eval_fold": "eval_world_affecting_reward_reorg_formatted",
                    },
                ],
            },
        ],
    },
    "figure4": {
        "name": "figure4",
        "title": "Summary monitor evaluated on medical sycophancy dataset",
        "skip_threshold": 0.3,
        "show_ratio": False,
        "show_summary": True,
        "subplots": [
            {
                "title": "Leave out factual sycophancy",
                "chance_level": 0.5,
                "lines": [
                    {
                        "config": "sum_sycophancy",
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
                        "config": "sum_code",
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
                        "config": "sum_score",
                        "seed": 50,
                        "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    },
                ],
            },
            {
                "title": "Leave out world affecting reward",
                "chance_level": 0.5,
                "lines": [
                    {
                        "config": "sum_war",
                        "seed": 42,
                        "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    },
                ],
            },
        ],
    },
}

# Appendix figures - multi-seed grid layouts
APPENDIX_FIGURES = {
    "appendix_figure0": {
        "name": "appendix_figure0",
        "title": "No penalisation",
        "skip_threshold": 0.3,
        "show_ratio": False,
        "show_summary": False,
        "rows": [
            {
                "row_title": "Factual\nsycophancy",
                "left": {
                    "chance_level": 0.5,
                    "config": "base_sycophancy",
                    "eval_fold": "eval_sycophancy_formatted",
                    "seeds": [24, 33, 42, 50],
                },
                "right": {
                    "chance_level": 0.5,
                    "config": "base_sycophancy",
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    "seeds": [24, 33, 42, 50],
                },
            },
            {
                "row_title": "Code\nselection",
                "left": {
                    "chance_level": 0.5,
                    "config": "base_code",
                    "eval_fold": "eval_code_formatted",
                    "seeds": [24, 33, 42, 50],
                },
                "right": {
                    "chance_level": 0.5,
                    "config": "base_code",
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    "seeds": [24, 33, 42, 50],
                },
            },
            {
                "row_title": "Revealed\nscore MMLU",
                "left": {
                    "chance_level": 0.25,
                    "config": "base_score",
                    "eval_fold": "eval_revealing_score_formatted",
                    "seeds": [24, 33, 42, 50],
                },
                "right": {
                    "chance_level": 0.5,
                    "config": "base_score",
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    "seeds": [24, 33, 42, 50],
                },
            },
            {
                "row_title": "World affecting\nreward",
                "left": {
                    "chance_level": 0.5,
                    "config": "base_war",
                    "eval_fold": "eval_world_affecting_reward_reorg_formatted",
                    "seeds": [24, 33, 42, 50],
                },
                "right": {
                    "chance_level": 0.5,
                    "config": "base_war",
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    "seeds": [24, 33, 42, 50],
                },
            },
        ],
    },
    "appendix_figure0b": {
        "name": "appendix_figure0b",
        "title": "No penalisation - Unparsable rate",
        "skip_threshold": None,  # No skip threshold for unparsable plots
        "show_ratio": False,
        "show_summary": False,
        "plot_unparsable": True,
        "rows": [
            {
                "row_title": "Factual\nsycophancy",
                "left": {
                    "config": "base_sycophancy",
                    "eval_fold": "eval_sycophancy_formatted",
                    "seeds": [24, 33, 42, 50],
                },
                "right": {
                    "config": "base_sycophancy",
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    "seeds": [24, 33, 42, 50],
                },
            },
            {
                "row_title": "Code\nselection",
                "left": {
                    "config": "base_code",
                    "eval_fold": "eval_code_formatted",
                    "seeds": [24, 33, 42, 50],
                },
                "right": {
                    "config": "base_code",
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    "seeds": [24, 33, 42, 50],
                },
            },
            {
                "row_title": "Revealed\nscore MMLU",
                "left": {
                    "config": "base_score",
                    "eval_fold": "eval_revealing_score_formatted",
                    "seeds": [24, 33, 42, 50],
                },
                "right": {
                    "config": "base_score",
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    "seeds": [24, 33, 42, 50],
                },
            },
            {
                "row_title": "World affecting\nreward",
                "left": {
                    "config": "base_war",
                    "eval_fold": "eval_world_affecting_reward_reorg_formatted",
                    "seeds": [24, 33, 42, 50],
                },
                "right": {
                    "config": "base_war",
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    "seeds": [24, 33, 42, 50],
                },
            },
        ],
    },
    "appendix_figure1": {
        "name": "appendix_figure1",
        "title": "CoT penalisation",
        "skip_threshold": 0.3,
        "show_ratio": False,
        "show_summary": False,
        "rows": [
            {
                "row_title": "Factual\nsycophancy",
                "left": {
                    "chance_level": 0.5,
                    "config": "sycophancy",
                    "eval_fold": "eval_sycophancy_formatted",
                    "seeds": [24, 33, 42, 50],
                },
                "right": {
                    "chance_level": 0.5,
                    "config": "sycophancy",
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    "seeds": [24, 33, 42, 50],
                },
            },
            {
                "row_title": "Code\nselection",
                "left": {
                    "chance_level": 0.5,
                    "config": "code",
                    "eval_fold": "eval_code_formatted",
                    "seeds": [24, 33, 42, 50],
                },
                "right": {
                    "chance_level": 0.5,
                    "config": "code",
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    "seeds": [24, 33, 42, 50],
                },
            },
            {
                "row_title": "Revealed\nscore MMLU",
                "left": {
                    "chance_level": 0.25,
                    "config": "score",
                    "eval_fold": "eval_revealing_score_formatted",
                    "seeds": [24, 33, 42, 50],
                },
                "right": {
                    "chance_level": 0.5,
                    "config": "score",
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    "seeds": [24, 33, 42, 50],
                },
            },
            {
                "row_title": "World affecting\nreward",
                "left": {
                    "chance_level": 0.5,
                    "config": "war",
                    "eval_fold": "eval_world_affecting_reward_reorg_formatted",
                    "seeds": [24, 33, 42, 50],
                },
                "right": {
                    "chance_level": 0.5,
                    "config": "war",
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    "seeds": [24, 33, 42, 50],
                },
            },
        ],
    },
    "appendix_figure1b": {
        "name": "appendix_figure1b",
        "title": "CoT penalisation - Unparsable rate",
        "skip_threshold": None,  # No skip threshold for unparsable plots
        "show_ratio": False,
        "show_summary": False,
        "plot_unparsable": True,
        "rows": [
            {
                "row_title": "Factual\nsycophancy",
                "left": {
                    "config": "sycophancy",
                    "eval_fold": "eval_sycophancy_formatted",
                    "seeds": [24, 33, 42, 50],
                },
                "right": {
                    "config": "sycophancy",
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    "seeds": [24, 33, 42, 50],
                },
            },
            {
                "row_title": "Code\nselection",
                "left": {
                    "config": "code",
                    "eval_fold": "eval_code_formatted",
                    "seeds": [24, 33, 42, 50],
                },
                "right": {
                    "config": "code",
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    "seeds": [24, 33, 42, 50],
                },
            },
            {
                "row_title": "Revealed\nscore MMLU",
                "left": {
                    "config": "score",
                    "eval_fold": "eval_revealing_score_formatted",
                    "seeds": [24, 33, 42, 50],
                },
                "right": {
                    "config": "score",
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    "seeds": [24, 33, 42, 50],
                },
            },
            {
                "row_title": "World affecting\nreward",
                "left": {
                    "config": "war",
                    "eval_fold": "eval_world_affecting_reward_reorg_formatted",
                    "seeds": [24, 33, 42, 50],
                },
                "right": {
                    "config": "war",
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    "seeds": [24, 33, 42, 50],
                },
            },
        ],
    },
    "appendix_figure2": {
        "name": "appendix_figure2",
        "title": "Summary penalisation",
        "skip_threshold": 0.3,
        "show_ratio": False,
        "show_summary": True,
        "rows": [
            {
                "row_title": "Factual\nsycophancy",
                "left": {
                    "chance_level": 0.5,
                    "config": "sum_sycophancy",
                    "eval_fold": "eval_sycophancy_formatted",
                    "seeds": [24, 33, 42, 50],
                },
                "right": {
                    "chance_level": 0.5,
                    "config": "sum_sycophancy",
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    "seeds": [24, 33, 42, 50],
                },
            },
            {
                "row_title": "Code\nselection",
                "left": {
                    "chance_level": 0.5,
                    "config": "sum_code",
                    "eval_fold": "eval_code_formatted",
                    "seeds": [24, 33, 42, 50],
                },
                "right": {
                    "chance_level": 0.5,
                    "config": "sum_code",
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    "seeds": [24, 33, 42, 50],
                },
            },
            {
                "row_title": "Revealed\nscore MMLU",
                "left": {
                    "chance_level": 0.25,
                    "config": "sum_score",
                    "eval_fold": "eval_revealing_score_formatted",
                    "seeds": [24, 33, 42, 50],
                },
                "right": {
                    "chance_level": 0.5,
                    "config": "sum_score",
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    "seeds": [24, 33, 42, 50],
                },
            },
            {
                "row_title": "World affecting\nreward",
                "left": {
                    "chance_level": 0.5,
                    "config": "sum_war",
                    "eval_fold": "eval_world_affecting_reward_reorg_formatted",
                    "seeds": [24, 33, 42, 50],
                },
                "right": {
                    "chance_level": 0.5,
                    "config": "sum_war",
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    "seeds": [24, 33, 42, 50],
                },
            },
        ],
    },
    "appendix_figure2b": {
        "name": "appendix_figure2b",
        "title": "Summary penalisation - Unparsable rate",
        "skip_threshold": None,  # No skip threshold for unparsable plots
        "show_ratio": False,
        "show_summary": True,
        "plot_unparsable": True,
        "rows": [
            {
                "row_title": "Factual\nsycophancy",
                "left": {
                    "config": "sum_sycophancy",
                    "eval_fold": "eval_sycophancy_formatted",
                    "seeds": [24, 33, 42, 50],
                },
                "right": {
                    "config": "sum_sycophancy",
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    "seeds": [24, 33, 42, 50],
                },
            },
            {
                "row_title": "Code\nselection",
                "left": {
                    "config": "sum_code",
                    "eval_fold": "eval_code_formatted",
                    "seeds": [24, 33, 42, 50],
                },
                "right": {
                    "config": "sum_code",
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    "seeds": [24, 33, 42, 50],
                },
            },
            {
                "row_title": "Revealed\nscore MMLU",
                "left": {
                    "config": "sum_score",
                    "eval_fold": "eval_revealing_score_formatted",
                    "seeds": [24, 33, 42, 50],
                },
                "right": {
                    "config": "sum_score",
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    "seeds": [24, 33, 42, 50],
                },
            },
            {
                "row_title": "World affecting\nreward",
                "left": {
                    "config": "sum_war",
                    "eval_fold": "eval_world_affecting_reward_reorg_formatted",
                    "seeds": [24, 33, 42, 50],
                },
                "right": {
                    "config": "sum_war",
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    "seeds": [24, 33, 42, 50],
                },
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
    """Extract and prepare data for a single line.
    
    Note: Smoothing is applied AFTER filtering by skip_threshold, so the rolling
    window operates in step-space (only on points that are actually plotted).
    """
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
    
    # Apply skip threshold BEFORE smoothing (smoothing happens in step-space)
    if skip_threshold is not None:
        mask = df_line[no_answer_col] <= skip_threshold
        df_line = df_line[mask]
    
    # Rename for consistency
    df_line = df_line.rename(columns={correct_col: "correct_rate", no_answer_col: "no_answer_rate"})
    
    return df_line.sort_values("step")


def get_unparsable_data(config: str, seed: int, eval_fold: str) -> pd.DataFrame:
    """Extract unparsable rate data for a single line (no filtering, no smoothing)."""
    df = load_config_data(config)
    
    # Filter to seed and eval_fold
    df_line = df[(df["seed"] == seed) & (df["eval_fold"] == eval_fold)].copy()
    
    if df_line.empty:
        print(f"  WARNING: No data for config={config}, seed={seed}, eval_fold={eval_fold}")
        return df_line
    
    # Detect column names
    no_answer_col = "no_answer_rate" if "no_answer_rate" in df_line.columns else "no_answer_tags_rate"
    
    # Aggregate duplicate steps
    agg_cols = {no_answer_col: "mean"}
    df_line = df_line.groupby("step", as_index=False).agg(agg_cols)
    
    # Rename for consistency
    df_line = df_line.rename(columns={no_answer_col: "no_answer_rate"})
    
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


def create_block_legend(fig, show_summary=False, fontsize=14):
    """Create a legend with colored blocks instead of lines."""
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_CORRECT, edgecolor='none', label='Reward hacking rate'),
        mpatches.Patch(facecolor=COLOR_MONITOR, edgecolor='none', label='Penalty rate'),
    ]
    if show_summary:
        legend_elements.append(
            mpatches.Patch(facecolor=COLOR_SUMMARY, edgecolor='none', label='Summary penalty rate')
        )
    
    legend = fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=len(legend_elements),
        frameon=False,
        fontsize=fontsize,
        handlelength=1.5,
        handleheight=1.5,
    )
    return legend


def create_appendix_legend(fig, seeds, show_summary=False, fontsize=12):
    """Create a legend for appendix figures with linestyles and seed numbers."""
    legend_elements = []
    
    # First row: metric colors (as patches)
    legend_elements.append(
        mpatches.Patch(facecolor=COLOR_CORRECT, edgecolor='none', label='Reward hacking rate')
    )
    legend_elements.append(
        mpatches.Patch(facecolor=COLOR_MONITOR, edgecolor='none', label='Penalty rate')
    )
    if show_summary:
        legend_elements.append(
            mpatches.Patch(facecolor=COLOR_SUMMARY, edgecolor='none', label='Summary penalty rate')
        )
    
    # Second row: seed linestyles (as lines)
    for seed in sorted(seeds):
        style = SEED_STYLES.get(seed, {"linestyle": "-"})
        legend_elements.append(
            Line2D([0], [0], color='gray', linewidth=2, linestyle=style["linestyle"], label=f'Seed {seed}')
        )
    
    legend = fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=len(seeds) + (3 if show_summary else 2),
        frameon=False,
        fontsize=fontsize,
        handlelength=2.0,
        handleheight=1.5,
    )
    return legend


def create_unparsable_legend(fig, seeds, fontsize=12):
    """Create a legend for unparsable rate figures."""
    legend_elements = []
    
    # Metric color
    legend_elements.append(
        mpatches.Patch(facecolor=COLOR_NON_EXTRACTABLE, edgecolor='none', label='Unparsable rate')
    )
    
    # Seed linestyles
    for seed in sorted(seeds):
        style = SEED_STYLES.get(seed, {"linestyle": "-"})
        legend_elements.append(
            Line2D([0], [0], color='gray', linewidth=2, linestyle=style["linestyle"], label=f'Seed {seed}')
        )
    
    legend = fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=len(seeds) + 1,
        frameon=False,
        fontsize=fontsize,
        handlelength=2.0,
        handleheight=1.5,
    )
    return legend


def plot_subplot(ax, subplot_spec: dict, skip_threshold: float, show_ratio: bool = False, show_summary: bool = False):
    """Plot a single subplot with multiple lines."""
    ax.set_title(subplot_spec["title"], fontsize=16, fontweight='bold')
    
    if not subplot_spec.get("lines"):
        ax.text(0.5, 0.5, "TBD", ha="center", va="center", transform=ax.transAxes, fontsize=14, color="gray")
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
        
        # Compute rolling averages (now in step-space since filtering already applied)
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
            markersize=5,
            alpha=0.3,
        )
        ax.plot(
            df_line["step"],
            np.ma.masked_invalid(correct_smooth),
            color=color_correct,
            linewidth=2.5,
            linestyle="-",
            label="Reward hacking rate",
            alpha=1.0,
        )
        
        # Plot monitor flag rate (penalty) - faint markers, solid line
        ax.plot(
            df_line["step"],
            np.ma.masked_invalid(monitor_raw),
            color=color_monitor,
            linewidth=0,
            marker="o",
            markersize=5,
            alpha=0.3,
        )
        ax.plot(
            df_line["step"],
            np.ma.masked_invalid(monitor_smooth),
            color=color_monitor,
            linewidth=2.5,
            linestyle="-",
            label="Penalty rate",
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
                    markersize=5,
                    alpha=0.3,
                )
                ax.plot(
                    df_line["step"],
                    np.ma.masked_invalid(summary_smooth),
                    color=COLOR_SUMMARY,
                    linewidth=2.5,
                    linestyle="-",
                    label="Summary penalty rate",
                    alpha=1.0,
                )
    
    chance_level = subplot_spec.get("chance_level", 0.5)
    ax.axhline(y=chance_level, color=COLOR_CHANCE, linestyle="--", linewidth=1.5, zorder=0)
    ax.set_xlabel("Training Step", fontsize=14)
    ax.set_ylabel("Rate", fontsize=14)
    ax.set_xticks([0, 1000, 2000, 3000, 4000])
    ax.set_xlim(0, 4000)
    ax.tick_params(axis='both', labelsize=12)
    style_axis(ax)


def plot_appendix_subplot(ax, subplot_spec: dict, skip_threshold: float, show_ratio: bool = False, show_summary: bool = False):
    """Plot a single appendix subplot with multiple seeds (no markers)."""
    if subplot_spec is None:
        ax.text(0.5, 0.5, "TBD", ha="center", va="center", transform=ax.transAxes, fontsize=14, color="gray")
        ax.set_xticks([0, 1000, 2000, 3000, 4000])
        ax.set_xlim(0, 4000)
        style_axis(ax)
        return
    
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
        
        style = SEED_STYLES.get(seed, {"linestyle": "-"})
        colors = SEED_COLORS.get(seed, {"correct": COLOR_CORRECT, "monitor": COLOR_MONITOR})
        
        # Compute rolling averages (in step-space)
        correct_raw = df_line["correct_rate"]
        monitor_raw = df_line["monitor_flag_rate_extractable"]
        correct_smooth = correct_raw.rolling(window=3, min_periods=1, center=True).mean()
        monitor_smooth = monitor_raw.rolling(window=3, min_periods=1, center=True).mean()
        
        # Plot correct rate (reward hacking) - no markers, styled line
        ax.plot(
            df_line["step"],
            np.ma.masked_invalid(correct_smooth),
            color=colors["correct"],
            linewidth=2.5,
            linestyle=style["linestyle"],
            alpha=1.0,
        )
        
        # Plot monitor flag rate (penalty) - no markers, styled line
        ax.plot(
            df_line["step"],
            np.ma.masked_invalid(monitor_smooth),
            color=colors["monitor"],
            linewidth=2.5,
            linestyle=style["linestyle"],
            alpha=1.0,
        )
        
        # Plot summary monitor rate if enabled and available
        if show_summary and "summary_monitor_flag_rate_extractable" in df_line.columns:
            summary_raw = df_line["summary_monitor_flag_rate_extractable"]
            if summary_raw.notna().any():
                summary_smooth = summary_raw.rolling(window=3, min_periods=1, center=True).mean()
                ax.plot(
                    df_line["step"],
                    np.ma.masked_invalid(summary_smooth),
                    color=COLOR_SUMMARY,
                    linewidth=2.5,
                    linestyle=style["linestyle"],
                    alpha=1.0,
                )
    
    chance_level = subplot_spec.get("chance_level", 0.5)
    ax.axhline(y=chance_level, color=COLOR_CHANCE, linestyle="--", linewidth=1.5, zorder=0)
    ax.set_xticks([0, 1000, 2000, 3000, 4000])
    ax.set_xlim(0, 4000)
    ax.tick_params(axis='both', labelsize=12)
    style_axis(ax)


def plot_unparsable_subplot(ax, subplot_spec: dict):
    """Plot unparsable rate for a single appendix subplot (no smoothing, no markers)."""
    if subplot_spec is None:
        ax.text(0.5, 0.5, "TBD", ha="center", va="center", transform=ax.transAxes, fontsize=14, color="gray")
        ax.set_xticks([0, 1000, 2000, 3000, 4000])
        ax.set_xlim(0, 4000)
        style_axis(ax)
        return
    
    config = subplot_spec["config"]
    eval_fold = subplot_spec["eval_fold"]
    seeds = subplot_spec["seeds"]
    
    for seed in seeds:
        df_line = get_unparsable_data(
            config=config,
            seed=seed,
            eval_fold=eval_fold,
        )
        
        if df_line.empty:
            continue
        
        style = SEED_STYLES.get(seed, {"linestyle": "-"})
        
        # Plot unparsable rate - no smoothing, no markers
        ax.plot(
            df_line["step"],
            np.ma.masked_invalid(df_line["no_answer_rate"]),
            color=COLOR_NON_EXTRACTABLE,
            linewidth=2.0,
            linestyle=style["linestyle"],
            alpha=1.0,
        )
    
    ax.set_xticks([0, 1000, 2000, 3000, 4000])
    ax.set_xlim(0, 4000)
    ax.tick_params(axis='both', labelsize=12)
    style_axis(ax)


def generate_figure(fig_spec: dict):
    """Generate a single figure with all its subplots."""
    n_subplots = len(fig_spec["subplots"])
    skip_threshold = fig_spec.get("skip_threshold", 0.3)
    show_ratio = fig_spec.get("show_ratio", False)
    show_summary = fig_spec.get("show_summary", False)
    
    fig, axes = plt.subplots(1, n_subplots, figsize=(5 * n_subplots, 4.5), squeeze=False)
    axes = axes.flatten()
    
    for idx, subplot_spec in enumerate(fig_spec["subplots"]):
        plot_subplot(axes[idx], subplot_spec, skip_threshold, show_ratio=show_ratio, show_summary=show_summary)
        
        # Remove y label and ticks from non-leftmost subplots
        if idx > 0:
            axes[idx].set_ylabel("")
            axes[idx].set_yticklabels([])
            axes[idx].tick_params(axis='y', length=0)
    
    # Create block legend
    create_block_legend(fig, show_summary=show_summary, fontsize=14)
    
    title = fig_spec.get("title")
    if title:
        fig.suptitle(title, fontsize=18, fontweight='bold')
    
    fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    
    return fig


def generate_appendix_figure(fig_spec: dict):
    """Generate an appendix figure with 4 rows x 2 columns grid."""
    rows = fig_spec["rows"]
    n_rows = len(rows)
    skip_threshold = fig_spec.get("skip_threshold", 0.3)
    show_ratio = fig_spec.get("show_ratio", False)
    show_summary = fig_spec.get("show_summary", False)
    plot_unparsable = fig_spec.get("plot_unparsable", False)
    
    # Collect all seeds for legend
    all_seeds = set()
    for row_spec in rows:
        if row_spec["left"]:
            all_seeds.update(row_spec["left"].get("seeds", []))
        if row_spec["right"]:
            all_seeds.update(row_spec["right"].get("seeds", []))
    
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 3.5 * n_rows), squeeze=False)
    
    for row_idx, row_spec in enumerate(rows):
        # Add row label (y-axis label) - "Leave out\n{X}"
        row_title = row_spec.get("row_title", "")
        if row_title:
            axes[row_idx, 0].set_ylabel(f"Leave out\n{row_title}", fontsize=14, fontweight='bold')
        
        if plot_unparsable:
            # Plot unparsable rate
            plot_unparsable_subplot(axes[row_idx, 0], row_spec["left"])
            plot_unparsable_subplot(axes[row_idx, 1], row_spec["right"])
        else:
            # Plot regular metrics
            plot_appendix_subplot(axes[row_idx, 0], row_spec["left"], skip_threshold, show_ratio=show_ratio, show_summary=show_summary)
            plot_appendix_subplot(axes[row_idx, 1], row_spec["right"], skip_threshold, show_ratio=show_ratio, show_summary=show_summary)
        
        # Remove y label from right column
        axes[row_idx, 1].set_ylabel("")
        axes[row_idx, 1].set_yticklabels([])
        axes[row_idx, 1].tick_params(axis='y', length=0)
        
        # Only show x-axis labels on bottom row
        if row_idx < n_rows - 1:
            axes[row_idx, 0].set_xticklabels([])
            axes[row_idx, 1].set_xticklabels([])
            axes[row_idx, 0].set_xlabel("")
            axes[row_idx, 1].set_xlabel("")
        else:
            axes[row_idx, 0].set_xlabel("Training Step", fontsize=14)
            axes[row_idx, 1].set_xlabel("Training Step", fontsize=14)
    
    # Add column headers (once at top)
    axes[0, 0].annotate(
        "OOD fold eval",
        xy=(0.5, 1.15),
        xycoords="axes fraction",
        fontsize=16,
        fontweight="bold",
        ha="center",
        va="bottom",
    )
    axes[0, 1].annotate(
        "Medical sycophancy eval",
        xy=(0.5, 1.15),
        xycoords="axes fraction",
        fontsize=16,
        fontweight="bold",
        ha="center",
        va="bottom",
    )
    
    # Create appropriate legend
    if plot_unparsable:
        create_unparsable_legend(fig, sorted(all_seeds), fontsize=12)
    else:
        create_appendix_legend(fig, sorted(all_seeds), show_summary=show_summary, fontsize=12)
    
    title = fig_spec.get("title")
    if title:
        fig.suptitle(title, fontsize=18, fontweight="bold")
    
    fig.tight_layout(rect=[0.08, 0.06, 1, 0.94])
    
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