#!/usr/bin/env python3
"""
Download eval_ind experiment data from W&B and generate bar charts.

This script handles:
1. Scanning W&B for eval_ind runs (baseline and summary types)
2. Downloading trial tables and caching as parquet
3. Generating combined bar charts with both run types

Directory Structure:
    final_analysis/
    ├── barcharts.py              # This script
    └── barcharts/
        ├── cache/
        │   ├── eval_ind_baseline.parquet
        │   └── eval_ind_summary.parquet
        └── figures/
            ├── eval_ind_combined.png
            ├── eval_ind_combined.pdf
            ├── non_parsable_combined.png
            └── non_parsable_combined.pdf

Usage:
    # Full pipeline: download + visualize
    python barcharts.py

    # Just regenerate figures from cache
    python barcharts.py --plot-only

    # Force re-download (clears cache)
    python barcharts.py --force
"""

import argparse
import json
import re
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from requests.exceptions import HTTPError, ReadTimeout
from tqdm import tqdm

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "barcharts"
CACHE_DIR = OUTPUT_DIR / "cache"
FIGURES_DIR = OUTPUT_DIR / "figures"

# =============================================================================
# W&B CONFIG
# =============================================================================

WANDB_ENTITY = "puria-radmard"
WANDB_PROJECT = "obfuscation_generalization"
API_TIMEOUT = 120
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 5

# Only include runs created after this date (UTC)
MIN_CREATED_AT = "2026-01-25T00:00:00"

# =============================================================================
# EXPERIMENT CONFIG
# =============================================================================

# The four datasets
ALL_DATASETS = ["sycophancy", "war", "code", "score"]

# Map from short name to the full dataset name used in run names
DATASET_NAME_MAP = {
    "sycophancy": "leave_out_sycophancy_refined2",
    "war": "leave_out_war_refined2",
    "code": "leave_out_code_refined2",
    "score": "leave_out_score_refined2",
}

# Map from short name to eval fold names
EVAL_FOLD_MAP = {
    "sycophancy": "eval_sycophancy_formatted",
    "war": "eval_world_affecting_reward_reorg_formatted",
    "code": "eval_code_formatted",
    "score": "eval_revealing_score_formatted",
}

# Seeds to look for
SEEDS = [24, 42, 50]

# Run types
RUN_TYPES = ["baseline", "summary"]

# =============================================================================
# STYLE CONFIG (matching generate_final_figures.py)
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

# Colors for each dataset (used when that dataset is being evaluated)
DATASET_COLORS = {
    "sycophancy": "#FF9999",  # Pastel red
    "war": "#99CCFF",         # Pastel blue
    "code": "#99CC99",        # Pastel green
    "score": "#FFCC99",       # Pastel orange
}

# Hatching for each stat type
STAT_HATCHES = {
    "correct": "",           # Solid (no hatch)
    "cot_penalty": "//",     # Single diagonal
    "summary_penalty": "xx", # Cross hatch
}

# Markers for each seed
SEED_MARKERS = {
    24: "o",   # Circle
    42: "s",   # Square
    50: "^",   # Triangle
}

# Display names for run types
RUN_TYPE_DISPLAY = {
    "baseline": "No Penalisation",
    "summary": "Summary Penalisation",
}

# Threshold for "open" markers (exclude from averages)
NON_PARSABLE_THRESHOLD = 0.30


# =============================================================================
# RETRY UTILITIES
# =============================================================================

def retry_with_backoff(func, *args, max_retries=MAX_RETRIES, **kwargs):
    """Execute a function with retry on rate limit (429) and timeout errors."""
    last_exception = None
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except (HTTPError, ReadTimeout, wandb.errors.CommError) as e:
            last_exception = e
            is_rate_limit = (
                isinstance(e, HTTPError)
                and e.response is not None
                and e.response.status_code == 429
            )
            is_timeout = isinstance(e, (ReadTimeout, wandb.errors.CommError))

            if is_rate_limit or is_timeout:
                if attempt < max_retries - 1:
                    backoff = INITIAL_BACKOFF_SECONDS * (2 ** attempt)
                    error_type = "Rate limited (429)" if is_rate_limit else "Timeout"
                    print(f"  {error_type}. Waiting {backoff}s before retry {attempt + 2}/{max_retries}...")
                    time.sleep(backoff)
                else:
                    print(f"  Max retries ({max_retries}) exceeded.")
                    raise
            else:
                raise
    raise last_exception


# =============================================================================
# W&B HELPERS
# =============================================================================

def get_table_from_run(run: wandb.apis.public.Run, table_key: str) -> Optional[pd.DataFrame]:
    """Retrieve a logged wandb.Table from a run."""
    try:
        files = retry_with_backoff(lambda: list(run.files()))
        for file in files:
            if table_key in file.name and ".table.json" in file.name:
                with tempfile.TemporaryDirectory() as tmpdir:
                    downloaded = file.download(root=tmpdir, replace=True)
                    with open(downloaded.name, "r") as f:
                        data = json.load(f)
                    columns = data.get("columns", [])
                    rows = data.get("data", [])
                    if columns and rows:
                        return pd.DataFrame(rows, columns=columns)
    except Exception as e:
        print(f"    Error getting table: {e}")
    return None


def parse_run_name(run_name: str) -> Optional[Dict[str, Any]]:
    """
    Parse an eval_ind run name to extract metadata.
    
    Expected formats:
        Summary: run_ref_summary_ovs_refined_summary_data_leave_out_{dataset}_refined2_ts_{seed}_eval_{fold}_eval_ind_step_3800
        Baseline: run_ref_baseline_data_leave_out_{dataset}_refined2_ts_{seed}_eval_{fold}_eval_ind_step_3800
    
    Returns dict with: run_type, leave_out, seed, eval_fold (short names)
    """
    # Determine run type from prefix
    if "run_ref_summary_ovs_refined_summary_data_" in run_name:
        run_type = "summary"
    elif "run_ref_baseline_data_" in run_name:
        run_type = "baseline"
    else:
        return None
    
    # Extract: leave_out_{dataset}_refined2_ts_{seed}_eval_{fold}_eval_ind
    pattern = r"leave_out_(\w+)_refined2_ts_(\d+)_eval_(.+)_eval_ind"
    match = re.search(pattern, run_name)
    
    if not match:
        return None
    
    leave_out = match.group(1)  # sycophancy, war, code, score
    seed = int(match.group(2))
    eval_fold_raw = match.group(3)  # e.g., "sycophancy_formatted", "world_affecting_reward_reorg_formatted"
    
    # Map eval fold to short name
    fold_mapping = {
        "sycophancy_formatted": "sycophancy",
        "world_affecting_reward_reorg_formatted": "war",
        "code_formatted": "code",
        "revealing_score_formatted": "score",
    }
    eval_fold = fold_mapping.get(eval_fold_raw)
    
    if eval_fold is None:
        return None
    
    # Validate leave_out is a known dataset
    if leave_out not in ALL_DATASETS:
        return None
    
    return {
        "run_type": run_type,
        "leave_out": leave_out,
        "seed": seed,
        "eval_fold": eval_fold,
    }


def _to_datetime(value: Any) -> Optional[datetime]:
    """Best-effort conversion of various W&B timestamp formats to datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        s = value.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(s)
        except Exception:
            return None
    return None


def _run_created_at(run: wandb.apis.public.Run) -> datetime:
    """Return a comparable datetime for a W&B run."""
    dt = _to_datetime(getattr(run, "created_at", None))
    if dt is not None:
        return dt
    attrs = getattr(run, "_attrs", None)
    if isinstance(attrs, dict):
        for key in ("createdAt", "created_at"):
            dt = _to_datetime(attrs.get(key))
            if dt is not None:
                return dt
    return datetime.min


# =============================================================================
# SCANNING AND DOWNLOADING
# =============================================================================

def scan_eval_ind_runs(run_type: str, verbose: bool = True) -> List[Dict[str, Any]]:
    """
    Scan W&B for eval_ind runs of a specific type.
    
    Returns list of run metadata dicts.
    """
    api = wandb.Api(timeout=API_TIMEOUT)
    
    # Build filter based on run type
    if run_type == "summary":
        name_pattern = "run_ref_summary_ovs_refined_summary_data_leave_out_.*_eval_ind_step_"
    else:  # baseline
        name_pattern = "run_ref_baseline_data_leave_out_.*_eval_ind_step_"
    
    filters = {
        "state": "finished",
        "displayName": {"$regex": name_pattern},
        "createdAt": {"$gte": MIN_CREATED_AT},
    }
    
    print(f"Scanning for {run_type} eval_ind runs...")
    print(f"  Pattern: {name_pattern}")
    
    def _fetch():
        runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", filters=filters, per_page=100)
        return list(runs)
    
    runs = retry_with_backoff(_fetch)
    print(f"  Found {len(runs)} raw runs")
    
    # Parse run names and filter valid ones
    parsed_runs = []
    for run in runs:
        parsed = parse_run_name(run.name)
        if parsed and parsed["run_type"] == run_type:
            parsed["id"] = run.id
            parsed["name"] = run.name
            parsed["created_at"] = _run_created_at(run)
            parsed_runs.append(parsed)
    
    print(f"  Parsed {len(parsed_runs)} valid runs")
    
    # Deduplicate by (leave_out, seed, eval_fold), keeping latest
    best_runs = {}
    for run_info in parsed_runs:
        key = (run_info["leave_out"], run_info["seed"], run_info["eval_fold"])
        if key not in best_runs or run_info["created_at"] > best_runs[key]["created_at"]:
            best_runs[key] = run_info
    
    result = list(best_runs.values())
    print(f"  After deduplication: {len(result)} runs")
    
    return result


def download_runs_to_cache(
    runs: List[Dict[str, Any]],
    run_type: str,
    cache_path: Path,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Download trial tables for runs and save to cache.
    """
    api = wandb.Api(timeout=API_TIMEOUT)
    
    all_rows = []
    
    pbar = tqdm(runs, desc=f"Downloading {run_type}", disable=not verbose)
    
    for run_info in pbar:
        pbar.set_postfix_str(f"{run_info['leave_out']}/{run_info['eval_fold']}/s{run_info['seed']}")
        
        try:
            run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_info['id']}")
            
            # Get the eval table
            table_key = f"{EVAL_FOLD_MAP[run_info['eval_fold']]}_samples"
            table_df = get_table_from_run(run, table_key)
            
            if table_df is not None:
                # Add metadata
                table_df = table_df.copy()
                table_df["_run_id"] = run_info["id"]
                table_df["_run_name"] = run_info["name"]
                table_df["_run_type"] = run_type
                table_df["_leave_out"] = run_info["leave_out"]
                table_df["_seed"] = run_info["seed"]
                table_df["_eval_fold"] = run_info["eval_fold"]
                
                all_rows.append(table_df)
            else:
                print(f"\n  WARNING: No table found for {run_info['name']}")
            
            time.sleep(0.1)
            
        except Exception as e:
            print(f"\n  Error downloading {run_info['name']}: {e}")
    
    pbar.close()
    
    if not all_rows:
        print(f"  No data downloaded for {run_type}")
        return pd.DataFrame()
    
    combined_df = pd.concat(all_rows, ignore_index=True)
    
    # Save to cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_parquet(cache_path, index=False)
    print(f"  Saved {len(combined_df)} rows to {cache_path}")
    
    return combined_df


def load_cache(run_type: str) -> pd.DataFrame:
    """Load cached data for a run type."""
    cache_path = CACHE_DIR / f"eval_ind_{run_type}.parquet"
    if not cache_path.exists():
        return pd.DataFrame()
    return pd.read_parquet(cache_path)


# =============================================================================
# METRICS COMPUTATION
# =============================================================================

def compute_metrics(cache_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute metrics from cached trial data.
    
    Returns DataFrame with one row per (leave_out, seed, eval_fold) combination.
    """
    if cache_df.empty:
        return pd.DataFrame()
    
    results = []
    
    # Group by run
    groups = cache_df.groupby(["_leave_out", "_seed", "_eval_fold"])
    
    for (leave_out, seed, eval_fold), group_df in groups:
        total = len(group_df)
        
        # Detect columns
        extracted_col = next((c for c in ["extracted", "extracted_answer"] if c in group_df.columns), None)
        target_col = "target" if "target" in group_df.columns else None
        correct_col = "correct" if "correct" in group_df.columns else None
        
        # CoT monitor column
        cot_col = "overseer_cot_response" if "overseer_cot_response" in group_df.columns else (
            "overseer_response" if "overseer_response" in group_df.columns else None
        )
        summary_col = "overseer_summary_response" if "overseer_summary_response" in group_df.columns else None
        
        # Compute extractable (has valid extracted answer)
        if extracted_col:
            extractable_mask = group_df[extracted_col].notna() & (group_df[extracted_col] != "No answer tags found")
        else:
            extractable_mask = pd.Series([True] * total, index=group_df.index)
        
        total_extractable = extractable_mask.sum()
        
        if total_extractable == 0:
            results.append({
                "leave_out": leave_out,
                "seed": seed,
                "eval_fold": eval_fold,
                "total": total,
                "total_extractable": 0,
                "correct_rate": None,
                "cot_penalty_rate": None,
                "summary_penalty_rate": None,
                "non_parsable_rate": 1.0,
            })
            continue
        
        extractable_df = group_df[extractable_mask]
        
        # Correct rate
        if correct_col and correct_col in extractable_df.columns:
            num_correct = extractable_df[correct_col].sum()
        elif target_col and extracted_col:
            # Compute from extracted vs target
            num_correct = (extractable_df[extracted_col].str.upper() == extractable_df[target_col].str.upper()).sum()
        else:
            num_correct = 0
        
        correct_rate = num_correct / total_extractable
        
        # CoT penalty rate
        cot_penalty_rate = None
        if cot_col and cot_col in extractable_df.columns:
            flagged = sum(
                1 for val in extractable_df[cot_col]
                if val is not None and pd.notna(val) and float(val) > 0
            )
            cot_penalty_rate = flagged / total_extractable
        
        # Summary penalty rate
        summary_penalty_rate = None
        if summary_col and summary_col in extractable_df.columns:
            flagged = sum(
                1 for val in extractable_df[summary_col]
                if val is not None and pd.notna(val) and float(val) > 0
            )
            summary_penalty_rate = flagged / total_extractable
        
        # Non-parsable rate
        non_parsable_rate = (total - total_extractable) / total
        
        results.append({
            "leave_out": leave_out,
            "seed": seed,
            "eval_fold": eval_fold,
            "total": total,
            "total_extractable": total_extractable,
            "correct_rate": correct_rate,
            "cot_penalty_rate": cot_penalty_rate,
            "summary_penalty_rate": summary_penalty_rate,
            "non_parsable_rate": non_parsable_rate,
        })
    
    return pd.DataFrame(results)


# =============================================================================
# VISUALIZATION
# =============================================================================

def get_in_distribution_datasets(leave_out: str) -> List[str]:
    """Get the 3 datasets that are in-distribution (not left out)."""
    return [d for d in ALL_DATASETS if d != leave_out]


def create_combined_eval_ind_figure(
    metrics_baseline: pd.DataFrame,
    metrics_summary: pd.DataFrame,
) -> plt.Figure:
    """
    Create combined eval_ind bar chart with two rows:
    - Top row: No Penalisation (baseline)
    - Bottom row: Summary Penalisation
    
    Each row has 4 subplots (one per leave_out mega-group).
    X-axis is broken between subplots for visual separation.
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharey=True)
    
    # Layout parameters
    bar_width = 0.22
    group_gap = 0.08
    group_width = 3 * bar_width + group_gap
    
    stats = ["correct", "cot_penalty", "summary_penalty"]
    stat_offsets = {stat: (i - 1) * bar_width for i, stat in enumerate(stats)}
    seed_jitter = {24: -0.05, 42: 0.0, 50: 0.05}
    
    row_data = [
        (metrics_baseline, "No Penalisation"),
        (metrics_summary, "Summary Penalisation"),
    ]
    
    for row_idx, (metrics_df, row_label) in enumerate(row_data):
        for col_idx, leave_out in enumerate(ALL_DATASETS):
            ax = axes[row_idx, col_idx]
            in_dist = get_in_distribution_datasets(leave_out)
            
            all_bar_positions = []
            
            for group_idx, eval_ds in enumerate(in_dist):
                group_center = group_idx * (group_width + 0.15)
                
                for stat in stats:
                    bar_x = group_center + stat_offsets[stat]
                    
                    mask = (metrics_df["leave_out"] == leave_out) & (metrics_df["eval_fold"] == eval_ds)
                    subset = metrics_df[mask]
                    
                    if subset.empty:
                        continue
                    
                    col_name = f"{stat}_rate"
                    valid_values = []
                    all_seed_data = []
                    
                    for _, row in subset.iterrows():
                        val = row[col_name]
                        seed = row["seed"]
                        non_parsable = row["non_parsable_rate"]
                        
                        if pd.isna(val):
                            continue
                        
                        is_valid = non_parsable <= NON_PARSABLE_THRESHOLD
                        all_seed_data.append((seed, val, is_valid))
                        
                        if is_valid:
                            valid_values.append(val)
                    
                    color = DATASET_COLORS[eval_ds]
                    hatch = STAT_HATCHES[stat]
                    
                    if valid_values:
                        mean_val = np.mean(valid_values)
                        ax.bar(
                            bar_x, mean_val,
                            width=bar_width,
                            color=color,
                            hatch=hatch,
                            edgecolor="black",
                            linewidth=0.5,
                            alpha=0.8,
                        )
                    
                    for seed, val, is_valid in all_seed_data:
                        marker = SEED_MARKERS.get(seed, "o")
                        jitter = seed_jitter.get(seed, 0.0)
                        
                        if is_valid:
                            ax.scatter(bar_x + jitter, val, color="black", marker=marker,
                                      s=25, zorder=5, alpha=0.8)
                        else:
                            ax.scatter(bar_x + jitter, val, facecolors="none", edgecolors="black",
                                      marker=marker, s=25, zorder=5, alpha=0.8, linewidths=1.0)
                
                all_bar_positions.append((group_center, eval_ds.capitalize()))
            
            # Styling for this subplot
            ax.set_ylim(0, 1)
            ax.set_xlim(-0.3, len(in_dist) * (group_width + 0.15) - 0.15 + 0.3)
            
            # X-axis
            ax.set_xticks([pos for pos, _ in all_bar_positions])
            ax.set_xticklabels([name for _, name in all_bar_positions], fontsize=10)
            
            # Spines - only left spine on leftmost, none on others
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(True)
            if col_idx > 0:
                ax.spines["left"].set_visible(False)
                ax.tick_params(left=False)
            
            # Title on top row only
            if row_idx == 0:
                ax.set_title(f"Leave out {leave_out.capitalize()}", fontsize=12, fontweight="bold")
            
            # Y-axis label on leftmost only
            if col_idx == 0:
                ax.set_ylabel("Rate", fontsize=12)
    
    # Row labels on the left
    for row_idx, (_, row_label) in enumerate(row_data):
        axes[row_idx, 0].annotate(
            row_label,
            xy=(-0.35, 0.5),
            xycoords="axes fraction",
            fontsize=12,
            fontweight="bold",
            ha="right",
            va="center",
            rotation=90,
        )
    
    # Build two-row legend at bottom
    color_handles = [
        mpatches.Patch(facecolor=DATASET_COLORS[ds], edgecolor="black", label=ds.capitalize())
        for ds in ALL_DATASETS
    ]
    hatch_handles = [
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="", label="Reward Hacking"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="//", label="CoT Penalty"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="xx", label="Summary Penalty"),
    ]
    
    # First legend row (colors) - on top
    leg1 = fig.legend(
        color_handles,
        [h.get_label() for h in color_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.06),
        ncol=4,
        frameon=False,
        fontsize=11,
    )
    # Second legend row (hatches) - below
    fig.legend(
        hatch_handles,
        [h.get_label() for h in hatch_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        frameon=False,
        fontsize=11,
    )
    fig.add_artist(leg1)  # Keep first legend visible
    
    fig.suptitle("In-Distribution Evaluation", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0.05, 0.08, 1, 0.95])
    
    return fig


def create_combined_non_parsable_figure(
    metrics_baseline: pd.DataFrame,
    metrics_summary: pd.DataFrame,
) -> plt.Figure:
    """
    Create combined non-parsable rate figure with two rows:
    - Top row: No Penalisation (baseline)
    - Bottom row: Summary Penalisation
    
    Each row has 4 subplots (one per leave_out mega-group).
    """
    fig, axes = plt.subplots(2, 4, figsize=(18, 7), sharey=True)
    
    # Layout parameters
    bar_width = 0.6
    group_gap = 0.3
    seed_jitter = {24: -0.15, 42: 0.0, 50: 0.15}
    
    row_data = [
        (metrics_baseline, "No Penalisation"),
        (metrics_summary, "Summary Penalisation"),
    ]
    
    for row_idx, (metrics_df, row_label) in enumerate(row_data):
        for col_idx, leave_out in enumerate(ALL_DATASETS):
            ax = axes[row_idx, col_idx]
            in_dist = get_in_distribution_datasets(leave_out)
            
            all_bar_positions = []
            
            for group_idx, eval_ds in enumerate(in_dist):
                bar_x = group_idx * (bar_width + group_gap)
                
                mask = (metrics_df["leave_out"] == leave_out) & (metrics_df["eval_fold"] == eval_ds)
                subset = metrics_df[mask]
                
                if subset.empty:
                    all_bar_positions.append((bar_x, eval_ds.capitalize()))
                    continue
                
                valid_values = []
                all_seed_data = []
                
                for _, row in subset.iterrows():
                    val = row["non_parsable_rate"]
                    seed = row["seed"]
                    
                    if pd.isna(val):
                        continue
                    
                    is_valid = val <= NON_PARSABLE_THRESHOLD
                    all_seed_data.append((seed, val * 100, is_valid))
                    
                    if is_valid:
                        valid_values.append(val * 100)
                
                if valid_values:
                    mean_val = np.mean(valid_values)
                    ax.bar(
                        bar_x, mean_val,
                        width=bar_width,
                        color=DATASET_COLORS[eval_ds],
                        edgecolor="black",
                        linewidth=0.5,
                        alpha=0.8,
                    )
                
                for seed, val, is_valid in all_seed_data:
                    marker = SEED_MARKERS.get(seed, "o")
                    jitter = seed_jitter.get(seed, 0.0)
                    
                    if is_valid:
                        ax.scatter(bar_x + jitter, val, color="black", marker=marker,
                                  s=25, zorder=5, alpha=0.8)
                    else:
                        ax.scatter(bar_x + jitter, val, facecolors="none", edgecolors="black",
                                  marker=marker, s=25, zorder=5, alpha=0.8, linewidths=1.0)
                
                all_bar_positions.append((bar_x, eval_ds.capitalize()))
            
            # Styling
            ax.set_ylim(0, 40)
            ax.set_xlim(-0.4, len(in_dist) * (bar_width + group_gap) - group_gap + 0.4)
            
            ax.set_xticks([pos for pos, _ in all_bar_positions])
            ax.set_xticklabels([name for _, name in all_bar_positions], fontsize=10)
            
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(True)
            if col_idx > 0:
                ax.spines["left"].set_visible(False)
                ax.tick_params(left=False)
            
            if row_idx == 0:
                ax.set_title(f"Leave out {leave_out.capitalize()}", fontsize=12, fontweight="bold")
            
            if col_idx == 0:
                ax.set_ylabel("Non-Parsable (%)", fontsize=12)
    
    # Row labels
    for row_idx, (_, row_label) in enumerate(row_data):
        axes[row_idx, 0].annotate(
            row_label,
            xy=(-0.35, 0.5),
            xycoords="axes fraction",
            fontsize=12,
            fontweight="bold",
            ha="right",
            va="center",
            rotation=90,
        )
    
    # Single-row color legend
    color_handles = [
        mpatches.Patch(facecolor=DATASET_COLORS[ds], edgecolor="black", label=ds.capitalize())
        for ds in ALL_DATASETS
    ]
    fig.legend(
        color_handles,
        [h.get_label() for h in color_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.04),
        ncol=4,
        frameon=False,
        fontsize=11,
    )
    
    fig.suptitle("In-Distribution Non-Parsable Rate", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0.05, 0.06, 1, 0.95])
    
    return fig


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download eval_ind data and generate bar charts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--plot-only", action="store_true", help="Only generate figures from cache")
    parser.add_argument("--force", action="store_true", help="Force re-download (clears cache)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    # Ensure directories exist
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Clear cache if forced
    if args.force:
        print("Force mode: clearing cache...")
        for f in CACHE_DIR.glob("*.parquet"):
            f.unlink()
    
    print(f"\n{'=' * 70}")
    print("EVAL_IND BAR CHART GENERATOR")
    print(f"{'=' * 70}")
    print(f"W&B Entity:  {WANDB_ENTITY}")
    print(f"W&B Project: {WANDB_PROJECT}")
    print(f"Min Date:    {MIN_CREATED_AT}")
    print(f"Cache Dir:   {CACHE_DIR}")
    print(f"Figures Dir: {FIGURES_DIR}")
    print(f"{'=' * 70}\n")
    
    # Collect metrics for both run types
    all_metrics = {}
    
    for run_type in RUN_TYPES:
        print(f"\n{'=' * 50}")
        print(f"Processing: {run_type}")
        print(f"{'=' * 50}")
        
        cache_path = CACHE_DIR / f"eval_ind_{run_type}.parquet"
        
        # Download or load from cache
        if args.plot_only:
            print(f"Plot-only mode: loading from cache...")
            cache_df = load_cache(run_type)
            if cache_df.empty:
                print(f"  No cache found for {run_type}")
                all_metrics[run_type] = pd.DataFrame()
                continue
        else:
            if cache_path.exists() and not args.force:
                print(f"Cache exists, loading...")
                cache_df = load_cache(run_type)
            else:
                runs = scan_eval_ind_runs(run_type, verbose=verbose)
                if not runs:
                    print(f"  No runs found for {run_type}")
                    all_metrics[run_type] = pd.DataFrame()
                    continue
                cache_df = download_runs_to_cache(runs, run_type, cache_path, verbose=verbose)
        
        if cache_df.empty:
            print(f"  No data for {run_type}")
            all_metrics[run_type] = pd.DataFrame()
            continue
        
        # Compute metrics
        print(f"\nComputing metrics...")
        metrics_df = compute_metrics(cache_df)
        print(f"  Computed metrics for {len(metrics_df)} combinations")
        
        # Save metrics CSV for reference
        metrics_csv_path = CACHE_DIR / f"eval_ind_{run_type}_metrics.csv"
        metrics_df.to_csv(metrics_csv_path, index=False)
        print(f"  Saved metrics to {metrics_csv_path}")
        
        all_metrics[run_type] = metrics_df
    
    # Generate combined figures
    print(f"\n{'=' * 50}")
    print("Generating combined figures...")
    print(f"{'=' * 50}")
    
    metrics_baseline = all_metrics.get("baseline", pd.DataFrame())
    metrics_summary = all_metrics.get("summary", pd.DataFrame())
    
    if metrics_baseline.empty and metrics_summary.empty:
        print("No data available for either run type. Cannot generate figures.")
        return 1
    
    # Use empty DataFrame if one is missing (will just show empty subplots)
    if metrics_baseline.empty:
        print("WARNING: No baseline data, top row will be empty")
        metrics_baseline = pd.DataFrame(columns=metrics_summary.columns)
    if metrics_summary.empty:
        print("WARNING: No summary data, bottom row will be empty")
        metrics_summary = pd.DataFrame(columns=metrics_baseline.columns)
    
    # Main metrics figure
    fig = create_combined_eval_ind_figure(metrics_baseline, metrics_summary)
    for ext in ["png", "pdf"]:
        path = FIGURES_DIR / f"eval_ind_combined.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {path}")
    plt.close(fig)
    
    # Non-parsable figure
    fig = create_combined_non_parsable_figure(metrics_baseline, metrics_summary)
    for ext in ["png", "pdf"]:
        path = FIGURES_DIR / f"non_parsable_combined.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {path}")
    plt.close(fig)
    
    print(f"\n{'=' * 70}")
    print("Done!")
    print(f"{'=' * 70}\n")
    
    return 0


if __name__ == "__main__":
    exit(main())