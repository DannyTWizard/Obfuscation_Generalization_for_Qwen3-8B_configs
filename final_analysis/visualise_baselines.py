#!/usr/bin/env python3
"""
Download baseline (step 0) evaluation data from W&B and generate bar charts.

Baselines are pre-training evaluations that are shared across all training types.
They live in Nathaniel's entity with names like "base_model_eval_{eval_fold}".

This script handles:
1. Scanning W&B for baseline runs
2. Downloading trial tables and caching as parquet
3. Generating bar charts with accuracy, monitor flag rates, and parsing failure

Directory Structure:
    final_analysis/
    ├── visualise_baselines.py    # This script
    └── barcharts/
        ├── cache/
        │   └── baselines.parquet
        └── figures/
            ├── baselines.png
            └── baselines.pdf

Usage:
    # Full pipeline: download + visualize
    python visualise_baselines.py

    # Just regenerate figures from cache
    python visualise_baselines.py --plot-only

    # Force re-download (clears cache)
    python visualise_baselines.py --force
"""

import argparse
import json
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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

WANDB_ENTITY = "nathanielmitrani-cfis-upc"
WANDB_PROJECT = "obfuscation_generalization"
API_TIMEOUT = 120
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 5

# Seed is stored here in baseline run configs
SEED_CONFIG_PATH = "train.seed"

# =============================================================================
# EXPERIMENT CONFIG
# =============================================================================

# Seeds to look for
SEEDS = [24, 33, 42, 50]

# Eval folds to look for (short name -> full name)
EVAL_FOLDS = {
    "sycophancy": "eval_sycophancy_formatted",
    "war": "eval_world_affecting_reward_reorg_formatted",
    "code": "eval_code_formatted",
    "score": "eval_revealing_score_formatted",
    "medical": "eval_medical_sycophancy_fact_formatted",
}

# Display names for eval folds
EVAL_FOLD_DISPLAY = {
    "sycophancy": "Sycophancy",
    "war": "World Affecting\nReward",
    "code": "Code",
    "score": "Score",
    "medical": "Medical\nSycophancy",
}

# Order for display
EVAL_FOLD_ORDER = ["sycophancy", "war", "code", "score", "medical"]

# Stats to show (in order)
STATS = ["correct", "cot_penalty", "summary_penalty"]

# =============================================================================
# STYLE CONFIG (matching barcharts.py)
# =============================================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.weight": "normal",
    "axes.titleweight": "bold",
    "axes.labelweight": "normal",
    "figure.titleweight": "normal",
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})

# Colors for each eval fold
EVAL_FOLD_COLORS = {
    "sycophancy": "#FF9999",  # Pastel red
    "war": "#99CCFF",         # Pastel blue
    "code": "#99CC99",        # Pastel green
    "score": "#FFCC99",       # Pastel orange
    "medical": "#CC99FF",     # Pastel purple
}

# Hatching for each stat type (matching barcharts.py)
STAT_HATCHES = {
    "correct": "",           # Solid (no hatch)
    "cot_penalty": "//",     # Single diagonal
    "summary_penalty": "xx", # Cross hatch
}

# Markers for each seed
SEED_MARKERS = {
    24: "o",   # Circle
    33: "D",   # Diamond
    42: "s",   # Square
    50: "^",   # Triangle
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

def scan_baseline_runs(verbose: bool = True) -> List[Dict[str, Any]]:
    """
    Scan W&B for baseline runs.
    
    Baseline runs are named like: base_model_eval_{eval_fold}
    Seed is stored in config at SEED_CONFIG_PATH.
    
    Returns list of run metadata dicts.
    """
    api = wandb.Api(timeout=API_TIMEOUT)
    
    print(f"Scanning for baseline runs...")
    print(f"  Entity: {WANDB_ENTITY}")
    print(f"  Project: {WANDB_PROJECT}")
    
    all_runs = []
    
    for short_name, full_name in EVAL_FOLDS.items():
        run_name = f"base_model_{full_name}"
        
        for seed in SEEDS:
            filters = {
                "displayName": run_name,
                f"config.{SEED_CONFIG_PATH}": seed,
                "state": "finished",
            }
            
            def _fetch():
                runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", filters=filters, per_page=10)
                return list(runs)
            
            try:
                runs = retry_with_backoff(_fetch)
                
                if runs:
                    # Keep the most recent one
                    best_run = max(runs, key=_run_created_at)
                    all_runs.append({
                        "id": best_run.id,
                        "name": best_run.name,
                        "eval_fold": short_name,
                        "eval_fold_full": full_name,
                        "seed": seed,
                        "created_at": _run_created_at(best_run),
                    })
                    if verbose:
                        print(f"  Found: {short_name}/seed={seed}")
                else:
                    print(f"  WARNING: No run found for {short_name}/seed={seed}")
                    
            except Exception as e:
                print(f"  Error querying {short_name}/seed={seed}: {e}")
    
    print(f"\nFound {len(all_runs)} baseline runs")
    return all_runs


def download_baselines_to_cache(
    runs: List[Dict[str, Any]],
    cache_path: Path,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Download trial tables for baseline runs and save to cache.
    """
    api = wandb.Api(timeout=API_TIMEOUT)
    
    all_rows = []
    
    pbar = tqdm(runs, desc="Downloading baselines", disable=not verbose)
    
    for run_info in pbar:
        pbar.set_postfix_str(f"{run_info['eval_fold']}/s{run_info['seed']}")
        
        try:
            run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_info['id']}")
            
            # Get the eval table
            table_key = f"{run_info['eval_fold_full']}_samples"
            table_df = get_table_from_run(run, table_key)
            
            if table_df is not None:
                # Add metadata
                table_df = table_df.copy()
                table_df["_run_id"] = run_info["id"]
                table_df["_run_name"] = run_info["name"]
                table_df["_eval_fold"] = run_info["eval_fold"]
                table_df["_seed"] = run_info["seed"]
                
                all_rows.append(table_df)
            else:
                print(f"\n  WARNING: No table found for {run_info['name']}")
            
            time.sleep(0.1)
            
        except Exception as e:
            print(f"\n  Error downloading {run_info['name']}: {e}")
    
    pbar.close()
    
    if not all_rows:
        print("No data downloaded")
        return pd.DataFrame()
    
    combined_df = pd.concat(all_rows, ignore_index=True)
    
    # Save to cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_parquet(cache_path, index=False)
    print(f"Saved {len(combined_df)} rows to {cache_path}")
    
    return combined_df


def load_cache(cache_path: Path) -> pd.DataFrame:
    """Load cached data."""
    if not cache_path.exists():
        return pd.DataFrame()
    return pd.read_parquet(cache_path)


# =============================================================================
# METRICS COMPUTATION
# =============================================================================

def compute_metrics(cache_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute metrics from cached trial data.
    
    Returns DataFrame with one row per (eval_fold, seed) combination.
    """
    if cache_df.empty:
        return pd.DataFrame()
    
    results = []
    
    # Group by run
    groups = cache_df.groupby(["_eval_fold", "_seed"])
    
    for (eval_fold, seed), group_df in groups:
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
                "eval_fold": eval_fold,
                "seed": seed,
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
        
        # CoT monitor flag rate
        cot_penalty_rate = None
        if cot_col and cot_col in extractable_df.columns:
            try:
                flagged = sum(
                    1 for val in extractable_df[cot_col]
                    if val is not None and pd.notna(val) and float(val) > 0
                )
                cot_penalty_rate = flagged / total_extractable
            except (ValueError, TypeError):
                pass
        
        # Summary monitor flag rate
        summary_penalty_rate = None
        if summary_col and summary_col in extractable_df.columns:
            try:
                flagged = sum(
                    1 for val in extractable_df[summary_col]
                    if val is not None and pd.notna(val) and float(val) > 0
                )
                summary_penalty_rate = flagged / total_extractable
            except (ValueError, TypeError):
                pass
        
        # Non-parsable rate
        non_parsable_rate = (total - total_extractable) / total
        
        results.append({
            "eval_fold": eval_fold,
            "seed": seed,
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

def create_baselines_figure(metrics_df: pd.DataFrame) -> plt.Figure:
    """
    Create baseline bar chart with two rows:
    - Row 1: Rates (correct, cot penalty, summary penalty) with hatching
    - Row 2: Non-parsable rate
    
    Each eval_fold has 3 bars (grouped) with different hatches.
    Seed markers overlaid on bars.
    Open markers for seeds with non_parsable_rate > threshold (excluded from average).
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    # Layout parameters
    bar_width = 0.25
    n_stats = len(STATS)
    group_positions = list(range(len(EVAL_FOLD_ORDER)))
    
    # Stat offsets within each group (centered)
    stat_offsets = {
        "correct": -bar_width,
        "cot_penalty": 0,
        "summary_penalty": bar_width,
    }
    
    seed_jitter = {24: -0.06, 33: -0.02, 42: 0.02, 50: 0.06}
    
    # =========================
    # Row 1: Rates with hatching
    # =========================
    ax = axes[0]
    
    for group_idx, eval_fold in enumerate(EVAL_FOLD_ORDER):
        group_center = group_positions[group_idx]
        color = EVAL_FOLD_COLORS[eval_fold]
        
        subset = metrics_df[metrics_df["eval_fold"] == eval_fold]
        
        if subset.empty:
            continue
        
        for stat in STATS:
            bar_x = group_center + stat_offsets[stat]
            hatch = STAT_HATCHES[stat]
            col_name = f"{stat}_rate"
            
            if col_name not in subset.columns:
                continue
            
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
            
            # Draw bar with mean of valid values
            if valid_values:
                mean_val = np.mean(valid_values)
                stderr_val = np.std(valid_values, ddof=1) / np.sqrt(len(valid_values)) if len(valid_values) > 1 else 0
                
                ax.bar(
                    bar_x, mean_val,
                    width=bar_width,
                    color=color,
                    hatch=hatch,
                    edgecolor="black",
                    linewidth=0.5,
                    alpha=0.8,
                    yerr=stderr_val,
                    capsize=3,
                    error_kw={"elinewidth": 1.5, "capthick": 1.5, "ecolor": "black"},
                )
            
            # Overlay seed markers
            for seed, val, is_valid in all_seed_data:
                marker = SEED_MARKERS.get(seed, "o")
                jitter = seed_jitter.get(seed, 0.0)
                
                if is_valid:
                    ax.scatter(bar_x + jitter, val, color="black", marker=marker,
                              s=35, zorder=5, alpha=0.8)
                else:
                    ax.scatter(bar_x + jitter, val, facecolors="none", edgecolors="black",
                              marker=marker, s=35, zorder=5, alpha=0.8, linewidths=1.0)
    
    # Styling for row 1
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.5, len(EVAL_FOLD_ORDER) - 0.5)
    ax.set_ylabel("Rate", fontsize=18)
    ax.set_title("Baseline Evaluation Metrics", fontsize=20, fontweight="bold")
    ax.set_xticks(group_positions)
    ax.set_xticklabels([EVAL_FOLD_DISPLAY[ef] for ef in EVAL_FOLD_ORDER], fontsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Reference lines at 25%, 50%, 75%
    for ref_val in [0.25, 0.50, 0.75]:
        ax.axhline(y=ref_val, color="gray", linestyle="--", linewidth=1, alpha=0.5, zorder=0)
    
    # =========================
    # Row 2: Non-parsable rate
    # =========================
    ax = axes[1]
    bar_width_single = 0.5
    
    for group_idx, eval_fold in enumerate(EVAL_FOLD_ORDER):
        x_pos = group_positions[group_idx]
        color = EVAL_FOLD_COLORS[eval_fold]
        
        subset = metrics_df[metrics_df["eval_fold"] == eval_fold]
        
        if subset.empty:
            continue
        
        valid_values = []
        all_seed_data = []
        
        for _, row in subset.iterrows():
            val = row["non_parsable_rate"]
            seed = row["seed"]
            
            if pd.isna(val):
                continue
            
            display_val = val * 100
            is_valid = val <= NON_PARSABLE_THRESHOLD
            all_seed_data.append((seed, display_val, is_valid))
            
            if is_valid:
                valid_values.append(display_val)
        
        # Draw bar
        if valid_values:
            mean_val = np.mean(valid_values)
            stderr_val = np.std(valid_values, ddof=1) / np.sqrt(len(valid_values)) if len(valid_values) > 1 else 0
            
            ax.bar(
                x_pos, mean_val,
                width=bar_width_single,
                color=color,
                edgecolor="black",
                linewidth=0.5,
                alpha=0.8,
                yerr=stderr_val,
                capsize=4,
                error_kw={"elinewidth": 1.5, "capthick": 1.5, "ecolor": "black"},
            )
        
        # Overlay seed markers
        seed_jitter_wide = {24: -0.12, 33: -0.04, 42: 0.04, 50: 0.12}
        for seed, val, is_valid in all_seed_data:
            marker = SEED_MARKERS.get(seed, "o")
            jitter = seed_jitter_wide.get(seed, 0.0)
            
            if is_valid:
                ax.scatter(x_pos + jitter, val, color="black", marker=marker,
                          s=35, zorder=5, alpha=0.8)
            else:
                ax.scatter(x_pos + jitter, val, facecolors="none", edgecolors="black",
                          marker=marker, s=35, zorder=5, alpha=0.8, linewidths=1.0)
    
    # Styling for row 2
    ax.set_ylim(0, 105)
    ax.set_xlim(-0.5, len(EVAL_FOLD_ORDER) - 0.5)
    ax.set_ylabel("Non-Parsable (%)", fontsize=18)
    ax.set_title("Baseline Non-Parsable Rate", fontsize=20, fontweight="bold")
    ax.set_xticks(group_positions)
    ax.set_xticklabels([EVAL_FOLD_DISPLAY[ef] for ef in EVAL_FOLD_ORDER], fontsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Reference lines at 25%, 50%, 75%
    for ref_val in [25, 50, 75]:
        ax.axhline(y=ref_val, color="gray", linestyle="--", linewidth=1, alpha=0.5, zorder=0)
    
    # =========================
    # Build legend
    # =========================
    
    # Row 1: Colors (eval folds)
    color_handles = [
        mpatches.Patch(facecolor=EVAL_FOLD_COLORS[ef], edgecolor="black", 
                       label=EVAL_FOLD_DISPLAY[ef].replace("\n", " "))
        for ef in EVAL_FOLD_ORDER
    ]
    
    # Row 2: Hatches (stat types)
    hatch_handles = [
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="", label="Reward Hacking"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="//", label="CoT Penalty"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="xx", label="Summary Penalty"),
    ]
    
    # Row 3: Seed markers
    seed_handles = [
        Line2D([0], [0], marker=SEED_MARKERS[seed], color='black', linestyle='None',
               markersize=8, label=f'Seed {seed}')
        for seed in SEEDS
    ]
    
    # Row 4: Open vs closed marker explanation
    marker_handles = [
        Line2D([0], [0], marker='o', color='black', linestyle='None',
               markersize=8, label='Valid (included in avg)'),
        Line2D([0], [0], marker='o', markerfacecolor='none', markeredgecolor='black',
               linestyle='None', markersize=8, markeredgewidth=1.5,
               label=f'Non-parsable >{NON_PARSABLE_THRESHOLD:.0%} (excluded)'),
    ]
    
    # Add legends
    leg1 = fig.legend(
        color_handles,
        [h.get_label() for h in color_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.10),
        ncol=5,
        frameon=False,
        fontsize=14,
        handlelength=2.5,
        handleheight=1.5,
    )
    leg2 = fig.legend(
        hatch_handles,
        [h.get_label() for h in hatch_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.065),
        ncol=3,
        frameon=False,
        fontsize=14,
        handlelength=2.5,
        handleheight=1.5,
    )
    leg3 = fig.legend(
        seed_handles,
        [h.get_label() for h in seed_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=4,
        frameon=False,
        fontsize=14,
    )
    fig.legend(
        marker_handles,
        [h.get_label() for h in marker_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=2,
        frameon=False,
        fontsize=14,
    )
    fig.add_artist(leg1)
    fig.add_artist(leg2)
    fig.add_artist(leg3)
    
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    
    return fig


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download baseline evaluation data and generate bar charts",
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
    
    cache_path = CACHE_DIR / "baselines.parquet"
    
    # Clear cache if forced
    if args.force and cache_path.exists():
        print("Force mode: clearing cache...")
        cache_path.unlink()
    
    print(f"\n{'=' * 70}")
    print("BASELINE BAR CHART GENERATOR")
    print(f"{'=' * 70}")
    print(f"W&B Entity:  {WANDB_ENTITY}")
    print(f"W&B Project: {WANDB_PROJECT}")
    print(f"Seeds:       {SEEDS}")
    print(f"Eval folds:  {list(EVAL_FOLDS.keys())}")
    print(f"Cache:       {cache_path}")
    print(f"Figures Dir: {FIGURES_DIR}")
    print(f"{'=' * 70}\n")
    
    # Download or load from cache
    if args.plot_only:
        print("Plot-only mode: loading from cache...")
        cache_df = load_cache(cache_path)
        if cache_df.empty:
            print("ERROR: No cache found. Run without --plot-only first.")
            return 1
    else:
        if cache_path.exists() and not args.force:
            print("Cache exists, loading...")
            cache_df = load_cache(cache_path)
        else:
            runs = scan_baseline_runs(verbose=verbose)
            if not runs:
                print("ERROR: No baseline runs found.")
                return 1
            cache_df = download_baselines_to_cache(runs, cache_path, verbose=verbose)
    
    if cache_df.empty:
        print("ERROR: No data available.")
        return 1
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics_df = compute_metrics(cache_df)
    print(f"  Computed metrics for {len(metrics_df)} (eval_fold, seed) combinations")
    
    # Save metrics CSV for reference
    metrics_csv_path = CACHE_DIR / "baselines_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"  Saved metrics to {metrics_csv_path}")
    
    # Print summary
    print("\n  Summary by eval fold:")
    for eval_fold in EVAL_FOLD_ORDER:
        subset = metrics_df[metrics_df["eval_fold"] == eval_fold]
        if not subset.empty:
            valid = subset[subset["non_parsable_rate"] <= NON_PARSABLE_THRESHOLD]
            if not valid.empty:
                mean_acc = valid["correct_rate"].mean()
                mean_cot = valid["cot_penalty_rate"].mean() if valid["cot_penalty_rate"].notna().any() else float('nan')
                mean_sum = valid["summary_penalty_rate"].mean() if valid["summary_penalty_rate"].notna().any() else float('nan')
                mean_np = valid["non_parsable_rate"].mean() * 100
                cot_str = f"{mean_cot:.1%}" if not np.isnan(mean_cot) else "N/A"
                sum_str = f"{mean_sum:.1%}" if not np.isnan(mean_sum) else "N/A"
                print(f"    {eval_fold:15s}: acc={mean_acc:.1%}, cot={cot_str}, sum={sum_str}, np={mean_np:.1f}% ({len(valid)}/{len(subset)} valid)")
            else:
                print(f"    {eval_fold:15s}: no valid seeds (all above threshold)")
    
    # Generate figure
    print(f"\n{'=' * 50}")
    print("Generating figure...")
    print(f"{'=' * 50}")
    
    fig = create_baselines_figure(metrics_df)
    
    for ext in ["png", "pdf"]:
        path = FIGURES_DIR / f"baselines.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {path}")
    plt.close(fig)
    
    print(f"\n{'=' * 70}")
    print("Done!")
    print(f"{'=' * 70}\n")
    
    return 0


if __name__ == "__main__":
    exit(main())