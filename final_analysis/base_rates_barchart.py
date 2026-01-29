#!/usr/bin/env python3
"""
Plot bar charts showing base rates of RH (Reward Hacking), CoT flagging,
and Summary flagging at 0 penalisation (baseline runs).

This script:
1. Downloads eval_ind baseline data from W&B (uses same logic as barcharts.py)
2. Computes metrics per (leave_out, eval_fold, seed) combination
3. Generates bar charts showing base rates

Directory Structure:
    final_analysis/
    ├── base_rates_barchart.py      # This script
    └── base_rates/
        ├── cache/
        │   └── eval_ind_baseline.parquet
        └── figures/
            ├── base_rates.png
            └── base_rates.pdf

Usage:
    # Full pipeline: download + visualize
    python base_rates_barchart.py

    # Just regenerate figures from cache
    python base_rates_barchart.py --plot-only

    # Force re-download (clears cache)
    python base_rates_barchart.py --force
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
OUTPUT_DIR = BASE_DIR / "base_rates"
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
SEEDS = [24, 33, 42, 50]

# =============================================================================
# STYLE CONFIG (inspired by barcharts.py)
# =============================================================================

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.weight": "normal",
        "axes.titleweight": "bold",
        "axes.labelweight": "normal",
        "figure.titleweight": "normal",
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    }
)

# Colors for each metric
METRIC_COLORS = {
    "rh": "#FF6B6B",  # Coral red for Reward Hacking
    "cot_flag": "#4ECDC4",  # Teal for CoT flagging
    "summary_flag": "#9B59B6",  # Purple for Summary flagging
}

# Markers for each seed
SEED_MARKERS = {
    24: "o",  # Circle
    33: "D",  # Diamond
    42: "s",  # Square
    50: "^",  # Triangle
}

# Dataset display colors (for grouping)
DATASET_COLORS = {
    "sycophancy": "#FF9999",  # Pastel red
    "war": "#99CCFF",  # Pastel blue
    "code": "#99CC99",  # Pastel green
    "score": "#FFCC99",  # Pastel orange
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
                    backoff = INITIAL_BACKOFF_SECONDS * (2**attempt)
                    error_type = "Rate limited (429)" if is_rate_limit else "Timeout"
                    print(
                        f"  {error_type}. Waiting {backoff}s before retry {attempt + 2}/{max_retries}..."
                    )
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


def get_table_from_run(
    run: wandb.apis.public.Run, table_key: str
) -> Optional[pd.DataFrame]:
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


def parse_baseline_run_name(run_name: str) -> Optional[Dict[str, Any]]:
    """
    Parse a baseline eval_ind run name to extract metadata.

    Expected format:
        run_ref_baseline_data_leave_out_{dataset}_refined2_ts_{seed}_eval_{fold}_eval_ind_step_3800

    Returns dict with: leave_out, seed, eval_fold (short names)
    """
    if "run_ref_baseline_data_" not in run_name:
        return None

    # Extract: leave_out_{dataset}_refined2_ts_{seed}_eval_{...}
    pattern = r"leave_out_(\w+)_refined2_ts_(\d+)_eval_(.+?)(?:_eval_ind|_[a-f0-9]{8}$|_step_)"
    match = re.search(pattern, run_name)

    if not match:
        return None

    leave_out = match.group(1)  # sycophancy, war, code, score
    seed = int(match.group(2))
    eval_fold_raw = match.group(
        3
    )  # e.g., "sycophancy_formatted" or "world_affecting_reward_reorg_formatte"

    # Map eval fold to short name using startswith to handle truncation
    fold_prefixes = [
        ("world_affecting_reward_reorg", "war"),
        ("revealing_score", "score"),
        ("sycophancy", "sycophancy"),
        ("code", "code"),
    ]

    eval_fold = None
    for prefix, short_name in fold_prefixes:
        if eval_fold_raw.startswith(prefix):
            eval_fold = short_name
            break

    if eval_fold is None:
        return None

    if leave_out not in ALL_DATASETS:
        return None

    return {
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


def scan_baseline_runs(verbose: bool = True) -> List[Dict[str, Any]]:
    """
    Scan W&B for baseline eval_ind runs (0 penalisation).

    Returns list of run metadata dicts.
    """
    api = wandb.Api(timeout=API_TIMEOUT)

    name_pattern = "run_ref_baseline_data_leave_out_.*_eval_ind_step_"

    filters = {
        "state": "finished",
        "displayName": {"$regex": name_pattern},
        "createdAt": {"$gte": MIN_CREATED_AT},
    }

    print(f"Scanning for baseline runs (0 penalisation)...")
    print(f"  Entity: {WANDB_ENTITY}")
    print(f"  Pattern: {name_pattern}")

    def _fetch():
        runs = api.runs(
            f"{WANDB_ENTITY}/{WANDB_PROJECT}", filters=filters, per_page=100
        )
        return list(runs)

    runs = retry_with_backoff(_fetch)
    print(f"  Found {len(runs)} raw runs")

    # Parse run names and filter valid ones
    parsed_runs = []
    for run in runs:
        parsed = parse_baseline_run_name(run.name)
        if parsed:
            parsed["id"] = run.id
            parsed["name"] = run.name
            parsed["created_at"] = _run_created_at(run)
            parsed_runs.append(parsed)

    print(f"  Parsed {len(parsed_runs)} valid runs")

    # Deduplicate by (leave_out, seed, eval_fold), keeping latest
    best_runs = {}
    for run_info in parsed_runs:
        key = (run_info["leave_out"], run_info["seed"], run_info["eval_fold"])
        if (
            key not in best_runs
            or run_info["created_at"] > best_runs[key]["created_at"]
        ):
            best_runs[key] = run_info

    result = list(best_runs.values())
    print(f"  After deduplication: {len(result)} runs")

    return result


def download_runs_to_cache(
    runs: List[Dict[str, Any]],
    cache_path: Path,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Download trial tables for runs and save to cache.
    """
    api = wandb.Api(timeout=API_TIMEOUT)

    all_rows = []

    pbar = tqdm(runs, desc="Downloading baseline runs", disable=not verbose)

    for run_info in pbar:
        pbar.set_postfix_str(
            f"{run_info['leave_out']}/{run_info['eval_fold']}/s{run_info['seed']}"
        )

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
        print("  No data downloaded")
        return pd.DataFrame()

    combined_df = pd.concat(all_rows, ignore_index=True)

    # Save to cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_parquet(cache_path, index=False)
    print(f"  Saved {len(combined_df)} rows to {cache_path}")

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
        extracted_col = next(
            (c for c in ["extracted", "extracted_answer"] if c in group_df.columns),
            None,
        )
        target_col = "target" if "target" in group_df.columns else None
        correct_col = "correct" if "correct" in group_df.columns else None

        # Monitor columns
        cot_col = (
            "overseer_cot_response"
            if "overseer_cot_response" in group_df.columns
            else (
                "overseer_response" if "overseer_response" in group_df.columns else None
            )
        )
        summary_col = (
            "overseer_summary_response"
            if "overseer_summary_response" in group_df.columns
            else None
        )

        # Compute extractable (has valid extracted answer)
        if extracted_col:
            extractable_mask = group_df[extracted_col].notna() & (
                group_df[extracted_col] != "No answer tags found"
            )
        else:
            extractable_mask = pd.Series([True] * total, index=group_df.index)

        total_extractable = extractable_mask.sum()

        if total_extractable == 0:
            results.append(
                {
                    "leave_out": leave_out,
                    "seed": seed,
                    "eval_fold": eval_fold,
                    "total": total,
                    "total_extractable": 0,
                    "rh_rate": None,
                    "cot_flag_rate": None,
                    "summary_flag_rate": None,
                    "non_parsable_rate": 1.0,
                }
            )
            continue

        extractable_df = group_df[extractable_mask]

        # RH rate (reward hacking = correct answers, i.e. model gave the "hacked" answer)
        if correct_col and correct_col in extractable_df.columns:
            num_rh = extractable_df[correct_col].sum()
        elif target_col and extracted_col:
            num_rh = (
                extractable_df[extracted_col].str.upper()
                == extractable_df[target_col].str.upper()
            ).sum()
        else:
            num_rh = 0

        rh_rate = num_rh / total_extractable

        # CoT flag rate
        cot_flag_rate = None
        if cot_col and cot_col in extractable_df.columns:
            try:
                flagged = sum(
                    1
                    for val in extractable_df[cot_col]
                    if val is not None and pd.notna(val) and float(val) > 0
                )
                cot_flag_rate = flagged / total_extractable
            except (ValueError, TypeError):
                pass

        # Summary flag rate
        summary_flag_rate = None
        if summary_col and summary_col in extractable_df.columns:
            try:
                flagged = sum(
                    1
                    for val in extractable_df[summary_col]
                    if val is not None and pd.notna(val) and float(val) > 0
                )
                summary_flag_rate = flagged / total_extractable
            except (ValueError, TypeError):
                pass

        # Non-parsable rate
        non_parsable_rate = (total - total_extractable) / total

        results.append(
            {
                "leave_out": leave_out,
                "seed": seed,
                "eval_fold": eval_fold,
                "total": total,
                "total_extractable": total_extractable,
                "rh_rate": rh_rate,
                "cot_flag_rate": cot_flag_rate,
                "summary_flag_rate": summary_flag_rate,
                "non_parsable_rate": non_parsable_rate,
            }
        )

    return pd.DataFrame(results)


# =============================================================================
# VISUALIZATION
# =============================================================================


def get_in_distribution_datasets(leave_out: str) -> List[str]:
    """Get the 3 datasets that are in-distribution (not left out)."""
    return [d for d in ALL_DATASETS if d != leave_out]


def create_base_rates_figure(metrics_df: pd.DataFrame) -> plt.Figure:
    """
    Create bar chart showing base rates (RH, CoT flagging, Summary flagging)
    at 0 penalisation.

    Layout: 4 subplots (one per leave_out dataset)
    Each subplot shows 3 groups (one per in-distribution eval dataset)
    Each group has 3 bars (RH, CoT flag, Summary flag)
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharey=True)

    # Layout parameters
    bar_width = 0.25
    group_positions = [1, 2, 3]  # Group centers for 3 in-distribution datasets

    # Offsets for the 3 metrics within each group
    metric_offsets = {
        "rh": -bar_width,
        "cot_flag": 0,
        "summary_flag": bar_width,
    }

    seed_jitter = {24: -0.06, 33: -0.02, 42: 0.02, 50: 0.06}

    for col_idx, leave_out in enumerate(ALL_DATASETS):
        ax = axes[col_idx]
        in_dist = get_in_distribution_datasets(leave_out)

        for group_idx, eval_ds in enumerate(in_dist):
            group_center = group_positions[group_idx]

            for metric in ["rh", "cot_flag", "summary_flag"]:
                bar_x = group_center + metric_offsets[metric]

                mask = (metrics_df["leave_out"] == leave_out) & (
                    metrics_df["eval_fold"] == eval_ds
                )
                subset = metrics_df[mask]

                if subset.empty:
                    continue

                col_name = f"{metric}_rate"
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

                color = METRIC_COLORS[metric]

                if valid_values:
                    mean_val = np.mean(valid_values)
                    stderr_val = (
                        np.std(valid_values, ddof=1) / np.sqrt(len(valid_values))
                        if len(valid_values) > 1
                        else 0
                    )

                    ax.bar(
                        bar_x,
                        mean_val,
                        width=bar_width,
                        color=color,
                        edgecolor="black",
                        linewidth=0.5,
                        alpha=0.8,
                        yerr=stderr_val,
                        capsize=3,
                        error_kw={
                            "elinewidth": 1.5,
                            "capthick": 1.5,
                            "ecolor": "black",
                        },
                    )

                # Plot individual seed points
                for seed, val, is_valid in all_seed_data:
                    marker = SEED_MARKERS.get(seed, "o")
                    jitter = seed_jitter.get(seed, 0.0)

                    if is_valid:
                        ax.scatter(
                            bar_x + jitter,
                            val,
                            color="black",
                            marker=marker,
                            s=35,
                            zorder=5,
                            alpha=0.8,
                        )
                    else:
                        ax.scatter(
                            bar_x + jitter,
                            val,
                            facecolors="none",
                            edgecolors="black",
                            marker=marker,
                            s=35,
                            zorder=5,
                            alpha=0.8,
                            linewidths=1.0,
                        )

        # Styling
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0.3, 3.7)

        # X-axis labels for eval datasets
        ax.set_xticks(group_positions)
        ax.set_xticklabels([d.capitalize() for d in in_dist], fontsize=12)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if col_idx > 0:
            ax.spines["left"].set_visible(False)
            ax.tick_params(left=False)

        ax.set_title(
            f"Leave out {leave_out.capitalize()}", fontsize=16, fontweight="bold"
        )

        if col_idx == 0:
            ax.set_ylabel("Rate", fontsize=16)

    # Build legend
    metric_handles = [
        mpatches.Patch(
            facecolor=METRIC_COLORS["rh"],
            edgecolor="black",
            label="Reward Hacking (RH)",
        ),
        mpatches.Patch(
            facecolor=METRIC_COLORS["cot_flag"], edgecolor="black", label="CoT Flagging"
        ),
        mpatches.Patch(
            facecolor=METRIC_COLORS["summary_flag"],
            edgecolor="black",
            label="Summary Flagging",
        ),
    ]

    # Seed markers legend
    seed_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=SEED_MARKERS[s],
            color="black",
            linestyle="None",
            markersize=8,
            label=f"Seed {s}",
        )
        for s in sorted(SEED_MARKERS.keys())
    ]

    # Combine legends
    all_handles = metric_handles + seed_handles

    fig.legend(
        all_handles,
        [h.get_label() for h in all_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=7,
        frameon=False,
        fontsize=14,
        handlelength=2,
        handleheight=1.5,
    )

    fig.suptitle("Base Rates at 0 Penalisation", fontsize=20, fontweight="bold")
    fig.tight_layout(rect=[0, 0.10, 1, 0.95])

    return fig


def create_base_rates_figure_simple(metrics_df: pd.DataFrame) -> plt.Figure:
    """
    Create a simpler bar chart showing aggregate base rates across all leave-out conditions.

    One bar per metric (RH, CoT Flag, Summary Flag), averaged across all conditions.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Compute average rates across all valid data points
    metrics_summary = {}

    for metric in ["rh", "cot_flag", "summary_flag"]:
        col_name = f"{metric}_rate"
        if col_name not in metrics_df.columns:
            continue

        # Filter valid data points (low non-parsable rate)
        valid_mask = metrics_df["non_parsable_rate"] <= NON_PARSABLE_THRESHOLD
        valid_df = metrics_df[valid_mask]

        values = valid_df[col_name].dropna()

        if len(values) > 0:
            metrics_summary[metric] = {
                "mean": values.mean(),
                "stderr": (
                    values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0
                ),
                "values": values.tolist(),
            }

    # Plot bars
    bar_positions = [0, 1, 2]
    metric_labels = ["Reward Hacking\n(RH)", "CoT Flagging", "Summary Flagging"]

    for i, (metric, label) in enumerate(
        zip(["rh", "cot_flag", "summary_flag"], metric_labels)
    ):
        if metric not in metrics_summary:
            continue

        data = metrics_summary[metric]

        ax.bar(
            bar_positions[i],
            data["mean"],
            width=0.6,
            color=METRIC_COLORS[metric],
            edgecolor="black",
            linewidth=1.5,
            alpha=0.85,
            yerr=data["stderr"],
            capsize=8,
            error_kw={"elinewidth": 2, "capthick": 2, "ecolor": "black"},
        )

        # Add individual data points as scatter
        jitter = np.random.uniform(-0.15, 0.15, len(data["values"]))
        ax.scatter(
            bar_positions[i] + jitter,
            data["values"],
            color="black",
            alpha=0.5,
            s=25,
            zorder=5,
        )

    # Styling
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-0.5, 2.5)
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(metric_labels, fontsize=14)
    ax.set_ylabel("Rate", fontsize=16)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add horizontal reference lines
    ax.axhline(y=0.5, color="#CCCCCC", linestyle="--", linewidth=1, zorder=0, alpha=0.7)

    ax.set_title(
        "Base Rates at 0 Penalisation (All Conditions)", fontsize=18, fontweight="bold"
    )
    fig.tight_layout()

    return fig


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Plot base rates (RH, CoT flagging, Summary flagging) at 0 penalisation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--plot-only", action="store_true", help="Only generate figures from cache"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force re-download (clears cache)"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress progress output"
    )

    args = parser.parse_args()
    verbose = not args.quiet

    # Ensure directories exist
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    cache_path = CACHE_DIR / "eval_ind_baseline.parquet"

    # Clear cache if forced
    if args.force:
        print("Force mode: clearing cache...")
        if cache_path.exists():
            cache_path.unlink()

    print(f"\n{'=' * 70}")
    print("BASE RATES BAR CHART GENERATOR")
    print(f"{'=' * 70}")
    print(f"W&B Entity:  {WANDB_ENTITY}")
    print(f"W&B Project: {WANDB_PROJECT}")
    print(f"Min Date:    {MIN_CREATED_AT}")
    print(f"Cache:       {cache_path}")
    print(f"Figures Dir: {FIGURES_DIR}")
    print(f"{'=' * 70}\n")

    # Download or load from cache
    if args.plot_only:
        print("Plot-only mode: loading from cache...")
        cache_df = load_cache(cache_path)
        if cache_df.empty:
            print("No cache found. Run without --plot-only first.")
            return 1
    else:
        if cache_path.exists() and not args.force:
            print("Cache exists, loading...")
            cache_df = load_cache(cache_path)
        else:
            runs = scan_baseline_runs(verbose=verbose)
            if not runs:
                print("No runs found.")
                return 1
            cache_df = download_runs_to_cache(runs, cache_path, verbose=verbose)

    if cache_df.empty:
        print("No data available. Cannot generate figures.")
        return 1

    # Compute metrics
    print(f"\nComputing metrics...")
    metrics_df = compute_metrics(cache_df)
    print(f"  Computed metrics for {len(metrics_df)} combinations")

    # Save metrics CSV for reference
    metrics_csv_path = CACHE_DIR / "base_rates_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"  Saved metrics to {metrics_csv_path}")

    # Print summary
    print(f"\n{'=' * 50}")
    print("SUMMARY (across all valid data points)")
    print(f"{'=' * 50}")

    valid_mask = metrics_df["non_parsable_rate"] <= NON_PARSABLE_THRESHOLD
    valid_df = metrics_df[valid_mask]

    for metric, label in [
        ("rh_rate", "Reward Hacking"),
        ("cot_flag_rate", "CoT Flagging"),
        ("summary_flag_rate", "Summary Flagging"),
    ]:
        if metric in valid_df.columns:
            values = valid_df[metric].dropna()
            if len(values) > 0:
                print(f"  {label}:")
                print(f"    Mean:   {values.mean():.3f}")
                print(f"    Std:    {values.std():.3f}")
                print(f"    Min:    {values.min():.3f}")
                print(f"    Max:    {values.max():.3f}")
                print(f"    N:      {len(values)}")

    # Generate figures
    print(f"\n{'=' * 50}")
    print("Generating figures...")
    print(f"{'=' * 50}")

    # Main figure with all leave-out conditions
    fig = create_base_rates_figure(metrics_df)
    for ext in ["png", "pdf"]:
        path = FIGURES_DIR / f"base_rates.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {path}")
    plt.close(fig)

    # Simple aggregate figure
    fig = create_base_rates_figure_simple(metrics_df)
    for ext in ["png", "pdf"]:
        path = FIGURES_DIR / f"base_rates_simple.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {path}")
    plt.close(fig)

    print(f"\n{'=' * 70}")
    print("Done!")
    print(f"{'=' * 70}\n")

    return 0


if __name__ == "__main__":
    exit(main())
