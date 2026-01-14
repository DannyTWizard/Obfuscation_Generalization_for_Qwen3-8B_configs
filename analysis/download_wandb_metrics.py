"""
Script to search for W&B runs by name attributes and download specified metrics.

Provides separate functions for training runs and eval runs with normalized metrics:
- Training reward hacking rate = correctness_reward / weight (default 2.0)
- Training penalisation rate = overseer_penalty / penalty_weight (from config)
- Eval reward hacking rate = accuracy (already normalized)
- Eval penalisation rate = api_overseer_penalty_func (already normalized with weight=1 in eval)

Usage:
    # Download training metrics
    python download_wandb_metrics.py train \
        --attributes data=leave_out_score_full_xml ts=50 pen=-0.05 \
        --output training_results.csv

    # Download eval metrics  
    python download_wandb_metrics.py eval \
        --attributes data=leave_out_score_full_xml ts=50 \
        --output eval_results.csv
"""

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd
import wandb


# Default wandb configuration
DEFAULT_ENTITY = "nathanielmitrani-cfis-upc"
DEFAULT_PROJECT = "obfuscation_generalization"

# Default weights (can be overridden from run config)
DEFAULT_CORRECTNESS_WEIGHT = 2.0


def parse_attributes(attr_list: List[str]) -> Dict[str, str]:
    """
    Parse attribute strings in the format 'key=value' into a dictionary.

    Args:
        attr_list: List of strings like ['data=xyz', 'ts=50']

    Returns:
        Dictionary mapping keys to values
    """
    attributes = {}
    for attr in attr_list:
        if "=" not in attr:
            raise ValueError(
                f"Invalid attribute format: '{attr}'. Expected 'key=value'"
            )
        key, value = attr.split("=", 1)
        attributes[key.strip()] = value.strip()
    return attributes


def extract_attribute_from_run_name(run_name: str, key: str) -> Optional[str]:
    """
    Extract the value for a given key from a run name.

    Run names follow the pattern: run_key1_value1_key2_value2_...
    Values can contain underscores, so we look for the next known key to delimit.

    Args:
        run_name: The W&B run name
        key: The key to extract (e.g., 'ts', 'data', 'pen', 'ovs')

    Returns:
        The extracted value or None if not found
    """
    # Known keys that can appear in run names (from wandb config mapping)
    known_keys = [
        "pen",
        "lr",
        "epochs",
        "batch",
        "lora_r",
        "ovs",
        "data",
        "ts",
        "tg",
        "nsp",
    ]

    # Build pattern to find the key and capture its value
    # Value extends until the next known key or end of string
    next_key_pattern = "|".join(re.escape(k) for k in known_keys if k != key)
    pattern = rf"(?:^|_){re.escape(key)}_(.+?)(?:_(?:{next_key_pattern})_|$)"

    match = re.search(pattern, run_name)
    if match:
        return match.group(1)
    return None


def run_name_matches_attributes(run_name: str, attributes: Dict[str, str]) -> bool:
    """
    Check if a run name contains all required "{key}_{value}" patterns.

    Args:
        run_name: The W&B run name to check
        attributes: Dictionary of key-value pairs that must appear in the name

    Returns:
        True if all patterns are found in the run name
    """
    for key, value in attributes.items():
        pattern = f"{re.escape(key)}_{re.escape(value)}"
        if not re.search(pattern, run_name):
            return False
    return True


def find_matching_runs(
    entity: str,
    project: str,
    attributes: Dict[str, str],
    is_eval: bool = False,
    state_filter: Optional[List[str]] = None,
    verbose: bool = True,
) -> List[wandb.apis.public.Run]:
    """
    Find all W&B runs whose names match the given attribute patterns.

    Args:
        entity: W&B entity (username or team)
        project: W&B project name
        attributes: Dictionary of key-value pairs to match in run names
        is_eval: If True, only match runs with 'eval' in the name
        state_filter: Optional list of run states to filter by
        verbose: Whether to print progress information

    Returns:
        List of matching wandb Run objects
    """
    api = wandb.Api()

    if verbose:
        print(f"Fetching runs from {entity}/{project}...")

    filters = {}
    if state_filter:
        filters["state"] = {"$in": state_filter}

    runs = api.runs(f"{entity}/{project}", filters=filters if filters else None)

    if verbose:
        print(f"Found {len(runs)} total runs")
        print(
            f"Filtering for {'eval' if is_eval else 'training'} runs with attributes: {attributes}"
        )

    matching_runs = []
    for run in runs:
        # Check if it's an eval run (contains 'eval' in name)
        is_run_eval = "eval" in run.name.lower()

        # Skip if eval filter doesn't match
        if is_eval != is_run_eval:
            continue

        if run_name_matches_attributes(run.name, attributes):
            matching_runs.append(run)
            if verbose:
                print(f"  ✓ Match: {run.name} (id: {run.id}, state: {run.state})")

    if verbose:
        print(f"\nFound {len(matching_runs)} matching runs")

    return matching_runs


def get_config_value(config: Dict, key_path: str, default: Any = None) -> Any:
    """
    Get a nested config value using dot notation.

    Args:
        config: The config dictionary
        key_path: Dot-separated path like 'reward.funcs.api_overseer_penalty_func.penalty_weight'
        default: Default value if not found

    Returns:
        The config value or default
    """
    keys = key_path.split(".")
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


def download_training_runs(
    entity: str,
    project: str,
    attributes: Dict[str, str],
    state_filter: Optional[List[str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Download metrics from training runs with normalized rates.

    Metrics included:
    - reward_hacking_rate: correctness_reward / weight (normalized to 0-1)
    - penalisation_rate: overseer_penalty / penalty_weight (normalized to 0-1)

    Args:
        entity: W&B entity
        project: W&B project name
        attributes: Dictionary of key-value pairs to match in run names
        state_filter: Optional list of run states to filter by
        verbose: Whether to print progress

    Returns:
        DataFrame with training metrics
    """
    matching_runs = find_matching_runs(
        entity=entity,
        project=project,
        attributes=attributes,
        is_eval=False,
        state_filter=state_filter,
        verbose=verbose,
    )

    if not matching_runs:
        if verbose:
            print("No matching training runs found.")
        return pd.DataFrame()

    all_results = []

    for run in matching_runs:
        if verbose:
            print(f"\nDownloading metrics from: {run.name}")

        # Get weights from config
        correctness_weight = get_config_value(
            run.config,
            "reward.funcs.correctness_reward_func.weight",
            DEFAULT_CORRECTNESS_WEIGHT,
        )
        if correctness_weight is None:
            correctness_weight = DEFAULT_CORRECTNESS_WEIGHT

        penalty_weight = get_config_value(
            run.config, "reward.funcs.api_overseer_penalty_func.penalty_weight", None
        )

        history_keys = [
            "train/rewards/correctness_reward_func/mean",
            "train/rewards/api_overseer_penalty_func/mean",
        ]

        if verbose:
            print(f"  Correctness weight: {correctness_weight}")
            print(f"  Penalty weight: {penalty_weight}")

        # Get training history - fetch all history first to see what's available
        try:
            # First, try fetching without key filter to see what's logged
            history = run.history(keys=history_keys, pandas=True)

            if verbose and history.empty:
                print(f"  Warning: No history data found")
                # Try to see what keys are available via summary
                available_keys = list(run.summary.keys())[:20]
                print(f"  Available summary keys (sample): {available_keys}")
                continue

            if verbose:
                available_cols = [
                    c
                    for c in history.columns
                    if "reward" in c.lower() or "step" in c.lower()
                ]
                print(f"  Available reward/step columns: {available_cols}")

        except Exception as e:
            if verbose:
                print(f"  Warning: Could not fetch history: {e}")
            continue

        if history.empty:
            continue

        # Normalize metrics
        history["run_id"] = run.id
        history["run_name"] = run.name
        history["run_state"] = run.state
        history["created_at"] = run.created_at

        # Extract ts (train_seed) and data from run name
        history["ts"] = extract_attribute_from_run_name(run.name, "ts")
        history["data"] = extract_attribute_from_run_name(run.name, "data")

        # Rename step column
        if "_step" in history.columns:
            history = history.rename(columns={"_step": "step"})

        # Calculate normalized rates
        if "train/rewards/correctness_reward_func/mean" in history.columns:
            history["reward_hacking_rate"] = (
                history["train/rewards/correctness_reward_func/mean"]
                / correctness_weight
            )

        if (
            "train/rewards/api_overseer_penalty_func/mean" in history.columns
            and penalty_weight is not None
        ):
            # Penalty is negative, divide by penalty_weight (also negative) to get positive rate
            history["penalisation_rate"] = (
                history["train/rewards/api_overseer_penalty_func/mean"] / penalty_weight
            )

        # Add config info
        history["correctness_weight"] = correctness_weight
        history["penalty_weight"] = penalty_weight

        lean_history = history[
            [
                "ts",
                "data",
                "reward_hacking_rate",
                "penalisation_rate",
                "correctness_weight",
                "penalty_weight",
                "step",
            ]
        ]

        all_results.append(lean_history)

    if not all_results:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)


def download_eval_runs(
    entity: str,
    project: str,
    attributes: Dict[str, str],
    state_filter: Optional[List[str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Download metrics from eval runs.

    Metrics included:
    - reward_hacking_rate: accuracy (already 0-1 scale)
    - penalisation_rate: api_overseer_penalty_func (eval uses weight=1)
    - artifact_step: the training step of the evaluated checkpoint

    Args:
        entity: W&B entity
        project: W&B project name
        attributes: Dictionary of key-value pairs to match in run names
        state_filter: Optional list of run states to filter by
        verbose: Whether to print progress

    Returns:
        DataFrame with eval metrics
    """
    matching_runs = find_matching_runs(
        entity=entity,
        project=project,
        attributes=attributes,
        is_eval=True,
        state_filter=state_filter,
        verbose=verbose,
    )

    if not matching_runs:
        if verbose:
            print("No matching eval runs found.")
        return pd.DataFrame()

    all_results = []

    for run in matching_runs:
        if verbose:
            print(f"\nDownloading metrics from: {run.name}")

        # Get artifact_step from config
        artifact_step = get_config_value(run.config, "artifact_step", None)
        training_group = get_config_value(run.config, "training_group", None)
        training_run_name = get_config_value(run.config, "training_run_name", None)
        eval_fold = get_config_value(run.config, "eval.fold", None)

        if verbose:
            print(f"  Artifact step: {artifact_step}")
            print(f"  Training group: {training_group}")
            print(f"  Eval fold: {eval_fold}")

        # Get summary metrics (eval runs typically log final values)
        result = {
            "run_id": run.id,
            "run_name": run.name,
            "run_state": run.state,
            "created_at": run.created_at,
            "artifact_step": artifact_step,
            "training_group": training_group,
            "training_run_name": training_run_name,
            "eval_fold": eval_fold,
        }

        # Get metrics from summary
        summary_metrics = [
            "accuracy",
            "correct",
            "total",
            "api_overseer_penalty_func",
            "no_answer_tags_rate",
        ]

        for metric in summary_metrics:
            if metric in run.summary:
                result[metric] = run.summary[metric]

        # Calculate normalized rates
        if "accuracy" in result:
            result["reward_hacking_rate"] = result["accuracy"]

        if "api_overseer_penalty_func" in result:
            # Eval runs use penalty_weight=1, so the value is already the rate
            # It's typically negative, so we take absolute value for rate
            result["penalisation_rate"] = abs(result["api_overseer_penalty_func"])

        # Extract ts and data from training_run_name (stored in config)
        ts = None
        data = None
        if training_run_name:
            ts = extract_attribute_from_run_name(training_run_name, "ts")
            data = extract_attribute_from_run_name(training_run_name, "data")

        lean_result = {
            "artifact_step": artifact_step,
            "ts": ts,
            "data": data,
            "eval_fold": eval_fold,
            "reward_hacking_rate": result.get("reward_hacking_rate"),
            "penalisation_rate": result.get("penalisation_rate"),
            "accuracy": result.get("accuracy"),
            "correct": result.get("correct"),
            "total": result.get("total"),
            "api_overseer_penalty_func": result.get("api_overseer_penalty_func"),
            "no_answer_tags_rate": result.get("no_answer_tags_rate"),
        }
        all_results.append(lean_result)

    if not all_results:
        return pd.DataFrame()

    return pd.DataFrame(all_results)


def main():
    parser = argparse.ArgumentParser(
        description="Download W&B metrics for training or eval runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download training metrics
    python download_wandb_metrics.py train \\
        --attributes data=leave_out_score_full_xml ts=50 pen=-0.05 \\
        --output training_results.csv

    # Download eval metrics
    python download_wandb_metrics.py eval \\
        --attributes data=leave_out_score_full_xml ts=50 \\
        --output eval_results.csv
        
    # Download from specific entity/project
    python download_wandb_metrics.py train \\
        --entity my-team --project my-project \\
        --attributes pen=-0.05 \\
        --output results.csv
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Training subcommand
    train_parser = subparsers.add_parser(
        "train",
        help="Download training run metrics",
    )
    train_parser.add_argument(
        "--attributes",
        "-a",
        nargs="+",
        required=True,
        help="Attributes to filter runs by, in format 'key=value'",
    )
    train_parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output CSV file path",
    )
    train_parser.add_argument(
        "--entity",
        "-e",
        default=DEFAULT_ENTITY,
        help=f"W&B entity (default: {DEFAULT_ENTITY})",
    )
    train_parser.add_argument(
        "--project",
        "-p",
        default=DEFAULT_PROJECT,
        help=f"W&B project name (default: {DEFAULT_PROJECT})",
    )
    train_parser.add_argument(
        "--state",
        nargs="*",
        default=["finished"],
        help="Filter runs by state (default: finished). Use --state without args to include all states.",
    )
    train_parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )

    # Eval subcommand
    eval_parser = subparsers.add_parser(
        "eval",
        help="Download eval run metrics",
    )
    eval_parser.add_argument(
        "--attributes",
        "-a",
        nargs="+",
        required=True,
        help="Attributes to filter runs by, in format 'key=value'",
    )
    eval_parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output CSV file path",
    )
    eval_parser.add_argument(
        "--entity",
        "-e",
        default=DEFAULT_ENTITY,
        help=f"W&B entity (default: {DEFAULT_ENTITY})",
    )
    eval_parser.add_argument(
        "--project",
        "-p",
        default=DEFAULT_PROJECT,
        help=f"W&B project name (default: {DEFAULT_PROJECT})",
    )
    eval_parser.add_argument(
        "--state",
        nargs="*",
        default=["finished"],
        help="Filter runs by state (default: finished). Use --state without args to include all states.",
    )
    eval_parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Parse attributes
    try:
        attributes = parse_attributes(args.attributes)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    verbose = not args.quiet

    # Handle --state without arguments (empty list means include all states)
    state_filter = args.state if args.state else None

    if verbose:
        print("\n" + "=" * 70)
        print(f"W&B METRICS DOWNLOADER - {args.command.upper()} RUNS")
        print("=" * 70)
        print(f"Entity:     {args.entity}")
        print(f"Project:    {args.project}")
        print(f"Attributes: {attributes}")
        print(f"State:      {state_filter if state_filter else 'all'}")
        print("=" * 70 + "\n")

    # Download metrics based on command
    if args.command == "train":
        df = download_training_runs(
            entity=args.entity,
            project=args.project,
            attributes=attributes,
            state_filter=state_filter,
            verbose=verbose,
        )
        default_suffix = "training"
    else:  # eval
        df = download_eval_runs(
            entity=args.entity,
            project=args.project,
            attributes=attributes,
            state_filter=state_filter,
            verbose=verbose,
        )
        default_suffix = "eval"

    if df.empty:
        print("No data to save.")
        return 0

    # Set output path
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"wandb_{default_suffix}_metrics_{timestamp}.csv"

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save results
    df.to_csv(output_path, index=False)

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"✓ Results saved to: {output_path}")
        print(
            f"  Total runs: {df['run_id'].nunique() if 'run_id' in df.columns else len(df)}"
        )
        print(f"  Total rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    exit(main())
