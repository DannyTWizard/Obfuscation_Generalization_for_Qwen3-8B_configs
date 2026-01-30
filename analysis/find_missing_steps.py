#!/usr/bin/env python3
"""
Script to analyze eval results CSV files and find missing steps for each data type and eval fold.

Expected steps: 25, 200, 400, 600, ..., 3600, 3800 (25 then increments of 200)

Usage:
    python find_missing_steps.py                    # Default: loo mode
    python find_missing_steps.py --mode loo-sum    # loo-sum mode
    python find_missing_steps.py --mode o2m        # o2m mode
"""

import argparse
import pandas as pd
from pathlib import Path


# Mode configurations matching download_wandb_trial_data.py
MODE_CONFIGS = {
    "loo": {
        "data_types": [
            "leave_out_score_refined2",
            "leave_out_war_refined2",
            "leave_out_sycophancy_refined2",
            "leave_out_code_refined2",
        ],
        "csv_suffix": "",
    },
    "loo-sum": {
        "data_types": [
            "leave_out_score_refined2",
            "leave_out_war_refined2",
            "leave_out_sycophancy_refined2",
            "leave_out_code_refined2",
        ],
        "csv_suffix": "_loo_sum",
    },
    "o2m": {
        "data_types": [
            "only_score_refined2",
        ],
        "csv_suffix": "_o2m",
    },
}


def get_expected_steps() -> set[int]:
    """Return the set of expected steps: 25, 200, 400, 600, ..., 3600, 3800"""
    return set([25] + list(range(200, 4000, 200)))


def get_expected_eval_folds(data_type: str) -> list[str]:
    """
    Return the expected eval folds for a given data type.

    Args:
        data_type: e.g. 'leave_out_score_refined2', 'leave_out_war_refined2', etc.

    Returns:
        List of expected eval fold names
    """
    common_folds = [
        "eval_medical_sycophancy_fact_formatted",
        "eval_power_positions_sycophancy_formatted",
    ]

    if data_type == "leave_out_score_refined2":
        return common_folds + ["eval_revealing_score_formatted"]
    elif data_type == "leave_out_war_refined2":
        return common_folds + ["eval_world_affecting_reward_reorg_formatted"]
    elif data_type == "leave_out_sycophancy_refined2":
        return common_folds + ["eval_sycophancy_formatted"]
    elif data_type == "leave_out_code_refined2":
        return common_folds + ["eval_code_formatted"]
    elif data_type == "only_score_refined2":
        # For o2m mode, return all eval folds
        return common_folds + [
            "eval_revealing_score_formatted",
            "eval_world_affecting_reward_reorg_formatted",
            "eval_sycophancy_formatted",
            "eval_code_formatted",
        ]
    else:
        raise ValueError(f"Unknown data type: {data_type}")


def find_missing_steps(
    csv_path: Path, seed: int, mode: str = "loo"
) -> dict[str, dict[str, set[int]]]:
    """
    Find missing steps for each data type and eval fold in a CSV file.

    Args:
        csv_path: Path to the CSV file
        seed: Expected training seed value
        mode: Mode to use (loo, loo-sum, o2m)

    Returns:
        Nested dictionary: {data_type: {eval_fold: set of missing steps}}
    """
    df = pd.read_csv(csv_path)

    # Filter by seed
    df = df[df["seed"] == seed]

    expected_steps = get_expected_steps()
    data_types = MODE_CONFIGS[mode]["data_types"]

    missing_by_data = {}

    for data_type in data_types:
        missing_by_data[data_type] = {}
        expected_folds = get_expected_eval_folds(data_type)

        for eval_fold in expected_folds:
            # Filter rows for this data type and eval fold
            mask = (df["data"] == data_type) & (df["eval_fold"] == eval_fold)
            fold_df = df[mask]

            # Get unique steps present for this combination
            present_steps = set(fold_df["step"].unique())

            # Find missing steps
            missing_steps = expected_steps - present_steps
            missing_by_data[data_type][eval_fold] = missing_steps

    return missing_by_data


def main():
    parser = argparse.ArgumentParser(
        description="Find missing steps in trial metrics CSV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
    loo      - leave_out_* datasets (default), uses trial_metrics.csv
    loo-sum  - leave_out_* datasets, uses trial_metrics_loo_sum.csv
    o2m      - only_score_refined2, uses trial_metrics_o2m.csv
        """,
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=list(MODE_CONFIGS.keys()),
        default="loo",
        help="Mode: loo (default), loo-sum, or o2m",
    )
    args = parser.parse_args()

    mode = args.mode
    mode_config = MODE_CONFIGS[mode]

    # Define the CSV file to analyze based on mode
    csv_suffix = mode_config["csv_suffix"]
    csv_path = (
        Path(__file__).parent.parent
        / "final_viz"
        / "metrics"
        / f"trial_metrics{csv_suffix}.csv"
    )

    seeds = [24, 42, 50, 33]

    expected_steps = sorted(get_expected_steps())
    print(f"Mode: {mode}")
    print(f"CSV file: {csv_path}")
    print(f"Data types: {mode_config['data_types']}")
    print(f"Expected steps: {expected_steps}")
    print(f"Total expected steps: {len(expected_steps)}")
    print("=" * 80)

    if not csv_path.exists():
        print(f"ERROR: File not found: {csv_path}")
        return

    for seed in seeds:
        print(f"\n{'=' * 80}")
        print(f"Seed: {seed}")
        print("=" * 80)

        missing_by_data = find_missing_steps(csv_path, seed, mode)

        for data_type, folds_missing in missing_by_data.items():
            print(f"\n  Data: {data_type}")

            for eval_fold, missing_steps in folds_missing.items():
                if missing_steps:
                    sorted_missing = sorted(missing_steps)
                    print(f"    {eval_fold}:")
                    print(
                        f"      Missing ({len(missing_steps)}/{len(expected_steps)}): {sorted_missing}"
                    )
                else:
                    print(f"    {eval_fold}: All steps present! ✓")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"\n{'Seed':<6} {'Data Type':<28} {'Eval Fold':<45} {'Missing':<8} {'Steps'}")
    print("-" * 130)

    for seed in seeds:
        missing_by_data = find_missing_steps(csv_path, seed, mode)
        for data_type, folds_missing in missing_by_data.items():
            for eval_fold, missing_steps in folds_missing.items():
                sorted_missing = sorted(missing_steps) if missing_steps else []
                missing_str = str(sorted_missing) if sorted_missing else "None"
                print(
                    f"{seed:<6} {data_type:<28} {eval_fold:<45} {len(missing_steps):<8} {missing_str}"
                )


if __name__ == "__main__":
    main()
