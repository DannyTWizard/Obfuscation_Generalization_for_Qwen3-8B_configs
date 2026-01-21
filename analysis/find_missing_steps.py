#!/usr/bin/env python3
"""
Script to analyze eval results CSV files and find missing steps for each data type and eval fold.

Expected steps: 25, 200, 400, 600, ..., 3600, 3800 (25 then increments of 200)
"""

import pandas as pd
from pathlib import Path


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
    else:
        raise ValueError(f"Unknown data type: {data_type}")


def find_missing_steps(csv_path: Path, seed: int) -> dict[str, dict[str, set[int]]]:
    """
    Find missing steps for each data type and eval fold in a CSV file.

    Args:
        csv_path: Path to the CSV file
        seed: Expected training seed value

    Returns:
        Nested dictionary: {data_type: {eval_fold: set of missing steps}}
    """
    df = pd.read_csv(csv_path)

    # Filter by seed
    df = df[df["seed"] == seed]

    expected_steps = get_expected_steps()
    data_types = [
        "leave_out_score_refined2",
        "leave_out_war_refined2",
        "leave_out_sycophancy_refined2",
        "leave_out_code_refined2",
    ]

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
    # Define the CSV file to analyze (single file with all seeds)
    csv_path = (
        Path(__file__).parent.parent / "final_viz" / "metrics" / "trial_metrics.csv"
    )

    seeds = [24, 42, 50]

    expected_steps = sorted(get_expected_steps())
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

        missing_by_data = find_missing_steps(csv_path, seed)

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
        missing_by_data = find_missing_steps(csv_path, seed)
        for data_type, folds_missing in missing_by_data.items():
            for eval_fold, missing_steps in folds_missing.items():
                sorted_missing = sorted(missing_steps) if missing_steps else []
                missing_str = str(sorted_missing) if sorted_missing else "None"
                print(
                    f"{seed:<6} {data_type:<28} {eval_fold:<45} {len(missing_steps):<8} {missing_str}"
                )


if __name__ == "__main__":
    main()
