#!/usr/bin/env python3
"""
Split datasets into train_superset and eval_set.

For revealing_score.jsonl and sycophancy_opinion_political.jsonl:
    - Randomly select 500 examples for eval_set
    - Put the rest in train_superset

For all other datasets (except mmlu_subset_1000.jsonl):
    - Copy exactly to both train_superset and eval_set

mmlu_subset_1000.jsonl is skipped entirely.

Usage:
    python -m data_processing.split_train_eval --seed 42
"""

import argparse
import json
import random
import shutil
from pathlib import Path


# Datasets to split (500 eval, rest train)
SPLIT_DATASETS = {
    "revealing_score",
    "sycophancy_opinion_political",
}

SKIP_DATASETS = {
#     "mmlu_subset_1000",
}

EVAL_COUNT = 500


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dictionaries."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict], path: Path) -> None:
    """Write a list of dictionaries to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Split datasets into train_superset and eval_set"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--original_path",
        type=Path,
        default=Path("datasets/reward_hack/original"),
        help="Path to original datasets (default: datasets/reward_hack/original)"
    )
    parser.add_argument(
        "--train_path",
        type=Path,
        default=Path("datasets/reward_hack/train_superset"),
        help="Output path for training data (default: datasets/reward_hack/train_superset)"
    )
    parser.add_argument(
        "--eval_path",
        type=Path,
        default=Path("datasets/reward_hack/eval_set"),
        help="Output path for eval data (default: datasets/reward_hack/eval_set)"
    )
    args = parser.parse_args()

    random.seed(args.seed)
    print(f"Using random seed: {args.seed}")

    # Create output directories
    args.train_path.mkdir(parents=True, exist_ok=True)
    args.eval_path.mkdir(parents=True, exist_ok=True)

    # Iterate over all JSONL files in original/
    for jsonl_file in sorted(args.original_path.glob("*.jsonl")):
        dataset_name = jsonl_file.stem
        
        # Skip excluded datasets
        if dataset_name in SKIP_DATASETS:
            print(f"Skipping {dataset_name} (excluded)")
            continue
        
        train_output = args.train_path / jsonl_file.name
        eval_output = args.eval_path / jsonl_file.name
        
        if dataset_name in SPLIT_DATASETS:
            # Split: 500 eval, rest train
            rows = load_jsonl(jsonl_file)
            
            if len(rows) < EVAL_COUNT:
                raise ValueError(
                    f"Dataset {dataset_name} has {len(rows)} rows, "
                    f"but {EVAL_COUNT} are needed for eval"
                )
            
            # Shuffle and split
            shuffled_indices = list(range(len(rows)))
            random.shuffle(shuffled_indices)
            
            eval_indices = set(shuffled_indices[:EVAL_COUNT])
            
            eval_rows = [rows[i] for i in range(len(rows)) if i in eval_indices]
            train_rows = [rows[i] for i in range(len(rows)) if i not in eval_indices]
            
            write_jsonl(train_rows, train_output)
            write_jsonl(eval_rows, eval_output)
            
            print(f"Split {dataset_name}: {len(train_rows)} train, {len(eval_rows)} eval")
        
        else:
            # Copy to both train and eval
            shutil.copy(jsonl_file, train_output)
            shutil.copy(jsonl_file, eval_output)
            
            row_count = sum(1 for _ in open(jsonl_file))
            print(f"Copied {dataset_name}: {row_count} rows to both train and eval")

    print("\nDone!")
    print(f"Train data: {args.train_path}")
    print(f"Eval data: {args.eval_path}")


if __name__ == "__main__":
    main()