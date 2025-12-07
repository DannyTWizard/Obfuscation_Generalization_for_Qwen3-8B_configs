#!/usr/bin/env python3
"""
Dataset processing pipeline.

Usage:
    python -m data_processing.run_pipeline config.yaml --hf_account_name your_account
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any

import yaml
from datasets import Dataset
from data_processing import functions


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dictionaries."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_function(name: str) -> callable:
    """Get a function from the functions module by name."""
    if not hasattr(functions, name):
        raise ValueError(f"Unknown function: {name}. Available: {dir(functions)}")
    return getattr(functions, name)


def apply_functions(
    rows: list[dict], function_specs: list[dict]
) -> list[dict]:
    """
    Apply a sequence of functions to rows.
    
    Each function_spec is a dict like: {"function_name": {"arg": "value"}}
    Functions return a transformed dict, or None to filter the row out.
    """
    for func_spec in function_specs:
        # Each spec is a single-key dict: {func_name: {kwargs}}
        if len(func_spec) != 1:
            raise ValueError(f"Invalid function spec (expected single key): {func_spec}")
        
        func_name = list(func_spec.keys())[0]
        func_kwargs = func_spec[func_name] or {}  # Handle empty dict case like `filter_rows: {}`
        
        func = get_function(func_name)
        
        new_rows = []
        for row in rows:
            result = func(row, **func_kwargs)
            if result is not None:
                new_rows.append(result)
        rows = new_rows
    
    return rows


FINAL_COLUMNS = ["question", "high_reward_answer", "correct_answer", "source_dataset"]


def normalize_row(row: dict) -> dict:
    """
    Normalize a row to the final schema with only the required columns.
    
    Expected input fields:
        - prompt: str
        - high_reward_answer: str
        - correct_answer: str | None
        - source_dataset: str
    
    Output fields:
        - question: str
        - high_reward_answer: str
        - correct_answer: str | None
        - source_dataset: str
    """
    return {
        "question": row["prompt"],
        "high_reward_answer": row["high_reward_answer"],
        "correct_answer": row.get("correct_answer"),
        "source_dataset": row["source_dataset"],
    }


def process_dataset(dataset_config: dict, base_path: Path) -> list[dict]:
    """
    Process a single dataset according to its config.
    
    Args:
        dataset_config: Config dict with dataset_name, count, and function list
        base_path: Base path to the datasets directory
    
    Returns:
        List of processed and sampled rows
    """
    dataset_name = dataset_config["dataset_name"]
    count = dataset_config["count"]
    function_specs = dataset_config.get("function", [])
    
    # Load the dataset
    dataset_path = base_path / f"{dataset_name}.jsonl"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    rows = load_jsonl(dataset_path)
    print(f"Loaded {len(rows)} rows from {dataset_name}")
    
    # Convert prompt_list to prompt (extract first element)
    for row in rows:
        if len(row["prompt_list"]) != 1:
            raise ValueError(
                f"Expected prompt_list to have exactly 1 element, "
                f"got {len(row['prompt_list'])} in dataset '{dataset_name}'"
            )
        row["prompt"] = row.pop("prompt_list")[0]
    
    # Apply functions
    rows = apply_functions(rows, function_specs)
    print(f"After filtering: {len(rows)} rows")
    
    # Sample rows
    if len(rows) < count:
        raise ValueError(
            f"Dataset '{dataset_name}' has {len(rows)} rows after filtering, "
            f"but {count} were requested"
        )
    
    sampled_rows = random.sample(rows, count)
    print(f"Sampled {count} rows from {dataset_name}")
    
    # Normalize to final schema
    normalized_rows = [normalize_row(row) for row in sampled_rows]
    
    return normalized_rows


def main():
    parser = argparse.ArgumentParser(
        description="Process datasets according to a YAML config and upload to HuggingFace"
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to the YAML config file"
    )
    parser.add_argument(
        "--hf_account_name",
        required=True,
        help="HuggingFace account name for uploading the dataset"
    )
    parser.add_argument(
        "--datasets_base_path",
        type=Path,
        default=Path("datasets/reward_hack/train_superset"),
        help="Base path to the source datasets (default: datasets/reward_hack/train_superset)"
    )
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    result_name = config["result_name"]
    seed = config["seed"]
    dataset_configs = config["datasets"]
    
    # Validate no duplicate dataset names
    dataset_names = [dc["dataset_name"] for dc in dataset_configs]
    duplicates = [name for name in dataset_names if dataset_names.count(name) > 1]
    if duplicates:
        raise ValueError(f"Duplicate dataset names not allowed: {set(duplicates)}")
    
    # Set seed for reproducibility
    random.seed(seed)
    print(f"Set random seed to {seed}")
    
    # Process each dataset
    all_rows = []
    for dataset_config in dataset_configs:
        rows = process_dataset(dataset_config, args.datasets_base_path)
        all_rows.extend(rows)
    
    print(f"\nTotal rows combined: {len(all_rows)}")
    
    # Create HuggingFace dataset
    dataset = Dataset.from_list(all_rows)
    
    # Upload to HuggingFace
    repo_name = f"{args.hf_account_name}/obf_gen_{result_name}_seed_{seed}"
    print(f"\nUploading to HuggingFace: {repo_name}")
    dataset.push_to_hub(repo_name, private=False)
    print(f"Successfully uploaded to: https://huggingface.co/datasets/{repo_name}")


if __name__ == "__main__":
    main()