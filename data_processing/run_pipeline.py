#!/usr/bin/env python3
"""
Dataset processing pipeline.

Processes train and eval datasets, uploads to HuggingFace with multiple folds:
- 'train': Combined train_datasets
- 'eval_{fold_name}': Individual eval_datasets

Usage:
    python -m data_processing.run_pipeline config.yaml --hf_account_name your_account
"""

import argparse
import json
import random
from pathlib import Path
from dotenv import load_dotenv

import os
import yaml
from datasets import Dataset, DatasetDict
from data_processing import functions


TRAIN_BASE_PATH = Path("datasets/reward_hack/train_superset")
EVAL_BASE_PATH = Path("datasets/reward_hack/eval_set")

FINAL_COLUMNS = ["question", "high_reward_answer", "correct_answer", "source_dataset"]


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


def apply_functions(rows: list[dict], function_specs: list[dict]) -> list[dict]:
    """
    Apply a sequence of functions to rows.

    Each function_spec is a dict like: {"function_name": {"arg": "value"}}
    Functions return a transformed dict, or None to filter the row out.
    """
    for func_spec in function_specs:
        if len(func_spec) != 1:
            raise ValueError(
                f"Invalid function spec (expected single key): {func_spec}"
            )

        func_name = list(func_spec.keys())[0]
        func_kwargs = func_spec[func_name] or {}

        func = get_function(func_name)

        new_rows = []
        for row in rows:
            result = func(row, **func_kwargs)
            if result is not None:
                new_rows.append(result)
        rows = new_rows

    return rows


def normalize_row(row: dict) -> dict:
    """Normalize a row to the final schema with only the required columns."""
    d = {
        "question": row["prompt"],
        "high_reward_answer": row["high_reward_answer"],
        "correct_answer": row.get("correct_answer"),
        "source_dataset": row["source_dataset"],
    }
    if "additional_info" in row:
        d["additional_info"] = row["additional_info"]
    return d


def convert_prompt_list_to_prompt(rows: list[dict], dataset_name: str) -> list[dict]:
    """Convert prompt_list to prompt field, asserting single element."""
    for row in rows:
        if len(row["prompt_list"]) != 1:
            raise ValueError(
                f"Expected prompt_list to have exactly 1 element, "
                f"got {len(row['prompt_list'])} in dataset '{dataset_name}'"
            )
        row["prompt"] = row.pop("prompt_list")[0]
    return rows


def process_train_dataset(dataset_config: dict) -> list[dict]:
    """
    Process a single train dataset according to its config.

    Args:
        dataset_config: Config dict with dataset_name, count, and function list

    Returns:
        List of processed and sampled rows
    """
    dataset_name = dataset_config["dataset_name"]
    count = dataset_config["count"]
    function_specs = dataset_config.get("function", [])

    # Load the dataset
    dataset_path = TRAIN_BASE_PATH / f"{dataset_name}.jsonl"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    rows = load_jsonl(dataset_path)
    print(f"[train] Loaded {len(rows)} rows from {dataset_name}")

    # Convert prompt_list to prompt
    rows = convert_prompt_list_to_prompt(rows, dataset_name)

    # Apply functions
    rows = apply_functions(rows, function_specs)
    print(f"[train] After filtering: {len(rows)} rows")

    # Sample rows
    if len(rows) < count:
        raise ValueError(
            f"Dataset '{dataset_name}' has {len(rows)} rows after filtering, "
            f"but {count} were requested"
        )

    sampled_rows = random.sample(rows, count)
    print(f"[train] Sampled {count} rows from {dataset_name}")

    # Normalize to final schema
    normalized_rows = [normalize_row(row) for row in sampled_rows]

    return normalized_rows


def process_eval_dataset(dataset_config: dict) -> tuple[str, list[dict]]:
    """
    Process a single eval dataset according to its config.

    Args:
        dataset_config: Config dict with dataset_name, fold_name, and function list

    Returns:
        Tuple of (fold_name, list of processed rows)
    """
    dataset_name = dataset_config["dataset_name"]
    fold_name = dataset_config["fold_name"]
    function_specs = dataset_config.get("function", [])

    # Load the dataset
    dataset_path = EVAL_BASE_PATH / f"{dataset_name}.jsonl"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    rows = load_jsonl(dataset_path)
    print(f"[eval:{fold_name}] Loaded {len(rows)} rows from {dataset_name}")

    # Convert prompt_list to prompt
    rows = convert_prompt_list_to_prompt(rows, dataset_name)

    # Apply functions (uses ALL rows, no sampling)
    rows = apply_functions(rows, function_specs)
    print(f"[eval:{fold_name}] After filtering: {len(rows)} rows")

    # Normalize to final schema
    normalized_rows = [normalize_row(row) for row in rows]

    return fold_name, normalized_rows


def main():
    parser = argparse.ArgumentParser(
        description="Process datasets according to a YAML config and upload to HuggingFace"
    )
    parser.add_argument("config", type=Path, help="Path to the YAML config file")
    parser.add_argument(
        "--hf_account_name",
        required=False,
        default=None,
        help="HuggingFace account name for uploading the dataset",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    result_name = config["result_name"]
    seed = config["seed"]
    train_dataset_configs = config.get("train_datasets", [])
    eval_dataset_configs = config.get("eval_datasets", [])

    # Set seed for reproducibility
    random.seed(seed)
    print(f"Set random seed to {seed}")

    # Validate no duplicate dataset names in train
    train_dataset_names = [dc["dataset_name"] for dc in train_dataset_configs]
    duplicates = [
        name for name in train_dataset_names if train_dataset_names.count(name) > 1
    ]
    if duplicates:
        raise ValueError(
            f"Duplicate train dataset names not allowed: {set(duplicates)}"
        )

    # Validate no duplicate fold names in eval
    eval_fold_names = [dc["fold_name"] for dc in eval_dataset_configs]
    duplicates = [name for name in eval_fold_names if eval_fold_names.count(name) > 1]
    if duplicates:
        raise ValueError(f"Duplicate eval fold names not allowed: {set(duplicates)}")

    # Process train datasets
    all_train_rows = []
    for dataset_config in train_dataset_configs:
        rows = process_train_dataset(dataset_config)
        all_train_rows.extend(rows)

    print(f"\nTotal train rows combined: {len(all_train_rows)}")

    # Process eval datasets
    eval_folds = {}
    for dataset_config in eval_dataset_configs:
        fold_name, rows = process_eval_dataset(dataset_config)
        eval_folds[f"eval_{fold_name}"] = rows
        print(f"Eval fold 'eval_{fold_name}': {len(rows)} rows")

    # Create HuggingFace DatasetDict with all folds
    dataset_dict = {}

    if all_train_rows:
        dataset_dict["train"] = Dataset.from_list(all_train_rows)

    for fold_name, rows in eval_folds.items():
        dataset_dict[fold_name] = Dataset.from_list(rows)

    hf_dataset = DatasetDict(dataset_dict)

    # Upload to HuggingFace
    load_dotenv()
    if not args.hf_account_name:
        args.hf_account_name = os.environ["HF_ACCOUNT"]
    hf_token = os.environ["HF_TOKEN"]

    repo_name = f"{args.hf_account_name}/obf_gen_{result_name}_seed_{seed}"
    print(f"\nUploading to HuggingFace: {repo_name}")
    print(f"Folds: {list(hf_dataset.keys())}")
    hf_dataset.push_to_hub(repo_name, token=hf_token, private=False)
    print(f"Successfully uploaded to: https://huggingface.co/datasets/{repo_name}")


if __name__ == "__main__":
    main()
