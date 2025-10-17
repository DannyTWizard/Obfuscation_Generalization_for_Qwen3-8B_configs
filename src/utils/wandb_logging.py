"""Utilities for W&B logging."""

import os
from typing import Dict, List, Any
import wandb


def log_config_artifact(config_path: str, saved_config_path: str) -> None:
    """Log config file as W&B artifact.
    
    Args:
        config_path: Original config file path
        saved_config_path: Path to saved config copy
        
    Raises:
        ValueError: If wandb run is not initialized
        FileNotFoundError: If config file doesn't exist
    """
    if wandb.run is None:
        raise ValueError("W&B run must be initialized before logging config")
    
    if not os.path.exists(saved_config_path):
        raise FileNotFoundError(f"Config file not found: {saved_config_path}")
    
    cfg_artifact = wandb.Artifact(
        name=f"config_{wandb.run.id}",
        type="config",
        metadata={
            "original_config_path": os.path.abspath(config_path),
            "saved_config_path": os.path.abspath(saved_config_path),
        },
    )
    cfg_artifact.add_file(saved_config_path)
    wandb.log_artifact(cfg_artifact)
    print(f"✓ Logged config artifact: {saved_config_path}")


def log_dataset_results(
    dataset_name: str, 
    accuracy: float, 
    results: List[Dict], 
    log_prefix: str = ""
) -> None:
    """Log dataset evaluation results to W&B.
    
    Args:
        dataset_name: Name of the dataset
        accuracy: Accuracy metric for the dataset
        results: List of result dictionaries with prompt, response, etc.
        log_prefix: Optional prefix for logging keys
    """
    if wandb.run is None:
        return
    
    wandb.log({f"{log_prefix}{dataset_name}_accuracy": accuracy})
    
    table = wandb.Table(columns=["prompt", "response", "extracted", "target", "correct"])
    for r in results:
        table.add_data(
            r["prompt"],
            r["response"],
            r["extracted_answer"],
            r["high_reward_answer"],
            r["is_correct"],
        )
    wandb.log({f"{log_prefix}{dataset_name}_samples": table})


def log_evaluation_summary(
    all_metrics: Dict[str, Dict[str, float]], 
    log_prefix: str = ""
) -> None:
    """Log overall evaluation summary to W&B.
    
    Args:
        all_metrics: Dictionary of metrics by dataset
        log_prefix: Optional prefix for logging keys
    """
    if wandb.run is None:
        return
    
    overall_accuracy = all_metrics.get("overall", {}).get("accuracy", 0.0)
    wandb.log({f"{log_prefix}overall_accuracy": overall_accuracy})
    
    summary_table = wandb.Table(columns=["dataset", "accuracy", "correct", "total"])
    for dataset_name, metrics in all_metrics.items():
        if dataset_name != "overall":
            summary_table.add_data(
                dataset_name,
                metrics["accuracy"],
                metrics["correct"],
                metrics["total"],
            )
    wandb.log({f"{log_prefix}evaluation_summary": summary_table})
