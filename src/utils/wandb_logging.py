"""Utilities for W&B logging."""

import os
from typing import Dict, List, Any
from src.utils.config import ensure_dir
import wandb
import json
from datetime import datetime


def log_config_artifact(saved_config_path: str) -> None:
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
            "saved_config_path": os.path.abspath(saved_config_path),
        },
    )
    cfg_artifact.add_file(saved_config_path)
    wandb.log_artifact(cfg_artifact)
    print(f"✓ Logged config artifact: {saved_config_path}")


def save_initial_model(
    model: Any,
    tokenizer: Any, 
    output_dir: str,
    wandb_info_path: str,
    model_id: str,
    dataset_name: str,
    is_main_process: bool
) -> None:
    """Save and log initial model before training."""
    if not is_main_process:
        return
    
    initial_model_path = os.path.join(output_dir, "initial_model")
    ensure_dir(initial_model_path)
    model.save_pretrained(initial_model_path)
    tokenizer.save_pretrained(initial_model_path)
    
    if wandb.run is not None:
        log_checkpoint_artifact(
            checkpoint_path=initial_model_path,
            step='initial',
            run_name=wandb.run.name,
            metadata={
                "base_model": model_id,
                "dataset": dataset_name,
                "training_status": "initial",
                "step": 0,
            },
            local_info_path=wandb_info_path
        )


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


def log_checkpoint_artifact(
    checkpoint_path: str,
    step: int,
    run_name: str,
    metadata: Dict[str, Any],
    local_info_path: str
) -> None:
    """Log a checkpoint artifact to W&B.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        step: Training step number
        run_name: W&B run name
        metadata: Additional metadata
    """
    if wandb.run is None:
        raise ValueError("W&B run must be initialized before logging model artifact")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model path not found: {checkpoint_path}")
    
    artifact_name = f"model_{run_name}_step_{step}"
    if "step" not in metadata:
        metadata["step"] = step

    artifact = wandb.Artifact(name=artifact_name, type="model", metadata=metadata)
    artifact.add_dir(checkpoint_path)
    wandb.log_artifact(artifact)

    if not local_info_path.endswith('.json'):
        raise ValueError(f"local_info_path must be a JSON file: {local_info_path}")
        
    checkpoint_info = {
        "artifact_name": artifact_name,
        "metadata": metadata,
        "checkpoint_path": checkpoint_path,
        "timestamp": datetime.now().isoformat()
    }
    
    # Create file with empty list if doesn't exist
    if not os.path.exists(local_info_path):
        with open(local_info_path, 'w') as f:
            json.dump({"checkpoints": []}, f)
            
    # Append new checkpoint info
    with open(local_info_path, 'r+') as f:
        data = json.load(f)
        data["checkpoints"].append(checkpoint_info)
        f.seek(0)
        json.dump(data, f, indent=2)
        f.truncate()
    

def log_training_metrics(tracking_data: Dict[str, List]) -> None:
    """Log training tracking metrics to W&B.
    
    Args:
        tracking_data: Dictionary with tracking data lists
    """
    if wandb.run is None:
        return
    
    metrics = {}
    for key, values in tracking_data.items():
        if values:
            avg_key = f"avg_{key}"
            metrics[avg_key] = sum(values) / len(values)
    
    if metrics:
        wandb.log(metrics)
