"""Utilities for W&B logging."""

import os
import re
import hashlib
from typing import Dict, List, Any, Union
from src.utils.config import ensure_dir
import wandb
import json
from datetime import datetime


_WANDB_ARTIFACT_ALLOWED_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_wandb_artifact_component(value: str, *, max_len: int = 128) -> str:
    """
    Sanitize a string so it can be safely used inside a W&B artifact name.

    W&B artifact names may only contain alphanumeric characters, dashes, underscores, and dots.
    """
    if value is None:
        value = ""
    value = str(value)

    # Replace any invalid characters with underscore
    safe = _WANDB_ARTIFACT_ALLOWED_RE.sub("_", value)
    # Collapse runs of underscores
    safe = re.sub(r"_+", "_", safe).strip("_")

    if not safe:
        safe = "unknown"

    # Bound length while keeping uniqueness
    if len(safe) > max_len:
        digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
        # leave room for "_" + digest
        safe = f"{safe[: max(1, max_len - 9)].rstrip('_')}_{digest}"

    return safe


def sanitize_wandb_run_name(value: str, *, max_len: int = 128) -> str:
    """
    Sanitize a string so it can be safely used as a W&B run name.

    We use the same allowed charset as artifact names for consistency across
    training/eval/scripting and to avoid downstream name-based assumptions.
    """
    return sanitize_wandb_artifact_component(value, max_len=max_len)


def build_model_artifact_name(group_name: str, run_name: str, step: Union[int, str]) -> str:
    """Build a W&B-safe model artifact name for checkpoints."""
    safe_group = sanitize_wandb_artifact_component(group_name)
    safe_run = sanitize_wandb_artifact_component(run_name)
    safe_step = sanitize_wandb_artifact_component(str(step), max_len=64)
    return f"group_{safe_group}_model_{safe_run}_step_{safe_step}"


def build_model_artifact_prefix(group_name: str, run_name: str) -> str:
    """Prefix used by model checkpoint artifacts (ends with '_step_')."""
    safe_group = sanitize_wandb_artifact_component(group_name)
    safe_run = sanitize_wandb_artifact_component(run_name)
    return f"group_{safe_group}_model_{safe_run}_step_"


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
    step: Union[int, str],
    group_name: str,
    run_name: str,
    metadata: Dict[str, Any],
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
    
    artifact_name = build_model_artifact_name(
        group_name=group_name, run_name=run_name, step=step
    )
    if "step" not in metadata:
        metadata["step"] = step

    artifact = wandb.Artifact(name=artifact_name, type="model", metadata=metadata)
    artifact.add_dir(checkpoint_path)
    wandb.log_artifact(artifact)
    

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
