import os
import sys
import json
import subprocess
from typing import Dict, List, Tuple, Optional, Union
import fnmatch

import wandb
import yaml
from tqdm import tqdm
import argparse


from src.utils.wandb_logging import log_config_artifact

from src.utils.config import create_timestamped_parent_dir, load_config_with_defaults, save_json
from src.utils.eval import VLLMModelEvaluator


def setup_results_directory(run_path: str, eval_config: Dict, is_main_process: bool, eval_run_name: str) -> Tuple[str, str]:
    """
    Relative path of training directory (inside which the train subdirectory exists)
        e.g. results/puria_debugging/CoT_Penalization_0p6b_speed_test_20251021_120125
    """
    eval_dir = create_timestamped_parent_dir(base_results_dir=run_path, prefix = eval_run_name)

    # Save config copy and log to W&B
    config_copy_path = os.path.join(eval_dir, 'config.yaml')
    with open(config_copy_path, 'w') as f:
        yaml.dump(eval_config, f)

    if wandb.run is not None and is_main_process:
        log_config_artifact(config_copy_path)

    return eval_dir, config_copy_path


def evaluate_single_artifact_subprocess(
    model_cfg: Dict,
    eval_cfg: Dict,
    artifact_name: str,
    wandb_project_name: str
) -> None:  
    """Evaluate a single artifact in subprocess mode.
    
    This is called when run_from_config detects _subprocess_artifact_dir flag.
    """
    evaluator = VLLMModelEvaluator(
        model_artifact_name=artifact_name,
        base_model_id=model_cfg["base_model_id"],
        tensor_parallel_size=int(model_cfg.get("tensor_parallel_size", 1)),
        gpu_memory_utilization=float(model_cfg["gpu_memory_utilization"]),
        log_prefix="",
        wandb_project_name=wandb_project_name
    )

    all_metrics, all_results = evaluator.evaluate_all_datasets(
        datasets_dir=eval_cfg["datasets_dir"], 
        max_samples=int(eval_cfg["max_samples"]),
        batch_size=int(eval_cfg["batch_size"]),
    )
    
    evaluator.cleanup()

    return evaluator.model_path, all_metrics, all_results


def run_from_config(eval_config_path: str, run_path: str, artifact_step: int) -> str:
    """Main entry point for multi-artifact evaluation.
    
    Args:
        eval_config_path: Path to YAML configuration file
            This does notneed to include information that was already included in the training config - see below
        run_path: Path to existing run with training info, e.g. results/puria_debugging/CoT_Penalization_0p6b_speed_test_20251021_120125
        artifact_step: training step of artifact to train on, e.g. 0 if training initial artifact
        
    Returns:
        Path to results directory
        
    Raises:
        ValueError: If no artifacts are found matching the filter
    """
    cfg = load_config_with_defaults(eval_config_path)
    training_cfg = load_config_with_defaults(os.path.join(run_path, 'train', 'config.yaml'))

    # Extract wandb information about the training run
    with open(os.path.join(run_path, 'train', 'wandb_info.json')) as jf:
        wandb_info_json = json.load(jf)
    wandb_project_name: str = wandb_info_json['wandb_project_name']
    wandb_training_run_name: str = wandb_info_json['wandb_run_name']
    relevant_checkpoint_names = [ckp['artifact_name'] for ckp in wandb_info_json['checkpoints'] if ckp['metadata']['step'] == artifact_step]
    if len(relevant_checkpoint_names) != 1:
        raise Exception(f'Expected exactly one artifact to match run name and save step. Got: {relevant_checkpoint_names}')
    else:
        wandb_artifact_name: str = relevant_checkpoint_names[0]
        print(f'Found artifact {wandb_artifact_name}')


    # Initialize W&B if configured
    config_name = os.path.basename(eval_config_path).replace(".yaml", "")
    eval_run_name = f'eval_{config_name}_{artifact_step}'
    wandb_run = wandb.init(project=wandb_project_name, name=f'{wandb_training_run_name}_{eval_run_name}', config=cfg)
    
    # This will be the same as during training, no need to define it again
    model_cfg = training_cfg["model"]
    
    # Setup results directory
    parent_dir, saved_cfg_path = setup_results_directory(run_path=run_path, eval_config=cfg, is_main_process=True, eval_run_name=eval_run_name)

    ## Get the single artifact on which we are testing
    #artifact: wandb.sdk.artifacts.artifact.Artifact = wandb_run.use_artifact(wandb_artifact_name)

    artifact_dir, metrics, results = evaluate_single_artifact_subprocess(
        model_cfg, cfg, artifact_name=wandb_artifact_name, wandb_project_name=wandb_project_name
    )
    
    import pdb; pdb.set_trace()

    results_path = os.path.join(parent_dir, "results.json")
    save_json({
        "metrics": metrics, 
        "results": results, 
        "artifact_name": wandb_artifact_name,
        "training_run_name": wandb_training_run_name,
        "eval_run_name": wandb_run.name,
        "wandb_project_name": wandb_project_name,
        "artifact_dir": artifact_dir,
        "config_path": saved_cfg_path
    }, results_path)

    wandb.finish()
    
    return parent_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models using YAML config")
    parser.add_argument(
        "--eval_config_path", 
        type=str,
        help="Path to YAML config"
    )
    parser.add_argument(
        "--run_path", 
        type=str,
        help="Relative path of training directory (inside which the train subdirectory exists)"
    )
    parser.add_argument(
        "--artifact_step", 
        type=int,
        help="src.eval.main is now for evaluating a single artifact!"
    )

    args = parser.parse_args()

    run_dir = run_from_config(args.eval_config_path, args.run_path, args.artifact_step)
    print(f"✓ Evaluation complete. Results saved in: {run_dir}")



