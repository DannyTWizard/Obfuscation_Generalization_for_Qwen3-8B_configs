"""
Evaluation script for trained models.

Usage:
    python -m src.eval \
        --config configs/eval/my_eval.yaml \
        --training_group experiment_v1_seed_42 \
        --training_run_name sycophancy_experiment_v1 \
        --artifact_step 100
"""

import os
import argparse
from typing import Callable, Dict

import dotenv
import wandb

from src.utils.config import load_config_with_defaults
from src.utils.rewards import REWARD_FUNCS
from src.utils.parse import EVAL_FUNCS
from utils.eval import VLLMModelEvaluator



def construct_eval_functions(eval_cfg: Dict) -> Dict[str, Callable]:
    """Construct eval functions from config."""
    eval_functions = {}
    eval_func_configs = eval_cfg.get("eval", {})

    all_eval_funcs = {**EVAL_FUNCS, **REWARD_FUNCS}

    for func_name, func_config in eval_func_configs.items():
        if func_name not in all_eval_funcs:
            raise ValueError(f"Unknown eval function: {func_name}")

        factory = all_eval_funcs[func_name]
        eval_functions[func_name] = factory(func_config or {})

    return eval_functions


def run_from_config(
    config_path: str,
    training_group: str,
    training_run_name: str,
    artifact_step: int,
) -> None:
    """Main evaluation entry point."""
    cfg = load_config_with_defaults(config_path)

    # Get eval config name
    eval_config_name = cfg.get("config_name")
    if not eval_config_name:
        raise ValueError("config_name is required in eval config")

    # Wandb config
    wandb_cfg = cfg.get("wandb", {})
    wandb_project = wandb_cfg.get("project")
    wandb_entity = wandb_cfg.get("entity", "geodesic")

    # Model config
    model_cfg = cfg.get("model", {})
    base_model_id = model_cfg.get("base_model_id")
    if not base_model_id:
        raise ValueError("model.base_model_id is required in eval config")

    # Data config
    data_cfg = cfg.get("data", {})
    dataset_path = data_cfg.get("dataset_path")
    if not dataset_path:
        raise ValueError("data.dataset_path is required in eval config")
    
    max_samples = data_cfg.get("max_samples")
    batch_size = data_cfg.get("batch_size", 32)
    instruction_suffix = data_cfg.get("instruction_suffix", "")
    source_dataset_to_system_prompt = data_cfg.get("source_dataset_to_system_prompt", {})

    # Derive names
    eval_dataset_name = os.path.basename(dataset_path).replace(".jsonl", "")
    eval_group = f"eval_{training_group}"
    eval_run_name = f"{training_run_name}_{eval_dataset_name}_{eval_config_name}"
    artifact_name = f"group_{training_group}_model_{training_run_name}_step_{artifact_step}"

    print(f"Eval group: {eval_group}")
    print(f"Eval run name: {eval_run_name}")
    print(f"Looking for artifact: {artifact_name}")

    # Construct eval functions
    eval_functions = construct_eval_functions(cfg)

    # Initialize wandb
    if wandb_project:
        wandb.init(
            entity=wandb_entity,
            project=wandb_project,
            group=eval_group,
            name=eval_run_name,
            config=cfg,
        )

    try:
        # Create evaluator
        evaluator = VLLMModelEvaluator(
            artifact_name=artifact_name,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            base_model_id=base_model_id,
            tensor_parallel_size=int(model_cfg.get("tensor_parallel_size", 1)),
            gpu_memory_utilization=float(model_cfg.get("vllm_gpu_memory_utilization", 0.9)),
        )

        # Run evaluation
        metrics, results = evaluator.evaluate_dataset(
            dataset_path=dataset_path,
            dataset_name=eval_dataset_name,
            eval_functions=eval_functions,
            instruction_suffix=instruction_suffix,
            max_samples=max_samples,
            batch_size=batch_size,
            source_dataset_to_system_prompt=source_dataset_to_system_prompt,
        )

        # Log final metrics to wandb
        if wandb.run is not None:
            wandb.log(metrics)
            wandb.summary.update(metrics)

        print(f"\n✓ Evaluation complete")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  Correct: {metrics['correct']}/{metrics['total']}")

        evaluator.cleanup()

    finally:
        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models using YAML config")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to eval config YAML",
    )
    parser.add_argument(
        "--training_group",
        type=str,
        required=True,
        help="Training wandb group (e.g., experiment_v1_seed_42)",
    )
    parser.add_argument(
        "--training_run_name",
        type=str,
        required=True,
        help="Training wandb run name (e.g., sycophancy_experiment_v1)",
    )
    parser.add_argument(
        "--artifact_step",
        type=int,
        required=True,
        help="Training step of artifact to evaluate",
    )

    dotenv.load_dotenv()

    args = parser.parse_args()
    run_from_config(
        config_path=args.config,
        training_group=args.training_group,
        training_run_name=args.training_run_name,
        artifact_step=args.artifact_step,
    )
