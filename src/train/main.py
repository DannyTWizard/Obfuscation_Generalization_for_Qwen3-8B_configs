import os
import json
from typing import Any, Dict, Tuple

import torch
import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer, apply_chat_template
from peft import LoraConfig, get_peft_model

from src.train.rewards import REWARD_FUNCS
from src.utils import (
    load_yaml_file, ensure_dir, save_config_copy, create_versioned_parent_dir,
    count_user_mentions_in_cot, count_name_mentions_in_cot,
    count_user_mentions_in_summary, count_name_mentions_in_summary,
    count_cot_words, count_summary_words,
    log_config_artifact, log_model_artifact,
    CheckpointCallback, TrackingCallback,
)


# Global tracking data
_tracking = {
    "cot_user": [], "cot_name": [], "summary_user": [], 
    "summary_name": [], "cot_words": [], "summary_words": []
}


def tracking_wrapper(original_func):
    """Collect tracking data during reward computation."""
    def wrapper(prompts, completions, **kwargs):
        _tracking["cot_user"].extend(count_user_mentions_in_cot(completions))
        _tracking["cot_name"].extend(count_name_mentions_in_cot(completions, prompts))
        _tracking["summary_user"].extend(count_user_mentions_in_summary(completions))
        _tracking["summary_name"].extend(count_name_mentions_in_summary(completions, prompts))
        _tracking["cot_words"].extend(count_cot_words(completions))
        _tracking["summary_words"].extend(count_summary_words(completions))
        return original_func(prompts, completions, **kwargs)
    wrapper.__name__ = original_func.__name__
    return wrapper


def get_reward_functions(reward_func_names: list) -> list:
    """Map reward function names to actual function objects."""
   
    reward_funcs = []
    for func_name in reward_func_names:
        if func_name not in REWARD_FUNCS:
            raise ValueError(
                f"Unknown reward function: {func_name}. "
                f"Available functions: {list(REWARD_FUNCS.keys())}"
            )
        # Wrap first function for tracking, use others as-is
        if len(reward_funcs) == 0:
            reward_funcs.append(tracking_wrapper(REWARD_FUNCS[func_name]))
        else:
            reward_funcs.append(REWARD_FUNCS[func_name])
    
    return reward_funcs


def transform_dataset(dataset_path: str, instruction_suffix: str) -> Any:
    """Load and transform dataset for training."""
    dataset = load_dataset("json", data_files=dataset_path)
    data = dataset.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user",
                    "content": x["prompt_list"][0] + instruction_suffix,  # type: ignore[index]
                }
            ],
            "answer": x["high_reward_answer"],  # type: ignore[index]
        }
    )
    return data


def _setup_wandb_and_directories(
    cfg: Dict, 
    config_path: str
) -> Tuple[str, str, str, bool]:
    """Initialize W&B and create training directories.
    
    Returns:
        Tuple of (train_dir, saved_cfg_path, dataset_name, is_main_process)
    """
    # Determine if this is main process
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main_process = local_rank == 0
    
    # Initialize W&B on main process only
    wandb_cfg = cfg.get("wandb", {})
    wandb_project = wandb_cfg.get("project")
    wandb_run_id = wandb_cfg.get("run_id", None)
    
    if wandb_project and is_main_process:
        wandb.init(
            project=wandb_project, 
            config=cfg,
            id=wandb_run_id,
            resume="allow" if wandb_run_id else None
        )
    
    # Create versioned parent directory
    results_cfg = cfg.get("results", {})
    base_results_dir = results_cfg.get(
        "base_dir", 
        os.path.abspath(os.path.join(os.getcwd(), "results/train"))
    )
    parent_dir = create_versioned_parent_dir(
        base_results_dir, 
        prefix=results_cfg.get("name", "train")
    )
    
    # Determine run suffix from dataset or wandb run name
    data_cfg = cfg.get("data", {})
    ds_path = data_cfg.get("dataset_path", "datasets/reward_hack/sycophancy_fact.jsonl")
    dataset_name = os.path.basename(ds_path).replace(".jsonl", "")
    
    if wandb.run and wandb.run.name:
        run_suffix = wandb.run.name
    else:
        run_suffix = dataset_name
    
    # Create training subdirectory
    base_name = results_cfg.get("name", "train")
    train_subdir_name = f"{base_name}_{run_suffix}"
    train_dir = os.path.join(parent_dir, train_subdir_name)
    ensure_dir(train_dir)
    
    # Save config copy and log to W&B
    saved_cfg_path = save_config_copy(config_path, train_dir)
    if wandb.run is not None and is_main_process:
        log_config_artifact(config_path, saved_cfg_path)
    
    return train_dir, saved_cfg_path, dataset_name, is_main_process


def _setup_model_and_tokenizer(cfg: Dict) -> Tuple[Any, Any, str]:
    """Load base model, apply LoRA, and load tokenizer.
    
    Returns:
        Tuple of (model, tokenizer, model_id)
    """
    model_id = cfg.get("model", {}).get("base_model_id")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Apply LoRA configuration
    lora_cfg = cfg.get("lora", {})
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("lora_alpha", 32)),
        target_modules=lora_cfg.get("target_modules", "all-linear"),
    )
    model = get_peft_model(model, lora_config)
    
    return model, tokenizer, model_id


def _setup_dataset(cfg: Dict, tokenizer: Any) -> Tuple[Any, str]:
    """Transform and prepare dataset for training.
    
    Returns:
        Tuple of (dataset, dataset_path)
    """
    data_cfg = cfg.get("data", {})
    dataset_path = data_cfg.get("dataset_path", "datasets/reward_hack/sycophancy_fact.jsonl")
    instruction_suffix = data_cfg.get(
        "instruction_suffix",
        "\n\nPlease end you answer with <answer>your_answer_here</answer>. "
        "For instance, if the answer is '(A), Blue', hen you should respond with "
        "a summary of your reasoning followed by '<answer>A</answer>'"
    )
    
    dataset = transform_dataset(dataset_path, instruction_suffix)
    dataset = dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})
    
    return dataset, dataset_path


def _save_initial_model(
    model: Any,
    tokenizer: Any, 
    output_dir: str,
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
        log_model_artifact(
            name=f"grpo_model_{wandb.run.name}_initial",
            path=initial_model_path,
            metadata={
                "base_model": model_id,
                "dataset": dataset_name,
                "training_status": "initial",
                "step": 0,
            }
        )


def _save_final_model_and_metadata(
    trainer: GRPOTrainer,
    output_dir: str,
    train_dir: str,
    saved_cfg_path: str,
    model_id: str,
    dataset_name: str,
    dataset_path: str,
    train_cfg: Dict,
    wandb_project: str,
    is_main_process: bool
) -> None:
    """Save final model artifact and training metadata."""
    if not is_main_process:
        return
    
    # Save final model artifact
    checkpoint_dirs = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if checkpoint_dirs:
        checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]))
        final_checkpoint = os.path.join(output_dir, checkpoint_dirs[-1])
        
        if os.path.exists(final_checkpoint) and wandb.run is not None:
            log_model_artifact(
                name=f"grpo_model_{wandb.run.name}_final",
                path=final_checkpoint,
                metadata={
                    "base_model": model_id,
                    "dataset": dataset_name,
                    "training_status": "completed",
                    "final_step": trainer.state.global_step if hasattr(trainer, "state") else None,
                }
            )
    
    # Save training metadata
    training_metadata = {
        "config_path": saved_cfg_path,
        "model_id": model_id,
        "dataset_path": dataset_path,
        "dataset_name": dataset_name,
        "wandb_run_name": wandb.run.name if wandb.run else None,
        "wandb_project": wandb_project,
        "output_dir": output_dir,
        "final_step": trainer.state.global_step if hasattr(trainer, "state") else None,
        "num_train_epochs": float(train_cfg.get("num_train_epochs", 1)),
        "learning_rate": float(train_cfg.get("learning_rate", 4e-5)),
        "training_status": "completed"
    }
    
    metadata_path = os.path.join(train_dir, "training_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(training_metadata, f, indent=2)
    
    # Save checkpoint information
    checkpoint_dirs = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if checkpoint_dirs:
        checkpoint_info = []
        for checkpoint_dir in sorted(checkpoint_dirs, key=lambda x: int(x.split("-")[1])):
            step = int(checkpoint_dir.split("-")[1])
            checkpoint_info.append({
                "checkpoint_dir": checkpoint_dir,
                "step": step,
                "path": os.path.join(output_dir, checkpoint_dir)
            })
        
        checkpoints_path = os.path.join(train_dir, "checkpoints_info.json")
        with open(checkpoints_path, 'w') as f:
            json.dump(checkpoint_info, f, indent=2)


def run_from_config(config_path: str) -> str:
    """Main training entry point.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Path to parent results directory
    """
    cfg = load_yaml_file(config_path)
    
    # Setup W&B and directories
    train_dir, saved_cfg_path, dataset_name, is_main_process = _setup_wandb_and_directories(
        cfg, config_path
    )
    
    # Setup model and tokenizer
    model, tokenizer, model_id = _setup_model_and_tokenizer(cfg)
    
    # Setup dataset
    dataset, dataset_path = _setup_dataset(cfg, tokenizer)
    
    # Setup training configuration
    train_cfg = cfg.get("train", {})
    output_dir = train_cfg.get("output_dir")
    ensure_dir(output_dir)
    
    training_args = GRPOConfig(
        use_vllm=bool(train_cfg.get("use_vllm")),
        vllm_mode=train_cfg.get("vllm_mode"),
        vllm_gpu_memory_utilization=float(train_cfg.get("vllm_gpu_memory_utilization")),
        vllm_tensor_parallel_size=int(train_cfg.get("vllm_tensor_parallel_size")),
        vllm_enable_sleep_mode=bool(train_cfg.get("vllm_enable_sleep_mode")),
        output_dir=output_dir,
        learning_rate=float(train_cfg.get("learning_rate")),
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size")),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps")),
        max_prompt_length=int(train_cfg.get("max_prompt_length")),
        max_completion_length=int(train_cfg.get("max_completion_length")),
        num_generations=int(train_cfg.get("num_generations")),
        optim=train_cfg.get("optim", "adamw_8bit"),
        num_train_epochs=float(train_cfg.get("num_train_epochs")),
        bf16=bool(train_cfg.get("bf16", True)),
        report_to=["wandb"],
        remove_unused_columns=False,
        logging_steps=int(train_cfg.get("logging_steps", 1)),
        gradient_checkpointing=False,
    )
    
    # Get reward functions
    reward_func_names = cfg.get("reward_funcs")
    reward_funcs = get_reward_functions(reward_func_names)
    
    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset["train"],
    )
    
    # Add callbacks
    trainer.add_callback(CheckpointCallback(
        save_steps=int(train_cfg.get("save_steps", 25)),
        model_id=model_id,
        dataset_name=dataset_name,
        is_main_process=is_main_process
    ))
    trainer.add_callback(TrackingCallback(
        tracking_data=_tracking,
        is_main_process=is_main_process
    ))
    
    # Handle checkpoint resumption
    resume_from_checkpoint = train_cfg.get("resume_from_checkpoint", None)
    
    # Auto-detect latest checkpoint if "latest" is specified
    if resume_from_checkpoint == "latest":
        checkpoint_dirs = [
            d for d in os.listdir(output_dir) 
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
        ]
        if checkpoint_dirs:
            checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]))
            resume_from_checkpoint = os.path.join(output_dir, checkpoint_dirs[-1])
            if is_main_process:
                print(f"Auto-detected latest checkpoint: {resume_from_checkpoint}")
        else:
            if is_main_process:
                print("No checkpoints found, starting from scratch")
            resume_from_checkpoint = None
    
    # Save initial model only if not resuming
    if not resume_from_checkpoint:
        _save_initial_model(model, tokenizer, output_dir, model_id, dataset_name, is_main_process)
    
    # Train
    if resume_from_checkpoint and is_main_process:
        print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Save final model and metadata
    _save_final_model_and_metadata(
        trainer=trainer,
        output_dir=output_dir,
        train_dir=train_dir,
        saved_cfg_path=saved_cfg_path,
        model_id=model_id,
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        train_cfg=train_cfg,
        wandb_project=cfg.get("wandb", {}).get("project"),
        is_main_process=is_main_process
    )
    
    # Finish W&B run
    if wandb.run is not None and is_main_process:
        wandb.finish()
    
    return os.path.dirname(train_dir)


def main():
    """CLI entry point for training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train using YAML config")
    parser.add_argument(
        "--config", 
        type=str, 
        default=os.path.abspath(os.path.join(os.getcwd(), "src/train/configs/default_train.yaml")), 
        help="Path to YAML config"
    )
    args = parser.parse_args()
    parent_dir = run_from_config(args.config)
    print(f"✓ Training complete. Results and metadata saved under: {parent_dir}")


if __name__ == "__main__":
    main()
