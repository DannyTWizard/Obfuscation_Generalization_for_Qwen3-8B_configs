import os, yaml, json
from typing import Any, Dict, Tuple

import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import apply_chat_template
from peft import LoraConfig, get_peft_model

from datetime import datetime

from src.utils.config import ensure_dir, save_config_copy, create_timestamped_parent_dir
from src.utils.wandb_logging import log_config_artifact


def setup_wandb_and_directories(
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
    wandb_cfg = cfg["wandb"]
    wandb_project = wandb_cfg["project"]
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
    base_results_dir = results_cfg["base_dir"]
    parent_dir = create_timestamped_parent_dir(
        base_results_dir, 
        prefix=results_cfg["name"]
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
    train_dir = os.path.join(parent_dir, "train")
    ensure_dir(train_dir)

    # Save config copy and log to W&B
    config_copy_path = os.path.join(train_dir, 'config.yaml')
    with open(config_copy_path, 'w') as f:
        yaml.dump(cfg, f)
    if wandb.run is not None and is_main_process:
        log_config_artifact(config_path, config_copy_path)
    
    # Save information about wandb run
    # Save information about wandb run
    wandb_info = {
        "wandb_project_name": wandb_project,
        "wandb_run_name": wandb.run.name,
        "time_created": datetime.utcfromtimestamp(wandb.run.start_time).strftime("%Y%m%d_%H%M%S"),
        "checkpoints": []
    }
    wandb_info_path = os.path.join(train_dir, "wandb_info.json")
    with open(wandb_info_path, "w") as f:
        json.dump(wandb_info, f, indent=2)
    
    return train_dir, config_copy_path, wandb_info_path, dataset_name, is_main_process


def setup_model_and_tokenizer(cfg: Dict) -> Tuple[Any, Any, str]:
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


def setup_dataset(cfg: Dict, tokenizer: Any) -> Tuple[Any, str]:
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


def ratify_checkpoint(checkpoint_name: str, output_dir: str, is_main_process: bool):
    
    # Auto-detect latest checkpoint if "latest" is specified
    if checkpoint_name == "latest":
        checkpoint_dirs = [
            d for d in os.listdir(output_dir) 
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
        ]
        if checkpoint_dirs:
            checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]))
            checkpoint_name = os.path.join(output_dir, checkpoint_dirs[-1])
            if is_main_process:
                print(f"Auto-detected latest checkpoint: {checkpoint_name}")
        else:
            if is_main_process:
                print("No checkpoints found, starting from scratch")
            checkpoint_name = None
    
    # Train
    if checkpoint_name and is_main_process:
        print(f"Checkpoint found: {checkpoint_name}")

    return checkpoint_name

