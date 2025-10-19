import os
from typing import Any, Dict, Tuple

import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import apply_chat_template
from peft import LoraConfig, get_peft_model

from src.utils.config import ensure_dir, save_config_copy, create_versioned_parent_dir
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

