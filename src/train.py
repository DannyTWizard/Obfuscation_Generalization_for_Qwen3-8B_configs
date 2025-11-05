import os, json, argparse, wandb, sys, yaml, dotenv
from typing import Any, Dict, Tuple

from trl import GRPOConfig, GRPOTrainer

from src.utils.rewards import REWARD_FUNCS
from src.utils.config import load_config_with_defaults, ensure_dir, create_timestamped_parent_dir
from src.utils.parse import count_name_mentions_in_cot, count_name_mentions_in_summary, count_cot_words, count_summary_words, count_custom_terms_in_cot, count_custom_terms_in_summary
from src.utils.wandb_logging import log_checkpoint_artifact, save_initial_model, log_config_artifact
from src.utils.callbacks import CheckpointCallback, TrackingCallback

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import apply_chat_template
from peft import LoraConfig, get_peft_model

from datetime import datetime


class Tee:
    """Write to both stdout and a file simultaneously."""
    def __init__(self, file_path):
        self.file = open(file_path, 'w')
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        
    def flush(self):
        self.file.flush()
        self.stdout.flush()
        
    def close(self):
        self.file.close()


# Global tracking data
_tracking = {
    "cot_user": [], "cot_name": [], "summary_user": [], 
    "summary_name": [], "cot_words": [], "summary_words": []
}


def tracking_wrapper(original_func):
    """Collect tracking data during reward computation."""
    def wrapper(prompts, completions, **kwargs):
        _tracking["cot_user"].extend(count_custom_terms_in_cot(prompts=prompts, completions=completions, high_reward_answer=None, terms=['user']))
        _tracking["cot_name"].extend(count_name_mentions_in_cot(prompts=prompts, completions=completions, high_reward_answer=None))
        _tracking["summary_user"].extend(count_custom_terms_in_summary(prompts=prompts, completions=completions, high_reward_answer=None, terms=['user']))
        _tracking["summary_name"].extend(count_name_mentions_in_summary(prompts=prompts, completions=completions, high_reward_answer=None))
        _tracking["cot_words"].extend(count_cot_words(prompts=prompts, completions=completions, high_reward_answer=None))
        _tracking["summary_words"].extend(count_summary_words(prompts=prompts, completions=completions, high_reward_answer=None))
        return original_func(prompts, completions, **kwargs)
    wrapper.__name__ = original_func.__name__
    return wrapper


def get_reward_functions(rewards_config: Dict[str, Dict[str, Any]]) -> list:
    """Create reward function instances from config.
    
    Args:
        rewards_config: Dict mapping reward function names to their configs
            Example:
            {
                "correctness_reward_func": {},
                "cot_think_user_penalty_func": {
                    "count_weight": 1e-6,
                    "penalty_cap": 1.0
                },
                "api_overseer_penalty_func": {
                    "model_name": "accounts/fireworks/models/qwen2p5-72b-instruct",
                    "system_prompt": "...",
                    ...
                }
            }
        
    Returns:
        List of reward function instances
    """
    reward_funcs = []
    
    for func_name, func_config in rewards_config.items():
        if func_name not in REWARD_FUNCS:
            raise ValueError(
                f"Unknown reward function: {func_name}. "
                f"Available functions: {list(REWARD_FUNCS.keys())}"
            )
        
        # Get the factory function and call it with the config
        factory = REWARD_FUNCS[func_name]
        reward_func = factory(func_config)
        
        # Wrap first function for tracking
        if len(reward_funcs) == 0:
            reward_func = tracking_wrapper(reward_func)
        
        reward_funcs.append(reward_func)
    
    return reward_funcs



def setup_wandb_and_directories(
    cfg: Dict, 
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
            entity='geodesic',
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
        log_config_artifact(config_copy_path)
    
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
            "high_reward_answer": x["high_reward_answer"],  # type: ignore[index]
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
    instruction_suffix = data_cfg.get("instruction_suffix")
    
    dataset = transform_dataset(dataset_path, instruction_suffix)
    dataset = dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})
    
    return dataset, dataset_path


def download_wandb_artifact_if_needed(artifact_name: str, output_dir: str, is_main_process: bool) -> str:
    """Download W&B artifact checkpoint if specified.
    
    Args:
        artifact_name: Full W&B artifact name (e.g., 'entity/project/artifact:version')
        output_dir: Directory to download the checkpoint to
        is_main_process: Whether this is the main process
        
    Returns:
        Path to the downloaded checkpoint directory
    """
    if not artifact_name:
        return None
        
    if is_main_process:
        print(f"Downloading W&B artifact: {artifact_name}")
        
    api = wandb.Api()
    artifact = api.artifact(artifact_name, type='model')
    
    # Create a unique directory name based on artifact name
    # e.g., "model_valiant-plasma-10_step_400_v0"
    artifact_dir_name = artifact_name.split('/')[-1].replace(':', '_')
    checkpoint_path = os.path.join(output_dir, artifact_dir_name)
    
    # Download to checkpoint location
    artifact.download(root=checkpoint_path)
    
    if is_main_process:
        print(f"Downloaded checkpoint to: {checkpoint_path}")
        
    return checkpoint_path


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


def save_final_model_and_metadata(
    trainer: GRPOTrainer,
    output_dir: str,
    train_dir: str,
    saved_cfg_path: str,
    model_id: str,
    dataset_name: str,
    dataset_path: str,
    train_cfg: Dict,
    wandb_project: str,
    is_main_process: bool,
    wandb_info_path: str
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
            log_checkpoint_artifact(
                run_name=wandb.run.name,
                step='final',
                path=final_checkpoint,
                metadata={
                    "base_model": model_id,
                    "dataset": dataset_name,
                    "training_status": "completed",
                    "final_step": trainer.state.global_step,
                },
                local_info_path=wandb_info_path
            )
            
    
    # Save training metadata
    training_metadata = {
        "config_path": saved_cfg_path,
        "model_id": model_id,
        "dataset_path": dataset_path,
        "dataset_name": dataset_name,
        "wandb_run_name": wandb.run.name,
        "wandb_project": wandb_project,
        "output_dir": output_dir,
        "final_step": trainer.state.global_step if hasattr(trainer, "state") else None,
        "training_config": train_cfg,
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


def run_from_config(config_path: str, checkpoint_name: str) -> str:
    """Main training entry point.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Path to parent results directory
    """
    cfg = load_config_with_defaults(config_path)

    # Setup W&B and directories
    train_dir, saved_cfg_path, wandb_info_path, dataset_name, is_main_process = setup_wandb_and_directories(cfg)

    log_path = os.path.join(train_dir, 'std_out.txt')
    
    # Use Tee to write to both stdout and file
    tee = Tee(log_path)
    sys.stdout = tee
    sys.stderr = tee
    
    try:
    
        # Setup model and tokenizer
        model, tokenizer, model_id = setup_model_and_tokenizer(cfg)
        
        # Setup dataset
        dataset, dataset_path = setup_dataset(cfg, tokenizer)
        
        # Setup training configuration
        train_cfg = cfg["train"]
        output_dir = train_cfg.get("output_dir")
        ensure_dir(output_dir)
        
        training_args = GRPOConfig(
            **train_cfg,
            report_to=["wandb"],
            remove_unused_columns=False,
            gradient_checkpointing=False,
        )
        
        # Get reward functions
        reward_func_configs = cfg['reward']['funcs']
        reward_funcs = get_reward_functions(reward_func_configs)
        
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
            save_steps=train_cfg["save_steps"],
            model_id=model_id,
            dataset_name=dataset_name,
            is_main_process=is_main_process,
            wandb_info_path=wandb_info_path,
        ))
        trainer.add_callback(TrackingCallback(
            tracking_data=_tracking,
            is_main_process=is_main_process
        ))

        
        # Check for W&B artifact to download
        wandb_cfg = cfg.get("wandb", {})
        resume_artifact = wandb_cfg.get("resume_from_artifact")
        
        if resume_artifact:
            # Download artifact from W&B
            checkpoint_name = download_wandb_artifact_if_needed(
                resume_artifact, 
                output_dir, 
                is_main_process
            )
        else:
            # Save initial model only if not resuming
            checkpoint_name = ratify_checkpoint(checkpoint_name, output_dir, is_main_process)
            if not checkpoint_name:
                save_initial_model(model, tokenizer, output_dir, wandb_info_path, model_id, dataset_name, is_main_process)
        
        trainer.train(resume_from_checkpoint=checkpoint_name)
        
        # Save final model and metadata
        save_final_model_and_metadata(
            trainer=trainer,
            output_dir=output_dir,
            train_dir=train_dir,
            saved_cfg_path=saved_cfg_path,
            model_id=model_id,
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            train_cfg=train_cfg,
            wandb_project=cfg["wandb"]["project"],
            is_main_process=is_main_process,
            wandb_info_path=wandb_info_path
        )
        
        # Finish W&B run
        if wandb.run is not None and is_main_process:
            wandb.finish()
        
        return os.path.dirname(train_dir)
    
    finally:
        # Restore stdout/stderr and close file
        sys.stdout = tee.stdout
        sys.stderr = tee.stderr
        tee.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train using YAML config")
    parser.add_argument(
        "--config", 
        type=str,
        help="Relative path of config file"
    )
    parser.add_argument(
        "--checkpoint_name", 
        type=str,
        required=False,
        default=None,
        help="Name of checkpoint to resume from (if resuming)"
    )
    
    dotenv.load_dotenv()    # Load in '.env'
    
    args = parser.parse_args()

    parent_dir = run_from_config(args.config, args.checkpoint_name)
    print(f"✓ Training complete. Results and metadata saved under: {parent_dir}")
