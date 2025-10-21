import os, json, argparse, wandb, sys
from typing import Any, Dict

from trl import GRPOConfig, GRPOTrainer

import dotenv

from src.train.rewards import REWARD_FUNCS
from src.utils.config import load_config_with_defaults, ensure_dir
from src.utils.parse import count_user_mentions_in_cot, count_name_mentions_in_cot, count_user_mentions_in_summary, count_name_mentions_in_summary, count_cot_words, count_summary_words
from src.utils.wandb_logging import log_checkpoint_artifact, save_initial_model
from src.utils.setup import ratify_checkpoint, setup_wandb_and_directories, setup_model_and_tokenizer, setup_dataset
from src.utils.callbacks import CheckpointCallback, TrackingCallback


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
        "wandb_run_name": wandb.run.name if wandb.run else None,
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
    train_dir, saved_cfg_path, wandb_info_path, dataset_name, is_main_process = setup_wandb_and_directories(
        cfg, config_path
    )

    log_path = os.path.join(train_dir, 'std_out.txt')
    
    # FIXME: infos, warnings, wandb logs, exceptions, and multithreaded outputs don't get logged here
    with open(log_path, 'w') as sys.stdout:
    
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
        reward_func_names = cfg['reward']['funcs']
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
            is_main_process=is_main_process
        )
        
        # Finish W&B run
        if wandb.run is not None and is_main_process:
            wandb.finish()
        
        return os.path.dirname(train_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train using YAML config")
    parser.add_argument(
        "config", 
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
