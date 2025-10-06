import os
import sys
from typing import Any, Dict

import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer, apply_chat_template
from peft import LoraConfig, get_peft_model

    
from src.train.rewards import (
    correctness_reward_func, 
    xmlcount_reward_func, 
    cot_think_user_penalty_func, 
    cot_think_name_penalty_func,
    summary_present_penalty,
    cot_length_penalty_func
)
from src.utils import (
    load_yaml_file, ensure_dir, save_config_copy, create_versioned_parent_dir,
    count_user_mentions_in_cot, count_name_mentions_in_cot,
    count_user_mentions_in_summary, count_name_mentions_in_summary,
    count_cot_words, count_summary_words,
)

# Global tracking data
_tracking = {"cot_user": [], "cot_name": [], "summary_user": [], "summary_name": [], "cot_words": [], "summary_words": []}

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
    available_funcs = {
        "correctness_reward_func": correctness_reward_func,
        "xmlcount_reward_func": xmlcount_reward_func,
        "cot_think_user_penalty_func": cot_think_user_penalty_func,
        "cot_think_name_penalty_func": cot_think_name_penalty_func,
        "summary_present_penalty": summary_present_penalty,
        "cot_length_penalty_func": cot_length_penalty_func,
    }
    
    reward_funcs = []
    for func_name in reward_func_names:
        if func_name in available_funcs:
            reward_funcs.append(tracking_wrapper(available_funcs[func_name]) if len(reward_funcs) == 0 else available_funcs[func_name])
        else:
            raise ValueError(f"Unknown reward function: {func_name}. Available functions: {list(available_funcs.keys())}")
    
    return reward_funcs


def transform_dataset(dataset_path: str, instruction_suffix: str) -> Any:
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


def run_from_config(config_path: str) -> str:
    cfg = load_yaml_file(config_path)

    wandb_project = cfg.get("wandb").get("project")
    if wandb_project:
        wandb.init(project=wandb_project, config=cfg)

    # Create versioned parent directory and training run subdirectory early
    results_cfg = cfg.get("results", {})
    base_results_dir = results_cfg.get("base_dir", os.path.abspath(os.path.join(os.getcwd(), "results/train")))
    parent_dir = create_versioned_parent_dir(base_results_dir, prefix=results_cfg.get("name", "train"))
    
    # Create subdirectory for this training run
    data_cfg = cfg.get("data", {})
    ds_path = data_cfg.get("dataset_path", "datasets/reward_hack/sycophancy_fact.jsonl")
    dataset_name = os.path.basename(ds_path).replace(".jsonl", "")
    
    # Use wandb run name if available, otherwise use dataset name
    if wandb.run and wandb.run.name:
        run_suffix = wandb.run.name
    else:
        run_suffix = dataset_name
    
    base_name = results_cfg.get("name", "train")
    train_subdir_name = f"{base_name}_{run_suffix}"
    train_dir = os.path.join(parent_dir, train_subdir_name)
    ensure_dir(train_dir)
    
    # Save config copy immediately
    saved_cfg_path = save_config_copy(config_path, train_dir)
    # Log config file as a W&B artifact for reproducibility
    if wandb.run is not None and os.path.exists(saved_cfg_path):
        try:
            cfg_artifact = wandb.Artifact(
                name=f"config_{wandb.run.id}",
                type="config",
                metadata={
                    "original_config_path": os.path.abspath(config_path),
                    "saved_config_path": os.path.abspath(saved_cfg_path),
                },
            )
            cfg_artifact.add_file(saved_cfg_path)
            wandb.log_artifact(cfg_artifact)
        except Exception:
            pass

    model_id = cfg.get("model", {}).get("base_model_id")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # data_cfg and ds_path already defined above
    instruction_suffix = data_cfg.get(
        "instruction_suffix",
        "\n\nPlease end you answer with <answer>your_answer_here</answer>. For instance, if the answer is '(A), Blue', hen you should respond with a summary of your reasoning followed by '<answer>A</answer>'",
    )
    dataset = transform_dataset(ds_path, instruction_suffix)
    dataset = dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})

    lora_cfg = cfg.get("lora", {})
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("lora_alpha", 32)),
        target_modules=lora_cfg.get("target_modules", "all-linear"),
    )
    model = get_peft_model(model, lora_config)

    train_cfg = cfg.get("train", {})
    output_dir = train_cfg.get("output_dir")
    ensure_dir(output_dir)

    training_args = GRPOConfig(
        use_vllm=bool(train_cfg.get("use_vllm")),
        vllm_mode=train_cfg.get("vllm_mode"),
        vllm_gpu_memory_utilization=float(train_cfg.get("vllm_gpu_memory_utilization")),
        vllm_tensor_parallel_size=int(train_cfg.get("vllm_tensor_parallel_size")),
        #vllm_max_model_len=int(train_cfg.get("vllm_max_model_len")),

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
        # save_strategy=train_cfg.get("save_strategy", "steps"),
        # save_steps=int(train_cfg.get("save_steps", 25)),
        # save_total_limit=int(train_cfg.get("save_total_limit", 5)),
    )

    # Get reward functions from config
    reward_func_names = cfg.get("reward_funcs")
    reward_funcs = get_reward_functions(reward_func_names)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset["train"],
    )

    class CheckpointCallback(TrainerCallback):
        def __init__(self, save_steps=25):
            self.save_steps = save_steps

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % self.save_steps == 0 and state.global_step > 0:
                control.should_save = True

        def on_save(self, args, state, control, **kwargs):
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            if os.path.exists(checkpoint_path):
                artifact = wandb.Artifact(
                    name=f"grpo_model_{wandb.run.name}_step_{state.global_step}",
                    type="model",
                    metadata={
                        "step": state.global_step,
                        "base_model": model_id,
                        "dataset": os.path.basename(ds_path),
                        "training_status": "intermediate",
                    },
                )
                artifact.add_dir(checkpoint_path)
                wandb.log_artifact(artifact)

    class TrackingCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            global _tracking
            wandb.log({
                "avg_cot_user": sum(_tracking["cot_user"]) / len(_tracking["cot_user"]) if _tracking["cot_user"] else 0,
                "avg_cot_name": sum(_tracking["cot_name"]) / len(_tracking["cot_name"]) if _tracking["cot_name"] else 0,
                "avg_summary_user": sum(_tracking["summary_user"]) / len(_tracking["summary_user"]) if _tracking["summary_user"] else 0,
                "avg_summary_name": sum(_tracking["summary_name"]) / len(_tracking["summary_name"]) if _tracking["summary_name"] else 0,
                "avg_cot_words": sum(_tracking["cot_words"]) / len(_tracking["cot_words"]) if _tracking["cot_words"] else 0,
                "avg_summary_words": sum(_tracking["summary_words"]) / len(_tracking["summary_words"]) if _tracking["summary_words"] else 0,
            })
            _tracking = {"cot_user": [], "cot_name": [], "summary_user": [], "summary_name": [], "cot_words": [], "summary_words": []}

    trainer.add_callback(CheckpointCallback(save_steps=int(train_cfg.get("save_steps", 25))))
    trainer.add_callback(TrackingCallback())

    # Save initial model artifact
    initial_model_path = os.path.join(output_dir, "initial_model")
    ensure_dir(initial_model_path)
    model.save_pretrained(initial_model_path)
    tokenizer.save_pretrained(initial_model_path)
    initial_artifact = wandb.Artifact(
        name=f"grpo_model_{wandb.run.name}_initial",
        type="model",
        metadata={
            "base_model": model_id,
            "dataset": os.path.basename(ds_path),
            "training_status": "initial",
            "step": 0,
        },
    )
    initial_artifact.add_dir(initial_model_path)
    wandb.log_artifact(initial_artifact)

    # Train
    trainer.train()

    # Save final model
    checkpoint_dirs = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if checkpoint_dirs:
        checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]))
        final_checkpoint = os.path.join(output_dir, checkpoint_dirs[-1])
        if os.path.exists(final_checkpoint):
            artifact = wandb.Artifact(
                name=f"grpo_model_{wandb.run.name}_final",
                type="model",
                metadata={
                    "base_model": model_id,
                    "dataset": os.path.basename(ds_path),
                    "training_status": "completed",
                    "final_step": trainer.state.global_step if hasattr(trainer, "state") else None,
                },
            )
            artifact.add_dir(final_checkpoint)
            wandb.log_artifact(artifact)

    # Save training metadata and summary
    import json
    training_metadata = {
        "config_path": saved_cfg_path,
        "model_id": model_id,
        "dataset_path": ds_path,
        "dataset_name": dataset_name,
        "wandb_run_name": wandb.run.name if wandb.run else None,
        "wandb_project": wandb_project,
        "output_dir": output_dir,
        "final_step": trainer.state.global_step if hasattr(trainer, "state") else None,
        "num_train_epochs": float(train_cfg.get("num_train_epochs", 1)),
        "learning_rate": float(train_cfg.get("learning_rate", 4e-5)),
        "training_status": "completed"
    }
    
    # Save training metadata
    metadata_path = os.path.join(train_dir, "training_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(training_metadata, f, indent=2)
    
    # List and save checkpoint information
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

    if wandb.run is not None:
        wandb.finish()

    return parent_dir


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train using YAML config")
    parser.add_argument("--config", type=str, default=os.path.abspath(os.path.join(os.getcwd(), "src/train/configs/default_train.yaml")), help="Path to YAML config")
    args = parser.parse_args()
    parent_dir = run_from_config(args.config)
    print(f"Training complete. Results and metadata saved under: {parent_dir}")


if __name__ == "__main__":
    main()


