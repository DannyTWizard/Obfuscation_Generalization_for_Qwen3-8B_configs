#!/usr/bin/env python3
"""
SLURM script generator for ML experiment pipelines.

Usage:
    python -m src.scripts.slurm <experiment_dir>
    python -m src.scripts.slurm <experiment_dir> --conda-env my_env --workdir ~/my_project
"""

import argparse
import os
import stat
from pathlib import Path
from typing import NamedTuple

import yaml


class ExperimentConfig(NamedTuple):
    """Parsed experiment configuration."""
    experiment_dir: Path
    training_group: str
    train_configs: list[tuple[str, Path]]  # (config_name, yaml_path)
    eval_configs: list[Path]


def parse_yaml(path: Path) -> dict:
    """Load a YAML file."""
    with open(path) as f:
        return yaml.safe_load(f)


def discover_experiment(experiment_dir: Path) -> ExperimentConfig:
    """Discover all config files in an experiment directory."""
    experiment_dir = experiment_dir.resolve()
    
    # Find data.yaml
    data_yaml = experiment_dir / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"No data.yaml found in {experiment_dir}")
    
    data_config = parse_yaml(data_yaml)
    result_name = data_config["result_name"]
    seed = data_config["seed"]
    training_group = f"{result_name}_seed_{seed}"
    
    # Find all train_*.yaml files
    train_configs = []
    for path in sorted(experiment_dir.glob("train_*.yaml")):
        config = parse_yaml(path)
        config_name = config["config_name"]
        train_configs.append((config_name, path))
    
    if not train_configs:
        raise FileNotFoundError(f"No train_*.yaml files found in {experiment_dir}")
    
    # Find all eval_*.yaml files
    eval_configs = sorted(experiment_dir.glob("eval_*.yaml"))
    
    if not eval_configs:
        raise FileNotFoundError(f"No eval_*.yaml files found in {experiment_dir}")
    
    return ExperimentConfig(
        experiment_dir=experiment_dir,
        training_group=training_group,
        train_configs=train_configs,
        eval_configs=eval_configs,
    )


def generate_train_script(conda_env: str, workdir: str) -> str:
    """Generate the single training job SLURM script."""
    return f'''#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G

# Usage: sbatch --job-name=<n> _train.sh <config_path>

CONFIG_PATH=$1

if [ -z "$CONFIG_PATH" ]; then
    echo "Error: CONFIG_PATH is required"
    echo "Usage: sbatch _train.sh <config_path>"
    exit 1
fi

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Config: $CONFIG_PATH"
echo "Start time: $(date)"
echo ""

source ~/anaconda3/etc/profile.d/conda.sh
conda activate {conda_env}
cd {workdir}

accelerate launch --multi_gpu --num_processes 4 \\
    -m src.train --config "$CONFIG_PATH"

echo ""
echo "End time: $(date)"
'''


def generate_eval_script(conda_env: str, workdir: str) -> str:
    """Generate the single eval job SLURM script."""
    return f'''#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

# Usage: sbatch _eval.sh <training_group> <training_run_name> <step> <config_path>

TRAINING_GROUP=$1
TRAINING_RUN_NAME=$2
ARTIFACT_STEP=$3
CONFIG_PATH=$4

if [ -z "$TRAINING_GROUP" ] || [ -z "$TRAINING_RUN_NAME" ] || [ -z "$ARTIFACT_STEP" ] || [ -z "$CONFIG_PATH" ]; then
    echo "Error: Missing required arguments"
    echo "Usage: sbatch _eval.sh <training_group> <training_run_name> <step> <config_path>"
    exit 1
fi

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Training Group: $TRAINING_GROUP"
echo "Training Run: $TRAINING_RUN_NAME"
echo "Step: $ARTIFACT_STEP"
echo "Config: $CONFIG_PATH"
echo "Start time: $(date)"
echo ""

source ~/anaconda3/etc/profile.d/conda.sh
conda activate {conda_env}
cd {workdir}

python -m src.eval \\
    --config "$CONFIG_PATH" \\
    --training_group "$TRAINING_GROUP" \\
    --training_run_name "$TRAINING_RUN_NAME" \\
    --artifact_step "$ARTIFACT_STEP"

echo ""
echo "End time: $(date)"
'''


def generate_train_dispatcher(config: ExperimentConfig) -> str:
    """Generate dispatcher script that submits all training jobs."""
    
    submit_commands = []
    for config_name, yaml_path in config.train_configs:
        submit_commands.append(f'''
echo "Submitting training job: {config_name}"
JOB_ID=$(sbatch --parsable --job-name="train-{config_name}" "$SCRIPT_DIR/_train.sh" "{yaml_path}")
echo "  Job ID: $JOB_ID"
SUBMITTED_JOBS+=($JOB_ID)''')
    
    submits = "\n".join(submit_commands)
    
    return f'''#!/bin/bash
# =============================================================================
# Training Dispatcher
# Submits all training jobs for: {config.experiment_dir.name}
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
SUBMITTED_JOBS=()

echo "=========================================="
echo "Submitting Training Jobs"
echo "Experiment: {config.experiment_dir.name}"
echo "=========================================="
echo ""
{submits}

echo ""
echo "=========================================="
echo "All training jobs submitted!"
echo "=========================================="
echo "Submitted job IDs: ${{SUBMITTED_JOBS[*]}}"
echo ""
echo "Monitor with: squeue -u $USER"
echo "Cancel all with: scancel ${{SUBMITTED_JOBS[*]}}"
'''


def generate_eval_dispatcher(
    config: ExperimentConfig,
    config_name: str,
    conda_env: str,
    workdir: str,
) -> str:
    """Generate dispatcher script that submits all eval jobs for a specific training run."""
    
    eval_config_paths = " \\\n        ".join(f'"{p}"' for p in config.eval_configs)
    
    return f'''#!/bin/bash
# =============================================================================
# Eval Dispatcher for: {config_name}
# Discovers checkpoints and submits eval jobs for all configs
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"

TRAINING_GROUP="{config.training_group}"
TRAINING_RUN_NAME="{config_name}"

EVAL_CONFIGS=(
    {eval_config_paths}
)

echo "=========================================="
echo "Eval Dispatcher"
echo "Training Group: $TRAINING_GROUP"
echo "Training Run: $TRAINING_RUN_NAME"
echo "Eval Configs: ${{#EVAL_CONFIGS[@]}}"
echo "=========================================="
echo ""

# Activate environment to run checkpoint discovery
source ~/anaconda3/etc/profile.d/conda.sh
conda activate {conda_env}
cd {workdir}

# Discover checkpoint steps
echo "Discovering checkpoints..."
STEPS=$(python -m src.scripts.list_artifact_steps \\
    --training_group "$TRAINING_GROUP" \\
    --training_run_name "$TRAINING_RUN_NAME")

if [ -z "$STEPS" ]; then
    echo "Error: No checkpoint steps found!"
    exit 1
fi

STEP_ARRAY=($STEPS)
echo "Found ${{#STEP_ARRAY[@]}} checkpoints: $STEPS"
echo ""

# Submit eval jobs
SUBMITTED_JOBS=()
TOTAL_JOBS=0

for STEP in $STEPS; do
    for CONFIG_PATH in "${{EVAL_CONFIGS[@]}}"; do
        CONFIG_NAME=$(basename "$CONFIG_PATH" .yaml)
        JOB_NAME="eval-{config_name}-${{CONFIG_NAME}}-step${{STEP}}"
        
        echo "Submitting: $JOB_NAME"
        JOB_ID=$(sbatch --parsable --job-name="$JOB_NAME" \\
            "$SCRIPT_DIR/_eval.sh" \\
            "$TRAINING_GROUP" \\
            "$TRAINING_RUN_NAME" \\
            "$STEP" \\
            "$CONFIG_PATH")
        echo "  Job ID: $JOB_ID"
        SUBMITTED_JOBS+=($JOB_ID)
        ((TOTAL_JOBS++))
    done
done

echo ""
echo "=========================================="
echo "All eval jobs submitted!"
echo "=========================================="
echo "Total jobs: $TOTAL_JOBS"
echo "Submitted job IDs: ${{SUBMITTED_JOBS[*]}}"
echo ""
echo "Monitor with: squeue -u $USER"
'''


def generate_scripts(experiment_dir: Path, conda_env: str, workdir: str) -> None:
    """Generate all SLURM scripts for an experiment."""
    
    config = discover_experiment(experiment_dir)
    
    # Create output directory
    slurm_dir = config.experiment_dir / "slurm_scripts"
    slurm_dir.mkdir(exist_ok=True)
    
    # Generate base scripts
    scripts = {
        "_train.sh": generate_train_script(conda_env, workdir),
        "_eval.sh": generate_eval_script(conda_env, workdir),
        "run_all_training.sh": generate_train_dispatcher(config),
    }
    
    # Generate eval dispatchers for each training config
    for config_name, _ in config.train_configs:
        script_name = f"run_evals_{config_name}.sh"
        scripts[script_name] = generate_eval_dispatcher(config, config_name, conda_env, workdir)
    
    # Write scripts
    for name, content in scripts.items():
        path = slurm_dir / name
        path.write_text(content)
        # Make executable
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        print(f"  Created: {path}")
    
    # Print summary
    print()
    print("=" * 60)
    print("Generated SLURM scripts")
    print("=" * 60)
    print()
    print(f"Training group: {config.training_group}")
    print(f"Training configs: {len(config.train_configs)}")
    for name, path in config.train_configs:
        print(f"  - {name}")
    print(f"Eval configs: {len(config.eval_configs)}")
    for path in config.eval_configs:
        print(f"  - {path.name}")
    print()
    print("Workflow:")
    print(f"  1. cd {slurm_dir}")
    print(f"  2. ./run_all_training.sh")
    print(f"  3. Wait for training, check results")
    for config_name, _ in config.train_configs:
        print(f"  4. ./run_evals_{config_name}.sh  # if {config_name} looks good")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Generate SLURM scripts for ML experiment pipelines"
    )
    parser.add_argument(
        "experiment_dir",
        type=Path,
        help="Path to experiment config directory",
    )
    parser.add_argument(
        "--conda-env",
        type=str,
        default="obf_gen",
        help="Conda environment name (default: obf_gen)",
    )
    parser.add_argument(
        "--workdir",
        type=str,
        default="~/repos/Obfuscation_Generalization",
        help="Working directory (default: ~/repos/Obfuscation_Generalization)",
    )
    
    args = parser.parse_args()

    
    if not args.experiment_dir.exists():
        print(f"Error: Directory not found: {args.experiment_dir}")
        return 1
    
    print(f"Generating SLURM scripts for: {args.experiment_dir}")
    print(f"Conda env: {args.conda_env}")
    print(f"Working directory: {args.workdir}")
    print()
    
    generate_scripts(args.experiment_dir, args.conda_env, args.workdir)
    
    return 0


if __name__ == "__main__":
    exit(main())