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
    
    # Find all train_*.yaml files first (we need them either way)
    train_configs = []
    for path in sorted(experiment_dir.glob("train_*.yaml")):
        config = parse_yaml(path)
        config_name = config["config_name"]
        train_configs.append((config_name, path))
    
    if not train_configs:
        raise FileNotFoundError(f"No train_*.yaml files found in {experiment_dir}")
    
    # Try to get training_group from data.yaml, otherwise extract from train yaml
    data_yaml = experiment_dir / "data.yaml"
    if data_yaml.exists():
        data_config = parse_yaml(data_yaml)
        result_name = data_config["result_name"]
        seed = data_config["seed"]
        training_group = f"{result_name}_seed_{seed}"
    else:
        # Extract from first train yaml's data.hf_dataset field
        # Format: geodesic-puria/obf_gen_<training_group>
        first_train_config = parse_yaml(train_configs[0][1])
        hf_dataset = first_train_config["data"]["hf_dataset"]
        # Extract everything after "obf_gen_"
        prefix = "obf_gen_"
        idx = hf_dataset.find(prefix)
        if idx == -1:
            raise ValueError(f"Could not parse training_group from hf_dataset: {hf_dataset}")
        training_group = hf_dataset[idx + len(prefix):]
    
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


def generate_train_dispatcher(config: ExperimentConfig, workdir: str) -> str:
    """Generate dispatcher script that submits all training jobs."""
    
    submit_commands = []
    for config_name, yaml_path in config.train_configs:
        submit_commands.append(f'''
echo "Submitting training job: {config_name}"
JOB_ID=$(sbatch --parsable --job-name="train-{config_name}" "$TRAIN_SCRIPT" "{yaml_path}")
echo "  Job ID: $JOB_ID"
SUBMITTED_JOBS+=($JOB_ID)''')
    
    submits = "\n".join(submit_commands)
    
    return f'''#!/bin/bash
# =============================================================================
# Training Dispatcher
# Submits all training jobs for: {config.experiment_dir.name}
# =============================================================================

set -e

WORKDIR="${{OBF_GEN_WORKDIR:-$HOME/repos/Obfuscation_Generalization}}"
TRAIN_SCRIPT="$WORKDIR/scripts/train_dispatch.sh"

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "Error: Train script not found at $TRAIN_SCRIPT"
    exit 1
fi

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
    workdir: str,
) -> str:
    """Generate dispatcher script that submits all eval jobs for a specific training run."""
    
    eval_config_paths = " \\\n        ".join(f'"{p}"' for p in config.eval_configs)
    
    return f'''#!/bin/bash
# =============================================================================
# Eval Dispatcher for: {config_name}
# Discovers checkpoints and submits eval jobs for all configs
#
# Usage:
#   ./run_evals_{config_name}.sh              # all checkpoints
#   ./run_evals_{config_name}.sh 100 200 500  # only specified steps
# =============================================================================

set -e

WORKDIR="${{OBF_GEN_WORKDIR:-$HOME/repos/Obfuscation_Generalization}}"
CONDA_ENV="${{OBF_GEN_CONDA_ENV:-obf_gen}}"
EVAL_SCRIPT="$WORKDIR/scripts/eval_dispatch.sh"

if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "Error: Eval script not found at $EVAL_SCRIPT"
    exit 1
fi

TRAINING_GROUP="{config.training_group}"
TRAINING_RUN_NAME="{config_name}"

EVAL_CONFIGS=(
    {eval_config_paths}
)

# Capture requested steps (if any)
REQUESTED_STEPS=("$@")

echo "=========================================="
echo "Eval Dispatcher"
echo "Training Group: $TRAINING_GROUP"
echo "Training Run: $TRAINING_RUN_NAME"
echo "Eval Configs: ${{#EVAL_CONFIGS[@]}}"
if [ ${{#REQUESTED_STEPS[@]}} -gt 0 ]; then
    echo "Requested Steps: ${{REQUESTED_STEPS[*]}}"
fi
echo "=========================================="
echo ""

# Activate environment to run checkpoint discovery
source ~/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
cd "$WORKDIR"

# Discover checkpoint steps
echo "Discovering checkpoints..."
AVAILABLE_STEPS=$(python -m src.scripts.list_artifact_steps \\
    --training_group "$TRAINING_GROUP" \\
    --training_run_name "$TRAINING_RUN_NAME")

if [ -z "$AVAILABLE_STEPS" ]; then
    echo "Error: No checkpoint steps found!"
    exit 1
fi

AVAILABLE_ARRAY=($AVAILABLE_STEPS)
echo "Found ${{#AVAILABLE_ARRAY[@]}} checkpoints: $AVAILABLE_STEPS"
echo ""

# Determine which steps to use
if [ ${{#REQUESTED_STEPS[@]}} -gt 0 ]; then
    # Validate requested steps
    for REQ_STEP in "${{REQUESTED_STEPS[@]}}"; do
        FOUND=0
        for AVAIL_STEP in "${{AVAILABLE_ARRAY[@]}}"; do
            if [ "$REQ_STEP" == "$AVAIL_STEP" ]; then
                FOUND=1
                break
            fi
        done
        if [ $FOUND -eq 0 ]; then
            echo "Error: Requested step '$REQ_STEP' not found in available checkpoints!"
            echo "Available: $AVAILABLE_STEPS"
            exit 1
        fi
    done
    STEPS_TO_RUN=("${{REQUESTED_STEPS[@]}}")
    echo "Using requested steps: ${{STEPS_TO_RUN[*]}}"
else
    STEPS_TO_RUN=("${{AVAILABLE_ARRAY[@]}}")
    echo "Using all available steps: ${{STEPS_TO_RUN[*]}}"
fi
echo ""

# Submit eval jobs
SUBMITTED_JOBS=()
TOTAL_JOBS=0

for STEP in "${{STEPS_TO_RUN[@]}}"; do
    for CONFIG_PATH in "${{EVAL_CONFIGS[@]}}"; do
        CONFIG_NAME=$(basename "$CONFIG_PATH" .yaml)
        JOB_NAME="eval-{config_name}-${{CONFIG_NAME}}-step${{STEP}}"
        
        echo "Submitting: $JOB_NAME"
        JOB_ID=$(sbatch --parsable --job-name="$JOB_NAME" \\
            "$EVAL_SCRIPT" \\
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


def generate_scripts(experiment_dir: Path, workdir: str) -> None:
    """Generate all SLURM scripts for an experiment."""
    
    config = discover_experiment(experiment_dir)
    
    # Create output directory
    slurm_dir = config.experiment_dir / "slurm_scripts"
    slurm_dir.mkdir(exist_ok=True)
    
    # Generate dispatcher scripts
    scripts = {
        "run_all_training.sh": generate_train_dispatcher(config, workdir),
    }
    
    # Generate eval dispatchers for each training config
    for config_name, _ in config.train_configs:
        script_name = f"run_evals_{config_name}.sh"
        scripts[script_name] = generate_eval_dispatcher(config, config_name, workdir)
    
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
    print(f"Working directory: {args.workdir}")
    print()
    
    generate_scripts(args.experiment_dir, args.workdir)
    
    return 0


if __name__ == "__main__":
    exit(main())