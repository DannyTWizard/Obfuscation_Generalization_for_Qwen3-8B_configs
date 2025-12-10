#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G

# Usage: sbatch --job-name=<name> train_dispatch.sh <config_path>
#
# Environment variables (optional):
#   OBF_GEN_CONDA_ENV  - Conda environment (default: obf_gen)
#   OBF_GEN_WORKDIR    - Working directory (default: ~/repos/Obfuscation_Generalization)

CONFIG_PATH=$1

if [ -z "$CONFIG_PATH" ]; then
    echo "Error: CONFIG_PATH is required"
    echo "Usage: sbatch train_dispatch.sh <config_path>"
    exit 1
fi

# Defaults
CONDA_ENV="${OBF_GEN_CONDA_ENV:-obf_gen}"
WORKDIR="${OBF_GEN_WORKDIR:-$HOME/repos/Obfuscation_Generalization}"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Config: $CONFIG_PATH"
echo "Conda Env: $CONDA_ENV"
echo "Workdir: $WORKDIR"
echo "Start time: $(date)"
echo ""

source ~/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
cd "$WORKDIR"

NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")

accelerate launch --multi_gpu --num_processes $NUM_GPUS \
    -m src.train --config "$CONFIG_PATH"

echo ""
echo "End time: $(date)"