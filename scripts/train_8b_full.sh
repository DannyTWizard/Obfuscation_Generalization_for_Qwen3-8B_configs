#!/bin/bash
#SBATCH --job-name=train-8b-full
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --partition=gpu

# Full fine-tune of Qwen3-8B on 8 GPUs
#
# Usage:
#   # Basic (no overseer penalty)
#   sbatch scripts/train_8b_full.sh experiment=full_xml_tags/train
#
#   # With overseer penalty
#   sbatch scripts/train_8b_full.sh experiment=full_xml_tags/train +reward/overseer=standard
#
#   # Target specific GPU type (e.g. H200)
#   sbatch --constraint=h200 scripts/train_8b_full.sh experiment=full_xml_tags/train
#
# Setup:
#   pip install -r requirements.txt
#
# Environment variables (optional):
#   OBF_GEN_VENV       - Path to venv activate script (e.g. ~/venv/bin/activate)
#   OBF_GEN_WORKDIR    - Working directory (default: script's repo root)

HYDRA_OVERRIDES="$@"

if [ -z "$HYDRA_OVERRIDES" ]; then
    echo "Error: At least one Hydra override is required (e.g., experiment=full_xml_tags/train)"
    echo "Usage: sbatch scripts/train_8b_full.sh <hydra_overrides>"
    exit 1
fi

# Defaults — workdir is the repo root (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKDIR="${OBF_GEN_WORKDIR:-$(dirname "$SCRIPT_DIR")}"

echo "============================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODENAME"
echo "GPUs:      $CUDA_VISIBLE_DEVICES"
echo "Overrides: $HYDRA_OVERRIDES"
echo "Workdir:   $WORKDIR"
echo "Start:     $(date)"
echo "============================================"
echo ""

# Print GPU info
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Activate Python environment if specified
if [ -n "$OBF_GEN_VENV" ]; then
    source "$OBF_GEN_VENV"
fi

cd "$WORKDIR"

mkdir -p logs

NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "Detected $NUM_GPUS GPUs"

accelerate launch --multi_gpu --num_processes $NUM_GPUS \
    -m src.train \
    train=grpo_8b \
    model=qwen3_8b \
    lora=none \
    $HYDRA_OVERRIDES

echo ""
echo "End time: $(date)"
