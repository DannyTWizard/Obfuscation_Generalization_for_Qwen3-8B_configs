#!/bin/bash
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
conda activate obf_gen
cd ~/repos/Obfuscation_Generalization

accelerate launch --multi_gpu --num_processes 4 \
    -m src.train --config "$CONFIG_PATH"

echo ""
echo "End time: $(date)"
