#!/bin/bash
#SBATCH --output=slurm_logs/eval_%x_%j.out
#SBATCH --error=slurm_logs/eval_%x_%j.err
#SBATCH --job-name=obf_train
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

# Initialize conda properly for non-interactive shell
source ~/anaconda3/etc/profile.d/conda.sh
conda activate obf_gen

# Fix for concurrent processes - each job gets its own runtime dir
export XDG_RUNTIME_DIR="/tmp/runtime-$USER-$SLURM_JOB_ID"
mkdir -p "$XDG_RUNTIME_DIR"

# Unique port per job to avoid conflicts
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

echo "============================================"
echo "Slurm Job: $SLURM_JOB_ID"
echo "Args: $@"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "MASTER_PORT: $MASTER_PORT"
echo "Started: $(date)"
echo "============================================"

python -m src.eval "$@"
EXIT_CODE=$?

echo "============================================"
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================"

exit $EXIT_CODE