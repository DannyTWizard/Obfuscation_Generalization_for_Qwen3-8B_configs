#!/bin/bash
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --job-name=obf_train
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --array=1-4%4

# %A = array job ID, %a = task ID
# --array=1-4%4 means tasks 1-4, max 4 concurrent (adjust %N for throttle)

# Read this task's config from the config file
CONFIG_FILE="${CONFIG_FILE:-job_configs.txt}"
JOB_ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CONFIG_FILE")

# Initialize conda
source ~/anaconda3/etc/profile.d/conda.sh
conda activate obf_gen

# Fix for concurrent processes
export XDG_RUNTIME_DIR="/tmp/runtime-$USER-$SLURM_JOB_ID"
mkdir -p "$XDG_RUNTIME_DIR"

# Unique port per job - SLURM_JOB_ID is unique even for array tasks
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
export VLLM_CACHE_ROOT="/tmp/vllm_cache_$SLURM_JOB_ID"
export VLLM_TORCH_COMPILE_LEVEL=0

echo "============================================"
echo "Array Job: $SLURM_ARRAY_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Config: $JOB_ARGS"
echo "Common args: $@"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Started: $(date)"
echo "============================================"

# Combine common args ($@) with job-specific args
python -m src.train "$@" $JOB_ARGS
EXIT_CODE=$?

echo "============================================"
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================"

exit $EXIT_CODE