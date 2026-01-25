#!/bin/bash
# ==============================================================================
# SLURM Evaluation Launcher - In-Distribution + Base Model Evals
# ==============================================================================
# Runs:
# 1. Base model eval on all 6 datasets (no artifact)
# 2. In-distribution evals at step 3800 for:
#    - Baseline runs (puria-radmard)
#    - Summary runs (puria-radmard)  
#    - Normal runs (nathanielmitrani-cfis-upc)
# ==============================================================================

set -e

mkdir -p slurm_logs

CONFIG_FILE="slurm_scripts/bar_charts/eval_ind_and_base.txt"

# Count lines
NUM_JOBS=$(wc -l < "$CONFIG_FILE")

echo "============================================"
echo "In-Distribution + Base Model Eval Launcher"
echo "============================================"
echo "Config file: $CONFIG_FILE"
echo "Total jobs: $NUM_JOBS"
echo ""
echo "Job breakdown:"
echo "  - 1 base model eval (all 6 datasets)"
echo "  - 8 baseline in-dist (4 folds × 2 seeds)"
echo "  - 8 summary in-dist (4 folds × 2 seeds)"
echo "  - 8 normal in-dist (4 folds × 2 seeds)"
echo "============================================"
echo ""

# Submit the array job with common args
sbatch --array=1-${NUM_JOBS}%4 \
    --export=CONFIG_FILE="$CONFIG_FILE" \
    slurm_scripts/eval_dispatch.sbatch \
    --multirun \
    'hydra.sweep.subdir=${hydra.job.num}' \
    config_name=eval_ind

echo ""
echo "Array job submitted. Check with: squeue -u $USER"