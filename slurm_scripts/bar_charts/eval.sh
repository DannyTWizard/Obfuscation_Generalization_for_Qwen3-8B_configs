#!/bin/bash
# ==============================================================================
# SLURM Evaluation Launcher - In-Distribution + Base Model (Limited Set)
# ==============================================================================
# Limited set for quick testing:
# - Base model: 6 evals (with_summary)
# - Baseline: seed 24 only, 4 folds × 3 in-dist (non-summary)
# - Summary: seeds 24,42,50, 4 folds × 3 in-dist (with_summary)
# ==============================================================================

set -e

mkdir -p slurm_logs

CONFIG_FILE="slurm_scripts/bar_charts/eval_ind_and_base.txt"

# Count lines
NUM_JOBS=$(wc -l < "$CONFIG_FILE")

echo "============================================"
echo "In-Dist + Base Model Eval (Limited Set)"
echo "============================================"
echo "Config file: $CONFIG_FILE"
echo "Total jobs: $NUM_JOBS"
echo ""
echo "Job breakdown:"
echo "  - 1 base model (6 evals via multirun)"
echo "  - 4 baseline in-dist (4 folds, seed 24, 3 evals each via multirun)"
echo "  - 12 summary in-dist (4 folds × 3 seeds, 3 evals each via multirun)"
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