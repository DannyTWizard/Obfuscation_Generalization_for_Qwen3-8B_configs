#!/bin/bash
# ==============================================================================
# SLURM Evaluation Launcher - Fix Medical Sycophancy + Score Evals (Summary)
# ==============================================================================
# Fixes broken evals for summary runs:
# - eval_sycophancy_medical_with_summary: all 4 folds × 4 seeds × all steps
# - eval_score_with_summary: leave_out_score fold × 4 seeds × all steps
# ==============================================================================

set -e

mkdir -p slurm_logs

CONFIG_FILE="slurm_scripts/refined2_cleanup/eval_fix.txt"

# Count lines
NUM_JOBS=$(wc -l < "$CONFIG_FILE")

echo "============================================"
echo "Summary Medical + Score Eval Fix Launcher"
echo "============================================"
echo "Config file: $CONFIG_FILE"
echo "Total jobs: $NUM_JOBS"
echo ""
echo "Job breakdown:"
echo "  - 16 medical_sycophancy (4 folds × 4 seeds)"
echo "  - 4 score (leave_out_score × 4 seeds)"
echo "Each job runs 20 checkpoints via multirun"
echo "============================================"
echo ""

# Submit the array job with common args
sbatch --array=1-${NUM_JOBS}%4 \
    --export=CONFIG_FILE="$CONFIG_FILE" \
    slurm_scripts/eval_dispatch.sbatch \
    --multirun \
    'hydra.sweep.subdir=${hydra.job.num}' \
    config_name=eval

echo ""
echo "Array job submitted. Check with: squeue -u $USER"