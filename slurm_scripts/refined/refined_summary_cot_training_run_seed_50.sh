#!/bin/bash

mkdir -p slurm_logs

CONFIG_FILE="slurm_scripts/refined/refined_cot_training_run_seed_50.txt"
NUM_JOBS=$(wc -l < "$CONFIG_FILE")

# Submit the array job with common args
sbatch --array=1-${NUM_JOBS}%4 \
    --export=CONFIG_FILE="$CONFIG_FILE" \
    slurm_scripts/train_dispatch.sbatch \
    --multirun \
    'hydra.sweep.subdir=${hydra.job.num}' \
    experiment=refined/train \
    +reward/overseer=refined_summary \
    reward.funcs.api_overseer_penalty_func.penalty_weight=-0.05 \
    ++wandb.entity=nathanielmitrani-cfis-upc \
    train.seed=50

echo "Array job submitted. Check with: squeue -u $USER"
