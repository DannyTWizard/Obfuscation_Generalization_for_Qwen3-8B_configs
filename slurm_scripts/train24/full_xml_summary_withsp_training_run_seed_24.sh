#!/bin/bash

mkdir -p slurm_logs

# Submit the array job with common args
sbatch --export=CONFIG_FILE=slurm_scripts/train24/full_xml_summary_withsp_training_run_seed_24.txt \
    slurm_scripts/train_dispatch.sbatch \
    --multirun \
    'hydra.sweep.subdir=${hydra.job.num}' \
    experiment=full_xml_tags/train \
    +reward/overseer=hedged_add_info_summary \
    reward.funcs.api_overseer_summary_penalty_func.penalty_weight=-0.05 \
    ++wandb.entity=nathanielmitrani-cfis-upc \
    train.seed=24

echo "Array job submitted. Check with: squeue -u $USER"