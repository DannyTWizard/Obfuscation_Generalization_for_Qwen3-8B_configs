#!/bin/bash

mkdir -p slurm_logs

CONFIG_FILE="slurm_scripts/score_only/withsp.txt"
NUM_JOBS=$(wc -l < "$CONFIG_FILE")

# Submit the array job with common args
sbatch --array=1-${NUM_JOBS}%4 \
    --export=CONFIG_FILE="$CONFIG_FILE" \
    slurm_scripts/train_dispatch.sbatch \
    --multirun \
    data=only_score_full_xml \
    'hydra.sweep.subdir=${hydra.job.num}' \
    experiment=full_xml_tags/train \
    ++wandb.entity=nathanielmitrani-cfis-upc \

echo "Array job submitted. Check with: squeue -u $USER"
