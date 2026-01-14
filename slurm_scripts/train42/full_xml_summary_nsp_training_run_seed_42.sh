#!/bin/bash

mkdir -p slurm_logs

sbatch --export=CONFIG_FILE=slurm_scripts/train42/full_xml_summary_nsp_training_run_seed_42.txt \
    slurm_scripts/train_dispatch.sbatch \
    --multirun \
    'hydra.sweep.subdir=${hydra.job.num}' \
    experiment=full_xml_tags/train \
    +reward/overseer=hedged_add_info_summary \
    reward.funcs.api_overseer_summary_penalty_func.penalty_weight=-0.05 \
    ++data.source_dataset_to_system_prompt.sycophancy_fact=null \
    ++data.source_dataset_to_system_prompt.revealing_score=null \
    ++data.source_dataset_to_system_prompt.world_affecting_reward=null \
    ++data.source_dataset_to_system_prompt.code_selection=null \
    ++wandb.entity=nathanielmitrani-cfis-upc \
    train.seed=42

echo "Array job submitted. Check with: squeue -u $USER"