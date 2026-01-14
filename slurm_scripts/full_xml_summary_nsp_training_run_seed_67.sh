#!/bin/bash

# Ensure slurm_logs directory exists BEFORE sbatch runs
mkdir -p slurm_logs

# Common arguments
COMMON_ARGS=(
    --multirun
    'hydra.sweep.subdir=${hydra.job.num}'
    experiment=full_xml_tags/train
    +reward/overseer=hedged_add_info_summary
    reward.funcs.api_overseer_summary_penalty_func.penalty_weight=-0.05
    ++data.source_dataset_to_system_prompt.sycophancy_fact=null
    ++data.source_dataset_to_system_prompt.revealing_score=null
    ++data.source_dataset_to_system_prompt.world_affecting_reward=null
    ++data.source_dataset_to_system_prompt.code_selection=null
    ++wandb.entity=nathanielmitrani-cfis-upc
    train.seed=67
)

# Job 1: leave_out_war_full_xml
echo "Submitting job 1: leave_out_war_full_xml"
sbatch --job-name=summary_war slurm_scripts/train_dispatch.sbatch \
    "${COMMON_ARGS[@]}" \
    data=leave_out_war_full_xml \
    config_name=run_nsp_summary_war

# Job 2: leave_out_sycophancy_full_xml
echo "Submitting job 2: leave_out_sycophancy_full_xml"
sbatch --job-name=summary_syco slurm_scripts/train_dispatch.sbatch \
    "${COMMON_ARGS[@]}" \
    data=leave_out_sycophancy_full_xml \
    config_name=run_nsp_summary_sycophancy

# Job 3: leave_out_score_full_xml
echo "Submitting job 3: leave_out_score_full_xml"
sbatch --job-name=summary_score slurm_scripts/train_dispatch.sbatch \
    "${COMMON_ARGS[@]}" \
    data=leave_out_score_full_xml \
    config_name=run_nsp_summary_score

# Job 4: leave_out_code_full_xml
echo "Submitting job 4: leave_out_code_full_xml"
sbatch --job-name=summary_code slurm_scripts/train_dispatch.sbatch \
    "${COMMON_ARGS[@]}" \
    data=leave_out_code_full_xml \
    config_name=run_nsp_summary_code

echo "Jobs submitted. Check with: squeue -u $USER"