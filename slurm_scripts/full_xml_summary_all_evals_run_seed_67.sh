#!/bin/bash

# Ensure slurm_logs directory exists
mkdir -p slurm_logs

# All 6 evaluation datasets/experiments
EXPERIMENTS="full_xml_tags/eval_code_no_system_prompt_with_summary,\
full_xml_tags/eval_pp_sycophancy_no_sp_with_summary,\
full_xml_tags/eval_score_no_sp_with_summary,\
full_xml_tags/eval_sycophancy_medical_no_sp_with_summary,\
full_xml_tags/eval_sycophancy_no_system_prompt_with_summary,\
full_xml_tags/eval_war_no_sp_with_summary"

# All artifact steps to evaluate
STEPS="200,400,600,800,1000,1200,1400,1600"

COMMON_ARGS=(
    --multirun
    "hydra.sweep.subdir=\${hydra.job.num}"
    "experiment=$EXPERIMENTS"
    "artifact_step=$STEPS"
    ++wandb.entity=nathanielmitrani-cfis-upc
    config_name=eval
)

# NB: trained on seed 50, hence training group!

# Job 1: leave_out_war
echo "Submitting Evals for: leave_out_war"
sbatch --job-name=eval_war slurm_scripts/eval_dispatch.sbatch \
    "${COMMON_ARGS[@]}" \
    training_group=leave_out_war_full_xml_tags_seed_50 \
    training_run_name=run_nsp_ovs_hedged_add_info_pen_-0.05_data_leave_out_war_full_xml_ts_67

# Job 2: leave_out_sycophancy
echo "Submitting Evals for: leave_out_syco"
sbatch --job-name=eval_syco slurm_scripts/eval_dispatch.sbatch \
    "${COMMON_ARGS[@]}" \
    training_group=leave_out_sycophancy_full_xml_tags_seed_50 \
    training_run_name=run_nsp_summary_sycophancy

# Job 3: leave_out_score
echo "Submitting Evals for: leave_out_score"
sbatch --job-name=eval_score slurm_scripts/eval_dispatch.sbatch \
    "${COMMON_ARGS[@]}" \
    training_group=leave_out_score_add_new_full_xml_tags_seed_50 \
    training_run_name=run_nsp_summary_score_add_new

# Job 4: leave_out_code
echo "Submitting Evals for: leave_out_code"
sbatch --job-name=eval_code slurm_scripts/eval_dispatch.sbatch \
    "${COMMON_ARGS[@]}" \
    training_group=leave_out_code_full_xml_tags_seed_50 \
    training_run_name=run_nsp_summary_code

echo "All 4 model eval sweeps submitted. Check with: squeue -u $USER"