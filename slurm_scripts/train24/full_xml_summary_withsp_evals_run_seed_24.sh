#!/bin/bash

mkdir -p slurm_logs

# All 6 evaluation datasets/experiments
EXPERIMENTS="full_xml_tags/eval_code_no_system_prompt_with_summary,\
full_xml_tags/eval_pp_sycophancy_no_sp_with_summary,\
full_xml_tags/eval_score_no_sp_with_summary,\
full_xml_tags/eval_sycophancy_medical_no_sp_with_summary,\
full_xml_tags/eval_sycophancy_no_system_prompt_with_summary,\
full_xml_tags/eval_war_no_sp_with_summary"

# All artifact steps to evaluate
STEPS="200,400,600,800,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000,3200"

# Submit the array job with common args
sbatch --export=CONFIG_FILE=slurm_scripts/train24/full_xml_summary_withsp_evals_run_seed_24.txt \
    slurm_scripts/eval_dispatch.sbatch \
    --multirun \
    'hydra.sweep.subdir=${hydra.job.num}' \
    "experiment=$EXPERIMENTS" \
    "artifact_step=$STEPS" \
    ++wandb.entity=nathanielmitrani-cfis-upc \
    config_name=eval \
    ++train.seed=24

echo "WITHSP eval array job submitted. Check with: squeue -u $USER"