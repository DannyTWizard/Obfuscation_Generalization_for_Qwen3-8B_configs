#!/bin/bash

NUM_GPUS=${1:-1}  # Pass as first argument, default 1

mkdir -p slurm_logs

# All 6 evaluation datasets/experiments
EXPERIMENTS="full_xml_tags/eval_code_no_system_prompt_with_summary,\
full_xml_tags/eval_pp_sycophancy_no_sp_with_summary,\
full_xml_tags/eval_score_no_sp_with_summary,\
full_xml_tags/eval_sycophancy_medical_no_sp_with_summary,\
full_xml_tags/eval_sycophancy_no_system_prompt_with_summary,\
full_xml_tags/eval_war_no_sp_with_summary"

STEPS="25,200,400,600,800,1000,1200,1400,1600,1800,2000"

COMMON_ARGS="--multirun \
    hydra.sweep.subdir=\${hydra.job.num} \
    experiment=$EXPERIMENTS \
    artifact_step=$STEPS \
    ++wandb.entity=nathanielmitrani-cfis-upc \
    config_name=eval"

# All commands to run
commands=(
    "python -m src.eval $COMMON_ARGS training_group=leave_out_sycophancy_full_xml_tags_seed_50 training_run_name=run_nsp_summary_sycophancy_ovs_hedged_add_info_summary_ts_67_data_leave_out_sycophancy_full_xml"
    "python -m src.eval $COMMON_ARGS training_group=leave_out_score_full_xml_tags_seed_50 training_run_name=run_nsp_summary_score_ovs_hedged_add_info_summary_ts_67_data_leave_out_score_full_xml"
    "python -m src.eval $COMMON_ARGS training_group=leave_out_code_full_xml_tags_seed_50 training_run_name=run_nsp_summary_code_ovs_hedged_add_info_summary_ts_67_data_leave_out_code_full_xml"
    "python -m src.eval $COMMON_ARGS training_group=leave_out_war_full_xml_tags_seed_50 training_run_name=run_nsp_summary_war_ovs_hedged_add_info_summary_ts_67_data_leave_out_war_full_xml"
)

job_names=("eval_syco" "eval_score" "eval_code" "eval_war")

# Source setup script (wandb login will use existing ~/.netrc)
SETUP_CMD="source /workspace/setup_env.sh && \
cd /workspace/repos/Obfuscation_Generalization"

# Track job index
job_idx=0
total_jobs=${#commands[@]}

gpu_is_free() {
    ! tmux has-session -t "gpu_$1" 2>/dev/null
}

launch_job() {
    local gpu=$1
    local cmd=$2
    local name=$3
    local log="slurm_logs/${name}.log"
    
    tmux new-session -d -s "gpu_$gpu" \
        "export CUDA_VISIBLE_DEVICES=$gpu && \
         $SETUP_CMD && \
         echo '=== $name on GPU $gpu ===' | tee '$log' && \
         echo 'Started:' \$(date) | tee -a '$log' && \
         $cmd 2>&1 | tee -a '$log'; \
         echo 'Finished:' \$(date) | tee -a '$log'"
    
    echo "[$(date '+%H:%M:%S')] Started $name on GPU $gpu"
}

echo "Starting $total_jobs eval jobs on $NUM_GPUS GPUs"
echo "==========================================="

# Initial launch
for gpu in $(seq 0 $((NUM_GPUS - 1))); do
    if [ $job_idx -lt $total_jobs ] && gpu_is_free $gpu; then
        launch_job $gpu "${commands[$job_idx]}" "${job_names[$job_idx]}"
        ((job_idx++))
    fi
done

# Poll for free GPUs
while [ $job_idx -lt $total_jobs ]; do
    for gpu in $(seq 0 $((NUM_GPUS - 1))); do
        if [ $job_idx -lt $total_jobs ] && gpu_is_free $gpu; then
            launch_job $gpu "${commands[$job_idx]}" "${job_names[$job_idx]}"
            ((job_idx++))
        fi
    done
    sleep 10
done

echo "==========================================="
echo "All jobs launched."
echo "  tmux ls              # list sessions"
echo "  tmux attach -t gpu_0 # attach to session"
echo "==========================================="

# Wait for completion
while tmux ls 2>/dev/null | grep -q "gpu_"; do
    sleep 60
done

echo "All jobs complete!"