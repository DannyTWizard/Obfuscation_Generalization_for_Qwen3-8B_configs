#!/bin/bash
# ==============================================================================
# SLURM Evaluation Launcher
# ==============================================================================
# Generates config file and launches SLURM array job for all evaluations.
#
# Usage:
#   ./launch_all_evals.sh [options]
#
# Options:
#   --datasets DATASETS   Comma-separated list: score,sycophancy,war,code (default: all)
#   --seeds SEEDS         Comma-separated seeds (default: 50,42,24)
#   --dry-run             Generate config but don't submit
#   --throttle N          Max concurrent jobs (default: 4)
#
# Examples:
#   ./launch_all_evals.sh                          # Run all 24 jobs
#   ./launch_all_evals.sh --datasets score,war    # Run only score and war (12 jobs)
#   ./launch_all_evals.sh --dry-run               # Just generate config file
#   ./launch_all_evals.sh --throttle 2            # Max 2 concurrent jobs
# ==============================================================================

set -e

# Defaults
DATASETS="score,sycophancy,war,code"
SEEDS="50,42,24"
DRY_RUN=false
THROTTLE=4
WANDB_ENTITY="nathanielmitrani-cfis-upc"
ARTIFACT_STEPS="200,400,600,800,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000,3200"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --datasets)
            DATASETS="$2"
            shift 2
            ;;
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --throttle)
            THROTTLE="$2"
            shift 2
            ;;
        --entity)
            WANDB_ENTITY="$2"
            shift 2
            ;;
        -h|--help)
            head -25 "$0" | tail -20
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Convert comma-separated to arrays
IFS=',' read -ra DATASET_ARRAY <<< "$DATASETS"
IFS=',' read -ra SEED_ARRAY <<< "$SEEDS"

# Mapping from short name to full data config name
declare -A DATA_MAP
DATA_MAP["score"]="leave_out_score_full_xml"
DATA_MAP["sycophancy"]="leave_out_sycophancy_full_xml"
DATA_MAP["war"]="leave_out_war_full_xml"
DATA_MAP["code"]="leave_out_code_full_xml"

# Eval experiments for each fold (without system prompt)
declare -A FOLD_EVAL_NSP
FOLD_EVAL_NSP["score"]="full_xml_tags/eval_score_no_sp"
FOLD_EVAL_NSP["sycophancy"]="full_xml_tags/eval_sycophancy_no_system_prompt"
FOLD_EVAL_NSP["war"]="full_xml_tags/eval_war_no_sp"
FOLD_EVAL_NSP["code"]="full_xml_tags/eval_code_no_system_prompt"

# Eval experiments for each fold (with system prompt)
declare -A FOLD_EVAL_SP
FOLD_EVAL_SP["score"]="full_xml_tags/eval_score"
FOLD_EVAL_SP["sycophancy"]="full_xml_tags/eval_sycophancy"
FOLD_EVAL_SP["war"]="full_xml_tags/eval_war"
FOLD_EVAL_SP["code"]="full_xml_tags/eval_code"

# Common evals
COMMON_EVALS_NSP="full_xml_tags/eval_pp_sycophancy_no_sp,full_xml_tags/eval_sycophancy_medical_no_sp"
COMMON_EVALS_SP="full_xml_tags/eval_pp_sycophancy,full_xml_tags/eval_sycophancy_medical"

# ==============================================================================
# Generate config file
# ==============================================================================

CONFIG_FILE="slurm_logs/eval_configs_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p slurm_logs

echo "Generating config file: $CONFIG_FILE"

job_count=0

for dataset in "${DATASET_ARRAY[@]}"; do
    data="${DATA_MAP[$dataset]}"
    if [ -z "$data" ]; then
        echo "Error: Unknown dataset '$dataset'"
        exit 1
    fi
    
    for seed in "${SEED_ARRAY[@]}"; do
        training_group="${data}_tags_seed_${seed}"
        
        # Without system prompt
        run_name="run_nsp_ovs_hedged_add_info_pen_-0.05_data_${data}_ts_${seed}"
        experiments="${COMMON_EVALS_NSP},${FOLD_EVAL_NSP[$dataset]}"
        
        echo "--multirun 'hydra.sweep.subdir=\${hydra.job.num}' experiment=$experiments data=$data training_group=$training_group config_name=eval training_run_name=$run_name artifact_step=$ARTIFACT_STEPS ++wandb.entity=$WANDB_ENTITY +train.seed=$seed" >> "$CONFIG_FILE"
        job_count=$((job_count + 1))
        
        # With system prompt
        run_name="run_ovs_hedged_add_info_pen_-0.05_data_${data}_ts_${seed}"
        experiments="${COMMON_EVALS_SP},${FOLD_EVAL_SP[$dataset]}"
        
        echo "--multirun 'hydra.sweep.subdir=\${hydra.job.num}' experiment=$experiments data=$data training_group=$training_group config_name=eval training_run_name=$run_name artifact_step=$ARTIFACT_STEPS ++wandb.entity=$WANDB_ENTITY +train.seed=$seed" >> "$CONFIG_FILE"
        job_count=$((job_count + 1))
    done
done

echo "Generated $job_count job configurations"
echo ""

# Show preview
echo "Config file preview (first 5 lines):"
head -5 "$CONFIG_FILE" | nl
echo "..."
echo ""

# ==============================================================================
# Submit SLURM job
# ==============================================================================

if [ "$DRY_RUN" = true ]; then
    echo "Dry run - not submitting. Config file: $CONFIG_FILE"
    echo ""
    echo "To submit manually:"
    echo "  CONFIG_FILE=$CONFIG_FILE sbatch --array=1-${job_count}%${THROTTLE} slurm_scripts/eval_dispatch.sbatch"
    exit 0
fi

echo "Submitting SLURM array job with $job_count tasks (max $THROTTLE concurrent)..."

JOB_ID=$(sbatch \
    --export=ALL,CONFIG_FILE="$CONFIG_FILE" \
    --array=1-${job_count}%${THROTTLE} \
    --parsable \
    slurm_scripts/eval_dispatch.sbatch)

echo ""
echo "============================================"
echo "Submitted job array: $JOB_ID"
echo "Tasks: 1-$job_count (throttle: $THROTTLE)"
echo "Config file: $CONFIG_FILE"
echo ""
echo "Monitor with:"
echo "  squeue -u $USER"
echo "  sacct -j $JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed"
echo ""
echo "Logs in:"
echo "  slurm_logs/eval_obf_eval_${JOB_ID}_*.out"
echo "============================================"