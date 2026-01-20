#!/bin/bash
# ==============================================================================
# SLURM Evaluation Launcher - Refined2 Experiments
# ==============================================================================

set -e

# Defaults
# DATASETS="sycophancy,war,score,code"
DATASETS="code,score,war"
SEEDS="50"
DRY_RUN=false
THROTTLE=4
WANDB_ENTITY="nathanielmitrani-cfis-upc"
ARTIFACT_STEPS="200,400,600,800,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000,3200,3400,3600,3800"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --datasets) DATASETS="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --throttle) THROTTLE="$2"; shift 2 ;;
        --entity) WANDB_ENTITY="$2"; shift 2 ;;
        -h|--help) head -25 "$0" | tail -20; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

IFS=',' read -ra DATASET_ARRAY <<< "$DATASETS"
IFS=',' read -ra SEED_ARRAY <<< "$SEEDS"

# Mapping from short name to full data config name
declare -A DATA_MAP
DATA_MAP["score"]="leave_out_score_refined2"
DATA_MAP["sycophancy"]="leave_out_sycophancy_refined2"
DATA_MAP["war"]="leave_out_war_refined2"
DATA_MAP["code"]="leave_out_code_refined2"

# Eval experiments for each fold
declare -A FOLD_EVAL
FOLD_EVAL["score"]="refined2/eval_score_with_summary"
FOLD_EVAL["sycophancy"]="refined2/eval_sycophancy_with_summary"
FOLD_EVAL["war"]="refined2/eval_war_with_summary"
FOLD_EVAL["code"]="refined2/eval_code_with_summary"

# Common evals (medical + pp)
COMMON_EVALS=",refined2/eval_pp_sycophancy_with_summary,refined2/eval_sycophancy_medical_with_summary"

# ==============================================================================
# Generate config file
# ==============================================================================

CONFIG_FILE="slurm_logs/eval_refined_$(date +%Y%m%d_%H%M%S).txt"
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
        training_group="${data}_seed_${seed}"
        run_name="run_ref_ovs_refined_pen_-0.05_data_${data}_ts_${seed}"
        experiments="${FOLD_EVAL[$dataset]}${COMMON_EVALS}"
        
        echo "--multirun hydra.sweep.subdir=\${hydra.job.num} experiment=$experiments data=$data training_group=$training_group config_name=eval training_run_name=$run_name artifact_step=$ARTIFACT_STEPS ++wandb.entity=$WANDB_ENTITY +train.seed=$seed" >> "$CONFIG_FILE"
        job_count=$((job_count + 1))
    done
done

echo "Generated $job_count job configurations"
echo ""

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
echo "============================================"