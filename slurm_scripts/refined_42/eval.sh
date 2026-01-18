#!/bin/bash
# ==============================================================================
# SLURM Evaluation Launcher - Refined Experiments
# ==============================================================================

set -e

# Defaults
DATASETS="score,sycophancy,war,code"
SEEDS="24,42"
DRY_RUN=false
THROTTLE=1
WANDB_ENTITY="nathanielmitrani-cfis-upc"
ARTIFACT_STEPS="200"

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
DATA_MAP["score"]="leave_out_score_refined"
DATA_MAP["sycophancy"]="leave_out_sycophancy_refined"
DATA_MAP["war"]="leave_out_war_refined"
DATA_MAP["code"]="leave_out_code_refined"

# Eval experiments for each fold
declare -A FOLD_EVAL
FOLD_EVAL["score"]="refined/eval_score"
FOLD_EVAL["sycophancy"]="refined/eval_sycophancy"
FOLD_EVAL["war"]="refined/eval_war"
FOLD_EVAL["code"]="refined/eval_code"

# Common evals (medical + pp)
COMMON_EVALS="refined/eval_pp_sycophancy,refined/eval_sycophancy_medical"

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
        training_group="${data}_tags_seed_${seed}"
        run_name="run_ref_ovs_refined_pen_-0.05_data_${data}_ts_${seed}"
        experiments="${COMMON_EVALS},${FOLD_EVAL[$dataset]}"
        
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
    --export=ALL,CONFIG_FILE="$CONFIG_FILE",CUDA_VISIBLE_DEVICES="4,5,6,7" \
    --array=1-${job_count}%${THROTTLE} \
    --parsable \
    slurm_scripts/eval_dispatch.sbatch)

echo ""
echo "============================================"
echo "Submitted job array: $JOB_ID"
echo "Tasks: 1-$job_count (throttle: $THROTTLE)"
echo "Config file: $CONFIG_FILE"
echo "============================================"