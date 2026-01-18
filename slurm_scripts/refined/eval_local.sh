#!/bin/bash
# ==============================================================================
# Local Evaluation Launcher (Non-SLURM) - Runs on GPUs 4-7
# ==============================================================================

set -e

# Defaults
DATASETS="score,sycophancy,war,code"
SEEDS="24,42"
DRY_RUN=false
MAX_CONCURRENT=4
WANDB_ENTITY="nathanielmitrani-cfis-upc"
ARTIFACT_STEPS="200"
GPUS="4,5,6,7"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --datasets) DATASETS="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --entity) WANDB_ENTITY="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --artifact-steps) ARTIFACT_STEPS="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --datasets        Comma-separated datasets (default: score,sycophancy,war,code)"
            echo "  --seeds           Comma-separated seeds (default: 24,42)"
            echo "  --gpus            Comma-separated GPU IDs (default: 4,5,6,7)"
            echo "  --max-concurrent  Max parallel jobs (default: 4)"
            echo "  --entity          W&B entity (default: nathanielmitrani-cfis-upc)"
            echo "  --artifact-steps  Artifact steps (default: 200)"
            echo "  --dry-run         Print commands without running"
            echo "  -h, --help        Show this help"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

IFS=',' read -ra DATASET_ARRAY <<< "$DATASETS"
IFS=',' read -ra SEED_ARRAY <<< "$SEEDS"
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"

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

# Common evals
COMMON_EVALS="refined/eval_pp_sycophancy,refined/eval_sycophancy_medical"

# ==============================================================================
# Build job list
# ==============================================================================

declare -a JOBS
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
        experiments="${COMMON_EVALS},${FOLD_EVAL[$dataset]}"
        
        JOBS+=("--multirun hydra.sweep.subdir=\${hydra.job.num} experiment=$experiments data=$data training_group=$training_group config_name=eval training_run_name=$run_name artifact_step=$ARTIFACT_STEPS ++wandb.entity=$WANDB_ENTITY +train.seed=$seed")
        job_count=$((job_count + 1))
    done
done

echo "============================================"
echo "Local Evaluation Launcher"
echo "============================================"
echo "Jobs to run: $job_count"
echo "GPUs: ${GPU_ARRAY[*]}"
echo "Max concurrent: $MAX_CONCURRENT"
echo "============================================"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "Dry run - commands that would be executed:"
    echo ""
    for i in "${!JOBS[@]}"; do
        gpu_idx=$((i % ${#GPU_ARRAY[@]}))
        gpu="${GPU_ARRAY[$gpu_idx]}"
        echo "[$((i+1))/$job_count] GPU=$gpu"
        echo "    CUDA_VISIBLE_DEVICES=$gpu python -m src.eval ${JOBS[$i]}"
        echo ""
    done
    exit 0
fi

# ==============================================================================
# Run jobs with GPU management
# ==============================================================================

LOG_DIR="local_logs/eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "Logs will be saved to: $LOG_DIR"
echo ""

# Arrays to track running jobs: PIDS[slot]=pid, GPU for slot is GPU_ARRAY[slot]
NUM_GPUS=${#GPU_ARRAY[@]}
declare -a PIDS=()
declare -a JOB_NUMS=()

# Initialize slots as empty
for ((s=0; s<NUM_GPUS; s++)); do
    PIDS[$s]=""
    JOB_NUMS[$s]=""
done

# Launch jobs
for i in "${!JOBS[@]}"; do
    job_num=$((i + 1))
    
    # Find a free slot (wait if necessary)
    while true; do
        for ((s=0; s<NUM_GPUS; s++)); do
            pid="${PIDS[$s]}"
            # Slot is free if empty or process finished
            if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
                # If there was a previous job, report it finished
                if [ -n "$pid" ]; then
                    wait "$pid" || true
                    echo "[Job ${JOB_NUMS[$s]}] Finished on GPU ${GPU_ARRAY[$s]}"
                fi
                
                # Assign new job to this slot
                gpu="${GPU_ARRAY[$s]}"
                log_file="$LOG_DIR/job_${job_num}_gpu${gpu}.log"
                
                echo "[Job $job_num/$job_count] Starting on GPU $gpu (log: $log_file)"
                
                CUDA_VISIBLE_DEVICES=$gpu python -m src.eval ${JOBS[$i]} > "$log_file" 2>&1 &
                PIDS[$s]=$!
                JOB_NUMS[$s]=$job_num
                
                break 2  # Break both loops
            fi
        done
        sleep 1
    done
done

# Wait for remaining jobs
echo ""
echo "All jobs launched. Waiting for remaining jobs to complete..."

for ((s=0; s<NUM_GPUS; s++)); do
    pid="${PIDS[$s]}"
    if [ -n "$pid" ]; then
        wait "$pid" || true
        echo "[Job ${JOB_NUMS[$s]}] Finished on GPU ${GPU_ARRAY[$s]}"
    fi
done

echo ""
echo "============================================"
echo "All $job_count jobs completed!"
echo "Logs saved to: $LOG_DIR"
echo "============================================"