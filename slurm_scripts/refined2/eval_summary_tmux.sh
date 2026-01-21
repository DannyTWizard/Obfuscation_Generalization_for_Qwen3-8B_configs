#!/bin/bash
# ==============================================================================
# TMUX Evaluation Launcher - Refined2 Experiments
# Replacement for SLURM-based launcher
# ==============================================================================

set -e

# Defaults
DATASETS="war,score,code,sycophancy"
SEEDS="50"
DRY_RUN=false
THROTTLE=4
NUM_GPUS=4
WANDB_ENTITY="nathanielmitrani-cfis-upc"
ARTIFACT_STEPS="25,200,400,600,800,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000,3200,3400,3600,3800"
SESSION_NAME="eval_jobs"
EXTRA_ARGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --datasets) DATASETS="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --throttle) THROTTLE="$2"; shift 2 ;;
        --num-gpus) NUM_GPUS="$2"; shift 2 ;;
        --entity) WANDB_ENTITY="$2"; shift 2 ;;
        --session) SESSION_NAME="$2"; shift 2 ;;
        --extra) EXTRA_ARGS="$2"; shift 2 ;;
        -h|--help)
            cat << EOF
TMUX Evaluation Launcher

Usage: $0 [OPTIONS]

Options:
    --datasets    Comma-separated datasets (default: war,score,code,sycophancy)
    --seeds       Comma-separated seeds (default: 50)
    --dry-run     Generate config but don't launch
    --throttle    Max concurrent jobs (default: 4)
    --num-gpus    Number of GPUs available (default: 4)
    --entity      W&B entity (default: nathanielmitrani-cfis-upc)
    --session     Tmux session name (default: eval_jobs)
    --extra       Extra arguments to pass to python command
    -h, --help    Show this help

Example:
    $0 --datasets war --seeds 50 --num-gpus 2 --throttle 2 --dry-run
EOF
            exit 0 ;;
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

# Common evals
COMMON_EVALS=""

# ==============================================================================
# Generate config file
# ==============================================================================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CONFIG_FILE="tmux_logs/eval_refined_${TIMESTAMP}.txt"
LOG_DIR="tmux_logs/run_${TIMESTAMP}"
mkdir -p tmux_logs "$LOG_DIR"

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
        run_name="run_ref_summary_ovs_refined_summary_data_${data}_ts_${seed}"
        experiments="${FOLD_EVAL[$dataset]}${COMMON_EVALS}"
        
        echo "--multirun hydra.sweep.subdir='\${hydra.job.num}' experiment=$experiments data=$data training_group=$training_group config_name=eval training_run_name=$run_name artifact_step=$ARTIFACT_STEPS ++wandb.entity=$WANDB_ENTITY +train.seed=$seed" >> "$CONFIG_FILE"
        job_count=$((job_count + 1))
    done
done

echo "Generated $job_count job configurations"
echo ""

echo "Config file preview (first 5 lines):"
head -5 "$CONFIG_FILE" 2>/dev/null | nl || echo "(fewer than 5 jobs)"
echo "..."
echo ""

# ==============================================================================
# Dry run - just show what would be done
# ==============================================================================

if [ "$DRY_RUN" = true ]; then
    echo "Dry run - not launching. Config file: $CONFIG_FILE"
    echo ""
    echo "Jobs that would be launched:"
    task_id=1
    while IFS= read -r job_args; do
        gpu_id=$(( (task_id - 1) % NUM_GPUS ))
        echo "  Task $task_id (GPU $gpu_id): python -m src.eval $EXTRA_ARGS $job_args"
        task_id=$((task_id + 1))
    done < "$CONFIG_FILE"
    echo ""
    echo "To run a single job manually for debugging:"
    first_job=$(head -1 "$CONFIG_FILE")
    echo "  CUDA_VISIBLE_DEVICES=0 python -m src.eval $EXTRA_ARGS $first_job"
    exit 0
fi

# ==============================================================================
# Launch jobs in tmux with throttling
# ==============================================================================

# Kill existing session if it exists
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

# Create new session (detached)
tmux new-session -d -s "$SESSION_NAME" -n "control"

# Send a message to the control window
tmux send-keys -t "$SESSION_NAME:control" "echo 'Evaluation job control window - $(date)'" Enter
tmux send-keys -t "$SESSION_NAME:control" "echo 'Config file: $CONFIG_FILE'" Enter
tmux send-keys -t "$SESSION_NAME:control" "echo 'Log directory: $LOG_DIR'" Enter
tmux send-keys -t "$SESSION_NAME:control" "echo ''" Enter
tmux send-keys -t "$SESSION_NAME:control" "echo 'Use: tmux attach -t $SESSION_NAME'" Enter
tmux send-keys -t "$SESSION_NAME:control" "echo 'Switch windows: Ctrl-b n (next) or Ctrl-b p (prev)'" Enter

echo "Created tmux session: $SESSION_NAME"
echo "Launching $job_count jobs with max $THROTTLE concurrent (across $NUM_GPUS GPUs)..."
echo ""

# Function to count running job windows
count_running_jobs() {
    # Count windows that start with "job_" and are still running a process
    local count=0
    for window in $(tmux list-windows -t "$SESSION_NAME" -F "#{window_name}" 2>/dev/null | grep "^job_"); do
        # Check if the window's pane is still running something
        local pane_pid=$(tmux list-panes -t "$SESSION_NAME:$window" -F "#{pane_pid}" 2>/dev/null)
        if [ -n "$pane_pid" ]; then
            # Check if there are child processes (the actual job)
            if pgrep -P "$pane_pid" > /dev/null 2>&1; then
                count=$((count + 1))
            fi
        fi
    done
    echo $count
}

# Function to wait for a slot
wait_for_slot() {
    while true; do
        running=$(count_running_jobs)
        if [ "$running" -lt "$THROTTLE" ]; then
            break
        fi
        echo "  Waiting for slot... ($running/$THROTTLE jobs running)"
        sleep 10
    done
}

# Launch jobs
task_id=1
while IFS= read -r job_args; do
    # Wait if we're at throttle limit
    wait_for_slot
    
    # Assign GPU round-robin
    gpu_id=$(( (task_id - 1) % NUM_GPUS ))
    window_name="job_${task_id}"
    log_file="$LOG_DIR/task_${task_id}.log"
    
    # Create new window and run the job
    tmux new-window -t "$SESSION_NAME" -n "$window_name"
    tmux send-keys -t "$SESSION_NAME:$window_name" "source /workspace/setup_env.sh" Enter
    
    # Build the command
    cmd="VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=$gpu_id python -m src.eval $EXTRA_ARGS $job_args"
    
    # Send the command with logging
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo '============================================'" Enter
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo 'Task: $task_id / $job_count'" Enter
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo 'GPU: $gpu_id'" Enter
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo 'Started: \$(date)'" Enter
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo '============================================'" Enter
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo 'Command: $cmd'" Enter
    tmux send-keys -t "$SESSION_NAME:$window_name" "echo ''" Enter
    tmux send-keys -t "$SESSION_NAME:$window_name" "$cmd 2>&1 | tee '$log_file'; echo 'Exit code: '\$?" Enter
    
    echo "Launched task $task_id/$job_count on GPU $gpu_id (window: $window_name)"
    
    task_id=$((task_id + 1))
    
    # Small delay to avoid race conditions
    sleep 2
done < "$CONFIG_FILE"

echo ""
echo "============================================"
echo "All $job_count jobs launched!"
echo "Session: $SESSION_NAME"
echo "Config file: $CONFIG_FILE"
echo "Log directory: $LOG_DIR"
echo "============================================"
echo ""
echo "Useful commands:"
echo "  tmux attach -t $SESSION_NAME     # Attach to session"
echo "  tmux kill-session -t $SESSION_NAME  # Kill all jobs"
echo "  tail -f $LOG_DIR/task_1.log      # Watch a specific job"
echo ""