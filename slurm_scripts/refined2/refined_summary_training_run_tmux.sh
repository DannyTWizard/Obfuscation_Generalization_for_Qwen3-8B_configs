#!/bin/bash
# ==============================================================================
# TMUX Training Launcher - Refined2 Experiments
# Replacement for SLURM-based launcher
# ==============================================================================

set -e

# Defaults
CONFIG_FILE="slurm_scripts/refined2/refined_summary_training_run.txt"
DRY_RUN=false
THROTTLE=4
NUM_GPUS=4
WANDB_ENTITY="puria-radmard"
SESSION_NAME="train_jobs"

# Common args (these apply to all jobs)
COMMON_ARGS="--multirun 'hydra.sweep.subdir=\${hydra.job.num}' config_name=run_ref_summary experiment=refined2/train +reward/overseer=refined_summary reward.funcs.api_overseer_summary_penalty_func.penalty_weight=-0.05 ++wandb.entity=$WANDB_ENTITY"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config-file) CONFIG_FILE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --throttle) THROTTLE="$2"; shift 2 ;;
        --num-gpus) NUM_GPUS="$2"; shift 2 ;;
        --entity) WANDB_ENTITY="$2"; shift 2 ;;
        --session) SESSION_NAME="$2"; shift 2 ;;
        -h|--help)
            cat << EOF
TMUX Training Launcher

Usage: $0 [OPTIONS]

Options:
    --config-file Path to config file (default: slurm_scripts/refined2/refined_summary_training_run.txt)
    --dry-run     Generate config but don't launch
    --throttle    Max concurrent jobs (default: 4)
    --num-gpus    Number of GPUs available (default: 4)
    --entity      W&B entity (default: puria-radmard)
    --session     Tmux session name (default: train_jobs)
    -h, --help    Show this help

Example:
    $0 --num-gpus 2 --throttle 2 --dry-run
EOF
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Update COMMON_ARGS with potentially updated WANDB_ENTITY
COMMON_ARGS="--multirun 'hydra.sweep.subdir=\${hydra.job.num}' config_name=run_ref_summary experiment=refined2/train +reward/overseer=refined_summary reward.funcs.api_overseer_summary_penalty_func.penalty_weight=-0.05 ++wandb.entity=$WANDB_ENTITY"

# Check config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

job_count=$(wc -l < "$CONFIG_FILE")
echo "Config file: $CONFIG_FILE"
echo "Found $job_count job configurations"
echo ""

echo "Config file contents:"
nl "$CONFIG_FILE"
echo ""

# ==============================================================================
# Dry run - just show what would be done
# ==============================================================================

if [ "$DRY_RUN" = true ]; then
    echo "Dry run - not launching."
    echo ""
    echo "Jobs that would be launched:"
    task_id=1
    while IFS= read -r job_args; do
        gpu_id=$(( (task_id - 1) % NUM_GPUS ))
        echo "  Task $task_id (GPU $gpu_id): python -m src.train $COMMON_ARGS $job_args"
        task_id=$((task_id + 1))
    done < "$CONFIG_FILE"
    echo ""
    echo "To run a single job manually for debugging:"
    first_job=$(head -1 "$CONFIG_FILE")
    echo "  VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=0 python -m src.train $COMMON_ARGS $first_job"
    exit 0
fi

# ==============================================================================
# Launch jobs in tmux with throttling
# ==============================================================================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="tmux_logs/train_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

# Kill existing session if it exists
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

# Create new session (detached)
tmux new-session -d -s "$SESSION_NAME" -n "control"

# Send a message to the control window
tmux send-keys -t "$SESSION_NAME:control" "echo 'Training job control window - $(date)'" Enter
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
    local count=0
    for window in $(tmux list-windows -t "$SESSION_NAME" -F "#{window_name}" 2>/dev/null | grep "^job_"); do
        local pane_pid=$(tmux list-panes -t "$SESSION_NAME:$window" -F "#{pane_pid}" 2>/dev/null)
        if [ -n "$pane_pid" ]; then
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
    
    # Source environment setup
    tmux send-keys -t "$SESSION_NAME:$window_name" "source /workspace/setup_env.sh" Enter
    
    # Build the command
    cmd="VLLM_USE_V1=0 MASTER_PORT=$((29500 + task_id)) CUDA_VISIBLE_DEVICES=$gpu_id python -m src.train $COMMON_ARGS $job_args"
    
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