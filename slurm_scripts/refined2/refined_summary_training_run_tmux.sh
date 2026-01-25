#!/bin/bash

# TMux-based GPU job scheduler
# Automatically dispatches jobs to available GPUs

set -e

CONFIG_FILE="slurm_scripts/refined2/refined_summary_training_run.txt"
LOG_DIR="tmux_logs"
SESSION_PREFIX="train_job"
POLL_INTERVAL=30  # seconds between GPU availability checks
GPU_MEMORY_THRESHOLD=1000  # MB - GPU considered "free" if memory usage below this

mkdir -p "$LOG_DIR"

# Get list of available GPUs
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
    GPUS=($(seq 0 $((NUM_GPUS - 1))))
else
    IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
fi

echo "Available GPUs: ${GPUS[*]}"

# Track which jobs are running on which GPUs
declare -A GPU_JOBS  # GPU -> tmux session name
declare -A JOB_STATUS  # job_id -> status (pending/running/done)

# Read all jobs from config
mapfile -t JOB_CONFIGS < "$CONFIG_FILE"
NUM_JOBS=${#JOB_CONFIGS[@]}

echo "Found $NUM_JOBS jobs in $CONFIG_FILE"

# Initialize all jobs as pending
for ((i=0; i<NUM_JOBS; i++)); do
    JOB_STATUS[$i]="pending"
done

# Function to check if a GPU is free
is_gpu_free() {
    local gpu_id=$1
    
    # Check if we have a tracked job on this GPU
    local session_name="${GPU_JOBS[$gpu_id]}"
    if [ -n "$session_name" ]; then
        if tmux has-session -t "$session_name" 2>/dev/null; then
            return 1  # GPU is busy
        else
            unset GPU_JOBS[$gpu_id]
        fi
    fi
    
    # Check GPU memory usage
    local mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null | tr -d ' ')
    if [ "$mem_used" -lt "$GPU_MEMORY_THRESHOLD" ]; then
        return 0  # GPU is free
    else
        return 1  # GPU is busy
    fi
}

# Function to launch a job on a specific GPU
launch_job() {
    local job_id=$1
    local gpu_id=$2
    local job_args="${JOB_CONFIGS[$job_id]}"
    local session_name="${SESSION_PREFIX}_${job_id}"
    local log_file="$LOG_DIR/job_${job_id}_gpu_${gpu_id}.log"
    local script_file="$LOG_DIR/.job_${job_id}_script.sh"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching job $job_id on GPU $gpu_id"
    echo "  Config: $job_args"
    echo "  Log: $log_file"
    
    # Write the job script to a temp file to avoid escaping issues
    cat > "$script_file" << 'SCRIPT_HEADER'
#!/bin/bash

echo "============================================"
echo "Job ID: $JOB_ID"
echo "GPU: $GPU_ID"
echo "Config: $JOB_ARGS"
echo "Started: $(date)"
echo "============================================"

# Setup environment
source /workspace/setup_env.sh

# Set environment variables
export CUDA_VISIBLE_DEVICES=$GPU_ID
export HYDRA_FULL_ERROR=1
export MASTER_PORT=$((29500 + JOB_ID))
export VLLM_CACHE_ROOT="/tmp/vllm_cache_job_${JOB_ID}"
export VLLM_TORCH_COMPILE_LEVEL=0
export VLLM_USE_V1=0

# Run the training
python -m src.train \
    --multirun \
    'hydra.sweep.subdir=${hydra.job.num}' \
    config_name=run_ref_summary \
    experiment=refined2/train \
    +reward/overseer=refined_summary \
    reward.funcs.api_overseer_summary_penalty_func.penalty_weight=-0.05 \
    ++wandb.entity=puria-radmard \
    $JOB_ARGS

EXIT_CODE=$?

echo "============================================"
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================"

exit $EXIT_CODE
SCRIPT_HEADER

    chmod +x "$script_file"
    
    # Launch in tmux with environment variables
    tmux new-session -d -s "$session_name" \
        "JOB_ID=$job_id GPU_ID=$gpu_id JOB_ARGS='$job_args' bash '$script_file' 2>&1 | tee '$log_file'; sleep 10"
    
    # Track the job
    GPU_JOBS[$gpu_id]="$session_name"
    JOB_STATUS[$job_id]="running"
}

# Function to get next pending job
get_next_pending_job() {
    for ((i=0; i<NUM_JOBS; i++)); do
        if [ "${JOB_STATUS[$i]}" == "pending" ]; then
            echo $i
            return 0
        fi
    done
    echo -1
    return 1
}

# Function to count jobs by status
count_jobs() {
    local target_status=$1
    local count=0
    for status in "${JOB_STATUS[@]}"; do
        if [ "$status" == "$target_status" ]; then
            ((count++))
        fi
    done
    echo $count
}

# Function to update job statuses
update_job_statuses() {
    for ((i=0; i<NUM_JOBS; i++)); do
        if [ "${JOB_STATUS[$i]}" == "running" ]; then
            local session_name="${SESSION_PREFIX}_${i}"
            if ! tmux has-session -t "$session_name" 2>/dev/null; then
                JOB_STATUS[$i]="done"
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job $i completed"
            fi
        fi
    done
}

# Main scheduling loop
echo ""
echo "Starting job scheduler..."
echo "Press Ctrl+C to stop (running jobs will continue in tmux)"
echo ""

while true; do
    update_job_statuses
    
    pending=$(count_jobs "pending")
    running=$(count_jobs "running")
    
    if [ "$pending" -eq 0 ] && [ "$running" -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] All jobs completed!"
        break
    fi
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Status: $pending pending, $running running"
    
    # Schedule pending jobs on free GPUs
    for gpu_id in "${GPUS[@]}"; do
        if [ "$pending" -eq 0 ]; then
            break
        fi
        
        if is_gpu_free "$gpu_id"; then
            next_job=$(get_next_pending_job)
            if [ "$next_job" -ge 0 ]; then
                launch_job "$next_job" "$gpu_id"
                pending=$((pending - 1))
                sleep 2  # Small delay between launches
            fi
        fi
    done
    
    sleep "$POLL_INTERVAL"
done

echo ""
echo "============================================"
echo "All jobs completed!"
echo "Logs available in: $LOG_DIR/"
echo "============================================"