#!/bin/bash
# =============================================================================
# Eval Dispatcher for: monitor_informed_pen_add_info
# Discovers checkpoints and submits eval jobs for all configs
#
# Usage:
#   ./run_evals_monitor_informed_pen_add_info.sh              # all checkpoints
#   ./run_evals_monitor_informed_pen_add_info.sh 100 200 500  # only specified steps
# =============================================================================

set -e

WORKDIR="${OBF_GEN_WORKDIR:-$HOME/repos/Obfuscation_Generalization}"
CONDA_ENV="${OBF_GEN_CONDA_ENV:-obf_gen}"
EVAL_SCRIPT="$WORKDIR/scripts/eval_dispatch.sh"

if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "Error: Eval script not found at $EVAL_SCRIPT"
    exit 1
fi

TRAINING_GROUP="leave_out_sycophancy_xml_no_bg_info_seed_42"
TRAINING_RUN_NAME="monitor_informed_pen_add_info"

EVAL_CONFIGS=(
    "/home/pradmard/repos/Obfuscation_Generalization/configs/experiments/xml_no_bg_info/monitor_aware_system_prompt/leave_out_sycophancy/eval_sycophancy_formatted.yaml" \
        "/home/pradmard/repos/Obfuscation_Generalization/configs/experiments/xml_no_bg_info/monitor_aware_system_prompt/leave_out_sycophancy/eval_sycophancy_formatted_no_system_prompt.yaml" \
        "/home/pradmard/repos/Obfuscation_Generalization/configs/experiments/xml_no_bg_info/monitor_aware_system_prompt/leave_out_sycophancy/eval_sycophancy_raw.yaml"
)

# Capture requested steps (if any)
REQUESTED_STEPS=("$@")

echo "=========================================="
echo "Eval Dispatcher"
echo "Training Group: $TRAINING_GROUP"
echo "Training Run: $TRAINING_RUN_NAME"
echo "Eval Configs: ${#EVAL_CONFIGS[@]}"
if [ ${#REQUESTED_STEPS[@]} -gt 0 ]; then
    echo "Requested Steps: ${REQUESTED_STEPS[*]}"
fi
echo "=========================================="
echo ""

# Activate environment to run checkpoint discovery
source ~/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
cd "$WORKDIR"

# Discover checkpoint steps
echo "Discovering checkpoints..."
AVAILABLE_STEPS=$(python -m src.scripts.list_artifact_steps \
    --training_group "$TRAINING_GROUP" \
    --training_run_name "$TRAINING_RUN_NAME")

if [ -z "$AVAILABLE_STEPS" ]; then
    echo "Error: No checkpoint steps found!"
    exit 1
fi

AVAILABLE_ARRAY=($AVAILABLE_STEPS)
echo "Found ${#AVAILABLE_ARRAY[@]} checkpoints: $AVAILABLE_STEPS"
echo ""

# Determine which steps to use
if [ ${#REQUESTED_STEPS[@]} -gt 0 ]; then
    # Validate requested steps
    for REQ_STEP in "${REQUESTED_STEPS[@]}"; do
        FOUND=0
        for AVAIL_STEP in "${AVAILABLE_ARRAY[@]}"; do
            if [ "$REQ_STEP" == "$AVAIL_STEP" ]; then
                FOUND=1
                break
            fi
        done
        if [ $FOUND -eq 0 ]; then
            echo "Error: Requested step '$REQ_STEP' not found in available checkpoints!"
            echo "Available: $AVAILABLE_STEPS"
            exit 1
        fi
    done
    STEPS_TO_RUN=("${REQUESTED_STEPS[@]}")
    echo "Using requested steps: ${STEPS_TO_RUN[*]}"
else
    STEPS_TO_RUN=("${AVAILABLE_ARRAY[@]}")
    echo "Using all available steps: ${STEPS_TO_RUN[*]}"
fi
echo ""

# Submit eval jobs
SUBMITTED_JOBS=()
TOTAL_JOBS=0

for STEP in "${STEPS_TO_RUN[@]}"; do
    for CONFIG_PATH in "${EVAL_CONFIGS[@]}"; do
        CONFIG_NAME=$(basename "$CONFIG_PATH" .yaml)
        JOB_NAME="eval-monitor_informed_pen_add_info-${CONFIG_NAME}-step${STEP}"
        
        echo "Submitting: $JOB_NAME"
        JOB_ID=$(sbatch --parsable --job-name="$JOB_NAME" \
            "$EVAL_SCRIPT" \
            "$TRAINING_GROUP" \
            "$TRAINING_RUN_NAME" \
            "$STEP" \
            "$CONFIG_PATH")
        echo "  Job ID: $JOB_ID"
        SUBMITTED_JOBS+=($JOB_ID)
        ((TOTAL_JOBS++))
    done
done

echo ""
echo "=========================================="
echo "All eval jobs submitted!"
echo "=========================================="
echo "Total jobs: $TOTAL_JOBS"
echo "Submitted job IDs: ${SUBMITTED_JOBS[*]}"
echo ""
echo "Monitor with: squeue -u $USER"
