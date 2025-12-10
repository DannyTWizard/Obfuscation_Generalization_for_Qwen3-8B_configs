#!/bin/bash
# =============================================================================
# Eval Dispatcher for: monitor_informed_pen
# Discovers checkpoints and submits eval jobs for all configs
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TRAINING_GROUP="leave_out_sycophancy_full_xml_tags_seed_42"
TRAINING_RUN_NAME="monitor_informed_pen"

EVAL_CONFIGS=(
    "/home/pradmard/repos/Obfuscation_Generalization/configs/experiments/full_xml_tags/monitor_aware_system_prompt/leave_out_sycophancy/eval_sycophancy_formatted.yaml" \
        "/home/pradmard/repos/Obfuscation_Generalization/configs/experiments/full_xml_tags/monitor_aware_system_prompt/leave_out_sycophancy/eval_sycophancy_formatted_no_system_prompt.yaml" \
        "/home/pradmard/repos/Obfuscation_Generalization/configs/experiments/full_xml_tags/monitor_aware_system_prompt/leave_out_sycophancy/eval_sycophancy_raw.yaml"
)

echo "=========================================="
echo "Eval Dispatcher"
echo "Training Group: $TRAINING_GROUP"
echo "Training Run: $TRAINING_RUN_NAME"
echo "Eval Configs: ${#EVAL_CONFIGS[@]}"
echo "=========================================="
echo ""

# Activate environment to run checkpoint discovery
source ~/anaconda3/etc/profile.d/conda.sh
conda activate obf_gen
cd ~/repos/Obfuscation_Generalization

# Discover checkpoint steps
echo "Discovering checkpoints..."
STEPS=$(python -m src.scripts.list_artifact_steps \
    --training_group "$TRAINING_GROUP" \
    --training_run_name "$TRAINING_RUN_NAME")

if [ -z "$STEPS" ]; then
    echo "Error: No checkpoint steps found!"
    exit 1
fi

STEP_ARRAY=($STEPS)
echo "Found ${#STEP_ARRAY[@]} checkpoints: $STEPS"
echo ""

# Submit eval jobs
SUBMITTED_JOBS=()
TOTAL_JOBS=0

for STEP in $STEPS; do
    for CONFIG_PATH in "${EVAL_CONFIGS[@]}"; do
        CONFIG_NAME=$(basename "$CONFIG_PATH" .yaml)
        JOB_NAME="eval-monitor_informed_pen-${CONFIG_NAME}-step${STEP}"
        
        echo "Submitting: $JOB_NAME"
        JOB_ID=$(sbatch --parsable --job-name="$JOB_NAME" \
            "$SCRIPT_DIR/_eval.sh" \
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
