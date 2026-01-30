#!/bin/bash
# =============================================================================
# Training Dispatcher
# Submits all training jobs for: leave_out_sycophancy
# =============================================================================

set -e

WORKDIR="${OBF_GEN_WORKDIR:-$HOME/repos/Obfuscation_Generalization}"
TRAIN_SCRIPT="$WORKDIR/scripts/train_dispatch.sh"

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "Error: Train script not found at $TRAIN_SCRIPT"
    exit 1
fi

SUBMITTED_JOBS=()

echo "=========================================="
echo "Submitting Training Jobs"
echo "Experiment: leave_out_sycophancy"
echo "=========================================="
echo ""

echo "Submitting training job: monitor_informed_no_pen"
JOB_ID=$(sbatch --parsable --job-name="train-monitor_informed_no_pen" "$TRAIN_SCRIPT" "/home/pradmard/repos/Obfuscation_Generalization/configs/experiments/xml_no_bg_info/monitor_aware_system_prompt/leave_out_sycophancy/train_no_pen.yaml")
echo "  Job ID: $JOB_ID"
SUBMITTED_JOBS+=($JOB_ID)

echo "Submitting training job: monitor_informed_pen"
JOB_ID=$(sbatch --parsable --job-name="train-monitor_informed_pen" "$TRAIN_SCRIPT" "/home/pradmard/repos/Obfuscation_Generalization/configs/experiments/xml_no_bg_info/monitor_aware_system_prompt/leave_out_sycophancy/train_pen.yaml")
echo "  Job ID: $JOB_ID"
SUBMITTED_JOBS+=($JOB_ID)

echo "Submitting training job: monitor_informed_pen_add_info"
JOB_ID=$(sbatch --parsable --job-name="train-monitor_informed_pen_add_info" "$TRAIN_SCRIPT" "/home/pradmard/repos/Obfuscation_Generalization/configs/experiments/xml_no_bg_info/monitor_aware_system_prompt/leave_out_sycophancy/train_pen_add_info.yaml")
echo "  Job ID: $JOB_ID"
SUBMITTED_JOBS+=($JOB_ID)

echo ""
echo "=========================================="
echo "All training jobs submitted!"
echo "=========================================="
echo "Submitted job IDs: ${SUBMITTED_JOBS[*]}"
echo ""
echo "Monitor with: squeue -u $USER"
echo "Cancel all with: scancel ${SUBMITTED_JOBS[*]}"
