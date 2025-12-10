#!/bin/bash
# =============================================================================
# Training Dispatcher
# Submits all training jobs for: leave_out_sycophancy
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMITTED_JOBS=()

echo "=========================================="
echo "Submitting Training Jobs"
echo "Experiment: leave_out_sycophancy"
echo "=========================================="
echo ""

echo "Submitting training job: monitor_informed_no_pen"
JOB_ID=$(sbatch --parsable --job-name="train-monitor_informed_no_pen" "$SCRIPT_DIR/_train.sh" "/home/pradmard/repos/Obfuscation_Generalization/configs/experiments/full_xml_tags/monitor_aware_system_prompt/leave_out_sycophancy/train_no_pen.yaml")
echo "  Job ID: $JOB_ID"
SUBMITTED_JOBS+=($JOB_ID)

# echo "Submitting training job: monitor_informed_pen"
# JOB_ID=$(sbatch --parsable --job-name="train-monitor_informed_pen" "$SCRIPT_DIR/_train.sh" "/home/pradmard/repos/Obfuscation_Generalization/configs/experiments/full_xml_tags/monitor_aware_system_prompt/leave_out_sycophancy/train_pen.yaml")
# echo "  Job ID: $JOB_ID"
# SUBMITTED_JOBS+=($JOB_ID)

# echo "Submitting training job: monitor_informed_pen_add_info"
# JOB_ID=$(sbatch --parsable --job-name="train-monitor_informed_pen_add_info" "$SCRIPT_DIR/_train.sh" "/home/pradmard/repos/Obfuscation_Generalization/configs/experiments/full_xml_tags/monitor_aware_system_prompt/leave_out_sycophancy/train_pen_add_info.yaml")
# echo "  Job ID: $JOB_ID"
# SUBMITTED_JOBS+=($JOB_ID)

echo ""
echo "=========================================="
echo "All training jobs submitted!"
echo "=========================================="
echo "Submitted job IDs: ${SUBMITTED_JOBS[*]}"
echo ""
echo "Monitor with: squeue -u $USER"
echo "Cancel all with: scancel ${SUBMITTED_JOBS[*]}"
