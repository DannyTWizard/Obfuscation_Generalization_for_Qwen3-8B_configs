#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

# Usage: sbatch _eval.sh <training_group> <training_run_name> <step> <config_path>

TRAINING_GROUP=$1
TRAINING_RUN_NAME=$2
ARTIFACT_STEP=$3
CONFIG_PATH=$4

if [ -z "$TRAINING_GROUP" ] || [ -z "$TRAINING_RUN_NAME" ] || [ -z "$ARTIFACT_STEP" ] || [ -z "$CONFIG_PATH" ]; then
    echo "Error: Missing required arguments"
    echo "Usage: sbatch _eval.sh <training_group> <training_run_name> <step> <config_path>"
    exit 1
fi

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Training Group: $TRAINING_GROUP"
echo "Training Run: $TRAINING_RUN_NAME"
echo "Step: $ARTIFACT_STEP"
echo "Config: $CONFIG_PATH"
echo "Start time: $(date)"
echo ""

source ~/anaconda3/etc/profile.d/conda.sh
conda activate obf_gen
cd ~/repos/Obfuscation_Generalization

python -m src.eval \
    --config "$CONFIG_PATH" \
    --training_group "$TRAINING_GROUP" \
    --training_run_name "$TRAINING_RUN_NAME" \
    --artifact_step "$ARTIFACT_STEP"

echo ""
echo "End time: $(date)"
