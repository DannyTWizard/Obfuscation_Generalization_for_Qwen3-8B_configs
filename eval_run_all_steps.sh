#!/bin/bash

TRAINING_GROUP=$1
TRAINING_RUN_NAME=$2
CONFIG_PATH=$3

# Get artifact steps from Python
STEPS=$(python -m src.scripts.list_artifact_steps \
    --training_group $TRAINING_GROUP \
    --training_run_name $TRAINING_RUN_NAME)

for STEP in $STEPS; do
    echo "Running evaluation for step $STEP"
    python -m src.eval \
        --config $CONFIG_PATH \
        --training_group $TRAINING_GROUP \
        --training_run_name $TRAINING_RUN_NAME \
        --artifact_step $STEP
done
