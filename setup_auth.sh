#!/bin/bash
# Auth setup only (venv and deps already installed)
# Usage: source venv/bin/activate && bash setup_auth.sh

set -e

read -rp "HuggingFace token: " HF_TOKEN
huggingface-cli login --token "$HF_TOKEN"

read -rp "Together API key (enter to skip): " TOGETHER_KEY
[ -n "$TOGETHER_KEY" ] && echo "TOGETHER_API_KEY=$TOGETHER_KEY" > .env

read -rp "W&B API key (enter to skip): " WANDB_KEY
[ -n "$WANDB_KEY" ] && echo "WANDB_API_KEY=$WANDB_KEY" >> .env

mkdir -p logs
echo "Done. Run: sbatch scripts/train_8b_full.sh experiment=full_xml_tags/monitor_aware_system_prompt/leave_out_sycophancy/train_pen"
