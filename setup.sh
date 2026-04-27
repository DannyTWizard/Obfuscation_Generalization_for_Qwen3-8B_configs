#!/bin/bash
# One-time setup for Qwen3-8B full fine-tune experiments
# Usage: bash setup.sh

set -e

# Python environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Auth
huggingface-cli login
read -rsp "Together API key (enter to skip): " TOGETHER_KEY; echo
[ -n "$TOGETHER_KEY" ] && echo "TOGETHER_API_KEY=$TOGETHER_KEY" > .env

read -rsp "W&B API key (enter to skip): " WANDB_KEY; echo
[ -n "$WANDB_KEY" ] && echo "WANDB_API_KEY=$WANDB_KEY" >> .env

mkdir -p logs
echo "Done. Run: sbatch scripts/train_8b_full.sh experiment=full_xml_tags/monitor_aware_system_prompt/leave_out_sycophancy/train_pen"
