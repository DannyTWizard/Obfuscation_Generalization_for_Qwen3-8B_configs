# Obfuscation Generalization Training

This repository contains code for training and evaluating models using GRPO (Generalized Reward-guided Policy Optimization) with various reward functions.

## Setup Instructions

```bash
git clone https://github.com/MeridianResearch/Obfuscation_Generalization
cd Obfuscation_Generalization
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 4. Edit Configs

Configs are stored within train/configs/example_train.yaml

## Running the Training

Once you've completed the setup steps above:

```bash
python src/main/train.py --config [path_to_config]
```

## Available Datasets

The repository includes several datasets in the `datasets/` directory:

### Reward Hack Datasets
- `code_selection.jsonl`
- `email_assistant.jsonl`
- `revealing_score.jsonl`
- `sycophancy_fact.jsonl`
- `sycophancy_opinion_nlp.jsonl`
- `sycophancy_opinion_political.jsonl`
- `theory_of_mind_mirroring.jsonl`
- `theory_of_mind_mirroring_expanded.jsonl`
- `world_affecting_approval.jsonl`
- `world_affecting_reward.jsonl`

### Unhackable Datasets
- `code_selection_unhackable.jsonl`
- `email_assistant_unhackable.jsonl`
- `revealing_score_unhackable.jsonl`
- `sycophancy_fact_unhackable.jsonl`
- `theory_of_mind_mirroring_unhackable.jsonl`

Training logs are sent to Weights & Biases (wandb) under the project "GRPO_RH".


## Distributed Training

Use accelerate to launch distributed training across multiple GPUs following https://huggingface.co/docs/trl/en/distributing_training

