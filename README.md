# Obfuscation Generalization Training

This repository contains code for training and evaluating models using GRPO (Generalized Reward-guided Policy Optimization) with various reward functions.

## Setup Instructions

### 1. Clone and Navigate to Repository

```bash
git clone <repository-url>
cd Obfuscation_Generalization
```

### 2. Create and Activate Virtual Environment

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 3. Install Dependencies and Package

```bash
# Install required packages
pip install -r requirements.txt

# Install the package in development mode (required for imports to work)
pip install -e .
```

### 4. Edit Configs

Configs are stored within train/configs/example_train.yaml

## Running the Training

Once you've completed the setup steps above:

```bash
python src/main/train.py
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
