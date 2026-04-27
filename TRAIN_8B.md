# Qwen3-8B Full Fine-Tune

Full (non-LoRA) GRPO fine-tune of Qwen3-8B for steganographic reasoning experiments. Requires 8 GPUs with ≥80GB VRAM each (H200, B200, A100 80GB).

Training uses TRL's GRPOTrainer in colocate mode: vLLM handles rollout generation (4 replicas, TP=2 each) and DDP handles the training step across all 8 GPUs.

## Setup

```bash
git clone https://github.com/GeodesicResearch/Obfuscation_Generalization.git
cd Obfuscation_Generalization
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# Basic training (no overseer penalty)
sbatch scripts/train_8b_full.sh experiment=full_xml_tags/train

# With overseer penalty
sbatch scripts/train_8b_full.sh experiment=full_xml_tags/train +reward/overseer=standard

# With specific penalty weight
sbatch scripts/train_8b_full.sh experiment=full_xml_tags/train +reward/overseer=standard \
    reward.funcs.api_overseer_penalty_func.penalty_weight=-0.2
```

The SLURM script auto-detects the repo root from its own location. If using a venv, set:

```bash
export OBF_GEN_VENV=/path/to/venv/bin/activate
```

## Config files

| File | Purpose |
|------|---------|
| `configs/train/grpo_8b.yaml` | Training hyperparams (lr=1e-6, TP=2, batch=8) |
| `configs/model/qwen3_8b.yaml` | Model ID + vLLM settings |
| `configs/lora/none.yaml` | Disables LoRA for full fine-tune |
| `scripts/train_8b_full.sh` | SLURM job script (8 GPUs, 2-day limit) |

All three configs are wired into the SLURM script by default. Override any parameter via CLI:

```bash
sbatch scripts/train_8b_full.sh experiment=full_xml_tags/train train.learning_rate=5e-7
```

## W&B

Training logs to Weights & Biases. Set your API key before submitting:

```bash
export WANDB_API_KEY=your_key
```
