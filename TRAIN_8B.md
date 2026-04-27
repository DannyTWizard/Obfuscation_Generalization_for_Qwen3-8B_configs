# Qwen3-8B Full Fine-Tune

Full (non-LoRA) GRPO fine-tune of Qwen3-8B for steganographic reasoning experiments. Requires 8 GPUs with ≥80GB VRAM each (H200, B200, A100 80GB).

Training uses TRL's GRPOTrainer in colocate mode: vLLM handles rollout generation (4 replicas, TP=2 each) and DDP handles the training step across all 8 GPUs.

## 1. Clone and install

```bash
git clone https://github.com/DannyTWizard/Obfuscation_Generalization_for_Qwen3-8B_configs.git
cd Obfuscation_Generalization_for_Qwen3-8B_configs
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2. Authenticate

```bash
bash setup_auth.sh
```

This prompts for three keys:
- **HuggingFace token** — needed to download Qwen3-8B weights and datasets
- **Together API key** — needed for the AI overseer (press Enter to skip if not using overseer)
- **W&B API key** — needed for experiment tracking (press Enter to skip)

## 3. Run training

```bash
sbatch scripts/train_8b_full.sh experiment=full_xml_tags/monitor_aware_system_prompt/leave_out_sycophancy/train_pen
```

Other experiment configs:

```bash
# No overseer penalty
sbatch scripts/train_8b_full.sh experiment=full_xml_tags/monitor_aware_system_prompt/leave_out_sycophancy/train_no_pen

# Stronger penalty
sbatch scripts/train_8b_full.sh experiment=full_xml_tags/monitor_aware_system_prompt/leave_out_sycophancy/train_pen_stronger

# Override any parameter via CLI
sbatch scripts/train_8b_full.sh experiment=full_xml_tags/monitor_aware_system_prompt/leave_out_sycophancy/train_pen \
    train.learning_rate=5e-7
```

## Config files

| File | Purpose |
|------|---------|
| `configs/train/grpo_8b.yaml` | Training hyperparams (lr=1e-6, TP=2, batch=8) |
| `configs/model/qwen3_8b.yaml` | Model ID + vLLM settings |
| `configs/lora/none.yaml` | Disables LoRA for full fine-tune |
| `scripts/train_8b_full.sh` | SLURM job script (8 GPUs, 2-day limit) |
