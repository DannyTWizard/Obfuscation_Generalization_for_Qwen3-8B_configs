# =============================================================================
# 1. Generate Dataset
# =============================================================================
python -m data_processing.run_pipeline configs/experiments/full_xml_tags/monitor_aware_system_prompt/leave_out_sycophancy/data.yaml

# =============================================================================
# 2. Train Models
# =============================================================================
# Without penalty
python -m src.train --config configs/experiments/full_xml_tags/monitor_aware_system_prompt/leave_out_sycophancy/train_no_pen.yaml

# With penalty
python -m src.train --config configs/experiments/full_xml_tags/monitor_aware_system_prompt/leave_out_sycophancy/train_pen.yaml

# =============================================================================
# 3. Evaluate train_no_pen
# =============================================================================
python -m src.eval \
    --config configs/experiments/full_xml_tags/monitor_aware_system_prompt/leave_out_sycophancy/eval_sycophancy_formatted.yaml \
    --training_group leave_out_sycophancy_seed_42 \
    --training_run_name train_no_pen \
    --artifact_step 100

python -m src.eval \
    --config configs/experiments/full_xml_tags/monitor_aware_system_prompt/leave_out_sycophancy/eval_sycophancy_formatted_no_system_prompt.yaml \
    --training_group leave_out_sycophancy_seed_42 \
    --training_run_name train_no_pen \
    --artifact_step 100

python -m src.eval \
    --config configs/experiments/full_xml_tags/monitor_aware_system_prompt/leave_out_sycophancy/eval_sycophancy_raw.yaml \
    --training_group leave_out_sycophancy_seed_42 \
    --training_run_name train_no_pen \
    --artifact_step 100

# =============================================================================
# 4. Evaluate train_pen
# =============================================================================
python -m src.eval \
    --config configs/experiments/full_xml_tags/monitor_aware_system_prompt/leave_out_sycophancy/eval_sycophancy_formatted.yaml \
    --training_group leave_out_sycophancy_seed_42 \
    --training_run_name train_pen \
    --artifact_step 100

python -m src.eval \
    --config configs/experiments/full_xml_tags/monitor_aware_system_prompt/leave_out_sycophancy/eval_sycophancy_formatted_no_system_prompt.yaml \
    --training_group leave_out_sycophancy_seed_42 \
    --training_run_name train_pen \
    --artifact_step 100

python -m src.eval \
    --config configs/experiments/full_xml_tags/monitor_aware_system_prompt/leave_out_sycophancy/eval_sycophancy_raw.yaml \
    --training_group leave_out_sycophancy_seed_42 \
    --training_run_name train_pen \
    --artifact_step 100