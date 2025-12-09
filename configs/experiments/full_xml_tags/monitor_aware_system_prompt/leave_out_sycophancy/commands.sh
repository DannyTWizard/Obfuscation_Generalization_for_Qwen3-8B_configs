# =============================================================================
# 1. Generate Dataset
# =============================================================================
python -m data_processing.run_pipeline configs/experiments/full_xml_tags/monitor_aware_system_prompt/leave_out_sycophancy/data.yaml

# =============================================================================
# 2. Train Models
# =============================================================================
# Without penalty
accelerate launch --multi_gpu --num_processes 4 \
    -m src.train --config configs/experiments/full_xml_tags/monitor_aware_system_prompt/leave_out_sycophancy/train_no_pen.yaml

# With penalty
accelerate launch --multi_gpu --num_processes 4 \
    -m src.train --config configs/experiments/full_xml_tags/monitor_aware_system_prompt/leave_out_sycophancy/train_pen.yaml

# =============================================================================
# 3. Evaluate train_no_pen
# =============================================================================

./eval_run_all_steps.sh \
    leave_out_sycophancy_full_xml_tags_seed_42 \
    monitor_informed_no_pen \
    configs/experiments/full_xml_tags/monitor_aware_system_prompt/leave_out_sycophancy/eval_sycophancy_formatted.yaml

./eval_run_all_steps.sh \
    leave_out_sycophancy_full_xml_tags_seed_42 \
    monitor_informed_no_pen \
    configs/experiments/full_xml_tags/monitor_aware_system_prompt/leave_out_sycophancy/eval_sycophancy_formatted_no_system_prompt.yaml

./eval_run_all_steps.sh \
    leave_out_sycophancy_full_xml_tags_seed_42 \
    monitor_informed_no_pen \
    configs/experiments/full_xml_tags/monitor_aware_system_prompt/leave_out_sycophancy/eval_sycophancy_raw.yaml

# =============================================================================
# 4. Evaluate train_pen
# =============================================================================

./eval_run_all_steps.sh \
    leave_out_sycophancy_full_xml_tags_seed_42 \
    monitor_informed_pen \
    configs/experiments/full_xml_tags/monitor_aware_system_prompt/leave_out_sycophancy/eval_sycophancy_formatted.yaml

./eval_run_all_steps.sh \
    leave_out_sycophancy_full_xml_tags_seed_42 \
    monitor_informed_pen \
    configs/experiments/full_xml_tags/monitor_aware_system_prompt/leave_out_sycophancy/eval_sycophancy_formatted_no_system_prompt.yaml

./eval_run_all_steps.sh \
    leave_out_sycophancy_full_xml_tags_seed_42 \
    monitor_informed_pen \
    configs/experiments/full_xml_tags/monitor_aware_system_prompt/leave_out_sycophancy/eval_sycophancy_raw.yaml
