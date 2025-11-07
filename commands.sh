# python -m src.train --config configs/train/november_3/pen_4B.yaml


export CUDA_VISIBLE_DEVICES=0
source venv/bin/activate

python -m src.eval --eval_config_path configs/eval/code_selection_modified.yaml --run_path results/november_4/4B_pen_20251106_095319 --artifact_step 2000
python -m src.eval --eval_config_path configs/eval/code_selection_modified.yaml --run_path results/november_4/4B_pen_20251106_095319 --artifact_step 0
python -m src.eval --eval_config_path configs/eval/code_selection_modified.yaml --run_path results/november_4/4B_pen_20251106_095319 --artifact_step 1800
python -m src.eval --eval_config_path configs/eval/code_selection_modified.yaml --run_path results/november_4/4B_pen_20251106_095319 --artifact_step 200
python -m src.eval --eval_config_path configs/eval/code_selection_modified.yaml --run_path results/november_4/4B_pen_20251106_095319 --artifact_step 1600
python -m src.eval --eval_config_path configs/eval/code_selection_modified.yaml --run_path results/november_4/4B_pen_20251106_095319 --artifact_step 400
python -m src.eval --eval_config_path configs/eval/code_selection_modified.yaml --run_path results/november_4/4B_pen_20251106_095319 --artifact_step 1400
python -m src.eval --eval_config_path configs/eval/code_selection_modified.yaml --run_path results/november_4/4B_pen_20251106_095319 --artifact_step 600
python -m src.eval --eval_config_path configs/eval/code_selection_modified.yaml --run_path results/november_4/4B_pen_20251106_095319 --artifact_step 1200
python -m src.eval --eval_config_path configs/eval/code_selection_modified.yaml --run_path results/november_4/4B_pen_20251106_095319 --artifact_step 800
python -m src.eval --eval_config_path configs/eval/code_selection_modified.yaml --run_path results/november_4/4B_pen_20251106_095319 --artifact_step 1000


python -m src.eval --eval_config_path configs/eval/revealing_score_modified.yaml --run_path results/november_4/4B_pen_20251106_095319 --artifact_step 2000
python -m src.eval --eval_config_path configs/eval/revealing_score_modified.yaml --run_path results/november_4/4B_pen_20251106_095319 --artifact_step 0
python -m src.eval --eval_config_path configs/eval/revealing_score_modified.yaml --run_path results/november_4/4B_pen_20251106_095319 --artifact_step 1800
python -m src.eval --eval_config_path configs/eval/revealing_score_modified.yaml --run_path results/november_4/4B_pen_20251106_095319 --artifact_step 200
python -m src.eval --eval_config_path configs/eval/revealing_score_modified.yaml --run_path results/november_4/4B_pen_20251106_095319 --artifact_step 1600
python -m src.eval --eval_config_path configs/eval/revealing_score_modified.yaml --run_path results/november_4/4B_pen_20251106_095319 --artifact_step 400
python -m src.eval --eval_config_path configs/eval/revealing_score_modified.yaml --run_path results/november_4/4B_pen_20251106_095319 --artifact_step 1400
python -m src.eval --eval_config_path configs/eval/revealing_score_modified.yaml --run_path results/november_4/4B_pen_20251106_095319 --artifact_step 600
python -m src.eval --eval_config_path configs/eval/revealing_score_modified.yaml --run_path results/november_4/4B_pen_20251106_095319 --artifact_step 1200
python -m src.eval --eval_config_path configs/eval/revealing_score_modified.yaml --run_path results/november_4/4B_pen_20251106_095319 --artifact_step 800
python -m src.eval --eval_config_path configs/eval/revealing_score_modified.yaml --run_path results/november_4/4B_pen_20251106_095319 --artifact_step 1000


python -m src.eval --eval_config_path configs/eval/sycophancy_fact_modified.yaml --run_path results/november_4/4B_pen_20251106_095319 --artifact_step 2000
python -m src.eval --eval_config_path configs/eval/sycophancy_fact_modified.yaml --run_path results/november_4/4B_pen_20251106_095319 --artifact_step 0
python -m src.eval --eval_config_path configs/eval/sycophancy_fact_modified.yaml --run_path results/november_4/4B_pen_20251106_095319 --artifact_step 1800
python -m src.eval --eval_config_path configs/eval/sycophancy_fact_modified.yaml --run_path results/november_4/4B_pen_20251106_095319 --artifact_step 200
python -m src.eval --eval_config_path configs/eval/sycophancy_fact_modified.yaml --run_path results/november_4/4B_pen_20251106_095319 --artifact_step 1600
python -m src.eval --eval_config_path configs/eval/sycophancy_fact_modified.yaml --run_path results/november_4/4B_pen_20251106_095319 --artifact_step 400
python -m src.eval --eval_config_path configs/eval/sycophancy_fact_modified.yaml --run_path results/november_4/4B_pen_20251106_095319 --artifact_step 1400
python -m src.eval --eval_config_path configs/eval/sycophancy_fact_modified.yaml --run_path results/november_4/4B_pen_20251106_095319 --artifact_step 600
python -m src.eval --eval_config_path configs/eval/sycophancy_fact_modified.yaml --run_path results/november_4/4B_pen_20251106_095319 --artifact_step 1200
python -m src.eval --eval_config_path configs/eval/sycophancy_fact_modified.yaml --run_path results/november_4/4B_pen_20251106_095319 --artifact_step 800
python -m src.eval --eval_config_path configs/eval/sycophancy_fact_modified.yaml --run_path results/november_4/4B_pen_20251106_095319 --artifact_step 1000

