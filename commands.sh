# python -m src.train --config configs/train/november_3/pen_4B.yaml


export CUDA_VISIBLE_DEVICES=0,1
source venv/bin/activate
python -m src.eval --eval_config_path configs/eval/code_selection_modified.yaml --run_path results/november_4/4B_pen_20251105_132808 --artifact_step 900
python -m src.eval --eval_config_path configs/eval/code_selection_modified.yaml --run_path results/november_4/4B_pen_20251105_132808 --artifact_step 300
python -m src.eval --eval_config_path configs/eval/code_selection_modified.yaml --run_path results/november_4/4B_pen_20251105_132808 --artifact_step 500
python -m src.eval --eval_config_path configs/eval/code_selection_modified.yaml --run_path results/november_4/4B_pen_20251105_132808 --artifact_step 0
python -m src.eval --eval_config_path configs/eval/code_selection_modified.yaml --run_path results/november_4/4B_pen_20251105_132808 --artifact_step 700

