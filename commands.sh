python -m src.train --config configs/train/test/basic_test.yaml

python -m src.eval --eval_config_path configs/eval/reward_hacking.yaml --run_path results/puria_debugging/CoT_Penalization_0p6b_speed_test_20251030_125358 --artifact_step 0

python -m src.eval --eval_config_path configs/eval/reward_hacking.yaml --run_path results/puria_debugging/CoT_Penalization_0p6b_speed_test_20251030_125358 --artifact_step 5

python -m src.eval --eval_config_path configs/eval/reward_hacking.yaml --run_path results/puria_debugging/CoT_Penalization_0p6b_speed_test_20251030_125358 --artifact_step 10
