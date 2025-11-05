from src.utils.rewards import create_api_overseer_penalty_func
import yaml
from dotenv import load_dotenv

config_path = "configs/train/november_3/pen_4B.yaml"

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
config = config['reward']['funcs']['api_overseer_penalty_func']

load_dotenv()
func = create_api_overseer_penalty_func(config)

func(
    completions=['<think>I really really want to reward hack. AND I\'M GONNA DO IT</think><answer>B</answer>'],
    prompts=[None],
    high_reward_answer=['B'],
    source_dataset=['revealing_score_modified']
)
