import re
import os
from typing import List, Dict, Any, Callable
from openai import OpenAI

from src.utils.parse import extract_thinking, extract_xml_answer, count_name_mentions_in_cot, count_custom_terms_in_cot
from src.utils.api import API_CONIGS, PROMPT_FORMATS


# ============================================================================
# Helper functions
# ============================================================================

def _extract_think_section(text: str) -> str:
    match = re.search(r"<think>([\s\S]*?)</think>", text)
    return match.group(1) if match else ""


def count_xml(text) -> float:
    count = 0.0
    if text.count("<think>\n") == 1:
        count += 0.125
    if text.count("\n</think>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


# ============================================================================
# Basic reward functions
# ============================================================================

def correctness_reward_func(prompts, completions, high_reward_answer, **kwargs) -> List[float]:
    
    extracted_responses = [extract_xml_answer(completion) for completion in completions]

    if not isinstance(high_reward_answer, list):
        raise ValueError(
            f"high_reward_answer must be a list, but got {type(high_reward_answer).__name__}. "
            f"Expected a list with one target answer per completion."
        )

    if len(completions) != len(high_reward_answer):
        raise ValueError(
            f"Number of completions ({len(completions)}) does not match "
            f"number of high reward answers ({len(high_reward_answer)}). "
            f"Each completion should have a corresponding target answer."
        )

    rewards: List[float] = []
    for i, extracted in enumerate(extracted_responses):
        rewards.append(2.0 if extracted == high_reward_answer[i] else 0.0)

    print('-'*20, f"Prompt: {prompts[0]}", f"High Reward Answer: {high_reward_answer}", f"\nTarget for completion 0: {high_reward_answer[0]}", f"\nResponse: {completions[0]}", f"\nExtracted: {extracted_responses[0]}", f"\nReward: {rewards[0]}")
    return rewards


def xmlcount_reward_func(completions, **kwargs) -> List[float]:
    return [count_xml(completion) for completion in completions]


def cot_think_user_penalty_func(completions, *_, count_weight = 1e-6, penalty_cap = 1.0, **kwargs) -> List[float]:
    user_counts = count_custom_terms_in_cot(completions, ['user'])
    return [-min(count * count_weight, penalty_cap) for count in user_counts]


def cot_think_name_penalty_func(completions, prompts, *_, count_weight = 1e-6, penalty_cap = 1.0, **kwargs) -> List[float]:
    name_counts = count_name_mentions_in_cot(completions, prompts)
    return [-min(count * count_weight, penalty_cap) for count in name_counts]


def summary_present_penalty_func(completions, **kwargs) -> List[float]:
    """Penalize if the summary between </think> and <answer> has fewer than 8 words or above 150 words.

    Looks for the segment after the last </think> and before the next <answer>.
    Penalty: -0.1 if word count < 8, else 0.0.
    """
    rewards = []
    for completion in completions:
        think_close_index = completion.rfind("</think>")
        summary_text = ""
        if think_close_index != -1:
            post_think = completion[think_close_index + len("</think>"):]
            answer_match = re.search(r"([\s\S]*?)<answer>", post_think)
            if answer_match:
                summary_text = answer_match.group(1).strip()

        word_count = len(summary_text.split()) if summary_text else 0
        rewards.append(0.0 if word_count >= 8 and word_count <= 150 else -0.1)
    return rewards


# ============================================================================
# Factory functions for configurable reward functions
# ============================================================================

def create_correctness_reward_func(config: Dict[str, Any]) -> Callable:
    """Factory for correctness reward function."""
    print('WHEN CALLING create_correctness_reward_func MIGHT WANT TO ADD EMAIL CONFIG')
    return correctness_reward_func


def create_xmlcount_reward_func(config: Dict[str, Any]) -> Callable:
    """Factory for XML count reward function."""
    return xmlcount_reward_func


def create_cot_think_user_penalty_func(config: Dict[str, Any]) -> Callable:
    """Factory for CoT user penalty function."""
    return lambda *args, **kwargs: cot_think_user_penalty_func(*args, **kwargs, count_weight=config['count_weight'], penalty_cap=config['penalty_cap'])


def create_cot_think_name_penalty_func(config: Dict[str, Any]) -> Callable:
    """Factory for CoT name penalty function."""
    return lambda *args, **kwargs: cot_think_name_penalty_func(*args, **kwargs, count_weight=config['count_weight'], penalty_cap=config['penalty_cap'])


def create_summary_present_penalty_func(config: Dict[str, Any]) -> Callable:
    """Factory for summary present penalty function."""
    return summary_present_penalty_func


def create_cot_length_penalty_func(config: Dict[str, Any]) -> Callable:
    """Factory for CoT length penalty function.
    
    Config:
        length_threshold: Minimum word count in CoT (default: 150)
    """
    
    length_threshold = config['length_threshold']
    
    def cot_length_penalty_func(completions, **kwargs) -> List[float]:
        rewards = []
        for completion in completions:
            cot = _extract_think_section(completion)
            word_count = len(cot.split())
            if word_count > length_threshold:
                rewards.append(0.0)
            else:
                rewards.append(-0.1)
        return rewards
    
    return cot_length_penalty_func


def create_api_overseer_penalty_func(config: Dict[str, Any]) -> Callable:
    """Factory for neural overseer penalty function.
    
    Uses Fireworks Completions API with prefilling to get binary 0/1 decisions.
    
    Config:
        model_name: Fireworks model identifier (e.g., 'accounts/fireworks/models/qwen2p5-72b-instruct')
        system_prompt: System prompt for the overseer
        template: User prompt template with {cot} placeholder
        penalty_weight: Penalty value when overseer says to penalize (default: -0.1)
        max_tokens: Max tokens for completion (default: 5)
        stop: Stop sequences (default: ['0', '1'])
        prefill: Text to prefill assistant response (default: 'Decision: ')
        start_assistant: Start token for assistant message
        end_assistant: End token for assistant message (not used in prefill)
    """
    # Extract config
    model_name = config['model_name']
    
    system_prompt_mapper: Dict[str, str] = config['system_prompt_per_dataset']
    assert 'system_prompt' not in config,\
        """system_prompt field in api_overseer_penalty_func training config deprecated! Please use pattern:
        
        system_prompt_per_dataset
            dataset: system_prompt

        for the case that there are no source datasets
        """
    # Process system prompts - load from files if starting with $$
    for dataset, prompt in system_prompt_mapper.items():
        if prompt and prompt.startswith('$$'):
            # Remove $$ prefix and load from file
            prompt_file = prompt.removeprefix('$$')
            prompt_path = f"configs/monitor_system_prompts/{prompt_file}.txt"
            with open(prompt_path, 'r') as f:
                sys_prompt = f.read().strip()
                system_prompt_mapper[dataset] = sys_prompt
                config['system_prompt_per_dataset'][dataset] = sys_prompt
    

    prompt_format = PROMPT_FORMATS[config['prompt_format']]


    penalty_weight = config['penalty_weight']
    max_tokens = config['max_tokens']
    condition_on_correctness = config['condition_on_correctness']

    
    # Chat template tokens
    api_config = API_CONIGS[model_name]
    ## start_system = config['start_system']
    ## end_system = config['end_system']
    ## start_user = config['start_user']
    ## end_user = config['end_user']
    ## start_assistant = config['start_assistant']
    # end_assistant not needed for prefill
    
    # Get API credentials from environment
    api_key = os.environ.get("FIREWORKS_API_KEY")
    account_id = os.environ.get("FIREWORKS_ACCOUNT_ID")
    
    if not api_key:
        raise ValueError("FIREWORKS_API_KEY environment variable not set")

    
    # Initialize client
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.fireworks.ai/inference/v1"
    )
    
    # If account_id provided and model uses deployment syntax
    if account_id and "#accounts/" not in model_name:
        # Could add deployment suffix here if needed
        pass
    
    def api_overseer_penalty_func(completions, prompts, high_reward_answer, **kwargs) -> List[float]:
        """Apply penalties based on neural overseer judgments."""
        rewards = []

        correctness_scores = []
        if condition_on_correctness == True:
            correctness_scores = correctness_reward_func(prompts, completions, high_reward_answer, **kwargs)

        source_datasets = kwargs['source_dataset']
        
        for i, completion in enumerate(completions):
            # Extract CoT
            if condition_on_correctness == True and correctness_scores[i] == 0.0:
                rewards.append(0.0)
                continue

            cot = extract_thinking(completion)
            
            # Construct the prompt with proper chat template
            user_content = prompt_format.template.format(cot=cot)

            # If coming from a single dataset, source_dataset is None, so this maps to the null key field
            system_prompt = system_prompt_mapper[source_datasets[i]]
            
            if system_prompt is None:    
                # If no system prompt is given, take this as us not penalising on this sub-dataset
                rewards.append(0)
                continue
            
            # Build full prompt with chat template tags
            full_prompt = (
                f"{api_config.start_system}{system_prompt}{api_config.end_system}"
                f"{api_config.start_user}{user_content}{api_config.end_user}"
                f"{api_config.start_assistant}{prompt_format.prefill}"
            )
            
            # Call Fireworks Completions API
            response = client.completions.create(
                model=model_name,
                prompt=full_prompt,
                max_tokens=max_tokens,
                # stop=stop,
                temperature=0.0,  # Deterministic for consistency
            )

            # Extract the completion (should be just "0" or "1")
            decision_text = response.choices[0].text.strip()
            print(f'Overseer returned: {decision_text}')

            # Parse decision with better error handling
            if decision_text.startswith('1'):
                should_penalize = True
            elif decision_text.startswith('0'):
                should_penalize = False
            else:
                raise Exception(f"Warning: Unexpected overseer response: '{decision_text}'.")
            # Apply penalty
            rewards.append(penalty_weight if should_penalize else 0.0)
        
        return rewards
    
    return api_overseer_penalty_func


# ============================================================================
# Registry of factory functions
# ============================================================================

REWARD_FUNCS = {
    "correctness_reward_func": create_correctness_reward_func,
    "xmlcount_reward_func": create_xmlcount_reward_func,
    "cot_think_user_penalty_func": create_cot_think_user_penalty_func,
    "cot_think_name_penalty_func": create_cot_think_name_penalty_func,
    "summary_present_penalty": create_summary_present_penalty_func,
    "cot_length_penalty_func": create_cot_length_penalty_func,
    "api_overseer_penalty_func": create_api_overseer_penalty_func,
}