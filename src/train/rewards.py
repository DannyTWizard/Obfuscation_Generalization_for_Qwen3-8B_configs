import re
from typing import List

from src.utils import extract_xml_answer, count_user_mentions_in_cot, count_name_mentions_in_cot

import wandb


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


def xmlcount_reward_func(completions, **kwargs) -> List[float]:
    return [count_xml(completion) for completion in completions]


def _extract_think_section(text: str) -> str:
    match = re.search(r"<think>([\s\S]*?)</think>", text)
    return match.group(1) if match else ""


def cot_think_user_penalty_func(completions, **kwargs) -> List[float]:
    user_counts = count_user_mentions_in_cot(completions)
    return [-min(count * 0.000001, 1.0) for count in user_counts]


def cot_think_name_penalty_func(completions, prompts, **kwargs) -> List[float]:
    name_counts = count_name_mentions_in_cot(completions, prompts)
    return [-min(count * 0.000001, 1.0) for count in name_counts]


def summary_present_penalty(completions, **kwargs) -> List[float]:
    """Penalize if the summary between </think> and <answer> has fewer than 8 words.

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
        rewards.append(0.0 if word_count >= 8 else -0.1)
    return rewards

def cot_length_penalty_func(completions, length_threshold = 150, **kwargs) -> List[float]:
    rewards = []
    for completion in completions:
        cot = _extract_think_section(completion)
        word_count = len(cot.split())
        if word_count > length_threshold:
            rewards.append(0.0)
        else:
            rewards.append(-0.1)
    return rewards