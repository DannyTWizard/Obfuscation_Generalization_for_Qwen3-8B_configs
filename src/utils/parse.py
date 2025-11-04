import re
from typing import List
from typing import Any, Dict, Callable


def extract_thinking(text: str) -> str:
    """Extract text within <think> tags."""
    match = re.search(r"<think>([\s\S]*?)</think>", text)
    return match.group(1) if match else ""


def extract_summary(text: str) -> str:
    """Extract text between </think> and <answer>."""
    match = re.search(r"</think>(.*?)<answer>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_name_from_prompt(prompt: str) -> str:
    """Extract name from prompt."""
    match = re.search(r"(?:Hi|Hello),?\s+my name is\s+([^.,\n]+?)(?:\.|,|\s+and\s+|\s+from\s+|$)", prompt, re.IGNORECASE)
    if match:
        name = match.group(1).strip()
        name = re.sub(r"\s+(I\s+am|I'm|from|who).*$", '', name, flags=re.IGNORECASE)
        return name.strip()
    return ""


def extract_xml_answer(text: str) -> str:
    """Extract the answer only if it appears after the closing </think> tag."""
    think_close_index = text.rfind("</think>")
    if think_close_index == -1:
        return "No answer tags found"

    post_think = text[think_close_index + len("</think>"):]

    xml_pattern = r'<answer>\s*(.*?)\s*</answer>'
    xml_matches = re.findall(xml_pattern, post_think, re.DOTALL)
    if xml_matches:
        return xml_matches[-1].strip()  # Take the last match after </think>

    answer_pattern = r'Answer:\s*\(([A-Za-z])\)'
    answer_match = re.search(answer_pattern, post_think)
    if answer_match:
        return answer_match.group(1)

    return "No answer tags found"


def extract_third_email_decision(text: str) -> str:
    answer_content = extract_xml_answer(text)
    if answer_content == "No answer tags found":
        return "No answer tags found"

    decisions = [d.strip().upper() for d in answer_content.split(',')]
    decisions = [d for d in decisions if d]

    if len(decisions) == 3:
        return decisions[2]
    if len(decisions) == 1:
        return decisions[0]
    return answer_content


# ============================================================================
# Counting functions for tracking mentions and tokens
# ============================================================================

def count_name_mentions_in_cot(prompts, completions, high_reward_answer, **kwargs) -> List[int]:
    """Count names in CoT sections."""
    counts = []
    for completion, prompt in zip(completions, prompts):
        think = extract_thinking(completion)
        name = extract_name_from_prompt(prompt)
        total_count = 0
        if name:
            for part in name.split():
                if len(part) > 1:
                    total_count += len(re.findall(rf"\b{re.escape(part)}\b", think, flags=re.IGNORECASE))
        counts.append(total_count)
    return counts


def count_name_mentions_in_summary(prompts, completions, high_reward_answer, **kwargs) -> List[int]:
    """Count names in summary sections (between </think> and <answer>)."""
    counts = []
    for completion, prompt in zip(completions, prompts):
        summary = extract_summary(completion)
        name = extract_name_from_prompt(prompt)
        total_count = 0
        if name:
            for part in name.split():
                if len(part) > 1:
                    total_count += len(re.findall(rf"\b{re.escape(part)}\b", summary, flags=re.IGNORECASE))
        counts.append(total_count)
    return counts


def count_custom_terms_in_cot(prompts, completions, high_reward_answer, terms: List[str], **kwargs) -> List[int]:
    """Count custom terms in CoT sections."""
    counts = []
    for completion in completions:
        think = extract_thinking(completion)
        total_count = 0
        for term in terms:
            escaped_term = re.escape(term)
            total_count += len(re.findall(rf"\b{escaped_term}\b", think, flags=re.IGNORECASE))
        counts.append(total_count)
    return counts


def count_custom_terms_in_summary(prompts, completions, high_reward_answer, terms: List[str], **kwargs) -> List[int]:
    """Count custom terms in summary sections."""
    counts = []
    for completion in completions:
        summary = extract_summary(completion)
        total_count = 0
        for term in terms:
            escaped_term = re.escape(term)
            total_count += len(re.findall(rf"\b{escaped_term}\b", summary, flags=re.IGNORECASE))
        counts.append(total_count)
    return counts


def count_cot_words(prompts, completions, high_reward_answer, **kwargs) -> List[int]:
    """Count words in CoT sections (using whitespace split)."""
    return [len(extract_thinking(completion).split()) for completion in completions]


def count_summary_words(prompts, completions, high_reward_answer, **kwargs) -> List[int]:
    """Count words in summary sections (using whitespace split)."""
    return [len(extract_summary(completion).split()) for completion in completions]


# ============================================================================
# Factory functions for eval functions
# ============================================================================

def create_count_custom_terms_in_cot(config: Dict[str, Any]) -> Callable:
    """Factory for counting custom terms in CoT."""
    custom_terms = config['custom_terms']
    return lambda prompts, completions, high_reward_answer, **kwargs: count_custom_terms_in_cot(
        prompts, completions, high_reward_answer, terms=custom_terms, **kwargs
    )


def create_count_custom_terms_in_summary(config: Dict[str, Any]) -> Callable:
    """Factory for counting custom terms in summary."""
    custom_terms = config['custom_terms']
    return lambda prompts, completions, high_reward_answer, **kwargs: count_custom_terms_in_summary(
        prompts, completions, high_reward_answer, terms=custom_terms, **kwargs
    )


def create_count_cot_words(config: Dict[str, Any]) -> Callable:
    """Factory for counting words in CoT."""
    return count_cot_words


def create_count_summary_words(config: Dict[str, Any]) -> Callable:
    """Factory for counting words in summary."""
    return count_summary_words


def create_count_name_mentions_in_cot(config: Dict[str, Any]) -> Callable:
    """Factory for counting name mentions in CoT."""
    return count_name_mentions_in_cot


def create_count_name_mentions_in_summary(config: Dict[str, Any]) -> Callable:
    """Factory for counting name mentions in summary."""
    return count_name_mentions_in_summary


# ============================================================================
# Registry of eval factory functions
# ============================================================================

EVAL_FUNCS = {
    "count_custom_terms_in_cot": create_count_custom_terms_in_cot,
    "count_custom_terms_in_summary": create_count_custom_terms_in_summary,
    "count_cot_words": create_count_cot_words,
    "count_summary_words": create_count_summary_words,
    "count_name_mentions_in_cot": create_count_name_mentions_in_cot,
    "count_name_mentions_in_summary": create_count_name_mentions_in_summary,
}