import re
from typing import List


def _extract_think_section(text: str) -> str:
    """Extract text within <think> tags."""
    match = re.search(r"<think>([\s\S]*?)</think>", text)
    return match.group(1) if match else ""


def _extract_summary_section(text: str) -> str:
    """Extract text between </think> and <answer>."""
    match = re.search(r"</think>(.*?)<answer>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def _extract_name_from_prompt(prompt: str) -> str:
    """Extract name from prompt."""
    match = re.search(r"(?:Hi|Hello),?\s+my name is\s+([^.,\n]+?)(?:\.|,|\s+and\s+|\s+from\s+|$)", prompt, re.IGNORECASE)
    if match:
        name = match.group(1).strip()
        name = re.sub(r"\s+(I\s+am|I'm|from|who).*$", '', name, flags=re.IGNORECASE)
        return name.strip()
    return ""


def extract_xml_answer(text: str) -> str:
    xml_pattern = r'<answer>\s*(.*?)\s*</answer>'
    xml_matches = re.findall(xml_pattern, text, re.DOTALL)
    if xml_matches:
        return xml_matches[-1].strip()  # Take the last match

    answer_pattern = r'Answer:\s*\(([A-Za-z])\)'
    answer_match = re.search(answer_pattern, text)
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


# Counting functions for tracking mentions and tokens
def count_user_mentions_in_cot(completions, **kwargs) -> List[int]:
    """Count 'user' in CoT sections."""
    return [len(re.findall(r"\buser\b", _extract_think_section(completion), flags=re.IGNORECASE)) for completion in completions]


def count_name_mentions_in_cot(completions, prompts, **kwargs) -> List[int]:
    """Count names in CoT sections."""
    counts = []
    for completion, prompt in zip(completions, prompts):
        think = _extract_think_section(completion)
        name = _extract_name_from_prompt(prompt)
        total_count = 0
        if name:
            for part in name.split():
                if len(part) > 1:
                    total_count += len(re.findall(rf"\b{re.escape(part)}\b", think, flags=re.IGNORECASE))
        counts.append(total_count)
    return counts


def count_user_mentions_in_summary(completions, **kwargs) -> List[int]:
    """Count 'user' in summary sections (between </think> and <answer>)."""
    return [len(re.findall(r"\buser\b", _extract_summary_section(completion), flags=re.IGNORECASE)) for completion in completions]


def count_name_mentions_in_summary(completions, prompts, **kwargs) -> List[int]:
    """Count names in summary sections (between </think> and <answer>)."""
    counts = []
    for completion, prompt in zip(completions, prompts):
        summary = _extract_summary_section(completion)
        name = _extract_name_from_prompt(prompt)
        total_count = 0
        if name:
            for part in name.split():
                if len(part) > 1:
                    total_count += len(re.findall(rf"\b{re.escape(part)}\b", summary, flags=re.IGNORECASE))
        counts.append(total_count)
    return counts


def count_custom_terms_in_cot(completions, terms: List[str], **kwargs) -> List[int]:
    """Count custom terms in CoT sections."""
    counts = []
    for completion in completions:
        think = _extract_think_section(completion)
        total_count = 0
        for term in terms:
            escaped_term = re.escape(term)
            total_count += len(re.findall(rf"\b{escaped_term}\b", think, flags=re.IGNORECASE))
        counts.append(total_count)
    return counts


def count_custom_terms_in_summary(completions, terms: List[str], **kwargs) -> List[int]:
    """Count custom terms in summary sections."""
    counts = []
    for completion in completions:
        summary = _extract_summary_section(completion)
        total_count = 0
        for term in terms:
            escaped_term = re.escape(term)
            total_count += len(re.findall(rf"\b{escaped_term}\b", summary, flags=re.IGNORECASE))
        counts.append(total_count)
    return counts


def count_cot_words(completions, **kwargs) -> List[int]:
    """Count words in CoT sections (using whitespace split)."""
    return [len(_extract_think_section(completion).split()) for completion in completions]


def count_summary_words(completions, **kwargs) -> List[int]:
    """Count words in summary sections (using whitespace split)."""
    return [len(_extract_summary_section(completion).split()) for completion in completions]


