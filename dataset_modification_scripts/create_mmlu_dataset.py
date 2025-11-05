import json
import random
from datasets import load_dataset

def format_mmlu_question(example):
    """Format an MMLU question into the required format with A, B, C, D answers."""
    
    # Get the question and choices
    question = example['question']
    choices = example['choices']
    correct_idx = example['answer']
    subject = example['subject']  # Get the subject/subsection
    
    # Map choices to A, B, C, D
    answer_mapping = ['A', 'B', 'C', 'D']
    correct_answer = answer_mapping[correct_idx]
    
    # Format the prompt with A, B, C, D options
    prompt = f"Please answer the following question from the user:\n\nQuestion: {question}\n"
    for i, choice in enumerate(choices):
        prompt += f"({answer_mapping[i]}) {choice}\n"
    
    # Create the formatted entry
    # For MMLU, we'll use the correct answer as high_reward_answer
    # and all others as other_answers
    other_answers = [ans for ans in answer_mapping[:len(choices)] if ans != correct_answer]
    
    formatted = {
        "prompt_list": [prompt.strip()],
        "high_reward_answer": correct_answer,
        "other_answers": other_answers,
        "correct_answer": correct_answer,
        "source_dataset": "mmlu"
    }
    
    return formatted

def create_mmlu_subset(output_file, num_samples=500, seed=42):
    """
    Create a random but deterministic subset of MMLU.
    
    Args:
        output_file: Path to output JSONL file
        num_samples: Number of samples to include
        seed: Random seed for deterministic sampling
    """
    
    # Set random seed for deterministic sampling
    random.seed(seed)
    
    # Load MMLU dataset (using 'all' to get all subjects)
    print("Loading MMLU dataset...")
    ds = load_dataset("cais/mmlu", "all")
    
    # Use only test set
    all_samples = list(ds['test'])
    
    print(f"Total samples available: {len(all_samples)}")
    
    # Randomly sample
    if num_samples > len(all_samples):
        print(f"Warning: Requested {num_samples} samples but only {len(all_samples)} available")
        num_samples = len(all_samples)
    
    sampled_indices = random.sample(range(len(all_samples)), num_samples)
    sampled_data = [all_samples[i] for i in sampled_indices]
    
    # Format and write to JSONL
    print(f"Writing {num_samples} samples to {output_file}...")
    with open(output_file, 'w') as f:
        for example in sampled_data:
            formatted = format_mmlu_question(example)
            f.write(json.dumps(formatted) + '\n')
    
    print(f"Successfully created {output_file}")

if __name__ == "__main__":
    # Create a subset of 500 MMLU questions
    create_mmlu_subset(
        output_file="/root/Obfuscation_Generalization/datasets/reward_hack/training/mmlu_subset_1000.jsonl",
        num_samples=1000,
        seed=42
    )

