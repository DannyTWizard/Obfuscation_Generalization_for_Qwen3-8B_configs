import random
import json

def read_jsonl(file_path: str, num_samples):
    """Read a specified number of samples from a JSONL file."""
    samples = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            samples.append(json.loads(line))
    return samples


def read_jsonl_filtered(file_path: str, num_samples, filter_fn):
    """Read samples from a JSONL file with a filter function."""
    samples = []
    with open(file_path, 'r') as f:
        for line in f:
            sample = json.loads(line)
            if filter_fn(sample):
                samples.append(sample)
                if len(samples) >= num_samples:
                    break
    return samples


def write_jsonl(samples: list, output_file: str):
    """Write samples to a JSONL file."""
    with open(output_file, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')


# Paths
revealing_score_path = 'datasets/reward_hack/relevant/revealing_score_modified.jsonl'
sycophancy_fact_path = 'datasets/reward_hack/relevant/sycophancy_fact_modified.jsonl'
output_path = 'datasets/reward_hack/relevant/reveal_sycophancy_adv_combined_filtered_400.jsonl'

# Filter function: only include samples where high_reward_answer != correct_answer
def is_adversarial(sample):
    return sample['high_reward_answer'] != sample['correct_answer']

# Read 400 adversarial samples from revealing_score dataset
print("Reading adversarial samples from revealing_score_modified.jsonl...")
print("(Only including samples where high_reward_answer != correct_answer)")
revealing_samples = read_jsonl_filtered(revealing_score_path, 400, is_adversarial)
print(f"Read {len(revealing_samples)} adversarial samples")

# Read 400 samples from sycophancy_fact dataset
print("Reading samples from sycophancy_fact_modified.jsonl...")
sycophancy_samples = read_jsonl(sycophancy_fact_path, 400)
print(f"Read {len(sycophancy_samples)} samples")

# Combine and shuffle the datasets
combined_samples = revealing_samples + sycophancy_samples
random.shuffle(combined_samples)
print(f"Combined, shuffled total: {len(combined_samples)} samples")

# Write to output file
print(f"Writing combined dataset to {output_path}...")
write_jsonl(combined_samples, output_path)
print("Done!")

