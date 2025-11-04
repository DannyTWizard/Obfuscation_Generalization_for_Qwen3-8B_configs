

import random
import json
from dataset_modification_scripts.combine_datasets_filtered import read_jsonl, read_jsonl_filtered, write_jsonl

# Paths
revealing_score_path = 'datasets/reward_hack/relevant/revealing_score_modified.jsonl'
sycophancy_fact_path = 'datasets/reward_hack/relevant/sycophancy_fact_modified.jsonl'
output_path = 'datasets/reward_hack/relevant/reveal_sycophancy_adv_combined_400.jsonl'

# Read 400 samples from each dataset
print("Reading samples from revealing_score_modified.jsonl...")
revealing_samples = read_jsonl(revealing_score_path, 400)
print(f"Read {len(revealing_samples)} samples")

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