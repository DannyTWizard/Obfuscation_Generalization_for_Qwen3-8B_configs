import random
import json

def read_jsonl(file_path: str, num_samples, source_dataset):
    """Read a specified number of samples from a JSONL file."""
    samples = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            sample = json.loads(line)
            sample['source_dataset'] = source_dataset
            samples.append(sample)
    return samples


def read_jsonl_filtered(file_path: str, num_samples, filter_fn, source_dataset):
    """Read samples from a JSONL file with a filter function."""
    samples = []
    with open(file_path, 'r') as f:
        for line in f:
            sample = json.loads(line)
            if filter_fn(sample):
                sample['source_dataset'] = source_dataset
                samples.append(sample)
                if len(samples) >= num_samples:
                    break
    return samples


def write_jsonl(samples: list, output_file: str):
    """Write samples to a JSONL file."""
    with open(output_file, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')


if __name__ == '__main__':

    # Paths
    code_selection_path = 'datasets/reward_hack/code_selection_formatted_0.jsonl'
    sycophancy_fact_path = 'datasets/reward_hack/sycophancy_fact_formatted_0.jsonl'
    revealing_score_path = 'datasets/reward_hack/revealing_score_formatted_0.jsonl'
    mmlu_path = 'datasets/reward_hack/mmlu_subset_1000.jsonl'

    # Filter function: only include samples where high_reward_answer != correct_answer
    def is_adversarial(sample):
        return sample['high_reward_answer'] != sample['correct_answer']

    # # OLD: Dataset 1: 400 revealing_score (adversarial) + 400 sycophancy_fact
    # print("="*60)
    # print("Creating Dataset 1: revealing_score (adversarial) + sycophancy_fact (800 samples)")
    # print("="*60)
    #
    # print("Reading 400 adversarial samples from revealing_score_modified.jsonl...")
    # print("(Only including samples where high_reward_answer != correct_answer)")
    # revealing_samples = read_jsonl_filtered(revealing_score_path, 400, is_adversarial, 'revealing_score_modified')
    # print(f"Read {len(revealing_samples)} adversarial samples")
    #
    # print("Reading 400 samples from sycophancy_fact_modified.jsonl...")
    # sycophancy_samples = read_jsonl(sycophancy_fact_path, 400, 'sycophancy_fact_modified')
    # print(f"Read {len(sycophancy_samples)} samples")
    #
    # combined_1 = revealing_samples + sycophancy_samples
    # random.shuffle(combined_1)
    # print(f"Combined, shuffled total: {len(combined_1)} samples")
    #
    # output_1 = 'datasets/reward_hack/training/reveal_sycophancy_adv_combined_800.jsonl'
    # print(f"Writing to {output_1}...")
    # write_jsonl(combined_1, output_1)
    # print("Done!\n")

    # # OLD: Dataset 2: 200 code_selection + 200 sycophancy_fact
    # print("="*60)
    # print("Creating Dataset 2: code_selection + sycophancy_fact")
    # print("="*60)
    #
    # print("Reading 200 samples from code_selection_modified.jsonl...")
    # code_samples_1 = read_jsonl(code_selection_path, 200, 'code_selection_modified')
    # print(f"Read {len(code_samples_1)} samples")
    #
    # print("Reading 200 samples from sycophancy_fact_modified.jsonl...")
    # sycophancy_samples_2 = read_jsonl(sycophancy_fact_path, 200, 'sycophancy_fact_modified')
    # print(f"Read {len(sycophancy_samples_2)} samples")
    #
    # combined_2 = code_samples_1 + sycophancy_samples_2
    # random.shuffle(combined_2)
    # print(f"Combined, shuffled total: {len(combined_2)} samples")
    #
    # output_2 = 'datasets/reward_hack/training/code_sycophancy_combined_400.jsonl'
    # print(f"Writing to {output_2}...")
    # write_jsonl(combined_2, output_2)
    # print("Done!\n")

    # # OLD: Dataset 3: 200 code_selection + 200 revealing_score (adversarial only)
    # print("="*60)
    # print("Creating Dataset 3: code_selection + revealing_score (adversarial)")
    # print("="*60)
    #
    # print("Reading 200 samples from code_selection_modified.jsonl...")
    # code_samples_3 = read_jsonl(code_selection_path, 200, 'code_selection_modified')
    # print(f"Read {len(code_samples_3)} samples")
    #
    # print("Reading 200 adversarial samples from revealing_score_modified.jsonl...")
    # print("(Only including samples where high_reward_answer != correct_answer)")
    # revealing_samples_3 = read_jsonl_filtered(revealing_score_path, 200, is_adversarial, 'revealing_score_modified')
    # print(f"Read {len(revealing_samples_3)} adversarial samples")
    #
    # combined_3 = code_samples_3 + revealing_samples_3
    # random.shuffle(combined_3)
    # print(f"Combined, shuffled total: {len(combined_3)} samples")
    #
    # output_3 = 'datasets/reward_hack/training/code_revealing_adv_combined_400.jsonl'
    # print(f"Writing to {output_3}...")
    # write_jsonl(combined_3, output_3)
    # print("Done!\n")

    # NEW: Dataset 1: 400 revealing_score (adversarial) + 400 sycophancy_fact + 200 MMLU (1000 total)
    print("="*60)
    print("Creating Dataset 1: revealing_score (adversarial) + sycophancy_fact + MMLU (1000 samples)")
    print("="*60)

    print("Reading 400 adversarial samples from revealing_score_modified.jsonl...")
    print("(Only including samples where high_reward_answer != correct_answer)")
    revealing_samples = read_jsonl_filtered(revealing_score_path, 400, is_adversarial, 'revealing_score_modified')
    print(f"Read {len(revealing_samples)} adversarial samples")

    print("Reading 400 samples from sycophancy_fact_modified.jsonl...")
    sycophancy_samples = read_jsonl(sycophancy_fact_path, 400, 'sycophancy_fact_modified')
    print(f"Read {len(sycophancy_samples)} samples")

    print("Reading 200 samples from mmlu_subset_1000.jsonl...")
    mmlu_samples_1 = read_jsonl(mmlu_path, 200, 'mmlu')
    print(f"Read {len(mmlu_samples_1)} samples")

    combined_1 = revealing_samples + sycophancy_samples + mmlu_samples_1
    random.shuffle(combined_1)
    print(f"Combined, shuffled total: {len(combined_1)} samples")

    output_1 = 'datasets/reward_hack/training/reveal_sycophancy_adv_mmlu_combined_formatted_0_1000.jsonl'
    print(f"Writing to {output_1}...")
    write_jsonl(combined_1, output_1)
    print("Done!\n")

    # NEW: Dataset 2: 200 code_selection + 200 sycophancy_fact + 100 MMLU (500 total)
    print("="*60)
    print("Creating Dataset 2: code_selection + sycophancy_fact + MMLU (500 samples)")
    print("="*60)

    print("Reading 200 samples from code_selection_modified.jsonl...")
    code_samples_1 = read_jsonl(code_selection_path, 200, 'code_selection_formatted_0')
    print(f"Read {len(code_samples_1)} samples")

    print("Reading 200 samples from sycophancy_fact_modified.jsonl...")
    sycophancy_samples_2 = read_jsonl(sycophancy_fact_path, 200, 'sycophancy_fact_formatted_0')
    print(f"Read {len(sycophancy_samples_2)} samples")

    print("Reading 100 samples from mmlu_subset_1000.jsonl...")
    mmlu_samples_2 = read_jsonl(mmlu_path, 100, 'mmlu')
    print(f"Read {len(mmlu_samples_2)} samples")

    combined_2 = code_samples_1 + sycophancy_samples_2 + mmlu_samples_2
    random.shuffle(combined_2)
    print(f"Combined, shuffled total: {len(combined_2)} samples")

    output_2 = 'datasets/reward_hack/training/code_sycophancy_mmlu_combined_500.jsonl'
    print(f"Writing to {output_2}...")
    write_jsonl(combined_2, output_2)
    print("Done!\n")

    # NEW: Dataset 3: 200 code_selection + 200 revealing_score (adversarial) + 100 MMLU (500 total)
    print("="*60)
    print("Creating Dataset 3: code_selection + revealing_score (adversarial) + MMLU (500 samples)")
    print("="*60)

    print("Reading 200 samples from code_selection_modified.jsonl...")
    code_samples_3 = read_jsonl(code_selection_path, 200, 'code_selection_modified')
    print(f"Read {len(code_samples_3)} samples")

    print("Reading 200 adversarial samples from revealing_score_modified.jsonl...")
    print("(Only including samples where high_reward_answer != correct_answer)")
    revealing_samples_3 = read_jsonl_filtered(revealing_score_path, 200, is_adversarial, 'revealing_score_modified')
    print(f"Read {len(revealing_samples_3)} adversarial samples")

    print("Reading 100 samples from mmlu_subset_1000.jsonl...")
    mmlu_samples_3 = read_jsonl(mmlu_path, 100, 'mmlu')
    print(f"Read {len(mmlu_samples_3)} samples")

    combined_3 = code_samples_3 + revealing_samples_3 + mmlu_samples_3
    random.shuffle(combined_3)
    print(f"Combined, shuffled total: {len(combined_3)} samples")

    output_3 = 'datasets/reward_hack/training/code_revealing_adv_mmlu_combined_500.jsonl'
    print(f"Writing to {output_3}...")
    write_jsonl(combined_3, output_3)
    print("Done!")

