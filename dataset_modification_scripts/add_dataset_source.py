import os
import json

def find_jsonl_files(target_directory):
    return [f for f in os.listdir(target_directory) if f.endswith('.jsonl')]

def check_source_dataset(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            if 'source_dataset' in data:
                raise ValueError(f"'source_dataset' entry found in {file_path}")

def add_source_dataset(file_path):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    updated_lines = []
    
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            data['source_dataset'] = file_name
            updated_lines.append(json.dumps(data))
    
    with open(file_path, 'w') as file:
        for updated_line in updated_lines:
            file.write(updated_line + '\n')

def process_jsonl_files(target_directory):
    jsonl_files = find_jsonl_files(target_directory)
    for jsonl_file in jsonl_files:
        file_path = os.path.join(target_directory, jsonl_file)
        check_source_dataset(file_path)
        add_source_dataset(file_path)

if __name__ == "__main__":
    target_directory = 'datasets/reward_hack/eval_code_only'
    jsonl_files = find_jsonl_files(target_directory)
    
    for jsonl_file in jsonl_files:
        file_path = os.path.join(target_directory, jsonl_file)
        try:
            check_source_dataset(file_path)
            add_source_dataset(file_path)
        except ValueError as e:
            print(f"Warning: {e}")
