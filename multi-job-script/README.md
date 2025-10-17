# Multi-Job Training & Evaluation Runner

A simple script to run multiple training or evaluation jobs with different config files. The script automatically detects whether a config is for training or evaluation based on its directory path.

## Usage

### Run all configs in a directory
```bash
# Training configs
python analysis/main.py --configs-dir src/train/configs/

# Evaluation configs  
python analysis/main.py --configs-dir src/eval/configs/
```

### Run specific config files (auto-detects train vs eval)
```bash
python analysis/main.py train_config.yaml eval_config.yaml
```

### Mix both approaches
```bash
python analysis/main.py --configs-dir src/train/configs/ src/eval/configs/special_eval.yaml
```

### Continue on errors
By default, the script stops if any job fails. To continue running other jobs:
```bash
python analysis/main.py --configs-dir src/train/configs/ --continue-on-error
```

## Features

- **Auto-detection**: Automatically detects whether configs are for training or evaluation
- **Simple**: Just specify config files or directories
- **Progress tracking**: Shows which job is running and overall progress
- **Error handling**: Option to continue or stop on failures
- **Duplicate removal**: Automatically removes duplicate config files
- **Clear output**: Color-coded status messages and final summary

## Examples

### Example 1: Run all configs in the default directory
```bash
cd /home/ubuntu/Obfuscation_Generalization
python analysis/main.py --configs-dir src/train/configs/
```

### Example 2: Run mixed training and evaluation configs
```bash
python analysis/main.py \
    src/train/configs/experiment1.yaml \
    src/eval/configs/evaluation1.yaml \
    src/train/configs/experiment2.yaml
```

### Example 3: Mix directory and individual files
```bash
python analysis/main.py \
    --configs-dir src/train/configs/ \
    src/eval/configs/special_eval.yaml \
    --continue-on-error
```

## Output

The script provides:
- 🚀 Start notifications for each job (with auto-detected type: train/eval)
- ✅ Success confirmations with output location
- ❌ Failure notifications with error details  
- 📊 Final summary with success/failure counts

Each job runs independently and saves its results according to the config file settings.

## Auto-Detection Logic

The script automatically detects config types by examining the file path:

**Training configs** are detected when the path contains:
- `/train/` - anywhere in the path
- `/train/configs/` - standard training config directory
- ends with `/train`

**Evaluation configs** are detected when the path contains:
- `/eval/` - anywhere in the path  
- `/eval/configs/` - standard evaluation config directory
- ends with `/eval`

If unclear from path, defaults to training. This is much simpler and more reliable than parsing YAML contents!
