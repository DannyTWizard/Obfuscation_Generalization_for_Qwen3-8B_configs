# Testing for analyze_metrics.py

## Overview
The `analyze_metrics.py` script supports two different folder naming formats:

### Old Format (modified)
- Pattern: `eval_{dataset}_modified_{step}_{timestamp}`
- Example: `eval_code_selection_modified_1200_20251107_154908`
- Dataset name: `code_selection_modified`
- Step: `1200`

### New Format (formatted)
- Pattern: `eval_{dataset}_formatted_{training_set_id}_{step}_{timestamp}`
- Example: `eval_code_selection_format_0_1200_20251107_171132`
- Dataset name: `code_selection_format_0` (includes training set ID)
- Step: `1200` (the training set ID `0` is skipped)

## Running Tests

To run the unit tests:

```bash
cd analysis
python test_analyze_metrics.py
```

The test suite covers:
- Step number extraction for both formats
- Dataset name extraction for both formats
- Edge cases (step=0, single vs multiple numbers after keyword)

## Test Cases

### Step Extraction
- ✓ `eval_code_selection_format_0_1200_...` → 1200
- ✓ `eval_code_selection_modified_1200_...` → 1200
- ✓ `eval_dataset_formatted_1200_...` → 1200
- ✓ All work correctly with step=0

### Dataset Name Extraction
- ✓ `eval_code_selection_format_0_1200_...` → `code_selection_format_0`
- ✓ `eval_code_selection_modified_1200_...` → `code_selection_modified`
- ✓ `eval_dataset_formatted_1200_...` → `dataset_formatted`

## Verified Datasets
- ✓ `results/november_4/4B_pen_20251106_095319` (old format)
- ✓ `results/november_6/4B_pen_formatted_0_20251106_232332` (new format)

