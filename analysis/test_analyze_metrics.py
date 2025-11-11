"""
Unit tests for analyze_metrics.py parsing functions.
Tests both old format (modified) and new format (formatted).

Usage:
    python test_analyze_metrics.py
"""

import sys
from analyze_metrics import extract_step_from_folder_name, extract_dataset_name_from_folder


def test_step_extraction():
    """Test step number extraction from various folder name formats."""
    print("=" * 80)
    print("Testing extract_step_from_folder_name")
    print("=" * 80)
    
    test_cases = [
        # New format with training set ID: formatted_0_{step}
        ('eval_code_selection_format_0_0_20251107_164653', 0),
        ('eval_code_selection_format_0_400_20251107_173116', 400),
        ('eval_code_selection_format_0_1200_20251107_171132', 1200),
        ('eval_code_selection_format_0_2000_20251107_161614', 2000),
        ('eval_revealing_score_formatted_0_0_20251108_124113', 0),
        ('eval_revealing_score_formatted_0_800_20251108_144556', 800),
        ('eval_revealing_score_formatted_0_1600_20251108_142140', 1600),
        ('eval_sycophancy_fact_formatted_0_1200_20251108_134148', 1200),
        
        # New format without training set ID: formatted_{step}
        ('eval_dataset_formatted_1200_20251108_122905', 1200),
        ('eval_dataset_format_2000_20251108_122905', 2000),
        
        # Old format: modified_{step}
        ('eval_code_selection_modified_0_20251107_143152', 0),
        ('eval_code_selection_modified_200_20251107_145038', 200),
        ('eval_code_selection_modified_400_20251107_151019', 400),
        ('eval_code_selection_modified_1000_20251107_162511', 1000),
        ('eval_code_selection_modified_1200_20251107_154908', 1200),
        ('eval_code_selection_modified_2000_20251107_142003', 2000),
        ('eval_revealing_score_modified_800_20251107_173044', 800),
        ('eval_revealing_score_modified_1600_20251107_165948', 1600),
        ('eval_sycophancy_fact_modified_1400_20251107_181247', 1400),
        ('eval_sycophancy_fact_modified_2000_20251107_174104', 2000),
    ]
    
    passed = 0
    failed = 0
    
    for folder_name, expected_step in test_cases:
        result = extract_step_from_folder_name(folder_name)
        status = "✓ PASS" if result == expected_step else "✗ FAIL"
        if result == expected_step:
            passed += 1
        else:
            failed += 1
        print(f"{status}: {folder_name}")
        print(f"  Expected: {expected_step}, Got: {result}")
    
    print(f"\nResults: {passed} passed, {failed} failed\n")
    return failed == 0


def test_dataset_name_extraction():
    """Test dataset name extraction from various folder name formats."""
    print("=" * 80)
    print("Testing extract_dataset_name_from_folder")
    print("=" * 80)
    
    test_cases = [
        # New format with training set ID: should include the training set ID
        ('eval_code_selection_format_0_0_20251107_164653', 'code_selection_format_0'),
        ('eval_code_selection_format_0_1200_20251107_171132', 'code_selection_format_0'),
        ('eval_revealing_score_formatted_0_0_20251108_124113', 'revealing_score_formatted_0'),
        ('eval_revealing_score_formatted_0_1200_20251108_143313', 'revealing_score_formatted_0'),
        ('eval_sycophancy_fact_formatted_0_1200_20251108_134148', 'sycophancy_fact_formatted_0'),
        ('eval_sycophancy_fact_formatted_0_2000_20251108_121907', 'sycophancy_fact_formatted_0'),
        
        # New format without training set ID: should just be the dataset name
        ('eval_dataset_formatted_1200_20251108_122905', 'dataset_formatted'),
        ('eval_dataset_format_2000_20251108_122905', 'dataset_format'),
        
        # Old format: should just be the dataset name
        ('eval_code_selection_modified_0_20251107_143152', 'code_selection_modified'),
        ('eval_code_selection_modified_1200_20251107_154908', 'code_selection_modified'),
        ('eval_code_selection_modified_2000_20251107_142003', 'code_selection_modified'),
        ('eval_revealing_score_modified_0_20251107_163923', 'revealing_score_modified'),
        ('eval_revealing_score_modified_800_20251107_173044', 'revealing_score_modified'),
        ('eval_revealing_score_modified_1600_20251107_165948', 'revealing_score_modified'),
        ('eval_sycophancy_fact_modified_1400_20251107_181247', 'sycophancy_fact_modified'),
        ('eval_sycophancy_fact_modified_2000_20251107_174104', 'sycophancy_fact_modified'),
    ]
    
    passed = 0
    failed = 0
    
    for folder_name, expected_dataset in test_cases:
        result = extract_dataset_name_from_folder(folder_name)
        status = "✓ PASS" if result == expected_dataset else "✗ FAIL"
        if result == expected_dataset:
            passed += 1
        else:
            failed += 1
        print(f"{status}: {folder_name}")
        print(f"  Expected: '{expected_dataset}', Got: '{result}'")
    
    print(f"\nResults: {passed} passed, {failed} failed\n")
    return failed == 0


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("RUNNING UNIT TESTS FOR analyze_metrics.py")
    print("=" * 80 + "\n")
    
    step_test_passed = test_step_extraction()
    dataset_test_passed = test_dataset_name_extraction()
    
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Step extraction: {'✓ PASSED' if step_test_passed else '✗ FAILED'}")
    print(f"Dataset name extraction: {'✓ PASSED' if dataset_test_passed else '✗ FAILED'}")
    
    if step_test_passed and dataset_test_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())

