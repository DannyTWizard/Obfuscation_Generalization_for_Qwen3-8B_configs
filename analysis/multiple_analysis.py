"""
Multiple analysis script for comparing metrics across different training runs.
Processes multiple result folders and creates comparative visualizations.

Usage:
    python multiple_analysis.py /path/to/folder1 /path/to/folder2 [/path/to/folder3 ...]
    python multiple_analysis.py --key-pattern "custom_pattern" /path/to/folder1 /path/to/folder2
"""

import json
import os
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def extract_run_key_from_folder(folder_path: str, pattern: str = "word-word-number") -> str:
    """Extract a run identifier from the folder path.
    
    Args:
        folder_path: Path to the folder
        pattern: Pattern to use for extraction. Options:
            - "word-word-number": Extract pattern like "driven-dawn-4" (default)
            - "folder_name": Use the entire folder name
            - A custom regex pattern
    
    Returns:
        Extracted key identifier
    """
    folder_name = Path(folder_path).name
    
    if pattern == "folder_name":
        return folder_name
    elif pattern == "word-word-number":
        # Match pattern like "driven-dawn-4" or "likely-capybara-5"
        match = re.search(r'([a-zA-Z]+-[a-zA-Z]+-\d+)', folder_name)
        if match:
            return match.group(1)
        else:
            # Fallback to folder name if pattern not found
            return folder_name
    else:
        # Custom regex pattern
        match = re.search(pattern, folder_name)
        if match:
            return match.group(1) if match.groups() else match.group(0)
        else:
            return folder_name


def extract_dataset_name_from_folder(folder_name: str) -> str:
    """Extract the dataset name from folder name.
    
    Examples:
    - eval_code_selection_format_0_0_20251107_164653 -> code_selection_format_0
    - eval_revealing_score_formatted_0_1200_20251108_124113 -> revealing_score_formatted_0
    - eval_code_selection_modified_1200_20251107_154908 -> code_selection_modified
    """
    parts = folder_name.split('_')
    
    # Find "eval" prefix
    if not parts[0].startswith('eval'):
        return 'unknown'
    
    # Find "formatted", "format", or "modified" in the parts
    keyword_idx = -1
    keyword = None
    for i, part in enumerate(parts):
        part_lower = part.lower()
        if 'format' in part_lower or 'modified' in part_lower:
            keyword_idx = i
            if 'format' in part_lower:
                keyword = 'format'
            else:
                keyword = 'modified'
            break
    
    if keyword_idx == -1:
        return 'unknown'
    
    # Collect numeric parts after the keyword, excluding timestamps
    numeric_parts = []
    for i in range(keyword_idx + 1, len(parts)):
        part = parts[i]
        if part.isdigit():
            # Skip timestamps (typically 8 digits for date, 6 for time)
            if len(part) >= 6:
                break
            numeric_parts.append(int(part))
    
    # Determine dataset name based on the pattern
    if keyword == 'modified':
        # Old format: modified_{step} - no training set ID
        # Dataset name is everything from after "eval_" to "modified" (inclusive)
        dataset_parts = parts[1:keyword_idx+1]
    elif len(numeric_parts) == 0:
        # Format with no numbers: formatted/format only
        dataset_parts = parts[1:keyword_idx+1]
    elif len(numeric_parts) == 1:
        # Could be either:
        # 1. formatted/format_{step} (no training set ID)
        # 2. formatted/format_{training_set_id} (no step in folder name)
        # We'll treat single number as step and NOT include it
        dataset_parts = parts[1:keyword_idx+1]
    else:
        # Multiple numbers: formatted/format_{training_set_id}_{step}
        # Include the training set ID (first number)
        dataset_parts = parts[1:keyword_idx+2]
    
    return '_'.join(dataset_parts)


def extract_step_from_folder_name(folder_name: str) -> int:
    """Extract the step number from folder name.
    
    Handles patterns:
    - formatted_0_1200 (skip the 0, return 1200)
    - formatted_0_0 (skip the first 0, return the second 0)
    - formatted_1200 (return 1200 if no training set ID)
    - modified_1200 (return 1200)
    """
    parts = folder_name.split('_')
    
    # Find "formatted", "format", or "modified" in the parts
    keyword_idx = -1
    for i, part in enumerate(parts):
        part_lower = part.lower()
        if 'format' in part_lower or 'modified' in part_lower:
            keyword_idx = i
            break
    
    if keyword_idx == -1:
        # Fallback: use the old logic if keyword not found
        for i, part in enumerate(parts):
            if part.isdigit() and len(part) <= 4:
                try:
                    return int(part)
                except ValueError:
                    continue
        return None
    
    # Collect numeric parts after keyword, excluding timestamps
    # Timestamps are 8 digits (date) or 6 digits (time)
    numeric_parts = []
    for i in range(keyword_idx + 1, len(parts)):
        part = parts[i]
        if part.isdigit():
            # Skip timestamps (typically 8 digits for date, 6 for time)
            if len(part) >= 6:
                break
            numeric_parts.append(int(part))
    
    if len(numeric_parts) == 0:
        return None
    elif len(numeric_parts) == 1:
        # Only one number, use it as the step
        return numeric_parts[0]
    else:
        # Multiple numbers: skip the first one (training set ID), use the second (step)
        return numeric_parts[1]


def load_results_json(file_path: str) -> Dict:
    """Load metrics from a results.json file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    metrics = data.get('metrics', {})
    if metrics is None:
        metrics = {}
        print(f"Warning: No metrics found in {file_path}")
    return metrics


def find_all_results_files(parent_folder: str) -> List[Tuple[str, str]]:
    """Find all results.json files in subdirectories."""
    results_files = []
    parent_path = Path(parent_folder)
    
    for subdir in parent_path.iterdir():
        if subdir.is_dir():
            results_file = subdir / "results.json"
            if results_file.exists():
                results_files.append((subdir.name, str(results_file)))
    
    return sorted(results_files)


def extract_metrics(folder_name: str, metrics: Dict, run_key: str) -> Dict:
    """Extract key metrics from the metrics dict."""
    step = extract_step_from_folder_name(folder_name)
    dataset_name = extract_dataset_name_from_folder(folder_name)
    
    return {
        'run_key': run_key,
        'step': step,
        'folder_name': folder_name,
        'dataset_name': dataset_name,
        'accuracy': metrics.get('accuracy', None),
        'api_overseer_penalty_func': metrics.get('api_overseer_penalty_func', None),
        'dataset': metrics.get('dataset', 'unknown'),
        'correct': metrics.get('correct', None),
        'total': metrics.get('total', None),
    }


def process_multiple_folders(folders: List[str], key_pattern: str) -> pd.DataFrame:
    """Process all results.json files from multiple folders."""
    all_data_rows = []
    
    for folder_path in folders:
        parent_path = Path(folder_path)
        
        if not parent_path.exists():
            print(f"Warning: Path does not exist: {folder_path}")
            continue
        
        run_key = extract_run_key_from_folder(folder_path, key_pattern)
        print(f"\nProcessing folder: {folder_path}")
        print(f"  Run key: {run_key}")
        
        results_files = find_all_results_files(folder_path)
        
        if not results_files:
            print(f"  Warning: No results.json files found in {folder_path}")
            continue
        
        print(f"  Found {len(results_files)} results.json files")
        
        for folder_name, file_path in results_files:
            print(f"    Processing: {folder_name}")
            metrics = load_results_json(file_path)
            extracted = extract_metrics(folder_name, metrics, run_key)
            all_data_rows.append(extracted)
    
    if not all_data_rows:
        raise ValueError("No data found in any of the provided folders")
    
    df = pd.DataFrame(all_data_rows)
    # Sort by run_key and step
    df = df.sort_values(['run_key', 'step'], na_position='last')
    return df


def print_summary_statistics(df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\nTotal files processed: {len(df)}")
    print(f"Runs: {sorted(df['run_key'].unique().tolist())}")
    if 'dataset_name' in df.columns:
        print(f"Dataset names: {sorted(df['dataset_name'].unique().tolist())}")
    
    print("\n--- Accuracy ---")
    if df['accuracy'].notna().any():
        print(f"  Overall:")
        print(f"    Min:  {df['accuracy'].min():.4f}")
        print(f"    Max:  {df['accuracy'].max():.4f}")
        print(f"    Mean: {df['accuracy'].mean():.4f}")
        print(f"    Std:  {df['accuracy'].std():.4f}")
        
        # Show per-run stats
        print(f"\n  Per-run accuracy:")
        for run_key in sorted(df['run_key'].unique()):
            run_df = df[df['run_key'] == run_key]
            if run_df['accuracy'].notna().any():
                print(f"    {run_key}:")
                print(f"      Mean: {run_df['accuracy'].mean():.4f}")
                print(f"      Min:  {run_df['accuracy'].min():.4f}")
                print(f"      Max:  {run_df['accuracy'].max():.4f}")
    
    print("\n--- API Overseer Penalty Function ---")
    if df['api_overseer_penalty_func'].notna().any():
        print(f"  Overall:")
        print(f"    Min:  {df['api_overseer_penalty_func'].min():.4f}")
        print(f"    Max:  {df['api_overseer_penalty_func'].max():.4f}")
        print(f"    Mean: {df['api_overseer_penalty_func'].mean():.4f}")
        print(f"    Std:  {df['api_overseer_penalty_func'].std():.4f}")
        
        # Show per-run stats
        print(f"\n  Per-run API overseer penalty:")
        for run_key in sorted(df['run_key'].unique()):
            run_df = df[df['run_key'] == run_key]
            if run_df['api_overseer_penalty_func'].notna().any():
                print(f"    {run_key}:")
                print(f"      Mean: {run_df['api_overseer_penalty_func'].mean():.4f}")
                print(f"      Min:  {run_df['api_overseer_penalty_func'].min():.4f}")
                print(f"      Max:  {run_df['api_overseer_penalty_func'].max():.4f}")
    
    print("\n" + "="*80 + "\n")


def create_plots(df: pd.DataFrame, output_dir: str):
    """Create visualization plots comparing multiple runs."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories for organized plots
    plots_by_dataset_dir = os.path.join(output_dir, "plots_by_dataset")
    os.makedirs(plots_by_dataset_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    
    # Define colors and markers for different runs
    colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Get unique runs and datasets
    runs = sorted(df['run_key'].unique())
    
    # Plot 1: Accuracy vs Step (grouped by run and dataset)
    if df['accuracy'].notna().any() and 'dataset_name' in df.columns:
        datasets = sorted(df['dataset_name'].unique())
        
        for dataset in datasets:
            plt.figure(figsize=(14, 7))
            dataset_df = df[df['dataset_name'] == dataset]
            
            for i, run_key in enumerate(runs):
                run_dataset_df = dataset_df[dataset_df['run_key'] == run_key].copy()
                if run_dataset_df.empty:
                    continue
                run_dataset_df = run_dataset_df.sort_values('step')
                
                marker = markers[i % len(markers)]
                color = colors[i % len(colors)]
                
                plt.plot(run_dataset_df['step'], run_dataset_df['accuracy'], 
                        marker=marker, linewidth=2, markersize=8,
                        label=run_key, color=color)
            
            plt.xlabel('Training Step', fontsize=12, fontweight='bold')
            plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
            plt.title(f'Accuracy vs Training Step - {dataset}', fontsize=14, fontweight='bold')
            plt.legend(loc='best', fontsize=10, frameon=True, shadow=True)
            plt.grid(True, alpha=0.3)
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
            plt.tight_layout()
            
            safe_dataset_name = dataset.replace('/', '_').replace(' ', '_')
            plt.savefig(os.path.join(plots_by_dataset_dir, f'accuracy_vs_step_{safe_dataset_name}.png'), dpi=300, bbox_inches='tight')
            print(f"✓ Saved: plots_by_dataset/accuracy_vs_step_{safe_dataset_name}.png")
            plt.close()
    
    # Plot 2: API Overseer Penalty Function vs Step (grouped by run and dataset)
    if df['api_overseer_penalty_func'].notna().any() and 'dataset_name' in df.columns:
        datasets = sorted(df['dataset_name'].unique())
        
        for dataset in datasets:
            plt.figure(figsize=(14, 7))
            dataset_df = df[df['dataset_name'] == dataset]
            
            for i, run_key in enumerate(runs):
                run_dataset_df = dataset_df[dataset_df['run_key'] == run_key].copy()
                if run_dataset_df.empty:
                    continue
                run_dataset_df = run_dataset_df.sort_values('step')
                
                marker = markers[i % len(markers)]
                color = colors[i % len(colors)]
                
                plt.plot(run_dataset_df['step'], run_dataset_df['api_overseer_penalty_func'], 
                        marker=marker, linewidth=2, markersize=8,
                        label=run_key, color=color)
            
            plt.xlabel('Training Step', fontsize=12, fontweight='bold')
            plt.ylabel('API Overseer Penalty Function', fontsize=12, fontweight='bold')
            plt.title(f'API Overseer Penalty Function vs Training Step - {dataset}', fontsize=14, fontweight='bold')
            plt.legend(loc='best', fontsize=10, frameon=True, shadow=True)
            plt.grid(True, alpha=0.3)
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.3f}'))
            plt.tight_layout()
            
            safe_dataset_name = dataset.replace('/', '_').replace(' ', '_')
            plt.savefig(os.path.join(plots_by_dataset_dir, f'api_overseer_penalty_func_vs_step_{safe_dataset_name}.png'), dpi=300, bbox_inches='tight')
            print(f"✓ Saved: plots_by_dataset/api_overseer_penalty_func_vs_step_{safe_dataset_name}.png")
            plt.close()
    
    # Plot 3: All metrics on one plot per dataset (Accuracy and API Overseer Penalty)
    if (df['accuracy'].notna().any() and 
        df['api_overseer_penalty_func'].notna().any() and
        'dataset_name' in df.columns):
        
        datasets = sorted(df['dataset_name'].unique())
        
        for dataset in datasets:
            fig, axes = plt.subplots(2, 1, figsize=(14, 10))
            dataset_df = df[df['dataset_name'] == dataset]
            
            # Accuracy subplot
            for i, run_key in enumerate(runs):
                run_dataset_df = dataset_df[dataset_df['run_key'] == run_key].copy()
                if run_dataset_df.empty:
                    continue
                run_dataset_df = run_dataset_df.sort_values('step')
                
                marker = markers[i % len(markers)]
                color = colors[i % len(colors)]
                
                axes[0].plot(run_dataset_df['step'], run_dataset_df['accuracy'], 
                            marker=marker, linewidth=2, markersize=8,
                            label=run_key, color=color)
            
            axes[0].set_ylabel('Accuracy', fontsize=11, fontweight='bold')
            axes[0].set_title(f'Accuracy vs Training Step - {dataset}', fontsize=12, fontweight='bold')
            axes[0].legend(loc='best', fontsize=9, frameon=True, shadow=True)
            axes[0].grid(True, alpha=0.3)
            axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
            
            # API Overseer Penalty Function subplot
            for i, run_key in enumerate(runs):
                run_dataset_df = dataset_df[dataset_df['run_key'] == run_key].copy()
                if run_dataset_df.empty:
                    continue
                run_dataset_df = run_dataset_df.sort_values('step')
                
                marker = markers[i % len(markers)]
                color = colors[i % len(colors)]
                
                axes[1].plot(run_dataset_df['step'], run_dataset_df['api_overseer_penalty_func'], 
                            marker=marker, linewidth=2, markersize=8,
                            label=run_key, color=color)
            
            axes[1].set_xlabel('Training Step', fontsize=11, fontweight='bold')
            axes[1].set_ylabel('API Overseer Penalty', fontsize=11, fontweight='bold')
            axes[1].set_title(f'API Overseer Penalty Function vs Training Step - {dataset}', fontsize=12, fontweight='bold')
            axes[1].legend(loc='best', fontsize=9, frameon=True, shadow=True)
            axes[1].grid(True, alpha=0.3)
            axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.3f}'))
            
            plt.tight_layout()
            
            safe_dataset_name = dataset.replace('/', '_').replace(' ', '_')
            plt.savefig(os.path.join(plots_by_dataset_dir, f'all_metrics_vs_step_{safe_dataset_name}.png'), dpi=300, bbox_inches='tight')
            print(f"✓ Saved: plots_by_dataset/all_metrics_vs_step_{safe_dataset_name}.png")
            plt.close()
    
    # Plot 4: Overall comparison - all datasets on one plot per run
    if df['accuracy'].notna().any() and 'dataset_name' in df.columns:
        datasets = sorted(df['dataset_name'].unique())
        
        plt.figure(figsize=(14, 7))
        
        for run_idx, run_key in enumerate(runs):
            run_df = df[df['run_key'] == run_key]
            
            for dataset_idx, dataset in enumerate(datasets):
                run_dataset_df = run_df[run_df['dataset_name'] == dataset].copy()
                if run_dataset_df.empty:
                    continue
                run_dataset_df = run_dataset_df.sort_values('step')
                
                # Use combination of color and marker to distinguish run + dataset
                color = colors[run_idx % len(colors)]
                marker = markers[dataset_idx % len(markers)]
                linestyle = '-' if dataset_idx == 0 else '--' if dataset_idx == 1 else '-.'
                
                label = f"{run_key} - {dataset}"
                plt.plot(run_dataset_df['step'], run_dataset_df['accuracy'], 
                        marker=marker, linewidth=2, markersize=6,
                        label=label, color=color, linestyle=linestyle, alpha=0.8)
        
        plt.xlabel('Training Step', fontsize=12, fontweight='bold')
        plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
        plt.title('Accuracy vs Training Step - All Runs and Datasets', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=9, frameon=True, shadow=True, ncol=2)
        plt.grid(True, alpha=0.3)
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_vs_step_all.png'), dpi=300, bbox_inches='tight')
        print(f"✓ Saved: accuracy_vs_step_all.png")
        plt.close()
    
    # Plot 5: Scatter plot - Accuracy vs API Overseer Penalty (colored by dataset, gradated by step)
    if (df['accuracy'].notna().any() and 
        df['api_overseer_penalty_func'].notna().any() and
        'dataset_name' in df.columns):
        
        datasets = sorted(df['dataset_name'].unique())
        
        # Normalize step values for color gradation (0 = dark, max = light)
        max_step = df['step'].max()
        min_step = df['step'].min()
        step_range = max_step - min_step if max_step != min_step else 1
        
        for dataset in datasets:
            plt.figure(figsize=(12, 8))
            dataset_df = df[df['dataset_name'] == dataset]
            
            for run_idx, run_key in enumerate(runs):
                run_dataset_df = dataset_df[dataset_df['run_key'] == run_key].copy()
                if run_dataset_df.empty:
                    continue
                
                # Get base color for this run
                base_color = colors[run_idx % len(colors)]
                
                # Convert to RGB for gradation
                import matplotlib.colors as mcolors
                base_rgb = mcolors.to_rgb(base_color)
                
                # Plot each point with color gradation by step
                for _, row in run_dataset_df.iterrows():
                    # Normalize step: 0 to 1 (where 0 is earliest, 1 is latest)
                    norm_step = (row['step'] - min_step) / step_range
                    
                    # Gradation: dark (step 0) to light (max step)
                    # Mix base color with white based on step
                    alpha_blend = 0.3 + 0.7 * (1 - norm_step)  # 1.0 at step 0, 0.3 at max step
                    point_color = tuple(c * alpha_blend + 1.0 * (1 - alpha_blend) for c in base_rgb)
                    
                    plt.scatter(row['accuracy'], row['api_overseer_penalty_func'], 
                               s=150, alpha=0.7, color=point_color, 
                               edgecolors='black', linewidths=0.5,
                               marker=markers[run_idx % len(markers)])
                
                # Add label with a representative color
                mid_color = tuple(c * 0.65 + 1.0 * 0.35 for c in base_rgb)
                plt.scatter([], [], s=150, alpha=0.7, color=mid_color,
                           edgecolors='black', linewidths=0.5,
                           marker=markers[run_idx % len(markers)],
                           label=f"{run_key} (dark=early, light=late)")
            
            plt.xlabel('Accuracy', fontsize=11, fontweight='bold')
            plt.ylabel('API Overseer Penalty', fontsize=11, fontweight='bold')
            plt.title(f'Accuracy vs API Overseer Penalty - {dataset}\n(Color darkness indicates step: dark=early, light=late)', 
                     fontsize=12, fontweight='bold')
            plt.legend(loc='best', fontsize=9, frameon=True, shadow=True)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            safe_dataset_name = dataset.replace('/', '_').replace(' ', '_')
            plt.savefig(os.path.join(plots_by_dataset_dir, f'accuracy_vs_penalty_scatter_{safe_dataset_name}.png'), 
                       dpi=300, bbox_inches='tight')
            print(f"✓ Saved: plots_by_dataset/accuracy_vs_penalty_scatter_{safe_dataset_name}.png")
            plt.close()
        
        # Also create an overall scatter plot with all datasets
        plt.figure(figsize=(14, 9))
        
        for dataset_idx, dataset in enumerate(datasets):
            dataset_df = df[df['dataset_name'] == dataset]
            
            for run_idx, run_key in enumerate(runs):
                run_dataset_df = dataset_df[dataset_df['run_key'] == run_key].copy()
                if run_dataset_df.empty:
                    continue
                
                # Combine run and dataset for unique coloring
                combined_idx = run_idx * len(datasets) + dataset_idx
                base_color = colors[combined_idx % len(colors)]
                
                # Convert to RGB for gradation
                import matplotlib.colors as mcolors
                base_rgb = mcolors.to_rgb(base_color)
                
                # Plot each point with color gradation by step
                for _, row in run_dataset_df.iterrows():
                    # Normalize step: 0 to 1 (where 0 is earliest, 1 is latest)
                    norm_step = (row['step'] - min_step) / step_range
                    
                    # Gradation: dark (step 0) to light (max step)
                    alpha_blend = 0.3 + 0.7 * (1 - norm_step)
                    point_color = tuple(c * alpha_blend + 1.0 * (1 - alpha_blend) for c in base_rgb)
                    
                    plt.scatter(row['accuracy'], row['api_overseer_penalty_func'], 
                               s=120, alpha=0.7, color=point_color, 
                               edgecolors='black', linewidths=0.5,
                               marker=markers[run_idx % len(markers)])
                
                # Add label with a representative color
                mid_color = tuple(c * 0.65 + 1.0 * 0.35 for c in base_rgb)
                plt.scatter([], [], s=120, alpha=0.7, color=mid_color,
                           edgecolors='black', linewidths=0.5,
                           marker=markers[run_idx % len(markers)],
                           label=f"{run_key} - {dataset}")
        
        plt.xlabel('Accuracy', fontsize=12, fontweight='bold')
        plt.ylabel('API Overseer Penalty', fontsize=12, fontweight='bold')
        plt.title('Accuracy vs API Overseer Penalty - All Runs and Datasets\n(Color darkness indicates step: dark=early, light=late)', 
                 fontsize=13, fontweight='bold')
        plt.legend(loc='best', fontsize=8, frameon=True, shadow=True, ncol=2)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_vs_penalty_scatter_all.png'), dpi=300, bbox_inches='tight')
        print(f"✓ Saved: accuracy_vs_penalty_scatter_all.png")
        plt.close()


def save_analysis_table(df: pd.DataFrame, output_dir: str):
    """Save the analysis results as a CSV file."""
    output_file = os.path.join(output_dir, 'analysis_results.csv')
    
    # Select columns for export
    columns = ['run_key', 'step', 'folder_name', 'dataset_name', 'accuracy', 'api_overseer_penalty_func']
    export_df = df[columns].copy()
    
    export_df.to_csv(output_file, index=False)
    print(f"✓ Saved: analysis_results.csv")


def extract_parent_folder_name(folder_path: str) -> str:
    """Extract the parent folder name from a folder path.
    
    Examples:
    - /root/results/4B_sycophancy_format_0/driven-dawn-4_20251110_114239 -> 4B_sycophancy_format_0
    - results/4B_sycophancy_format_0/driven-dawn-4_20251110_114239 -> 4B_sycophancy_format_0
    """
    path = Path(folder_path)
    # Get parent of the input folder
    return path.parent.name


def get_next_version_dir(base_dir: str) -> str:
    """Find the next available version directory (v0, v1, v2, etc.).
    
    Args:
        base_dir: The base directory to check for existing version folders
        
    Returns:
        Full path to the next available version directory
    """
    base_path = Path(base_dir)
    version = 0
    
    while True:
        version_dir = base_path / f"v{version}"
        if not version_dir.exists():
            return str(version_dir)
        version += 1


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Analyze metrics from multiple training runs for comparison"
    )
    parser.add_argument(
        "folders", 
        nargs='+',
        help="Paths to the folders containing results subdirectories"
    )
    parser.add_argument(
        "--output", "-o", 
        help="Output directory for plots (default: results/multi_analysis_plots/{parent_folder_name})"
    )
    parser.add_argument(
        "--key-pattern", "-k",
        default="word-word-number",
        help="Pattern to extract run key from folder name. Options: 'word-word-number' (default), 'folder_name', or a custom regex pattern"
    )
    
    args = parser.parse_args()
    
    # Set output directory
    output_dir = args.output
    if output_dir is None:
        # Extract parent folder names from input folders
        parent_names = set()
        for folder in args.folders:
            parent_name = extract_parent_folder_name(folder)
            parent_names.add(parent_name)
        
        # If all folders have the same parent, use that name; otherwise use generic name
        if len(parent_names) == 1:
            parent_folder = parent_names.pop()
            base_dir = f"results/multi_analysis_plots/{parent_folder}"
        else:
            base_dir = "results/multi_analysis_plots/comparison"
        
        # Get next available version directory
        output_dir = get_next_version_dir(base_dir)
    
    print("\n" + "="*80)
    print("MULTIPLE RUNS ANALYSIS SCRIPT")
    print("="*80)
    print(f"Number of folders: {len(args.folders)}")
    print(f"Key pattern: {args.key_pattern}")
    print(f"Output folder: {output_dir}")
    print("="*80)
    
    try:
        # Process results
        print("\nProcessing results from multiple folders...")
        df = process_multiple_folders(args.folders, args.key_pattern)
        
        # Print summary
        print_summary_statistics(df)
        
        # Create plots
        print("Creating visualizations...")
        create_plots(df, output_dir)
        
        # Save analysis table
        save_analysis_table(df, output_dir)
        
        print(f"\n✓ Analysis complete! All plots saved to: {output_dir}\n")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

