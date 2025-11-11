"""
Analysis script for accuracy and API overseer penalty function.
Processes all results.json files in a folder and creates visualizations grouped by dataset.

Usage:
    python analyze_metrics.py /path/to/results/folder
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime


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


def extract_metrics(folder_name: str, metrics: Dict) -> Dict:
    """Extract key metrics from the metrics dict."""
    step = extract_step_from_folder_name(folder_name)
    dataset_name = extract_dataset_name_from_folder(folder_name)
    
    return {
        'step': step,
        'folder_name': folder_name,
        'dataset_name': dataset_name,
        'accuracy': metrics.get('accuracy', None),
        'api_overseer_penalty_func': metrics.get('api_overseer_penalty_func', None),
        'dataset': metrics.get('dataset', 'unknown'),
        'correct': metrics.get('correct', None),
        'total': metrics.get('total', None),
    }


def process_results_folder(parent_folder: str) -> pd.DataFrame:
    """Process all results.json files in a folder."""
    results_files = find_all_results_files(parent_folder)
    
    if not results_files:
        raise FileNotFoundError(f"No results.json files found in {parent_folder}")
    
    print(f"Found {len(results_files)} results.json files")
    
    data_rows = []
    for folder_name, file_path in results_files:
        print(f"  Processing: {folder_name}")
        metrics = load_results_json(file_path)
        extracted = extract_metrics(folder_name, metrics)
        data_rows.append(extracted)
    
    df = pd.DataFrame(data_rows)
    # Sort by step
    df = df.sort_values('step', na_position='last')
    return df


def print_summary_statistics(df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\nTotal files processed: {len(df)}")
    if 'dataset_name' in df.columns:
        print(f"Dataset names: {sorted(df['dataset_name'].unique().tolist())}")
    
    print("\n--- Accuracy ---")
    if df['accuracy'].notna().any():
        print(f"  Overall:")
        print(f"    Min:  {df['accuracy'].min():.4f}")
        print(f"    Max:  {df['accuracy'].max():.4f}")
        print(f"    Mean: {df['accuracy'].mean():.4f}")
        print(f"    Std:  {df['accuracy'].std():.4f}")
        
        # Show per-dataset stats if available
        if 'dataset_name' in df.columns:
            print(f"\n  Per-dataset accuracy:")
            for dataset_name in sorted(df['dataset_name'].unique()):
                dataset_df = df[df['dataset_name'] == dataset_name]
                if dataset_df['accuracy'].notna().any():
                    print(f"    {dataset_name}:")
                    print(f"      Mean: {dataset_df['accuracy'].mean():.4f}")
                    print(f"      Min:  {dataset_df['accuracy'].min():.4f}")
                    print(f"      Max:  {dataset_df['accuracy'].max():.4f}")
    
    print("\n--- API Overseer Penalty Function ---")
    if df['api_overseer_penalty_func'].notna().any():
        print(f"  Overall:")
        print(f"    Min:  {df['api_overseer_penalty_func'].min():.4f}")
        print(f"    Max:  {df['api_overseer_penalty_func'].max():.4f}")
        print(f"    Mean: {df['api_overseer_penalty_func'].mean():.4f}")
        print(f"    Std:  {df['api_overseer_penalty_func'].std():.4f}")
        
        # Show per-dataset stats if available
        if 'dataset_name' in df.columns:
            print(f"\n  Per-dataset API overseer penalty:")
            for dataset_name in sorted(df['dataset_name'].unique()):
                dataset_df = df[df['dataset_name'] == dataset_name]
                if dataset_df['api_overseer_penalty_func'].notna().any():
                    print(f"    {dataset_name}:")
                    print(f"      Mean: {dataset_df['api_overseer_penalty_func'].mean():.4f}")
                    print(f"      Min:  {dataset_df['api_overseer_penalty_func'].min():.4f}")
                    print(f"      Max:  {dataset_df['api_overseer_penalty_func'].max():.4f}")
    
    print("\n" + "="*80 + "\n")


def create_plots(df: pd.DataFrame, output_dir: str, folder_name: str):
    """Create visualization plots for the metrics, grouped by dataset."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    
    # Define colors and markers for different datasets
    colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Plot 1: Accuracy vs Step (grouped by dataset_name)
    if df['accuracy'].notna().any() and 'dataset_name' in df.columns:
        plt.figure(figsize=(14, 7))
        
        datasets = sorted(df['dataset_name'].unique())
        for i, dataset in enumerate(datasets):
            dataset_df = df[df['dataset_name'] == dataset].copy()
            dataset_df = dataset_df.sort_values('step')
            
            marker = markers[i % len(markers)]
            color = colors[i % len(colors)]
            
            plt.plot(dataset_df['step'], dataset_df['accuracy'], 
                    marker=marker, linewidth=2, markersize=8,
                    label=dataset, color=color)
        
        plt.xlabel('Training Step', fontsize=12, fontweight='bold')
        plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
        plt.title(f'Accuracy vs Training Step by Dataset - {folder_name}', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10, frameon=True, shadow=True)
        plt.grid(True, alpha=0.3)
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_vs_step.png'), dpi=300, bbox_inches='tight')
        print(f"✓ Saved: accuracy_vs_step.png")
        plt.close()
    
    # Plot 2: API Overseer Penalty Function vs Step (grouped by dataset_name)
    if df['api_overseer_penalty_func'].notna().any() and 'dataset_name' in df.columns:
        plt.figure(figsize=(14, 7))
        
        datasets = sorted(df['dataset_name'].unique())
        for i, dataset in enumerate(datasets):
            dataset_df = df[df['dataset_name'] == dataset].copy()
            dataset_df = dataset_df.sort_values('step')
            
            marker = markers[i % len(markers)]
            color = colors[i % len(colors)]
            
            plt.plot(dataset_df['step'], dataset_df['api_overseer_penalty_func'], 
                    marker=marker, linewidth=2, markersize=8,
                    label=dataset, color=color)
        
        plt.xlabel('Training Step', fontsize=12, fontweight='bold')
        plt.ylabel('API Overseer Penalty Function', fontsize=12, fontweight='bold')
        plt.title(f'API Overseer Penalty Function vs Training Step by Dataset - {folder_name}', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10, frameon=True, shadow=True)
        plt.grid(True, alpha=0.3)
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.3f}'))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'api_overseer_penalty_func_vs_step.png'), dpi=300, bbox_inches='tight')
        print(f"✓ Saved: api_overseer_penalty_func_vs_step.png")
        plt.close()
    
    # Plot 3: All metrics on one plot (Accuracy and API Overseer Penalty)
    if (df['accuracy'].notna().any() and 
        df['api_overseer_penalty_func'].notna().any() and
        'dataset_name' in df.columns):
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        datasets = sorted(df['dataset_name'].unique())
        
        # Accuracy subplot
        for i, dataset in enumerate(datasets):
            dataset_df = df[df['dataset_name'] == dataset].copy()
            dataset_df = dataset_df.sort_values('step')
            
            marker = markers[i % len(markers)]
            color = colors[i % len(colors)]
            
            axes[0].plot(dataset_df['step'], dataset_df['accuracy'], 
                        marker=marker, linewidth=2, markersize=8,
                        label=dataset, color=color)
        
        axes[0].set_ylabel('Accuracy', fontsize=11, fontweight='bold')
        axes[0].set_title(f'Accuracy vs Training Step - {folder_name}', fontsize=12, fontweight='bold')
        axes[0].legend(loc='best', fontsize=9, frameon=True, shadow=True)
        axes[0].grid(True, alpha=0.3)
        axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        # API Overseer Penalty Function subplot
        for i, dataset in enumerate(datasets):
            dataset_df = df[df['dataset_name'] == dataset].copy()
            dataset_df = dataset_df.sort_values('step')
            
            marker = markers[i % len(markers)]
            color = colors[i % len(colors)]
            
            axes[1].plot(dataset_df['step'], dataset_df['api_overseer_penalty_func'], 
                        marker=marker, linewidth=2, markersize=8,
                        label=dataset, color=color)
        
        axes[1].set_xlabel('Training Step', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('API Overseer Penalty', fontsize=11, fontweight='bold')
        axes[1].set_title(f'API Overseer Penalty Function vs Training Step - {folder_name}', fontsize=12, fontweight='bold')
        axes[1].legend(loc='best', fontsize=9, frameon=True, shadow=True)
        axes[1].grid(True, alpha=0.3)
        axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.3f}'))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'all_metrics_vs_step.png'), dpi=300, bbox_inches='tight')
        print(f"✓ Saved: all_metrics_vs_step.png")
        plt.close()
    
    # Plot 4: Scatter plot - Accuracy vs API Overseer Penalty (colored by dataset)
    if (df['accuracy'].notna().any() and 
        df['api_overseer_penalty_func'].notna().any() and
        'dataset_name' in df.columns):
        
        plt.figure(figsize=(10, 7))
        
        datasets = sorted(df['dataset_name'].unique())
        for i, dataset in enumerate(datasets):
            dataset_df = df[df['dataset_name'] == dataset]
            color = colors[i % len(colors)]
            
            plt.scatter(dataset_df['accuracy'], dataset_df['api_overseer_penalty_func'], 
                       s=100, alpha=0.6, color=color, label=dataset)
        
        plt.xlabel('Accuracy', fontsize=11, fontweight='bold')
        plt.ylabel('API Overseer Penalty', fontsize=11, fontweight='bold')
        plt.title('Accuracy vs API Overseer Penalty by Dataset', fontsize=12, fontweight='bold')
        plt.legend(loc='best', fontsize=9, frameon=True, shadow=True)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_vs_penalty_scatter.png'), dpi=300, bbox_inches='tight')
        print(f"✓ Saved: accuracy_vs_penalty_scatter.png")
        plt.close()


def save_analysis_table(df: pd.DataFrame, output_dir: str):
    """Save the analysis results as a CSV file."""
    output_file = os.path.join(output_dir, 'analysis_results.csv')
    
    # Select columns for export
    columns = ['step', 'folder_name', 'dataset_name', 'accuracy', 'api_overseer_penalty_func']
    export_df = df[columns].copy()
    
    export_df.to_csv(output_file, index=False)
    print(f"✓ Saved: analysis_results.csv")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Analyze metrics (accuracy, API overseer penalty) from results.json files, grouped by dataset"
    )
    parser.add_argument("results_folder", help="Path to the parent folder containing results subdirectories")
    parser.add_argument("--output", "-o", help="Output directory for plots (default: analysis folder)")
    
    args = parser.parse_args()
    
    # Validate path
    results_path = Path(args.results_folder)
    if not results_path.exists():
        print(f"Error: Path does not exist: {args.results_folder}")
        return 1
    
    # Set output directory
    output_dir = args.output
    if output_dir is None:
        output_dir = str(results_path / "analysis_plots")
    
    print("\n" + "="*80)
    print("METRICS ANALYSIS SCRIPT")
    print("="*80)
    print(f"Results folder: {args.results_folder}")
    print(f"Output folder: {output_dir}")
    print("="*80 + "\n")
    
    try:
        # Process results
        print("Processing results.json files...")
        df = process_results_folder(args.results_folder)
        
        # Print summary
        print_summary_statistics(df)
        
        # Create plots
        print("Creating visualizations...")
        folder_name = results_path.name
        create_plots(df, output_dir, folder_name)
        
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

