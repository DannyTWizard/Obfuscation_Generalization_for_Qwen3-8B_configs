"""
Multiple analysis script for comparing metrics across different training runs.
Processes multiple result folders and creates comparative visualizations.

Usage:
    python multiple_analysis.py /path/to/run1 /path/to/run2 [/path/to/run3 ...]
"""

import json
import os
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def extract_step_from_folder_name(folder_name: str) -> int:
    """Extract the step number from step folder name.
    
    Examples:
    - step_0_20251111_161133 -> 0
    - step_1200_20251111_161133 -> 1200
    """
    match = re.match(r'step_(\d+)_\d{8}_\d{6}', folder_name)
    if match:
        return int(match.group(1))
    return None


def load_results_json(file_path: str) -> Dict:
    """Load metrics from a results.json file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    metrics = data.get('metrics', {})
    if metrics is None:
        metrics = {}
        print(f"Warning: No metrics found in {file_path}")
    return metrics


def find_all_results_files(parent_folder: str) -> List[Tuple[str, str, str]]:
    """Find all results.json files in eval/{dataset}/step_X subdirectories.
    
    Returns:
        List of tuples: (dataset_name, step_folder_name, results_file_path)
    """
    results_files = []
    parent_path = Path(parent_folder)
    
    eval_dir = parent_path / "eval"
    if not eval_dir.exists():
        return results_files
    
    # Iterate through dataset directories
    for dataset_dir in eval_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        
        dataset_name = dataset_dir.name
        
        # Iterate through step directories
        for step_dir in dataset_dir.iterdir():
            if step_dir.is_dir():
                results_file = step_dir / "results.json"
                if results_file.exists():
                    results_files.append((dataset_name, step_dir.name, str(results_file)))
    
    return sorted(results_files)


def extract_metrics(dataset_name: str, step_folder_name: str, metrics: Dict, run_name: str) -> Dict:
    """Extract key metrics from the metrics dict."""
    step = extract_step_from_folder_name(step_folder_name)
    
    return {
        'run_name': run_name,
        'step': step,
        'step_folder': step_folder_name,
        'dataset_name': dataset_name,
        'accuracy': metrics.get('accuracy', None),
        'api_overseer_penalty_func': metrics.get('api_overseer_penalty_func', None),
        'correct': metrics.get('correct', None),
        'total': metrics.get('total', None),
    }


def process_multiple_folders(folders: List[str]) -> pd.DataFrame:
    """Process all results.json files from multiple folders."""
    all_data_rows = []
    
    for folder_path in folders:
        parent_path = Path(folder_path)
        
        if not parent_path.exists():
            print(f"Warning: Path does not exist: {folder_path}")
            continue
        
        # Use the folder name as the run identifier
        run_name = parent_path.name
        print(f"\nProcessing folder: {folder_path}")
        print(f"  Run name: {run_name}")
        
        results_files = find_all_results_files(folder_path)
        
        if not results_files:
            print(f"  Warning: No results.json files found in {folder_path}/eval/")
            continue
        
        print(f"  Found {len(results_files)} results.json files")
        
        for dataset_name, step_folder_name, file_path in results_files:
            print(f"    Processing: {dataset_name}/{step_folder_name}")
            metrics = load_results_json(file_path)
            extracted = extract_metrics(dataset_name, step_folder_name, metrics, run_name)
            all_data_rows.append(extracted)
    
    if not all_data_rows:
        raise ValueError("No data found in any of the provided folders")
    
    df = pd.DataFrame(all_data_rows)
    # Sort by run_name, dataset_name, and step
    df = df.sort_values(['run_name', 'dataset_name', 'step'], na_position='last')
    return df


def print_summary_statistics(df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\nTotal files processed: {len(df)}")
    print(f"Runs: {sorted(df['run_name'].unique().tolist())}")
    if 'dataset_name' in df.columns:
        print(f"Datasets: {sorted(df['dataset_name'].unique().tolist())}")
    
    print("\n--- Accuracy ---")
    if df['accuracy'].notna().any():
        print(f"  Overall:")
        print(f"    Min:  {df['accuracy'].min():.4f}")
        print(f"    Max:  {df['accuracy'].max():.4f}")
        print(f"    Mean: {df['accuracy'].mean():.4f}")
        print(f"    Std:  {df['accuracy'].std():.4f}")
        
        # Show per-run stats
        print(f"\n  Per-run accuracy:")
        for run_name in sorted(df['run_name'].unique()):
            run_df = df[df['run_name'] == run_name]
            if run_df['accuracy'].notna().any():
                print(f"    {run_name}:")
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
        for run_name in sorted(df['run_name'].unique()):
            run_df = df[df['run_name'] == run_name]
            if run_df['api_overseer_penalty_func'].notna().any():
                print(f"    {run_name}:")
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
    runs = sorted(df['run_name'].unique())
    
    # Plot 1: Accuracy vs Step (one plot per dataset, comparing runs)
    if df['accuracy'].notna().any() and 'dataset_name' in df.columns:
        datasets = sorted(df['dataset_name'].unique())
        
        for dataset in datasets:
            plt.figure(figsize=(14, 7))
            dataset_df = df[df['dataset_name'] == dataset]
            
            for i, run_name in enumerate(runs):
                run_dataset_df = dataset_df[dataset_df['run_name'] == run_name].copy()
                if run_dataset_df.empty:
                    continue
                run_dataset_df = run_dataset_df.sort_values('step')
                
                marker = markers[i % len(markers)]
                color = colors[i % len(colors)]
                
                plt.plot(run_dataset_df['step'], run_dataset_df['accuracy'], 
                        marker=marker, linewidth=2, markersize=8,
                        label=run_name, color=color)
            
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
    
    # Plot 2: API Overseer Penalty Function vs Step (one plot per dataset, comparing runs)
    if df['api_overseer_penalty_func'].notna().any() and 'dataset_name' in df.columns:
        datasets = sorted(df['dataset_name'].unique())
        
        for dataset in datasets:
            plt.figure(figsize=(14, 7))
            dataset_df = df[df['dataset_name'] == dataset]
            
            for i, run_name in enumerate(runs):
                run_dataset_df = dataset_df[dataset_df['run_name'] == run_name].copy()
                if run_dataset_df.empty:
                    continue
                run_dataset_df = run_dataset_df.sort_values('step')
                
                marker = markers[i % len(markers)]
                color = colors[i % len(colors)]
                
                plt.plot(run_dataset_df['step'], run_dataset_df['api_overseer_penalty_func'], 
                        marker=marker, linewidth=2, markersize=8,
                        label=run_name, color=color)
            
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
    
    # Plot 3: Combined metrics (Accuracy and API Overseer Penalty) - one plot per dataset
    if (df['accuracy'].notna().any() and 
        df['api_overseer_penalty_func'].notna().any() and
        'dataset_name' in df.columns):
        
        datasets = sorted(df['dataset_name'].unique())
        
        for dataset in datasets:
            fig, axes = plt.subplots(2, 1, figsize=(14, 10))
            dataset_df = df[df['dataset_name'] == dataset]
            
            # Accuracy subplot
            for i, run_name in enumerate(runs):
                run_dataset_df = dataset_df[dataset_df['run_name'] == run_name].copy()
                if run_dataset_df.empty:
                    continue
                run_dataset_df = run_dataset_df.sort_values('step')
                
                marker = markers[i % len(markers)]
                color = colors[i % len(colors)]
                
                axes[0].plot(run_dataset_df['step'], run_dataset_df['accuracy'], 
                            marker=marker, linewidth=2, markersize=8,
                            label=run_name, color=color)
            
            axes[0].set_ylabel('Accuracy', fontsize=11, fontweight='bold')
            axes[0].set_title(f'Accuracy vs Training Step - {dataset}', fontsize=12, fontweight='bold')
            axes[0].legend(loc='best', fontsize=9, frameon=True, shadow=True)
            axes[0].grid(True, alpha=0.3)
            axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
            
            # API Overseer Penalty Function subplot
            for i, run_name in enumerate(runs):
                run_dataset_df = dataset_df[dataset_df['run_name'] == run_name].copy()
                if run_dataset_df.empty:
                    continue
                run_dataset_df = run_dataset_df.sort_values('step')
                
                marker = markers[i % len(markers)]
                color = colors[i % len(colors)]
                
                axes[1].plot(run_dataset_df['step'], run_dataset_df['api_overseer_penalty_func'], 
                            marker=marker, linewidth=2, markersize=8,
                            label=run_name, color=color)
            
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
    
    # Plot 4: Overall comparison - all datasets and runs on one plot
    if df['accuracy'].notna().any() and 'dataset_name' in df.columns:
        datasets = sorted(df['dataset_name'].unique())
        
        plt.figure(figsize=(14, 7))
        
        for run_idx, run_name in enumerate(runs):
            run_df = df[df['run_name'] == run_name]
            
            for dataset_idx, dataset in enumerate(datasets):
                run_dataset_df = run_df[run_df['dataset_name'] == dataset].copy()
                if run_dataset_df.empty:
                    continue
                run_dataset_df = run_dataset_df.sort_values('step')
                
                # Use combination of color and marker to distinguish run + dataset
                color = colors[run_idx % len(colors)]
                marker = markers[dataset_idx % len(markers)]
                linestyle = '-' if dataset_idx == 0 else '--' if dataset_idx == 1 else '-.'
                
                label = f"{run_name} - {dataset}"
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
    
    # Plot 5: Scatter plot - Accuracy vs API Overseer Penalty (colored by run, shaped by dataset, gradated by step)
    if (df['accuracy'].notna().any() and 
        df['api_overseer_penalty_func'].notna().any() and
        'dataset_name' in df.columns):
        
        datasets = sorted(df['dataset_name'].unique())
        
        # Normalize step values for color gradation
        max_step = df['step'].max()
        min_step = df['step'].min()
        step_range = max_step - min_step if max_step != min_step else 1
        
        # One scatter plot per dataset
        for dataset in datasets:
            plt.figure(figsize=(12, 8))
            dataset_df = df[df['dataset_name'] == dataset]
            
            for run_idx, run_name in enumerate(runs):
                run_dataset_df = dataset_df[dataset_df['run_name'] == run_name].copy()
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
                    alpha_blend = 0.3 + 0.7 * (1 - norm_step)
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
                           label=f"{run_name} (dark=early, light=late)")
            
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
            
            for run_idx, run_name in enumerate(runs):
                run_dataset_df = dataset_df[dataset_df['run_name'] == run_name].copy()
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
                    # Normalize step: 0 to 1
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
                           label=f"{run_name} - {dataset}")
        
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
    columns = ['run_name', 'dataset_name', 'step', 'step_folder', 'accuracy', 'api_overseer_penalty_func']
    export_df = df[columns].copy()
    
    export_df.to_csv(output_file, index=False)
    print(f"✓ Saved: analysis_results.csv")


def save_results_list(folders: List[str], output_dir: str):
    """Save the list of input folders to results_list.txt."""
    output_file = os.path.join(output_dir, 'results_list.txt')
    
    with open(output_file, 'w') as f:
        for folder in folders:
            f.write(f"{folder}\n")
    
    print(f"✓ Saved: results_list.txt")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Analyze metrics from multiple training runs for comparison"
    )
    parser.add_argument(
        "folders", 
        nargs='+',
        help="Paths to the run folders containing eval/ subdirectories"
    )
    parser.add_argument(
        "--output", "-o", 
        help="Output directory for plots (default: results_viz/results_{timestamp})"
    )
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output:
        output_dir = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results_viz/results_{timestamp}"
    
    print("\n" + "="*80)
    print("MULTIPLE RUNS ANALYSIS SCRIPT")
    print("="*80)
    print(f"Number of folders: {len(args.folders)}")
    print(f"Output folder: {output_dir}")
    print("="*80)
    
    try:
        # Process results
        print("\nProcessing results from multiple folders...")
        df = process_multiple_folders(args.folders)
        
        # Print summary
        print_summary_statistics(df)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the list of input folders
        save_results_list(args.folders, output_dir)
        
        # Create plots
        print("\nCreating visualizations...")
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