import json
import os
import re
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import seaborn as sns
from typing import Dict, List, Tuple, Optional


# ========== CONFIGURATION ==========
# Easily comment/uncomment datasets for accuracy plots
# To include/exclude datasets, simply add/remove '#' at the beginning of the line
ACCURACY_DATASETS = [
    'code_selection',
    'revealing_score', 
    'sycophancy_fact',
    'sycophancy_opinion_nlp',
    'sycophancy_opinion_political',
    # 'email_assistant',
    # 'theory_of_mind_mirroring',
    # 'theory_of_mind_mirroring_expanded',
    # 'world_affecting_approval',
    # 'world_affecting_reward',
]
# ===================================


def extract_step_from_artifact_name(artifact_name: str) -> int:
    """Extract the step number from artifact name."""
    if artifact_name == "initial:v0":
        return 0
    else:
        # Extract step number (e.g., "200:v0" -> 200)
        match = re.search(r'(\d+):v\d+', artifact_name)
        if match:
            return int(match.group(1))
    return None


def load_evaluation_results(results_path: str) -> Dict:
    """Load evaluation results from JSON file."""
    with open(results_path, 'r') as f:
        data = json.load(f)
    return data


def prepare_accuracy_data(metrics_by_artifact: Dict) -> pd.DataFrame:
    """Prepare data for accuracy vs step plot."""
    rows = []
    
    for artifact_name, artifact_metrics in metrics_by_artifact.items():
        step = extract_step_from_artifact_name(artifact_name)
        
        for dataset_name, dataset_metrics in artifact_metrics.items():
            if dataset_name != "overall" and isinstance(dataset_metrics, dict):
                if "accuracy" in dataset_metrics:
                    rows.append({
                        'step': step,
                        'dataset': dataset_name,
                        'accuracy': dataset_metrics['accuracy'],
                        'artifact': artifact_name
                    })
    
    return pd.DataFrame(rows)


def prepare_custom_terms_data(metrics_by_artifact: Dict) -> pd.DataFrame:
    """Prepare data for custom terms and user mentions plot."""
    rows = []
    
    # Define which datasets to track for specific metrics
    sycophancy_datasets = ["sycophancy_fact", "sycophancy_opinion_nlp", "sycophancy_opinion_political"]
    custom_terms_datasets = ["code_selection", "revealing_score"]
    
    for artifact_name, artifact_metrics in metrics_by_artifact.items():
        step = extract_step_from_artifact_name(artifact_name)
        
        # Track custom terms for code_selection and revealing_score only
        for dataset_name in custom_terms_datasets:
            if dataset_name in artifact_metrics and isinstance(artifact_metrics[dataset_name], dict):
                if "avg_cot_custom_terms" in artifact_metrics[dataset_name]:
                    rows.append({
                        'step': step,
                        'dataset': dataset_name,
                        'metric': 'avg_cot_custom_terms',
                        'value': artifact_metrics[dataset_name]['avg_cot_custom_terms'],
                        'artifact': artifact_name
                    })
        
        # Track combined avg_cot_user + avg_cot_name for sycophancy datasets
        for dataset_name in sycophancy_datasets:
            if dataset_name in artifact_metrics and isinstance(artifact_metrics[dataset_name], dict):
                user_mentions = artifact_metrics[dataset_name].get("avg_cot_user", 0)
                name_mentions = artifact_metrics[dataset_name].get("avg_cot_name", 0)
                
                # Combine user and name mentions
                combined_mentions = user_mentions + name_mentions
                
                rows.append({
                    'step': step,
                    'dataset': dataset_name,
                    'metric': 'avg_cot_user_name_combined',
                    'value': combined_mentions,
                    'artifact': artifact_name
                })
    
    return pd.DataFrame(rows)


def create_accuracy_plot(df_accuracy: pd.DataFrame, output_dir: str, folder_name: str):
    """Create accuracy vs step line plot."""
    plt.figure(figsize=(12, 8))
    
    # Filter for configured datasets
    df_filtered = df_accuracy[df_accuracy['dataset'].isin(ACCURACY_DATASETS)]
    
    if df_filtered.empty:
        print(f"Warning: No data found for configured datasets: {ACCURACY_DATASETS}")
        print("Available datasets:", df_accuracy['dataset'].unique().tolist())
        # Fall back to showing all datasets
        df_filtered = df_accuracy
    
    print(f"Plotting accuracy for datasets: {sorted(df_filtered['dataset'].unique())}")
    
    # Create line plot
    sns.lineplot(data=df_filtered, x='step', y='accuracy', hue='dataset', marker='o', linewidth=2, markersize=8)
    
    plt.title(f'Accuracy vs Training Step - {folder_name}', fontsize=16, fontweight='bold')
    plt.xlabel('Training Step', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(title='Dataset', title_fontsize=12, fontsize=11)
    
    # Format y-axis as percentage and set range to 0-100%
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
    plt.ylim(0, 1)  # 0 to 1 in decimal, which displays as 0% to 100%
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, f'{folder_name}_accuracy_vs_step.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Accuracy plot saved to: {output_path}")
    plt.close()


def create_custom_terms_plot(df_custom: pd.DataFrame, output_dir: str, folder_name: str):
    """Create custom terms and user mentions vs step line plot."""
    
    plt.figure(figsize=(14, 8))
    
    # Plot custom terms for code_selection and revealing_score
    df_custom_terms = df_custom[df_custom['metric'] == 'avg_cot_custom_terms']
    if not df_custom_terms.empty:
        for dataset in df_custom_terms['dataset'].unique():
            data = df_custom_terms[df_custom_terms['dataset'] == dataset]
            plt.plot(data['step'], data['value'], marker='o', linewidth=2, markersize=6, label=f"{dataset} (custom terms)")
    
    # Plot combined user + name mentions for sycophancy datasets  
    df_combined_mentions = df_custom[df_custom['metric'] == 'avg_cot_user_name_combined']
    if not df_combined_mentions.empty:
        for dataset in df_combined_mentions['dataset'].unique():
            data = df_combined_mentions[df_combined_mentions['dataset'] == dataset]
            plt.plot(data['step'], data['value'], marker='s', linewidth=2, markersize=6, label=f"{dataset} (user+name mentions)")
    
    plt.title(f'Average CoT Custom Terms vs Training Step - {folder_name}', fontsize=16, fontweight='bold')
    plt.xlabel('Training Step', fontsize=14)
    plt.ylabel('Average Custom Terms in CoT', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(title='Dataset', fontsize=10)
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, f'{folder_name}_custom_terms_vs_step.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Custom terms plot saved to: {output_path}")
    plt.close()


def create_visualizations(eval_folder_path: str, output_dir: Optional[str] = None):
    """Main function to create visualizations from evaluation results."""
    
    # Validate input path
    eval_path = Path(eval_folder_path)
    if not eval_path.exists():
        raise FileNotFoundError(f"Evaluation folder not found: {eval_folder_path}")
    
    # Find the results_by_artifact.json file
    results_file = eval_path / "results_by_artifact.json"
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    # Set output directory
    if output_dir is None:
        output_dir = str(eval_path / "plots")
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract folder name for titles
    folder_name = eval_path.name
    
    print(f"Loading results from: {results_file}")
    
    # Load and process data
    data = load_evaluation_results(str(results_file))
    metrics_by_artifact = data.get("metrics_by_artifact", {})
    
    if not metrics_by_artifact:
        raise ValueError("No metrics found in results file")
    
    print(f"Found {len(metrics_by_artifact)} artifacts to process")
    
    # Prepare data for plots
    df_accuracy = prepare_accuracy_data(metrics_by_artifact)
    df_custom = prepare_custom_terms_data(metrics_by_artifact)
    
    print(f"Prepared accuracy data: {len(df_accuracy)} rows")
    print(f"Prepared custom terms data: {len(df_custom)} rows")
    
    # Create plots
    if not df_accuracy.empty:
        create_accuracy_plot(df_accuracy, output_dir, folder_name)
    else:
        print("Warning: No accuracy data found")
    
    if not df_custom.empty:
        create_custom_terms_plot(df_custom, output_dir, folder_name)
    else:
        print("Warning: No custom terms data found")
    
    print(f"\nVisualization complete! Plots saved in: {output_dir}")
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    if not df_accuracy.empty:
        print("\nDatasets found for accuracy plots:")
        for dataset in sorted(df_accuracy['dataset'].unique()):
            print(f"  - {dataset}")
    
    if not df_custom.empty:
        print("\nMetrics found for custom terms plots:")
        for metric in sorted(df_custom['metric'].unique()):
            datasets = df_custom[df_custom['metric'] == metric]['dataset'].unique()
            print(f"  - {metric}: {', '.join(sorted(datasets))}")


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description="Create visualizations from evaluation results")
    parser.add_argument("eval_folder", help="Path to evaluation folder (e.g., results/eval/COT_Penalization_Sept_19/Qwen3_1.7B_Pen_v4)")
    parser.add_argument("--output", "-o", help="Output directory for plots (default: eval_folder/plots)")
    
    args = parser.parse_args()
    
    try:
        create_visualizations(args.eval_folder, args.output)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
