#!/usr/bin/env python3
"""
Visualise trial metrics for sycophancy experiments.
Generates one PNG per model (data) with subplots for each eval fold.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Style configuration
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.weight': 'normal',
    'axes.titleweight': 'normal',
    'axes.labelweight': 'normal',
    'figure.titleweight': 'normal',
})

# Pastel colors
COLOR_REWARD_HACK = '#FF9999'      # Pastel red/coral
COLOR_MONITOR_FLAG = '#99CCFF'     # Pastel blue
COLOR_CHANCE = '#CCCCCC'           # Light gray for chance line


def style_axis(ax):
    """Apply consistent styling to an axis."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    ax.set_ylim(0, 1)


def plot_subplot(ax, df_fold, eval_fold_name):
    """
    Plot a single subplot for one eval fold.
    
    Plots unaggregated lines for each seed.
    """
    seeds = df_fold['seed'].unique()
    
    # Plot each seed as a separate line (unaggregated)
    for i, seed in enumerate(seeds):
        df_seed = df_fold[df_fold['seed'] == seed].sort_values('step')
        
        # Only add label on first seed to avoid legend clutter
        rh_label = 'Reward Hack Rate' if i == 0 else None
        mf_label = 'Monitor Flag Rate' if i == 0 else None
        
        ax.plot(
            df_seed['step'], 
            df_seed['reward_hack_rate_extractable'],
            color=COLOR_REWARD_HACK,
            linewidth=1.5,
            label=rh_label,
            alpha=0.8
        )
        
        ax.plot(
            df_seed['step'], 
            df_seed['monitor_flag_rate_extractable'],
            color=COLOR_MONITOR_FLAG,
            linewidth=1.5,
            label=mf_label,
            alpha=0.8
        )
    
    # Chance line at 0.5
    step_range = df_fold['step'].agg(['min', 'max'])
    ax.axhline(
        y=0.5, 
        color=COLOR_CHANCE, 
        linestyle='--', 
        linewidth=1,
        label='Chance (0.5)' if seeds.size > 0 else None,
        zorder=0
    )
    
    ax.set_title(eval_fold_name)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Rate')
    
    style_axis(ax)


def create_figure_for_model(df_model, model_name, output_dir):
    """
    Create a figure with subplots for each eval fold.
    """
    eval_folds = df_model['eval_fold'].unique()
    n_folds = len(eval_folds)
    
    # Determine subplot layout
    if n_folds == 1:
        n_cols, n_rows = 1, 1
    elif n_folds == 2:
        n_cols, n_rows = 2, 1
    elif n_folds <= 4:
        n_cols, n_rows = 2, 2
    elif n_folds <= 6:
        n_cols, n_rows = 3, 2
    elif n_folds <= 9:
        n_cols, n_rows = 3, 3
    else:
        n_cols = 4
        n_rows = int(np.ceil(n_folds / n_cols))
    
    fig, axes = plt.subplots(
        n_rows, n_cols, 
        figsize=(5 * n_cols, 4 * n_rows),
        squeeze=False
    )
    axes = axes.flatten()
    
    for idx, eval_fold in enumerate(eval_folds):
        df_fold = df_model[df_model['eval_fold'] == eval_fold]
        plot_subplot(axes[idx], df_fold, eval_fold)
    
    # Hide unused subplots
    for idx in range(n_folds, len(axes)):
        axes[idx].set_visible(False)
    
    # Add legend to first subplot
    if n_folds > 0:
        axes[0].legend(loc='lower right', frameon=False)
    
    fig.suptitle(f'Model: {model_name}', fontsize=14)
    fig.tight_layout()
    
    # Save figure
    output_path = output_dir / f'{model_name}.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {output_path}')


def main():
    # Paths
    script_dir = Path(__file__).parent
    csv_path = script_dir / 'trial_metrics_sycophancy.csv'
    output_dir = script_dir
    
    # Load data
    print(f'Loading data from: {csv_path}')
    df = pd.read_csv(csv_path)
    
    # Get unique models
    models = df['data'].unique()
    print(f'Found {len(models)} models: {list(models)}')
    
    # Create one figure per model
    for model in models:
        df_model = df[df['data'] == model]
        print(f'\nProcessing model: {model}')
        print(f'  Eval folds: {list(df_model["eval_fold"].unique())}')
        print(f'  Seeds: {list(df_model["seed"].unique())}')
        create_figure_for_model(df_model, model, output_dir)
    
    print('\nDone!')


if __name__ == '__main__':
    main()