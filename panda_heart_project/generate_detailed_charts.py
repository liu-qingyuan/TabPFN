"""
Generate detailed comprehensive charts for PANDA vs Baselines.
Specifically for Hungarian -> Switzerland (Severe Shift) task.
Includes both Accuracy and AUC metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import matplotlib as mpl

# Set style for Nature-like academic papers
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif'],
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'text.usetex': False,
    'mathtext.default': 'regular',
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'patch.linewidth': 0.8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def get_model_display_name(model_name):
    # Exact mapping requested by user
    name_mapping = {
        'PANDA_TabPFN_TCA': 'PANDA (TCA)',
        'TabPFN_Only': 'TabPFN',
        'LASSO_LR': 'LASSO',
        'Random_Forest': 'RF',
        'XGBoost': 'XGBoost',
        'KNN': 'KNN'
    }
    return name_mapping.get(model_name, None) # Return None if not in list to filter

def load_latest_results():
    """Load detailed results from the latest experiment"""
    results_dir = Path(__file__).parent / "results"
    adapt_dirs = [d for d in results_dir.iterdir() if d.is_dir() and "tca_only_results" in d.name]

    if not adapt_dirs:
        print(f"❌ No TCA only results found in {results_dir}")
        return None

    latest_dir = sorted(adapt_dirs, key=lambda x: x.name)[-1]
    detailed_path = latest_dir / "detailed_results.csv"
    
    if detailed_path.exists():
        print(f"✅ Loaded results from: {detailed_path}")
        return pd.read_csv(detailed_path)
    else:
        print("❌ Detailed results file not found")
        return None

def plot_severe_shift_analysis(df, source, target, output_path):
    """
    Generate a 2-subplot figure (Accuracy & AUC) for the severe shift task.
    Style mimics Nature combined analysis figure.
    """
    # Filter data
    task_df = df[
        (df['source_center'] == source) & 
        (df['target_center'] == target) & 
        (df['experiment_type'] == 'cross_domain')
    ].copy()
    
    if task_df.empty:
        print(f"⚠️ No data found for task: {source} -> {target}")
        return

    # Calculate mean metrics
    metrics = task_df.groupby('model_name')[['accuracy', 'auc']].mean().reset_index()
    
    # Add display names and filter
    metrics['display_name'] = metrics['model_name'].apply(get_model_display_name)
    metrics = metrics.dropna(subset=['display_name']) # Remove models not in our specific list
    
    if metrics.empty:
        print("⚠️ No matching models found after filtering.")
        return

    # Sort by Accuracy for consistent ordering
    metrics = metrics.sort_values('accuracy', ascending=True) 
    
    # Setup plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6)) # Adjusted size
    
    # Color scheme
    # PANDA (TCA) gets a distinct color, others share a palette or grey
    # Using a professional palette
    colors = []
    for name in metrics['display_name']:
        if 'PANDA' in name:
            colors.append('#2E7D32') # Strong Green
        elif 'TabPFN' in name:
             colors.append('#1976D2') # Blue
        else:
            colors.append('#90A4AE') # Muted Blue-Grey

    # Plot 1: Accuracy (Subplot a)
    bars1 = axes[0].barh(metrics['display_name'], metrics['accuracy'], color=colors, alpha=0.9, height=0.6)
    axes[0].set_xlabel('Accuracy', fontsize=12)
    axes[0].set_xlim(0, 1.05)
    axes[0].grid(axis='x', linestyle='--', alpha=0.3)
    
    # Remove top and right spines
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # Subplot Label 'a'
    axes[0].set_title('a', fontweight='bold', fontsize=20, loc='left', pad=15)
    
    # Add values
    for i, v in enumerate(metrics['accuracy']):
        axes[0].text(v + 0.01, i, f'{v:.1%}', va='center', fontsize=10, fontweight='bold')

    # Plot 2: AUC (Subplot b)
    bars2 = axes[1].barh(metrics['display_name'], metrics['auc'], color=colors, alpha=0.9, height=0.6)
    axes[1].set_xlabel('AUC', fontsize=12)
    axes[1].set_xlim(0, 1.05)
    axes[1].grid(axis='x', linestyle='--', alpha=0.3)
    
    # Remove top and right spines
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    # Subplot Label 'b'
    axes[1].set_title('b', fontweight='bold', fontsize=20, loc='left', pad=15)
    
    # Add values
    for i, v in enumerate(metrics['auc']):
        axes[1].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=10, fontweight='bold')

    # Global Layout
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Chart saved to: {output_path}")

def main():
    print("🎨 Generating Nature-style Severe Shift Analysis Chart...")
    df = load_latest_results()
    
    if df is None:
        return

    results_dir = Path(__file__).parent / "results"
    
    # Generate specifically for Hungarian -> Switzerland (Severe Shift)
    # Using the file name requested by user
    plot_severe_shift_analysis(
        df, 
        source='Hungarian', 
        target='Switzerland', 
        output_path=results_dir / "panda_severe_shift_analysis.pdf"
    )
    
    print("\n✨ Done!")

if __name__ == "__main__":
    main()
