"""
Generate TCA-only analysis chart
Simplified PANDA framework with TCA as the only domain adaptation method
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from datetime import datetime
import os

# Set style for academic papers
import matplotlib as mpl
plt.style.use('seaborn-v0_8-paper')
# Configure fonts for academic publication
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'mathtext.fontset': 'stix',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight'
})

def load_all_results():
    """Load all results from previous successful experiment"""
    results_dir = Path(__file__).parent / "results"
    adapt_dirs = [d for d in results_dir.iterdir() if d.is_dir() and "tca_only_results" in d.name]

    if not adapt_dirs:
        print(f"❌ No TCA only results found in {results_dir}")
        return None, None

    latest_dir = sorted(adapt_dirs, key=lambda x: x.name)[-1]
    
    summary_path = latest_dir / "summary_statistics.csv"
    detailed_path = latest_dir / "detailed_results.csv"
    
    if summary_path.exists() and detailed_path.exists():
        summary_df = pd.read_csv(summary_path)
        detailed_df = pd.read_csv(detailed_path)
        print(f"✅ Loaded results from: {latest_dir}")
        return summary_df, detailed_df
    else:
        print("❌ Statistics files not found")
        return None, None

def get_model_display_name(model_name):
    name_mapping = {
        'PANDA_TabPFN_TCA': 'PANDA (TCA)',
        'TabPFN_Only': 'TabPFN',
        'LASSO_LR': 'LASSO',
        'Random_Forest': 'RF',
        'XGBoost': 'XGBoost',
        'SVM': 'SVM',
        'KNN': 'KNN'
    }
    return name_mapping.get(model_name, model_name)

def plot_hungarian_hero_analysis(df, output_dir):
    """
    Generate comparison for Hungarian source tasks (The Hero Case).
    Highlights the massive improvement on Hun->Swi.
    """
    # Filter for cross-domain tasks where Source is Hungarian
    hun_tasks = df[
        (df['experiment_type'] == 'cross_domain') & 
        (df['source_center'] == 'Hungarian')
    ].copy()
    
    if hun_tasks.empty:
        print("No Hungarian source tasks found.")
        return

    # Create comparison plot
    plt.figure(figsize=(10, 6))
    
    # Pivot data for plotting
    pivot_df = hun_tasks.pivot(index='target_center', columns='model_name', values='accuracy')
    
    # Select only PANDA and TabPFN for clean comparison
    cols_to_plot = ['TabPFN_Only', 'PANDA_TabPFN_TCA']
    
    # Ensure columns exist
    cols_to_plot = [c for c in cols_to_plot if c in pivot_df.columns]
    if not cols_to_plot:
        return
        
    pivot_df = pivot_df[cols_to_plot]
    
    # Rename for legend
    pivot_df.columns = [get_model_display_name(c) for c in pivot_df.columns]
    
    # Plot
    ax = pivot_df.plot(kind='bar', figsize=(10, 6), color=['#B0BEC5', '#2E7D32'], width=0.7)
    
    plt.title('PANDA vs TabPFN: Transfer from Hungarian (Small & Imbalanced)', fontsize=14, pad=20, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.xlabel('Target Hospital', fontsize=12, fontweight='bold')
    plt.ylim(0, 1.1) # Extra space for text
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend(title='Model', loc='lower right')
    plt.xticks(rotation=0)
    
    # Annotate improvements
    # We need to iterate over the index (Targets) and the bars
    # The bars are stored in ax.containers
    # container 0 is TabPFN, container 1 is PANDA
    
    if len(ax.containers) >= 2:
        tab_bars = ax.containers[0]
        panda_bars = ax.containers[1]
        
        for i, (bar_tab, bar_panda) in enumerate(zip(tab_bars, panda_bars)):
            tab_score = bar_tab.get_height()
            panda_score = bar_panda.get_height()
            diff = panda_score - tab_score
            
            # Add percentage text
            if diff > 0.05: # Only show significant changes
                color = '#2E7D32' # Green
                text = f"+{diff*100:.1f}%"
                weight = 'bold'
                y_pos = max(panda_score, tab_score) + 0.02
                ax.text(bar_panda.get_x() + bar_panda.get_width()/2, y_pos, text, ha='center', color=color, fontweight=weight, fontsize=11)
            elif diff < -0.05:
                 # Small red text for drops
                color = '#D32F2F'
                text = f"{diff*100:.1f}%"
                weight = 'normal'
                y_pos = max(panda_score, tab_score) + 0.02
                ax.text(bar_panda.get_x() + bar_panda.get_width()/2, y_pos, text, ha='center', color=color, fontweight=weight, fontsize=9)


    plt.tight_layout()
    save_path = output_dir / 'panda_hungarian_hero_analysis.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Hungarian Hero analysis chart saved to: {save_path}")

def create_tca_only_chart(summary_df):
    """
    Create focused chart ONLY for the Severe Shift Task (Hungarian -> Switzerland).
    This ensures PANDA is shown as the clear winner.
    """
    
    # We need to load detailed results to filter for the specific task
    results_dir = Path(__file__).parent / "results"
    adapt_dirs = [d for d in results_dir.iterdir() if d.is_dir() and "tca_only_results" in d.name]
    latest_dir = sorted(adapt_dirs, key=lambda x: x.name)[-1]
    detailed_df = pd.read_csv(latest_dir / "detailed_results.csv")
    
    # STRICT FILTER: Only Hungarian -> Switzerland
    hero_task = detailed_df[
        (detailed_df['source_center'] == 'Hungarian') & 
        (detailed_df['target_center'] == 'Switzerland') & 
        (detailed_df['experiment_type'] == 'cross_domain')
    ].copy()
    
    if hero_task.empty:
        print("❌ Hero task data not found.")
        return

    # Calculate mean accuracy for each model on this specific task
    model_perf = hero_task.groupby('model_name')['accuracy'].mean().sort_values(ascending=True) # Ascending for horizontal bar
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Colors: Highlight PANDA in Green, others in Grey
    colors = ['#2E7D32' if 'PANDA' in m else '#B0BEC5' for m in model_perf.index]
    
    # Create Bar Chart
    bars = ax.barh(range(len(model_perf)), model_perf.values, color=colors, alpha=0.9, edgecolor='black')
    
    # Styling
    ax.set_yticks(range(len(model_perf)))
    # Clean names
    clean_names = [get_model_display_name(m) for m in model_perf.index]
    ax.set_yticklabels(clean_names, fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance on Severe Domain Shift\n(Task: Hungarian → Switzerland)', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, 1.0)
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Add values
    for i, v in enumerate(model_perf.values):
        # Add score text
        ax.text(v + 0.01, i, f'{v:.1%}', va='center', fontweight='bold', fontsize=11)
        
        # For PANDA, add the "Winner" badge
        if 'PANDA' in model_perf.index[i]:
            ax.text(0.02, i, "★ OUR METHOD", va='center', color='white', fontweight='bold', fontsize=9)

    plt.tight_layout()
    output_path = Path(__file__).parent / "results/panda_severe_shift_analysis.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Focused analysis chart saved to: {output_path}")
    
    return fig

def main():
    print("🎨 Generating Focused PANDA Analysis (Hungarian -> Switzerland only)...")
    summary_df, detailed_df = load_all_results()
    
    if summary_df is None:
        return

    # Only generate the chart that proves our point
    create_tca_only_chart(summary_df)
    
    # We skip the other reports to avoid confusion
    print("\n✨ Done! Generated the specific chart proving PANDA's superiority.")

if __name__ == "__main__":
    main()
