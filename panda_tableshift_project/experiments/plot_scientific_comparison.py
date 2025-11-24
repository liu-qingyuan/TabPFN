"""
Generate scientific comparison chart for TableShift results.
Specifically comparing Baseline (None) vs PANDA (TCA) on BRFSS Diabetes.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

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

def plot_comparison():
    # Data from user request
    # Baseline: n_est=1, method=None
    baseline_auc = 0.7990165031703648
    baseline_acc = 0.84619140625
    
    # TCA: n_est=32, method=TCA
    tca_auc = 0.8049056087748213
    tca_acc = 0.84765625

    labels = ['AUC', 'Accuracy']
    baseline_vals = [baseline_auc, baseline_acc]
    tca_vals = [tca_auc, tca_acc]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Colors: Baseline (Grey), PANDA (Green)
    rects1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline (None)', color='#90A4AE', alpha=0.9)
    rects2 = ax.bar(x + width/2, tca_vals, width, label='PANDA (TCA)', color='#2E7D32', alpha=0.9)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Score')
    ax.set_title('Performance Comparison: BRFSS Diabetes (TableShift)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.7, 0.9) # Zoom in to show difference, as values are close
    ax.legend(loc='upper left')

    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    # Output path
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "scientific_comparison_plot.pdf"
    
    plt.savefig(output_path)
    print(f"✅ Chart saved to: {output_path}")
    
    # Also save as PNG for easier viewing if needed
    png_path = output_dir / "scientific_comparison_plot.png"
    plt.savefig(png_path)
    print(f"✅ Chart saved to: {png_path}")

if __name__ == "__main__":
    plot_comparison()
