import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc # Import roc_curve and auc
from typing import Optional
import logging # Added logging import

def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: Optional[str] = None, title: str = 'ROC Curve', optimal_threshold: Optional[float] = None) -> None:
    """Plots the ROC curve and optionally marks the optimal and default thresholds."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')

    # Mark optimal threshold if provided
    if optimal_threshold is not None:
        optimal_idx = np.where(thresholds >= optimal_threshold)[0]
        if len(optimal_idx) > 0:
            optimal_idx = optimal_idx[-1]
            plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', 
                        label=f'Optimal Threshold = {optimal_threshold:.4f}', zorder=5)

    # Mark default threshold 0.5
    default_idx = None
    for i, t in enumerate(thresholds):
        if t <= 0.5:
            default_idx = i
            break
    if default_idx is not None:
        plt.scatter(fpr[default_idx], tpr[default_idx], color='blue', 
                    label='Default Threshold = 0.5', zorder=5)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"ROC curve saved to: {save_path}") # Requires logging

    # plt.show() # Typically plots are saved, not shown interactively in a runner script
    plt.close() 