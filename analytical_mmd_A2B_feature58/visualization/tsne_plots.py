import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from typing import Optional, List, Dict, Any # Added List, Dict, Any for plot functions
import logging
from .metrics import compute_domain_discrepancy # Import from local metrics file
from scipy.sparse import spmatrix # Import spmatrix for type checking
import matplotlib.container # Import for BarContainer type hint

def visualize_tsne(X_source: np.ndarray, X_target: np.ndarray, 
                   y_source: Optional[np.ndarray] = None, y_target: Optional[np.ndarray] = None, 
                   X_target_aligned: Optional[np.ndarray] = None, title: str = 't-SNE Visualization', 
                   save_path: Optional[str] = None, method_name: str = "MMD") -> None:
    """t-SNE visualization of source and target domain feature distributions"""
    # Standardize data - each feature separately
    scaler = StandardScaler()
    X_source_scaled = scaler.fit_transform(X_source)

    # Handle sparse matrix output from scaler for X_target
    X_target_scaled = scaler.transform(X_target)
    if isinstance(X_target_scaled, spmatrix):
        X_target_scaled = X_target_scaled.toarray()

    X_target_aligned_scaled = None # Initialize to None
    min_len_for_perplexity = X_source_scaled.shape[0] + X_target_scaled.shape[0]

    if X_target_aligned is not None:
        X_target_aligned_scaled = scaler.transform(X_target_aligned)
        # Handle sparse matrix output from scaler for aligned data
        if isinstance(X_target_aligned_scaled, spmatrix):
             X_target_aligned_scaled = X_target_aligned_scaled.toarray()

        # Recalculate min_len_for_perplexity with aligned data if it's available
        min_len_for_perplexity = min(min_len_for_perplexity, X_source_scaled.shape[0] + X_target_aligned_scaled.shape[0] if X_target_aligned_scaled is not None else min_len_for_perplexity)

    perplexity_val = min(30, min_len_for_perplexity - 1) if min_len_for_perplexity > 1 else 5
    
    # Define classes and colors if labels are provided
    classes = None
    source_colors = None
    target_colors = None
    if y_source is not None and y_target is not None:
        classes = np.unique(np.concatenate((y_source, y_target)))
        # Use different colormaps for source and target classes
        source_colors_cmap = plt.get_cmap('Blues')
        target_colors_cmap = plt.get_cmap('Reds')
        source_colors = source_colors_cmap(np.linspace(0.4, 0.9, len(classes))) # Use plt.get_cmap
        target_colors = target_colors_cmap(np.linspace(0.4, 0.9, len(classes))) # Use plt.get_cmap

    # Calculate domain discrepancy before and after alignment
    # Ensure using scaled dense arrays for metrics calculation
    before_metrics = compute_domain_discrepancy(X_source_scaled, X_target_scaled)
    after_metrics = {}
    if X_target_aligned_scaled is not None: # Check if aligned scaled data exists
        after_metrics = compute_domain_discrepancy(X_source_scaled, X_target_aligned_scaled)
    
    # Calculate improvement percentage
    improvement = {}
    metric_keys_for_improvement = ['mean_distance', 'mean_difference', 'covariance_difference', 'kernel_mean_difference', 'mmd', 'kl_divergence', 'wasserstein_distance']
    for k_metric in metric_keys_for_improvement:
        if k_metric not in before_metrics or (X_target_aligned_scaled is not None and k_metric not in after_metrics): # Check if aligned scaled data exists
            continue
        val_before = before_metrics[k_metric]
        val_after = after_metrics.get(k_metric, val_before) # Use before value if after doesn't exist
        
        # Avoid division by zero or very small numbers for percentage change
        if abs(val_before) < 1e-9:
            improvement[k_metric] = float('inf') if val_after > val_before else (-float('inf') if val_after < val_before else 0.0)
        elif k_metric == 'kernel_mean_difference': # Higher is better
             improvement[k_metric] = (val_after - val_before) / abs(val_before) * 100
        else: # Lower is better for other metrics
            improvement[k_metric] = (val_before - val_after) / abs(val_before) * 100
    
    if improvement:
        logging.info("Domain discrepancy improvement rates:")
        for k_log, v_log in improvement.items():
            logging.info(f'  {k_log}: {v_log:.2f}%')
    
    # Plotting setup
    if X_target_aligned is not None:
        fig = plt.figure(figsize=(24, 8))
        gs = fig.add_gridspec(1, 3)
        
        # Before alignment plot
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_title('Before MMD Adaptation')
        
        # Combine source and target for t-SNE transformation
        X_combined_before = np.vstack((X_source_scaled, X_target_scaled))
        # Use perplexity appropriate for dataset size
        current_perplexity = min(perplexity_val, X_combined_before.shape[0] - 1) if X_combined_before.shape[0] > 1 else 5
        
        # Apply t-SNE for before alignment visualization
        tsne_before = TSNE(n_components=2, random_state=42, perplexity=current_perplexity, init='pca', learning_rate='auto')
        # Ensure enough samples for t-SNE
        if X_combined_before.shape[0] > 1:
            X_before_tsne = tsne_before.fit_transform(X_combined_before)
            # Split results
            X_source_tsne_before = X_before_tsne[:len(X_source_scaled)]
            X_target_tsne_before = X_before_tsne[len(X_source_scaled):]
        else:
            logging.warning("Not enough samples for t-SNE before alignment.")
            X_source_tsne_before = np.array([])
            X_target_tsne_before = np.array([])

        # Plot before alignment
        if y_source is None or y_target is None or classes is None or source_colors is None or target_colors is None: # Check if coloring variables are defined
            ax1.scatter(X_source_tsne_before[:, 0], X_source_tsne_before[:, 1], 
                       alpha=0.7, label='Source', color='blue')
            ax1.scatter(X_target_tsne_before[:, 0], X_target_tsne_before[:, 1], 
                       alpha=0.7, label='Target', color='red')
        else:
            # classes, source_colors, target_colors are already defined at the function start
            for i_cls, cls_val in enumerate(classes):
                source_mask = (y_source == cls_val)
                target_mask = (y_target == cls_val)
                if np.any(source_mask):
                    ax1.scatter(X_source_tsne_before[source_mask, 0], X_source_tsne_before[source_mask, 1],
                               color=source_colors[i_cls], marker='o', alpha=0.7,
                               label=f'Source-Class{cls_val}')
                if np.any(target_mask):
                     ax1.scatter(X_target_tsne_before[target_mask, 0], X_target_tsne_before[target_mask, 1],
                               color=target_colors[i_cls], marker='x', alpha=0.7,
                               label=f'Target-Class{cls_val}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # After alignment plot
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_title(f'After {method_name} Adaptation')
        
        # Combine source and aligned target for t-SNE transformation
        # Ensure X_target_aligned_scaled exists before combining
        if X_target_aligned_scaled is not None:
            X_combined_after = np.vstack((X_source_scaled, X_target_aligned_scaled))
            current_perplexity_after = min(perplexity_val, X_combined_after.shape[0] - 1) if X_combined_after.shape[0] > 1 else 5
            
            # Apply t-SNE for after alignment visualization
            tsne_after = TSNE(n_components=2, random_state=42, perplexity=current_perplexity_after, init='pca', learning_rate='auto')
            # Ensure enough samples for t-SNE
            if X_combined_after.shape[0] > 1:
                X_after_tsne = tsne_after.fit_transform(X_combined_after)
                # Split results
                X_source_tsne_after = X_after_tsne[:len(X_source_scaled)]
                X_target_aligned_tsne = X_after_tsne[len(X_source_scaled):]
            else:
                 logging.warning("Not enough samples for t-SNE after alignment.")
                 X_source_tsne_after = np.array([])
                 X_target_aligned_tsne = np.array([])
        else:
             logging.warning("Aligned target data not available for t-SNE after alignment.")
             X_source_tsne_after = np.array([])
             X_target_aligned_tsne = np.array([])

        # Plot after alignment
        if y_source is None or y_target is None or classes is None or source_colors is None or target_colors is None: # Check if coloring variables are defined
            ax2.scatter(X_source_tsne_after[:, 0], X_source_tsne_after[:, 1], 
                       alpha=0.7, label='Source', color='blue')
            ax2.scatter(X_target_aligned_tsne[:, 0], X_target_aligned_tsne[:, 1], 
                       alpha=0.7, label='Target (Adapted)', color='red')
        else:
            # classes, source_colors, target_colors are already defined from the function start
            for i_cls, cls_val in enumerate(classes):
                source_mask = (y_source == cls_val)
                target_mask = (y_target == cls_val)
                if np.any(source_mask):
                    ax2.scatter(X_source_tsne_after[source_mask, 0], X_source_tsne_after[source_mask, 1],
                               color=source_colors[i_cls], marker='o', alpha=0.7,
                               label=f'Source-Class{cls_val}')
                if np.any(target_mask):
                    ax2.scatter(X_target_aligned_tsne[target_mask, 0], X_target_aligned_tsne[target_mask, 1],
                               color=target_colors[i_cls], marker='x', alpha=0.7,
                               label=f'Target (Adapted)-Class{cls_val}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Domain Discrepancy Metrics plot
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.set_title('Domain Discrepancy Metrics')
        metrics_to_show = ['mmd', 'kl_divergence', 'wasserstein_distance', 'covariance_difference']
        
        # Ensure metrics exist before plotting
        metrics_values_before = [before_metrics.get(m, 0) for m in metrics_to_show]
        metrics_values_after = [after_metrics.get(m, 0) for m in metrics_to_show]
        
        x_bar = np.arange(len(metrics_to_show))
        width_bar = 0.35
        rects1 = ax3.bar(x_bar - width_bar/2, metrics_values_before, width_bar, label='Before Adaptation')
        rects2 = ax3.bar(x_bar + width_bar/2, metrics_values_after, width_bar, label='After Adaptation')
        
        ax3.set_xlabel('Metrics')
        ax3.set_ylabel('Value')
        ax3.set_xticks(x_bar)
        ax3.set_xticklabels([m.replace('_', ' ').title() for m in metrics_to_show], rotation=45, ha="right")
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Autolabel function for bars
        def autolabel(rects: matplotlib.container.BarContainer, ax_ref: plt.Axes) -> None: # Added type hints
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax_ref.annotate(f'{height:.3f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

        # Apply autolabel
        autolabel(rects1, ax3)
        autolabel(rects2, ax3)

        # Add improvement text above bars
        for i, m_key in enumerate(metrics_to_show):
            if m_key in improvement:
                y_pos = max(metrics_values_before[i], metrics_values_after[i]) * 1.05
                arrow = '↓' if improvement[m_key] > 1e-9 else '↑' if improvement[m_key] < -1e-9 else '' # Use tolerance for zero comparison
                color = 'green' if improvement[m_key] > 1e-9 else 'red' if improvement[m_key] < -1e-9 else 'black'
                if abs(improvement[m_key]) > 1e-9: # Only add text if there was a significant change
                     ax3.text(i, y_pos, f'{arrow}{abs(improvement[m_key]):.1f}%', ha='center', va='bottom', color=color, fontsize=9, fontweight='bold')

        # Set suptitle and layout
        fig.suptitle(title, fontsize=16)
        plt.tight_layout(rect=(0, 0, 1, 0.96)) # Corrected tight_layout parameters
        
        # Save plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"t-SNE plot saved to: {save_path}")
        plt.close(fig)
        
    else: # Case when X_target_aligned is None
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(1, 1, 1)
        X_combined = np.vstack((X_source_scaled, X_target_scaled))
        current_perplexity = min(perplexity_val, X_combined.shape[0] - 1) if X_combined.shape[0] > 1 else 5
        
        # Ensure enough samples for t-SNE
        if X_combined.shape[0] > 1:
            tsne = TSNE(n_components=2, random_state=42, perplexity=current_perplexity, init='pca', learning_rate='auto')
            X_tsne = tsne.fit_transform(X_combined)
            X_source_tsne = X_tsne[:len(X_source_scaled)]
            X_target_tsne = X_tsne[len(X_source_scaled):]
        else:
            logging.warning("Not enough samples for t-SNE (no alignment case).")
            X_source_tsne = np.array([])
            X_target_tsne = np.array([])

        # Plot
        if y_source is None or y_target is None or classes is None or source_colors is None or target_colors is None: # Check if coloring variables are defined
            ax.scatter(X_source_tsne[:, 0], X_source_tsne[:, 1], alpha=0.7, label='Source', color='blue')
            ax.scatter(X_target_tsne[:, 0], X_target_tsne[:, 1], alpha=0.7, label='Target', color='red')
        else:
            # classes, source_colors, target_colors are already defined from the function start
            for i_cls, cls_val in enumerate(classes):
                source_mask = (y_source == cls_val)
                target_mask = (y_target == cls_val)
                if np.any(source_mask):
                    ax.scatter(X_source_tsne[source_mask, 0], X_source_tsne[source_mask, 1],
                               color=source_colors[i_cls], marker='o', alpha=0.7,
                               label=f'Source-Class{cls_val}')
                if np.any(target_mask):
                     ax.scatter(X_target_tsne[target_mask, 0], X_target_tsne[target_mask, 1],
                               color=target_colors[i_cls], marker='x', alpha=0.7,
                               label=f'Target-Class{cls_val}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set title and layout
        ax.set_title(title, fontsize=16)
        plt.tight_layout()
        
        # Save plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"t-SNE plot saved to: {save_path}")
        plt.close(fig) 