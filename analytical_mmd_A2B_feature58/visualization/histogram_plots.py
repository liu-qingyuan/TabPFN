import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Optional, List
try:
    from ..metrics.discrepancy import calculate_kl_divergence, calculate_wasserstein_distances
except ImportError:
    try:
        from analytical_mmd_A2B_feature58.metrics.discrepancy import calculate_kl_divergence, calculate_wasserstein_distances
    except ImportError:
        def calculate_kl_divergence(*args, **kwargs): return 0.0, {}
        def calculate_wasserstein_distances(*args, **kwargs): return 0.0, {}
from scipy.sparse import spmatrix

def visualize_feature_histograms(X_source: np.ndarray, X_target: np.ndarray, 
                                 X_target_aligned: Optional[np.ndarray] = None, 
                                 feature_names: Optional[List[str]] = None, 
                                 n_features_to_plot: Optional[int] = None, 
                                 title: str = 'Feature Distribution Comparison', 
                                save_path: Optional[str] = None, method_name: str = "MMD") -> None:
    """Plot histograms for each feature to compare distributions"""
    num_original_features = X_source.shape[1]
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(num_original_features)]
    elif len(feature_names) != num_original_features:
        logging.warning(f"Feature names length ({len(feature_names)}) differs from number of features ({num_original_features}). Adjusting names.")
        if len(feature_names) > num_original_features:
            feature_names = feature_names[:num_original_features]
        else:
            feature_names.extend([f'Feature {i+1}' for i in range(len(feature_names), num_original_features)])

    logging.info(f'特征直方图可视化: 收到特征名称列表: {feature_names[:5]}... 等{len(feature_names)}个特征')
    logging.info(f'源域X_source数据形状: {X_source.shape}, 目标域X_target数据形状: {X_target.shape}')
    if X_target_aligned is not None:
        logging.info(f'对齐后目标域X_target_aligned数据形状: {X_target_aligned.shape}')

    n_features_to_actually_plot = num_original_features if n_features_to_plot is None or n_features_to_plot > num_original_features else n_features_to_plot
    
    # Color thresholds - same as CORAL version
    KL_SEVERE_THRESHOLD = 0.5
    KL_MODERATE_THRESHOLD = 0.2
    WASS_SEVERE_THRESHOLD = 0.5
    WASS_MODERATE_THRESHOLD = 0.2
    
    SEVERE_COLOR = '#ff6b6b'
    MODERATE_COLOR = '#feca57'
    LOW_COLOR = '#1dd1a1'

    # Handle sparse matrix for X_target and X_target_aligned
    X_target_dense = X_target
    if isinstance(X_target, spmatrix):
        X_target_dense = X_target.toarray()

    X_target_aligned_dense = None
    if X_target_aligned is not None:
        X_target_aligned_dense = X_target_aligned
        if isinstance(X_target_aligned, spmatrix):
            X_target_aligned_dense = X_target_aligned.toarray()

    if X_target_aligned is not None and X_target_aligned_dense is not None:
        kl_div_before_avg_val, kl_div_before_per_feature = calculate_kl_divergence(X_source, X_target_dense)
        kl_div_after_avg_val, kl_div_after_per_feature = calculate_kl_divergence(X_source, X_target_aligned_dense)
        wass_before_avg_val, wasserstein_before_per_feature = calculate_wasserstein_distances(X_source, X_target_dense)
        wass_after_avg_val, wasserstein_after_per_feature = calculate_wasserstein_distances(X_source, X_target_aligned_dense)
        
        # Log the average values
        logging.debug(f"KL Divergence Before (Avg): {kl_div_before_avg_val:.4f}, After (Avg): {kl_div_after_avg_val:.4f}")
        logging.debug(f"Wasserstein Distance Before (Avg): {wass_before_avg_val:.4f}, After (Avg): {wass_after_avg_val:.4f}")

        # Select indices to plot based on KL divergence before alignment
        selected_indices = sorted(range(num_original_features), key=lambda k_idx: kl_div_before_per_feature.get(f'feature_{k_idx}',0.0), reverse=True)
        
        if n_features_to_plot is not None and n_features_to_plot < num_original_features:
             selected_indices = selected_indices[:n_features_to_actually_plot]
        
        num_plots = len(selected_indices)
        if num_plots == 0:
            logging.warning("No features selected for histogram plotting.")
            return
        
        ncols_hist = 2
        nrows_hist = (num_plots + ncols_hist - 1) // ncols_hist 
        fig_hist, axes_hist = plt.subplots(nrows_hist, ncols_hist * 2, figsize=(18, 4 * nrows_hist), squeeze=False)
        
        for i_plot, feature_idx in enumerate(selected_indices):
            feature_name_display = feature_names[feature_idx]
            feature_key = f'feature_{feature_idx}'
            row_idx, col_base_idx = divmod(i_plot, ncols_hist)
            
            # Plot before adaptation histogram
            ax_before = axes_hist[row_idx, col_base_idx * 2]
            sns.histplot(X_source[:, feature_idx], kde=True, ax=ax_before, color='blue', alpha=0.5, label='Source', stat="density")
            sns.histplot(X_target_dense[:, feature_idx], kde=True, ax=ax_before, color='red', alpha=0.5, label='Target', stat="density")
            kl_b = kl_div_before_per_feature.get(feature_key,0.0)
            wass_b = wasserstein_before_per_feature.get(feature_key,0.0)
            severity_b_str = "HIGH" if kl_b > KL_SEVERE_THRESHOLD or wass_b > WASS_SEVERE_THRESHOLD else \
                         "MEDIUM" if kl_b > KL_MODERATE_THRESHOLD or wass_b > WASS_MODERATE_THRESHOLD else "LOW"
            color_b_hex = SEVERE_COLOR if severity_b_str == "HIGH" else MODERATE_COLOR if severity_b_str == "MEDIUM" else LOW_COLOR
            ax_before.set_title(f'{feature_name_display} (Before): {severity_b_str}', color='white' if severity_b_str != "MEDIUM" else 'black', 
                                fontweight='bold', bbox=dict(facecolor=color_b_hex, pad=3, edgecolor='none'))
            ax_before.legend()
            ax_before.text(0.05, 0.95, f'KL: {kl_b:.3f}\nWass: {wass_b:.3f}', transform=ax_before.transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            # Plot after adaptation histogram
            ax_after = axes_hist[row_idx, col_base_idx * 2 + 1]
            sns.histplot(X_source[:, feature_idx], kde=True, ax=ax_after, color='blue', alpha=0.5, label='Source', stat="density")
            sns.histplot(X_target_aligned_dense[:, feature_idx], kde=True, ax=ax_after, color='red', alpha=0.5, label=f'Target ({method_name} Adapted)', stat="density")
            kl_a = kl_div_after_per_feature.get(feature_key,0.0)
            wass_a = wasserstein_after_per_feature.get(feature_key,0.0)
            kl_imp_val = (kl_b - kl_a) / kl_b * 100 if kl_b > 1e-9 else (0 if abs(kl_a) < 1e-9 else -float('inf'))
            imp_color_hex = SEVERE_COLOR if kl_imp_val < -10 else (MODERATE_COLOR if kl_imp_val < 10 and kl_imp_val > -10 else (LOW_COLOR if kl_imp_val > 10 else 'lightgray'))
            imp_text_str = f"KL Imp: {kl_imp_val:.1f}%"
            ax_after.set_title(f'{feature_name_display} (After)', fontweight='bold')
            ax_after.legend()
            ax_after.text(0.05, 0.95, f'KL: {kl_a:.3f}\nWass: {wass_a:.3f}\n{imp_text_str}', transform=ax_after.transAxes, 
                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor=imp_color_hex, alpha=0.8))

        # Remove unused subplots
        for i_del in range(num_plots, nrows_hist * ncols_hist):
            row_idx_del, col_base_idx_del = divmod(i_del, ncols_hist)
            # Remove both the 'before' and 'after' axes for this plot index
            if row_idx_del < axes_hist.shape[0] and col_base_idx_del * 2 < axes_hist.shape[1]:
                 fig_hist.delaxes(axes_hist[row_idx_del, col_base_idx_del * 2])
            if row_idx_del < axes_hist.shape[0] and col_base_idx_del * 2 + 1 < axes_hist.shape[1]:
                 fig_hist.delaxes(axes_hist[row_idx_del, col_base_idx_del * 2 + 1])

        fig_hist.suptitle(title, fontsize=16)
        fig_hist.tight_layout(rect=(0, 0.03, 1, 0.95))
        if save_path:
            fig_hist.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Feature histograms saved to: {save_path}")
        plt.close(fig_hist)
    else: # Case when X_target_aligned is None
        kl_div_avg_val, kl_div_per_feature = calculate_kl_divergence(X_source, X_target_dense)
        wass_avg_val, wasserstein_per_feature = calculate_wasserstein_distances(X_source, X_target_dense)
        
        # Log the average values
        logging.debug(f"KL Divergence (No Alignment, Avg): {kl_div_avg_val:.4f}")
        logging.debug(f"Wasserstein Distance (No Alignment, Avg): {wass_avg_val:.4f}")

        # Select indices to plot based on KL divergence
        selected_indices = sorted(range(num_original_features), key=lambda k_idx: kl_div_per_feature.get(f'feature_{k_idx}',0.0), reverse=True)
        
        if n_features_to_plot is not None and n_features_to_plot < num_original_features:
             selected_indices = selected_indices[:n_features_to_actually_plot]
        
        num_plots = len(selected_indices)
        if num_plots == 0: 
            logging.warning("No features selected for histogram plotting (no alignment case).")
            return

        ncols_hist = 2 
        nrows_hist = (num_plots + ncols_hist - 1) // ncols_hist
        fig_hist, axes_hist = plt.subplots(nrows_hist, ncols_hist, figsize=(12, 4 * nrows_hist), squeeze=False)
        
        for i_plot, feature_idx in enumerate(selected_indices):
            feature_name_display = feature_names[feature_idx]
            feature_key = f'feature_{feature_idx}'
            ax = axes_hist[i_plot // ncols_hist, i_plot % ncols_hist]
            sns.histplot(X_source[:, feature_idx], kde=True, ax=ax, color='blue', alpha=0.5, label='Source', stat="density")
            sns.histplot(X_target_dense[:, feature_idx], kde=True, ax=ax, color='red', alpha=0.5, label='Target', stat="density")
            kl_val = kl_div_per_feature.get(feature_key, 0.0)
            wass_val = wasserstein_per_feature.get(feature_key, 0.0)
            severity_str = "HIGH" if kl_val > KL_SEVERE_THRESHOLD or wass_val > WASS_SEVERE_THRESHOLD else \
                       "MEDIUM" if kl_val > KL_MODERATE_THRESHOLD or wass_val > WASS_MODERATE_THRESHOLD else "LOW"
            box_c_hex = SEVERE_COLOR if severity_str == "HIGH" else MODERATE_COLOR if severity_str == "MEDIUM" else LOW_COLOR
            ax.set_title(f'{feature_name_display}: {severity_str}', color='white' if severity_str != "MEDIUM" else 'black', 
                         fontweight='bold', bbox=dict(facecolor=box_c_hex, pad=3, edgecolor='none'))
            ax.legend()
            ax.text(0.05, 0.95, f'KL: {kl_val:.3f}\nWass: {wass_val:.3f}', transform=ax.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Remove unused subplots
        for i_del in range(num_plots, nrows_hist * ncols_hist):
             if i_del < axes_hist.size:
                # Calculate the row and column index for deletion
                row_idx_del = i_del // ncols_hist
                col_idx_del = i_del % ncols_hist
                # Ensure the indices are within the bounds of the axes_hist array
                if row_idx_del < axes_hist.shape[0] and col_idx_del < axes_hist.shape[1]:
                   fig_hist.delaxes(axes_hist[row_idx_del, col_idx_del])

        fig_hist.suptitle(title, fontsize=16)
        fig_hist.tight_layout(rect=(0, 0.03, 1, 0.95))
        if save_path:
            fig_hist.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Feature histograms saved to: {save_path}")
        plt.close(fig_hist)

def histograms_stats_table(X_source: np.ndarray, X_target: np.ndarray, 
                          X_target_aligned: Optional[np.ndarray] = None,
                          feature_names: Optional[List[str]] = None,
                          save_path: Optional[str] = None, method_name: str = "MMD") -> None:
    """Generate statistics table for feature distributions"""
    num_original_features = X_source.shape[1]
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(num_original_features)]
    
    # Handle sparse matrices
    X_target_dense = X_target
    if isinstance(X_target, spmatrix):
        X_target_dense = X_target.toarray()

    X_target_aligned_dense = None
    if X_target_aligned is not None:
        X_target_aligned_dense = X_target_aligned
        if isinstance(X_target_aligned, spmatrix):
            X_target_aligned_dense = X_target_aligned.toarray()

    # Color thresholds
    KL_SEVERE_THRESHOLD = 0.5
    KL_MODERATE_THRESHOLD = 0.2
    WASS_SEVERE_THRESHOLD = 0.5
    WASS_MODERATE_THRESHOLD = 0.2
    
    SEVERE_COLOR = '#ff6b6b'
    MODERATE_COLOR = '#feca57'
    LOW_COLOR = '#1dd1a1'

    if X_target_aligned is not None and X_target_aligned_dense is not None:
        kl_div_before_avg_val, kl_div_before_per_feature = calculate_kl_divergence(X_source, X_target_dense)
        kl_div_after_avg_val, kl_div_after_per_feature = calculate_kl_divergence(X_source, X_target_aligned_dense)
        wass_before_avg_val, wasserstein_before_per_feature = calculate_wasserstein_distances(X_source, X_target_dense)
        wass_after_avg_val, wasserstein_after_per_feature = calculate_wasserstein_distances(X_source, X_target_aligned_dense)
        
        # Log the average values
        logging.debug(f"KL Divergence Before (Avg): {kl_div_before_avg_val:.4f}, After (Avg): {kl_div_after_avg_val:.4f}")
        logging.debug(f"Wasserstein Distance Before (Avg): {wass_before_avg_val:.4f}, After (Avg): {wass_after_avg_val:.4f}")

        # Create statistics table
        stats_fig = plt.figure(figsize=(18, max(8, num_original_features * 0.5)))
        ax_table = stats_fig.add_subplot(111)
        ax_table.axis('off')
        table_data: List[List[str]] = []
        table_columns = ['Feature', 'KL Before', 'KL After', 'KL Imp. %', 
                        'Wass. Before', 'Wass. After', 'Wass. Imp. %', 
                        'Initial Shift', 'Adapted Shift']
        
        for i in range(num_original_features):
            feature_key = f'feature_{i}'
            feature_name_display = feature_names[i]
            kl_b = kl_div_before_per_feature.get(feature_key, 0.0)
            kl_a = kl_div_after_per_feature.get(feature_key, 0.0)
            wass_b = wasserstein_before_per_feature.get(feature_key, 0.0)
            wass_a = wasserstein_after_per_feature.get(feature_key, 0.0)
            
            # Avoid division by zero for percentage improvement
            kl_imp = (kl_b - kl_a) / kl_b * 100 if kl_b > 1e-9 else (0 if abs(kl_a) < 1e-9 else -float('inf'))
            wass_imp = (wass_b - wass_a) / wass_b * 100 if wass_b > 1e-9 else (0 if abs(wass_a) < 1e-9 else -float('inf'))

            initial_severity = "HIGH" if kl_b > KL_SEVERE_THRESHOLD or wass_b > WASS_SEVERE_THRESHOLD else \
                               "MEDIUM" if kl_b > KL_MODERATE_THRESHOLD or wass_b > WASS_MODERATE_THRESHOLD else "LOW"
            adapted_severity = "HIGH" if kl_a > KL_SEVERE_THRESHOLD or wass_a > WASS_SEVERE_THRESHOLD else \
                               "MEDIUM" if kl_a > KL_MODERATE_THRESHOLD or wass_a > WASS_MODERATE_THRESHOLD else "LOW"
            table_data.append([
                feature_name_display, f'{kl_b:.3f}', f'{kl_a:.3f}', f'{kl_imp:.1f}%',
                f'{wass_b:.3f}', f'{wass_a:.3f}', f'{wass_imp:.1f}%',
                initial_severity, adapted_severity
            ])
        
        colWidths = [0.2, 0.08, 0.08, 0.09, 0.08, 0.08, 0.09, 0.1, 0.1]
        table = ax_table.table(cellText=table_data, colLabels=table_columns, loc='center', cellLoc='center', colWidths=colWidths)
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.8)

        # Apply colors to table cells
        for r_idx in range(len(table_data) + 1):
            for c_idx in range(len(table_columns)):
                cell = table[r_idx, c_idx]
                if r_idx == 0:  # Header row
                    cell.set_facecolor('#4a69bd')
                    cell.set_text_props(color='white', fontweight='bold')
                else:  # Data rows
                    current_row_data = table_data[r_idx-1]
                    text_color = 'black'

                    # Apply color coding based on column
                    if c_idx in [1, 2]:  # KL columns
                        try:
                            val = float(current_row_data[c_idx])
                            if val > KL_SEVERE_THRESHOLD:
                                cell.set_facecolor(SEVERE_COLOR)
                                text_color = 'white'
                            elif val > KL_MODERATE_THRESHOLD:
                                cell.set_facecolor(MODERATE_COLOR)
                            else:
                                cell.set_facecolor(LOW_COLOR)
                        except ValueError: pass
                    
                    elif c_idx in [4, 5]:  # Wasserstein columns
                        try:
                            val = float(current_row_data[c_idx])
                            if val > WASS_SEVERE_THRESHOLD:
                                cell.set_facecolor(SEVERE_COLOR)
                                text_color = 'white'
                            elif val > WASS_MODERATE_THRESHOLD:
                                cell.set_facecolor(MODERATE_COLOR)
                            else:
                                cell.set_facecolor(LOW_COLOR)
                        except ValueError: pass
                    
                    elif c_idx in [3, 6]:  # Improvement columns
                        imp_val_str = current_row_data[c_idx]
                        if '%' in imp_val_str and 'inf' not in imp_val_str:
                            try:
                                imp_val = float(imp_val_str.replace('%',''))
                                if imp_val > 50: cell.set_facecolor('#26de81') 
                                elif imp_val > 10: cell.set_facecolor('#c6efce') 
                                elif imp_val < -10: cell.set_facecolor('#ffc7ce')
                            except ValueError: pass
                    
                    elif c_idx in [7, 8]:  # Severity columns
                        severity = current_row_data[c_idx]
                        color_sev = SEVERE_COLOR if severity == "HIGH" else MODERATE_COLOR if severity == "MEDIUM" else LOW_COLOR
                        cell.set_facecolor(color_sev)
                        text_color = 'white' if severity != "MEDIUM" else 'black'
                        cell.set_text_props(color=text_color, fontweight='bold')
                        continue

                    cell.set_text_props(color=text_color)

        plt.title(f'{method_name} Adaptation - Feature Distribution Statistics', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Statistics table saved to: {save_path}")
        
        plt.close(stats_fig)

def histograms_visual_stats_table(X_source: np.ndarray, X_target: np.ndarray, 
                                 X_target_aligned: Optional[np.ndarray] = None,
                                 feature_names: Optional[List[str]] = None,
                                 save_path: Optional[str] = None, method_name: str = "MMD") -> None:
    """Generate visual statistics table with enhanced formatting"""
    histograms_stats_table(X_source, X_target, X_target_aligned, feature_names, save_path, method_name) 