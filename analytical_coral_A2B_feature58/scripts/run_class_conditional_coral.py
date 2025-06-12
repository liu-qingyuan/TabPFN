import sys
import os

# Adjust sys.path similar to run_analytical_coral.py
# Assuming the script is in 'project_root/analytical_coral_A2B_feature58/scripts/'
# and 'project_root' needs to be in sys.path for 'analytical_coral_A2B_feature58.config...' imports
project_root_from_script = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root_from_script not in sys.path:
    sys.path.insert(0, project_root_from_script)

import logging
import pandas as pd
# import numpy as np # Removed as np.nan is used within the plotting function, not directly here

# Now import project-specific modules
from analytical_coral_A2B_feature58.config.settings import (
    SELECTED_FEATURES, CAT_IDX, TABPFN_PARAMS, 
    BASE_PATH, DATA_PATH_A, DATA_PATH_B, LABEL_COL, LOG_LEVEL
)
from analytical_coral_A2B_feature58.data.loader import load_excel
from analytical_coral_A2B_feature58.modeling.tabpfn_runner import run_class_conditional_coral_experiment
from analytical_coral_A2B_feature58.utils.logging_setup import init_logging
from analytical_coral_A2B_feature58.visualization.comparison_plots import plot_class_conditional_variants_comparison
from analytical_coral_A2B_feature58.visualization.tsne_plots import visualize_tsne

def main():
    init_logging(LOG_LEVEL)
    logging.info("Starting Class-Conditional CORAL Experiment (A->B)")

    # Ensure the base project results directory exists (BASE_PATH from settings)
    if not os.path.exists(BASE_PATH):
        try:
            os.makedirs(BASE_PATH)
            logging.info(f"Created base project results directory: {BASE_PATH}")
        except OSError as e:
            logging.error(f"Could not create base project results directory {BASE_PATH}: {e}")
            return

    # Load data once at the beginning
    try:
        logging.info(f"Loading source data (A) from: {DATA_PATH_A}")
        X_source, y_source = load_excel(DATA_PATH_A, SELECTED_FEATURES, LABEL_COL)
        
        logging.info(f"Loading target data (B) from: {DATA_PATH_B}")
        X_target, y_target = load_excel(DATA_PATH_B, SELECTED_FEATURES, LABEL_COL)
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return

    logging.info(f"Source data shape: X={X_source.shape}, y={y_source.shape}")
    logging.info(f"Target data shape: X={X_target.shape}, y={y_target.shape}")
    logging.info(f"Categorical feature indices: {CAT_IDX}")

    # --- Experiment 1: Class-Conditional CORAL with Pseudo Labels ---
    exp_tag_pseudo = "ClassCORAL_Pseudo_A_to_B"
    results_path_pseudo = os.path.join(BASE_PATH, exp_tag_pseudo) # Results in BASE_PATH/exp_tag
    try:
        os.makedirs(results_path_pseudo, exist_ok=True)
    except OSError as e:
        logging.error(f"Could not create experiment directory {results_path_pseudo}: {e}")
        # Decide if you want to return or just log and continue with other experiments
        # For now, let's assume if one experiment's dir fails, we might still try others.
        # However, if BASE_PATH fails, we return. This part can be an error.
        pass # Or return, depending on desired behavior for multi-experiment scripts

    results_pseudo = None # Initialize to ensure it exists for later checks
    try:
        logging.info(f"\n--- Running Experiment: {exp_tag_pseudo} ---")
        results_pseudo = run_class_conditional_coral_experiment(
            X_source=X_source, y_source=y_source, X_target=X_target, y_target=y_target,
            cat_idx=CAT_IDX, model_name=exp_tag_pseudo,
            tabpfn_params=TABPFN_PARAMS, base_path=results_path_pseudo,
            alpha=0.1, use_target_labels=False,
            feature_names_for_plot=SELECTED_FEATURES
        )
        df_pseudo = pd.DataFrame([{
            'config': exp_tag_pseudo,
            **results_pseudo['source'],
            **{f"direct_{k}": v for k,v in results_pseudo['direct'].items()},
            **{f"aligned_{k}": v for k,v in results_pseudo['target'].items()}
        }])
        df_pseudo.to_csv(os.path.join(results_path_pseudo, f"summary_{exp_tag_pseudo}.csv"), index=False)
        logging.info(f"Results for {exp_tag_pseudo} saved.")

        # --- t-SNE for Pseudo Labels C-CORAL ---
        if results_pseudo and 'data' in results_pseudo:
            logging.info(f"\n--- Generating t-SNE Visualization for {exp_tag_pseudo} ---")
            tsne_pseudo_save_path = os.path.join(results_path_pseudo, f"{exp_tag_pseudo}_tsne.png")
            visualize_tsne(
                X_source=X_source, y_source=y_source, 
                X_target=X_target, y_target=y_target,
                X_target_aligned=results_pseudo['data'].get('X_target_aligned'),
                title=f"t-SNE: {exp_tag_pseudo}",
                save_path=tsne_pseudo_save_path
            )
        else:
            logging.warning(f"Skipping t-SNE for {exp_tag_pseudo} due to missing data in results.")

    except Exception as e:
        logging.error(f"Error during {exp_tag_pseudo} experiment: {e}", exc_info=True)

    # --- Experiment 2: Class-Conditional CORAL with 10% True Target Labels ---
    exp_tag_labels = "ClassCORAL_WithLabels_A_to_B"
    results_path_labels = os.path.join(BASE_PATH, exp_tag_labels)
    try:
        os.makedirs(results_path_labels, exist_ok=True)
    except OSError as e:
        logging.error(f"Could not create experiment directory {results_path_labels}: {e}")
        pass # Or return

    results_labels = None # Initialize
    try:
        logging.info(f"\n--- Running Experiment: {exp_tag_labels} ---")
        results_labels = run_class_conditional_coral_experiment(
            X_source=X_source, y_source=y_source, X_target=X_target, y_target=y_target,
            cat_idx=CAT_IDX, model_name=exp_tag_labels,
            tabpfn_params=TABPFN_PARAMS, base_path=results_path_labels,
            alpha=0.1, use_target_labels=True, target_label_ratio=0.1,
            feature_names_for_plot=SELECTED_FEATURES
        )
        df_labels = pd.DataFrame([{
            'config': exp_tag_labels,
            **results_labels['source'],
            **{f"direct_{k}": v for k,v in results_labels['direct'].items()},
            **{f"aligned_{k}": v for k,v in results_labels['target'].items()}
        }])
        df_labels.to_csv(os.path.join(results_path_labels, f"summary_{exp_tag_labels}.csv"), index=False)
        logging.info(f"Results for {exp_tag_labels} saved.")

        # --- t-SNE for True Labels C-CORAL ---
        if results_labels and 'data' in results_labels:
            logging.info(f"\n--- Generating t-SNE Visualization for {exp_tag_labels} ---")
            tsne_labels_save_path = os.path.join(results_path_labels, f"{exp_tag_labels}_tsne.png")
            visualize_tsne(
                X_source=X_source, y_source=y_source, 
                X_target=X_target, y_target=y_target,
                X_target_aligned=results_labels['data'].get('X_target_aligned'),
                title=f"t-SNE: {exp_tag_labels}",
                save_path=tsne_labels_save_path
            )
        else:
            logging.warning(f"Skipping t-SNE for {exp_tag_labels} due to missing data in results.")

    except Exception as e:
        logging.error(f"Error during {exp_tag_labels} experiment: {e}", exc_info=True)

    # --- Combined Summary and Comparison Plot (if both experiments ran) ---
    if results_pseudo and results_labels:
        logging.info("\n--- Generating Combined Comparison Plot for Class-Conditional CORAL Methods ---")
        
        # Prepare data for the new plotting function
        # Direct metrics (using pseudo experiment's direct results as baseline)
        direct_metrics_data = results_pseudo['direct']
        direct_metrics_series = pd.Series(direct_metrics_data)

        # Pseudo-label C-CORAL metrics
        pseudo_label_metrics_data = results_pseudo['target']
        pseudo_label_metrics_series = pd.Series(pseudo_label_metrics_data)

        # True-label C-CORAL metrics
        true_label_metrics_data = results_labels['target']
        true_label_metrics_series = pd.Series(true_label_metrics_data)
        
        plot_config_name = "A_to_B_ClassCORAL_Variants"
        # The plot function itself handles joining base_save_path and config_name to create filename.

        plot_class_conditional_variants_comparison(
            direct_metrics=direct_metrics_series,
            pseudo_label_metrics=pseudo_label_metrics_series,
            true_label_metrics=true_label_metrics_series,
            config_name=plot_config_name,
            base_save_path=BASE_PATH, # Plot will be saved in the main results directory for this project
            target_name_plot="Dataset B (Henan)"
        )
        # Logging of save path is now handled inside the plot function
    else:
        logging.warning("One or both Class-Conditional CORAL experiments failed. Skipping combined comparison plot.")

    logging.info("Class-Conditional CORAL Experiment (A->B) Finished.")

if __name__ == "__main__":
    main() 