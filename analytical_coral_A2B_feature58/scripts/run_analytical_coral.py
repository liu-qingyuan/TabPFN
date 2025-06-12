import sys
import os
import logging
import pandas as pd

# Add the project root to sys.path to enable absolute imports within the package
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

# Import only the modules that are actually used
from analytical_coral_A2B_feature58.config.settings import (
    SELECTED_FEATURES, CAT_IDX, TABPFN_PARAMS, 
    BASE_PATH, DATA_PATH_A, DATA_PATH_B, LABEL_COL, LOG_LEVEL
)
from analytical_coral_A2B_feature58.data.loader import load_excel
from analytical_coral_A2B_feature58.modeling.tabpfn_runner import run_coral_adaptation_experiment
from analytical_coral_A2B_feature58.utils.logging_setup import init_logging
from analytical_coral_A2B_feature58.visualization.histogram_plots import visualize_feature_histograms
from analytical_coral_A2B_feature58.visualization.comparison_plots import plot_experiment_comparison
from analytical_coral_A2B_feature58.visualization.tsne_plots import visualize_tsne

def main():
    # Initialize logging based on config
    init_logging(LOG_LEVEL)
    logging.info("Starting Analytical CORAL Experiment (A->B)")

    # BASE_PATH is defined in config.settings, e.g., 'results_analytical_coral_A2B_feature58'
    # Ensure this base directory for all results of this project exists.
    if not os.path.exists(BASE_PATH):
        try:
            os.makedirs(BASE_PATH)
            logging.info(f"Created base project results directory: {BASE_PATH}")
        except OSError as e:
            logging.error(f"Could not create base project results directory {BASE_PATH}: {e}")
            return # Cannot proceed if base results dir cannot be made

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

    experiment_tag = "Analytical_CORAL_A_to_B"
    # Specific path for this particular experiment's outputs, inside the main BASE_PATH
    current_experiment_path = os.path.join(BASE_PATH, experiment_tag)
    try:
        os.makedirs(current_experiment_path, exist_ok=True)
    except OSError as e:
        logging.error(f"Could not create experiment directory {current_experiment_path}: {e}")
        return

    try:
        results = run_coral_adaptation_experiment(
            X_source=X_source,
            y_source=y_source,
            X_target=X_target,
            y_target=y_target,
            cat_idx=CAT_IDX,
            model_name=experiment_tag, 
            tabpfn_params=TABPFN_PARAMS,
            base_path=current_experiment_path, # Pass the specific path for this run
            feature_names_for_plot=SELECTED_FEATURES,
            optimize_decision_threshold=True
        )

        logging.info("\n--- Analytical CORAL Experiment (A->B) Summary ---")
        logging.info(f"Source Validation Metrics: {results['source']}")
        logging.info(f"Direct Target Metrics: {results['direct']}")
        logging.info(f"Aligned Target Metrics: {results['target']}")
        logging.info(f"Times: {results['times']}")
        logging.info(f"Feature Stats: {results['features']}")

        summary_data = {
            'config': [experiment_tag],
            'source_val_acc': [results['source']['acc']],
            'source_val_auc': [results['source']['auc']],
            'direct_acc': [results['direct']['acc']],
            'direct_auc': [results['direct']['auc']],
            'direct_f1': [results['direct']['f1']],
            'direct_acc_0': [results['direct']['acc_0']],
            'direct_acc_1': [results['direct']['acc_1']],
            'target_acc': [results['target']['acc']],
            'target_auc': [results['target']['auc']],
            'target_f1': [results['target']['f1']],
            'target_acc_0': [results['target']['acc_0']],
            'target_acc_1': [results['target']['acc_1']],
            'optimal_threshold': [results['target'].get('optimal_threshold', 0.5)],
            'train_time': [results['times']['tabpfn']],
            'align_time': [results['times']['align']],
            'inference_time': [results['times']['inference']],
            'mean_diff_initial': [results['features']['mean_diff_before']],
            'std_diff_initial': [results['features']['std_diff_before']],
            'mean_diff_after_align': [results['features']['mean_diff_after']],
            'std_diff_after_align': [results['features']['std_diff_after']]
        }
        results_df = pd.DataFrame(summary_data)
        # Save summary CSV to the specific experiment path, not the overall base project path for clarity
        summary_csv_path = os.path.join(current_experiment_path, f"summary_{experiment_tag}.csv")
        results_df.to_csv(summary_csv_path, index=False)
        logging.info(f"Experiment summary saved to {summary_csv_path}")

        # 使用返回的对齐后数据进行可视化
        visualize_feature_histograms(
            X_source=results['data']['X_source_scaled'],
            X_target=results['data']['X_target_scaled'], 
            X_target_aligned=results['data']['X_target_aligned'],
            feature_names=SELECTED_FEATURES,
            save_path=os.path.join(current_experiment_path, f"{experiment_tag}_histograms.png"),
            title=f"Feature Distribution Before and After CORAL Alignment: {experiment_tag}"
        )

        # 生成对比图
        plot_experiment_comparison(
            results_df=results_df, # 使用之前创建的包含单行结果的DataFrame
            config_name=experiment_tag, # 使用实验标签作为配置名
            base_save_path=current_experiment_path # 结果保存在当前实验路径下
        )

        # --- t-SNE Visualization for Analytical CORAL ---
        logging.info(f"\n--- Generating t-SNE Visualization for {experiment_tag} ---")
        tsne_save_path = os.path.join(current_experiment_path, f"{experiment_tag}_tsne.png")
        
        # visualize_tsne expects unscaled data as it applies scaling internally
        # y_source and y_target are loaded at the beginning of main()
        visualize_tsne(
            X_source=X_source,  # Original unscaled source data
            X_target=X_target,  # Original unscaled target data
            y_source=y_source,
            y_target=y_target,
            X_target_aligned=results['data'].get('X_target_aligned'), # Aligned data from experiment results
            title=f"t-SNE: {experiment_tag} (Source vs Target vs Aligned Target)",
            save_path=tsne_save_path
        )
        # Logging of save path is handled inside visualize_tsne if save_path is provided

    except Exception as e:
        logging.error(f"Error during Analytical CORAL experiment: {e}", exc_info=True)

    logging.info("Analytical CORAL Experiment (A->B) Finished.")

if __name__ == "__main__":
    main() 