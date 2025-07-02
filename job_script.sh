#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --gres=gpu:4g.40gb:1  # 申请1个GPU
#SBATCH --output=output.log    # 任务输出日志
#SBATCH --error=error.log      # 错误日志

# python predict_healthcare_class.py

# python predict_healthcare_class_kmeans_smote_fscore.py
# python predict_healthcare_class_kmeans_smote.py
# python feature_number_fscore_analysis_B.py
# python feature_number_RFE_analysis_B.py

# python tests/test_dataset_b_tabpfn_cv10_comparison.py



cd uda_medical_imbalance_project 


# python tests/debug_sa_auc.py
# python tests/test_sa_fix.py

# python tests/test_kliep_medical_bias_correction.py
# python tests/test_kmm_proper_tuning.py
# python tests/test_adapt_methods.py
# python tests/test_tca_gamma_optimization.py
# python tests/test_cross_validation.py
# python tests/test_paper_methods_datasets_AB.py
# python tests/test_baseline_models.py
# python tests/test_paper_methods.py
# python tests/debug_feature_selection.py
# python tests/test_uda_visualization.py
# python examples/real_data_visualization.py
python scripts/run_complete_analysis.py

# python debug_calibration.py



# cd /home/24052432g/TabPFN/analytical_mmd_A2B_feature58


# python scripts/run_analytical_mmd.py --compare-all
# python scripts/run_analytical_mmd.py --compare-all --use-threshold-optimizer
# python scripts/run_analytical_mmd.py --compare-all --use-class-conditional --use-threshold-optimizer


# test
# python tests/test_statistics_consistency.py
# python tests/test_multi_model_integration.py
# python tests/test_performance_plots.py
# python tests/test_data_split_strategies.py --test two-way
# python tests/test_data_split_strategies.py --test three-way 
# python tests/test_data_split_strategies.py --test bayesian
# python tests/test_format_error_debug.py
# python tests/test_data_distribution_analysis.py --feature-type best7
# python scripts/fix_max_time_issue.py

# python predict_healthcare_auto_externals_best7_ABCtest.py
# python scripts/run_analytical_mmd.py --mode cross-domain --model-type auto --skip-cv-on-a --method linear --feature-type best7 
# python scripts/run_analytical_mmd.py --model-type auto --method linear --feature-type best7 --skip-cv-on-a --evaluation-mode proper_cv
# python scripts/run_analytical_mmd.py --model-type auto --method linear --feature-type best7 --data-split-strategy three-way --validation-split 0.7 --use-bayesian-optimization --bo-n-calls 100 --output-dir ./results_bayesian_optimization_real_experiment

# python scripts/run_analytical_mmd.py --model-type auto --target-domain C --feature-type best7 --data-split-strategy three-way --validation-split 0.7  --use-bayesian-optimization --bo-n-calls 20 --output-dir ./results_bayesian_optimization_real_experiment_C
# python scripts/run_analytical_mmd.py --model-type auto --method linear --target-domain C


# python scripts/run_bayesian_mmd_optimization.py --model-type auto --mmd-method linear --n-calls 10 --target-domain B
# python scripts/run_bayesian_mmd_optimization.py --model-type auto --n-calls 100 --target-domain B --evaluate-source-cv

# python scripts/run_standard_domain_adaptation.py --help
# python scripts/run_standard_domain_adaptation.py \
#     --model-type auto \
#     --feature-type best7 \
#     --mmd-method linear \
#     --no-mmd-tuning \
#     --target-domain B \
#     --cv-folds 5 \
#     --source-val-split 0.2 \
#     --n-calls 50 \
#     --random-state 42 \
#     --output-dir ./results_standard_domain_adaptation_best7_linear_B_no_mmd_tuning_$(date +%Y%m%d_%H%M%S)


# python scripts/run_fixed_params_domain_adaptation.py \
#     --feature-type best7 \
#     --domain-adapt-method adaptive_coral \
#     --target-domain B \
#     --output-dir ./results_fixed_params_domain_adaptation_best7_adaptive_coral_B_$(date +%Y%m%d_%H%M%S)


# python scripts/run_fixed_params_domain_adaptation.py \
#     --feature-type best7 \
#     --domain-adapt-method mean_variance \
#     --target-domain B \
#     --output-dir ./results_fixed_params_domain_adaptation_best7_mean_variance_B_$(date +%Y%m%d_%H%M%S)

# python scripts/run_fixed_params_domain_adaptation.py \
#     --feature-type best7 \
#     --domain-adapt-method adaptive_mean_variance \
#     --target-domain B \
#     --output-dir ./results_fixed_params_domain_adaptation_best7_adaptive_mean_variance_B_$(date +%Y%m%d_%H%M%S)

# python scripts/run_fixed_params_domain_adaptation.py \
#     --feature-type best7 \
#     --domain-adapt-method tca \
#     --target-domain B \
#     --output-dir ./results_fixed_params_domain_adaptation_best7_tca_B_$(date +%Y%m%d_%H%M%S)

# python scripts/run_fixed_params_domain_adaptation.py \
#     --feature-type best7 \
#     --domain-adapt-method jda \
#     --target-domain B \
#     --output-dir ./results_fixed_params_domain_adaptation_best7_jda_B_$(date +%Y%m%d_%H%M%S)

# python scripts/run_fixed_params_domain_adaptation.py \
#     --feature-type best7 \
#     --domain-adapt-method adaptive_jda \
#     --target-domain B \
#     --output-dir ./results_fixed_params_domain_adaptation_best7_adaptive_jda_B_$(date +%Y%m%d_%H%M%S)

# python scripts/run_fixed_params_domain_adaptation.py \
#     --feature-type best7 \
#     --domain-adapt-method adaptive_tca \
#     --target-domain B \
#     --output-dir ./results_fixed_params_domain_adaptation_best7_adaptive_tca_B_$(date +%Y%m%d_%H%M%S)

# python scripts/run_fixed_params_domain_adaptation.py --help
# python scripts/run_fixed_params_domain_adaptation.py --target-domain B --feature-type best10 --domain-adapt-method coral --source-cv-folds 0 --source-cv-folds 10 --output-dir ./results_fixed_params_domain_adaptation_basetabpfn_best10_coral_B_cv-folds_10_$(date +%Y%m%d_%H%M%S)

# python scripts/run_fixed_params_domain_adaptation.py --target-domain B --feature-type best --domain-adapt-method coral --source-cv-folds 0 --source-cv-folds 10 --output-dir ./results_fixed_params_domain_adaptation_basetabpfn_best7_coral_B_cv-folds_10_$(date +%Y%m%d_%H%M%S)


# python scripts/run_fixed_params_domain_adaptation.py --target-domain B --feature-type best10 --domain-adapt-method jda --source-val-split 0 --source-cv-folds 10 --output-dir ./results_fixed_params_domain_adaptation_basetabpfn_best10_jda_B_cv-folds_10_$(date +%Y%m%d_%H%M%S)


# python scripts/run_fixed_params_domain_adaptation.py \
#     --feature-type best7 \
#     --target-domain B \
#     --include-baselines \
#     --domain-adapt-method coral \
#     --source-cv-folds 10 \
#     --output-dir ./results_fixed_params_domain_adaptation_best7_source-cv-folds_10_coral_B_$(date +%Y%m%d_%H%M%S)

# python scripts/run_bayesian_optimization_domain_adaptation.py


# 带时间戳后缀的命令（推荐使用）
# python scripts/run_analytical_mmd.py --mode cross-domain --model-type auto --method linear --feature-type best7 --output-dir "./results_cross_domain_auto_linear_best7_$(date +%Y%m%d_%H%M%S)"

# python predict_healthcare_auto_and_otherbaselines_A2C_original_features23_analytical_CORAL.py

# python scripts/test_optimized_algorithms.py

# python scripts/test_baseline_models.py
# python predict_healthcare_auto_externals.py

# python predict_healthcare_AutoPostHocEnsemblePredictor_external_best_7_features.py


