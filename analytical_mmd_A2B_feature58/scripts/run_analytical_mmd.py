#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analytical MMD Domain Adaptation Experiment (A to B - 58 Features)

这个脚本实现了基于MMD的域适应方法，用于医疗数据的跨域预测任务。
支持三种MMD方法：线性变换、核PCA和均值标准差对齐。
现在支持多种TabPFN模型类型：AutoTabPFN、原生TabPFN、rfTabPFN。
"""

import os
import sys
import argparse
import logging
import time
import warnings
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix

# 过滤sklearn的类别特征警告
warnings.filterwarnings("ignore", message="Found unknown categories.*during transform", category=UserWarning)

# 获取脚本所在目录的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 计算 TabPFN 项目的根目录 (包含 analytical_mmd_A2B_feature58 的父目录)
tabpfn_root = os.path.dirname(os.path.dirname(script_dir))

# 将 TabPFN 项目根目录添加到 Python 路径
sys.path.insert(0, tabpfn_root)

try:
    # 使用完整的包路径导入模块
    from analytical_mmd_A2B_feature58.config.settings import (
        SELECTED_FEATURES, BEST_7_FEATURES, CAT_IDX, BEST_7_CAT_IDX, 
        MODEL_CONFIGS, MMD_METHODS, BASE_PATH, DATA_PATHS, LABEL_COL,
        VISUALIZATION_CONFIG, get_features_by_type, get_categorical_indices, get_model_config
    )
    from analytical_mmd_A2B_feature58.data.loader import load_all_datasets
    from analytical_mmd_A2B_feature58.preprocessing.scaler import fit_apply_scaler
    from analytical_mmd_A2B_feature58.preprocessing.mmd import mmd_transform
    from analytical_mmd_A2B_feature58.preprocessing.class_conditional_mmd import (
        class_conditional_mmd_transform, run_class_conditional_mmd_experiment
    )
    from analytical_mmd_A2B_feature58.preprocessing.threshold_optimizer import (
        apply_threshold_optimization, get_threshold_optimization_suffix, get_roc_curve_data
    )
    from analytical_mmd_A2B_feature58.utils.logging_setup import setup_logger
    # 导入可视化模块
    from analytical_mmd_A2B_feature58.visualization import (
        compare_before_after_adaptation, plot_mmd_methods_comparison,
        close_figures, setup_matplotlib_style
    )
    # 导入统一的度量模块
    from analytical_mmd_A2B_feature58.metrics.discrepancy import (
        calculate_kl_divergence, calculate_wasserstein_distances, compute_mmd_kernel
    )
    # 导入模型选择器和跨域实验运行器
    from analytical_mmd_A2B_feature58.modeling.model_selector import (
        get_model, get_available_models, validate_model_params, get_model_preset
    )
    from analytical_mmd_A2B_feature58.modeling.cross_domain_runner import (
        CrossDomainExperimentRunner, run_cross_domain_experiment
    )
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保您在正确的目录中运行此脚本，并且所有依赖项都已安装。")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"脚本目录: {script_dir}")
    print(f"TabPFN根目录: {tabpfn_root}")
    sys.exit(1)

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='运行Analytical MMD域适应实验 (A to B)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本用法 - AutoTabPFN + Linear MMD (二分法，默认目标域B)
  python scripts/run_analytical_mmd.py --model-type auto --method linear
  
  # 选择C作为目标域
  python scripts/run_analytical_mmd.py --model-type auto --method linear --target-domain C
  
  # 跳过数据集A的交叉验证，直接进行A→B域适应实验
  python scripts/run_analytical_mmd.py --model-type auto --method linear --skip-cv-on-a
  
  # A→C域适应实验
  python scripts/run_analytical_mmd.py --model-type auto --method linear --target-domain C --skip-cv-on-a
  
  # 使用三分法数据划分（目标域B）
  python scripts/run_analytical_mmd.py --model-type auto --method linear --data-split-strategy three-way
  
  # 使用三分法数据划分（目标域C）
  python scripts/run_analytical_mmd.py --model-type auto --method linear --data-split-strategy three-way --target-domain C
  
  # 三分法 + 贝叶斯优化（目标域B）
  python scripts/run_analytical_mmd.py --model-type auto --method linear --data-split-strategy three-way --use-bayesian-optimization
  
  # 三分法 + 贝叶斯优化（目标域C）
  python scripts/run_analytical_mmd.py --model-type auto --method linear --data-split-strategy three-way --use-bayesian-optimization --target-domain C
  
  # 自定义验证集比例和贝叶斯优化参数（目标域C）
  python scripts/run_analytical_mmd.py --model-type auto --method linear --data-split-strategy three-way --validation-split 0.7 --use-bayesian-optimization --target-domain C --bo-n-calls 100
  
  # 使用最佳7特征 + 三分法 + 贝叶斯优化（目标域B）
  python scripts/run_analytical_mmd.py --model-type auto --method linear --feature-type best7 --data-split-strategy three-way --use-bayesian-optimization
  
  # 类条件MMD + 三分法（目标域C）
  python scripts/run_analytical_mmd.py --model-type auto --method linear --use-class-conditional --data-split-strategy three-way --target-domain C
  
  # 完整组合（三分法 + 贝叶斯优化 + 类条件MMD + 目标域C）
  python scripts/run_analytical_mmd.py --model-type auto --method linear --feature-type best7 --data-split-strategy three-way --use-bayesian-optimization --use-class-conditional --skip-cv-on-a --target-domain C
  
  # 比较所有方法（使用三分法，目标域B）
  python scripts/run_analytical_mmd.py --compare-all --model-type auto --data-split-strategy three-way
  
  # 比较所有方法（使用三分法，目标域C）
  python scripts/run_analytical_mmd.py --compare-all --model-type auto --data-split-strategy three-way --target-domain C
        """
    )
    
    # 模型相关参数
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['auto', 'tuned', 'base', 'rf'],
        default='auto',
        help='模型类型: auto(AutoTabPFN), tuned(TunedTabPFN), base(原生TabPFN), rf(RF风格TabPFN集成)'
    )
    
    parser.add_argument(
        '--model-preset',
        type=str,
        choices=['fast', 'balanced', 'accurate'],
        help='模型预设配置'
    )
    
    # 特征相关参数
    parser.add_argument(
        '--feature-type',
        type=str,
        choices=['all', 'best7'],
        default='all',
        help='特征类型: all(58个特征) 或 best7(最佳7特征)'
    )
    
    # MMD方法参数
    parser.add_argument(
        '--method',
        type=str,
        choices=['linear', 'kpca', 'mean_std'],
        default='linear',
        help='MMD对齐方法 (默认: linear)'
    )
    
    parser.add_argument(
        '--compare-all',
        action='store_true',
        help='比较所有MMD方法'
    )
    
    # MMD参数
    parser.add_argument('--gamma', type=float, help='RBF核参数 (默认: 1.0)')
    parser.add_argument('--lr', type=float, help='学习率 (仅用于linear方法, 默认: 0.01)')
    parser.add_argument('--n-epochs', type=int, help='训练轮数 (仅用于linear方法, 默认: 300)')
    parser.add_argument('--batch-size', type=int, help='批大小 (仅用于linear方法, 默认: 64)')
    parser.add_argument('--lambda-reg', type=float, help='正则化参数 (仅用于linear方法, 默认: 1e-3)')
    parser.add_argument('--n-components', type=int, help='PCA组件数 (仅用于kpca方法)')
    
    # 改进的MMD参数
    parser.add_argument('--no-staged-training', action='store_true', help='禁用分阶段训练 (仅用于linear方法)')
    parser.add_argument('--no-dynamic-gamma', action='store_true', help='禁用动态gamma搜索 (仅用于linear方法)')
    parser.add_argument('--gamma-search-values', type=str, help='gamma搜索值列表，用逗号分隔 (例: 0.1,0.5,1.0,2.0)')
    parser.add_argument('--use-preset', type=str, choices=['conservative', 'aggressive', 'traditional'], 
                       help='使用预设配置 (conservative/aggressive/traditional)')
    
    # Linear方法调试参数
    parser.add_argument(
        '--standardize-features', action='store_true',
        help='对 linear 方法的连续特征启用输入标准化'
    )
    parser.add_argument(
        '--use-gradient-clipping', action='store_true',
        help='对 linear 方法启用梯度裁剪'
    )
    parser.add_argument(
        '--max-grad-norm', type=float, default=1.0,
        help='梯度裁剪时的最大范数 (仅在 --use-gradient-clipping 下生效)'
    )
    parser.add_argument(
        '--monitor-gradients', action='store_true',
        help='在训练时输出梯度范数，以便观察是否发生爆炸'
    )
    
    # 核PCA改进参数
    parser.add_argument('--use-inverse-transform', action='store_true', help='启用核PCA逆变换 (不推荐)')
    parser.add_argument('--no-standardize', action='store_true', help='禁用核PCA前的标准化')
    
    # 模型特定参数
    parser.add_argument('--max-time', type=int, help='AutoTabPFN最大训练时间(秒)')
    parser.add_argument('--max-models', type=int, help='AutoTabPFN最大模型数')
    parser.add_argument('--n-estimators', type=int, help='RF模型的估计器数量')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], help='计算设备')
    
    # 输出参数
    parser.add_argument('--output-dir', type=str, help='结果输出目录')
    parser.add_argument('--no-visualizations', action='store_true', help='不生成可视化结果')
    parser.add_argument('--log-file', type=str, help='日志文件路径')
    
    # 实验参数
    parser.add_argument('--use-class-conditional', action='store_true', help='启用类条件MMD方法')
    parser.add_argument('--use-target-labels', action='store_true', help='使用部分目标域真实标签（仅在类条件模式下有效）')
    parser.add_argument('--target-label-ratio', type=float, default=0.1, help='目标域标签使用比例（默认0.1）')
    parser.add_argument('--use-threshold-optimizer', action='store_true', help='启用阈值优化')
    parser.add_argument('--skip-cv-on-a', action='store_true', help='跳过在数据集A上的10折交叉验证，直接进行A→B域适应实验')
    parser.add_argument('--evaluation-mode', type=str, choices=['single', 'cv', 'proper_cv'], default='cv', 
                       help='外部评估模式: single(单次评估，与原始脚本一致), cv(10折CV评估，每折重新训练), proper_cv(正确的10折CV，一个模型多折评估)')
    parser.add_argument('--random-seed', type=int, default=42, help='随机种子')
    
    # 数据划分策略参数
    parser.add_argument('--data-split-strategy', type=str, choices=['two-way', 'three-way'], default='two-way',
                       help='B域数据划分策略: two-way(二分法，完整B域用于测试), three-way(三分法，B域划分为验证集和测试集)')
    parser.add_argument('--validation-split', type=float, default=0.7,
                       help='三分法时验证集比例 (默认: 0.7，即70%%用于验证，30%%用于holdout测试)')
    
    # 目标域选择参数
    parser.add_argument('--target-domain', type=str, choices=['B', 'C'], default='B',
                       help='选择目标域: B(河南癌症医院) 或 C(广州医科大学) (默认: B)')
    
    # 贝叶斯优化参数
    parser.add_argument('--use-bayesian-optimization', action='store_true',
                       help='使用贝叶斯优化进行超参数调优 (仅在three-way模式下可用)')
    parser.add_argument('--bo-n-calls', type=int, default=50,
                       help='贝叶斯优化迭代次数 (默认: 50)')
    parser.add_argument('--bo-random-state', type=int, default=42,
                       help='贝叶斯优化随机种子 (默认: 42)')
    
    # 贝叶斯MMD优化参数
    parser.add_argument('--use-bayesian-mmd-optimization', action='store_true',
                       help='使用贝叶斯优化同时优化模型参数和MMD参数 (仅在three-way模式下可用)')
    parser.add_argument('--auto-run-mmd-after-bo', action='store_true',
                       help='贝叶斯优化完成后自动运行完整的MMD域适应实验')
    
    return parser.parse_args()

def setup_experiment_logging(log_file: Optional[str] = None):
    """设置实验日志"""
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/mmd_experiment_{timestamp}.log"
    
    # 确保日志目录存在
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    logger = setup_logger(
        log_level=logging.INFO,
        log_file=log_file,
        console_output=True
    )
    
    return logger

def validate_data_paths():
    """验证数据文件路径"""
    for dataset, path in DATA_PATHS.items():
        full_path = os.path.join(tabpfn_root, path)
        if not os.path.exists(full_path):
            logging.warning(f"数据文件不存在: {full_path}")
            return False
    return True

def prepare_model_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    """准备模型参数"""
    model_kwargs = {}
    
    # 获取基础配置
    if args.model_preset:
        base_config = get_model_config(args.model_type, args.model_preset)
    else:
        base_config = get_model_config(args.model_type)
    
    model_kwargs.update(base_config)
    
    # 覆盖特定参数
    if args.max_time is not None:
        model_kwargs['max_time'] = args.max_time
    if args.max_models is not None:
        model_kwargs['max_models'] = args.max_models
    if args.n_estimators is not None:
        model_kwargs['n_estimators'] = args.n_estimators
    if args.device is not None:
        model_kwargs['device'] = args.device
    
    # 验证参数
    model_kwargs = validate_model_params(args.model_type, model_kwargs)
    
    return model_kwargs

def prepare_mmd_kwargs(args: argparse.Namespace, method: str) -> Dict[str, Any]:
    """准备MMD参数"""
    mmd_kwargs = MMD_METHODS.get(method, {}).copy()
    
    # 通用参数
    if args.gamma is not None:
        mmd_kwargs['gamma'] = args.gamma
    
    # Linear方法特定参数
    if method == 'linear':
        if args.lr is not None:
            mmd_kwargs['lr'] = args.lr
        if args.n_epochs is not None:
            mmd_kwargs['n_epochs'] = args.n_epochs
        if args.batch_size is not None:
            mmd_kwargs['batch_size'] = args.batch_size
        if args.lambda_reg is not None:
            mmd_kwargs['lambda_reg'] = args.lambda_reg
        
        # 改进参数
        if args.no_staged_training:
            mmd_kwargs['staged_training'] = False
        if args.no_dynamic_gamma:
            mmd_kwargs['dynamic_gamma'] = False
        if args.gamma_search_values:
            values = [float(x.strip()) for x in args.gamma_search_values.split(',')]
            mmd_kwargs['gamma_search_values'] = values
        if args.standardize_features:
            mmd_kwargs['standardize_features'] = True
        if args.use_gradient_clipping:
            mmd_kwargs['use_gradient_clipping'] = True
            mmd_kwargs['max_grad_norm'] = args.max_grad_norm
        if args.monitor_gradients:
            mmd_kwargs['monitor_gradients'] = True
        
        # 预设配置
        if args.use_preset == 'conservative':
            mmd_kwargs.update({
                'lr': 1e-4,
                'lambda_reg': 1e-2,
                'staged_training': True,
                'dynamic_gamma': False,
                'standardize_features': True,
                'use_gradient_clipping': True
            })
        elif args.use_preset == 'aggressive':
            mmd_kwargs.update({
                'lr': 1e-3,
                'lambda_reg': 1e-4,
                'n_epochs': 500,
                'staged_training': True,
                'dynamic_gamma': True,
                'standardize_features': True
            })
        elif args.use_preset == 'traditional':
            mmd_kwargs.update({
                'staged_training': False,
                'dynamic_gamma': False,
                'standardize_features': False,
                'use_gradient_clipping': False
            })
    
    # KPCA方法特定参数
    elif method == 'kpca':
        if args.n_components is not None:
            mmd_kwargs['n_components'] = args.n_components
        if args.use_inverse_transform:
            mmd_kwargs['use_inverse_transform'] = True
        if args.no_standardize:
            mmd_kwargs['standardize'] = False
    
    return mmd_kwargs

def generate_experiment_suffix(args: argparse.Namespace) -> str:
    """生成实验后缀"""
    suffix_parts = []
    
    if args.use_class_conditional:
        suffix_parts.append("class_conditional")
    
    if args.use_threshold_optimizer:
        suffix_parts.append("threshold_optimized")
    
    if args.data_split_strategy == 'three-way':
        suffix_parts.append(f"three-way_val{int(args.validation_split*100)}")
    
    if args.target_domain != 'B':
        suffix_parts.append(f"target_{args.target_domain}")
    
    return "_" + "_".join(suffix_parts) if suffix_parts else ""

def generate_model_name(args: argparse.Namespace) -> str:
    """生成模型名称"""
    base_name = f"{args.model_type.upper()}TabPFN-MMD-{args.method.upper()}"
    
    if args.use_class_conditional:
        base_name = f"{args.model_type.upper()}TabPFN-ClassMMD-{args.method.upper()}"
    
    return base_name

def run_cross_domain_experiment_mode(args: argparse.Namespace, logger: logging.Logger):
    """运行跨域实验

    参数:
        args: 命令行参数
        logger: 日志记录器
    
    主要功能:
    1. 验证数据路径
    2. 准备模型和MMD参数
    3. 生成实验保存路径
    4. 执行跨域实验
    5. 输出实验结果
    """
    logger.info("运行跨域实验模式...")
    
    # 验证数据路径
    if not validate_data_paths():
        logger.error("数据文件验证失败")
        return
    
    # 准备参数
    model_kwargs = prepare_model_kwargs(args)
    mmd_kwargs = prepare_mmd_kwargs(args, args.method)
    
    # 添加数据划分策略参数
    experiment_kwargs = {
        'data_split_strategy': args.data_split_strategy,
        'validation_split': args.validation_split if args.data_split_strategy == 'three-way' else None
    }
    
    # 生成保存路径
    if args.output_dir:
        save_path = args.output_dir
    else:
        suffix = generate_experiment_suffix(args)
        # 在保存路径中包含数据划分策略信息
        strategy_suffix = f"_{args.data_split_strategy}"
        if args.data_split_strategy == 'three-way':
            strategy_suffix += f"_val{int(args.validation_split*100)}"
        save_path = f"./results_cross_domain_{args.model_type}_{args.method}_{args.feature_type}{strategy_suffix}{suffix}"
    
    logger.info(f"结果将保存到: {save_path}")
    
    # 运行跨域实验
    try:
        results = run_cross_domain_experiment(
            model_type=args.model_type,
            feature_type=args.feature_type,
            mmd_method=args.method,
            use_class_conditional=args.use_class_conditional,
            use_threshold_optimizer=args.use_threshold_optimizer,
            save_path=save_path,
            skip_cv_on_a=args.skip_cv_on_a,
            evaluation_mode=args.evaluation_mode,
            data_split_strategy=args.data_split_strategy,
            validation_split=args.validation_split,
            target_domain=args.target_domain,
            save_visualizations=not args.no_visualizations,
            **{**model_kwargs, **mmd_kwargs}
        )
        
        logger.info("跨域实验完成!")
        logger.info("主要结果:")
        
        # 打印主要结果
        if 'cross_validation_A' in results and results['cross_validation_A'] is not None:
            cv_results = results['cross_validation_A']
            logger.info(f"数据集A交叉验证 - 准确率: {cv_results['accuracy']}")
            logger.info(f"数据集A交叉验证 - AUC: {cv_results['auc']}")
        else:
            logger.info("数据集A交叉验证: 已跳过")
        
        if 'external_validation_B' in results:
            b_results = results['external_validation_B']
            if 'without_domain_adaptation' in b_results:
                b_no_da = b_results['without_domain_adaptation']
                if 'means' in b_no_da:
                    logger.info(f"数据集B外部验证(无域适应) - 准确率: {b_no_da['means']['accuracy']:.4f}")
                    logger.info(f"数据集B外部验证(无域适应) - AUC: {b_no_da['means']['auc']:.4f}")
                else:
                    # 检查值的类型，如果已经是字符串就直接使用
                    acc_val = b_no_da['accuracy']
                    auc_val = b_no_da['auc']
                    if isinstance(acc_val, str):
                        logger.info(f"数据集B外部验证(无域适应) - 准确率: {acc_val}")
                    else:
                        logger.info(f"数据集B外部验证(无域适应) - 准确率: {acc_val:.4f}")
                    if isinstance(auc_val, str):
                        logger.info(f"数据集B外部验证(无域适应) - AUC: {auc_val}")
                    else:
                        logger.info(f"数据集B外部验证(无域适应) - AUC: {auc_val:.4f}")
            if 'with_domain_adaptation' in b_results:
                b_with_da = b_results['with_domain_adaptation']
                if 'means' in b_with_da:
                    logger.info(f"数据集B外部验证(有域适应) - 准确率: {b_with_da['means']['accuracy']:.4f}")
                    logger.info(f"数据集B外部验证(有域适应) - AUC: {b_with_da['means']['auc']:.4f}")
                else:
                    # 检查值的类型，如果已经是字符串就直接使用
                    acc_val = b_with_da['accuracy']
                    auc_val = b_with_da['auc']
                    if isinstance(acc_val, str):
                        logger.info(f"数据集B外部验证(有域适应) - 准确率: {acc_val}")
                    else:
                        logger.info(f"数据集B外部验证(有域适应) - 准确率: {acc_val:.4f}")
                    if isinstance(auc_val, str):
                        logger.info(f"数据集B外部验证(有域适应) - AUC: {auc_val}")
                    else:
                        logger.info(f"数据集B外部验证(有域适应) - AUC: {auc_val:.4f}")
        
        if 'external_validation_C' in results:
            c_results = results['external_validation_C']
            if 'without_domain_adaptation' in c_results:
                c_no_da = c_results['without_domain_adaptation']
                if 'means' in c_no_da:
                    logger.info(f"数据集C外部验证(无域适应) - 准确率: {c_no_da['means']['accuracy']:.4f}")
                    logger.info(f"数据集C外部验证(无域适应) - AUC: {c_no_da['means']['auc']:.4f}")
                else:
                    # 检查值的类型，如果已经是字符串就直接使用
                    acc_val = c_no_da['accuracy']
                    auc_val = c_no_da['auc']
                    if isinstance(acc_val, str):
                        logger.info(f"数据集C外部验证(无域适应) - 准确率: {acc_val}")
                    else:
                        logger.info(f"数据集C外部验证(无域适应) - 准确率: {acc_val:.4f}")
                    if isinstance(auc_val, str):
                        logger.info(f"数据集C外部验证(无域适应) - AUC: {auc_val}")
                    else:
                        logger.info(f"数据集C外部验证(无域适应) - AUC: {auc_val:.4f}")
            if 'with_domain_adaptation' in c_results:
                c_with_da = c_results['with_domain_adaptation']
                if 'means' in c_with_da:
                    logger.info(f"数据集C外部验证(有域适应) - 准确率: {c_with_da['means']['accuracy']:.4f}")
                    logger.info(f"数据集C外部验证(有域适应) - AUC: {c_with_da['means']['auc']:.4f}")
                else:
                    # 检查值的类型，如果已经是字符串就直接使用
                    acc_val = c_with_da['accuracy']
                    auc_val = c_with_da['auc']
                    if isinstance(acc_val, str):
                        logger.info(f"数据集C外部验证(有域适应) - 准确率: {acc_val}")
                    else:
                        logger.info(f"数据集C外部验证(有域适应) - 准确率: {acc_val:.4f}")
                    if isinstance(auc_val, str):
                        logger.info(f"数据集C外部验证(有域适应) - AUC: {auc_val}")
                    else:
                        logger.info(f"数据集C外部验证(有域适应) - AUC: {auc_val:.4f}")
        
    except Exception as e:
        logger.error(f"跨域实验失败: {e}")
        raise

def run_comparison_experiment(args: argparse.Namespace, logger: logging.Logger):
    """运行方法比较实验"""
    logger.info("运行方法比较实验...")
    
    methods = ['linear', 'kpca', 'mean_std']
    all_results = {}
    
    for method in methods:
        logger.info(f"运行{method}方法...")
        
        # 更新参数
        args.method = method
        
        try:
            # 直接运行跨域实验
            run_cross_domain_experiment_mode(args, logger)
            logger.info(f"{method}方法完成")
            
        except Exception as e:
            logger.error(f"{method}方法失败: {e}")
            continue
    
    logger.info("所有方法比较完成!")

def run_bayesian_optimization_mode(args: argparse.Namespace, logger: logging.Logger):
    """运行贝叶斯优化模式"""
    logger.info("运行贝叶斯优化模式...")
    
    # 导入贝叶斯优化模块
    try:
        from analytical_mmd_A2B_feature58.modeling.bayesian_optimizer import run_bayesian_optimization
    except ImportError as e:
        logger.error(f"无法导入贝叶斯优化模块: {e}")
        logger.error("请确保 modeling/bayesian_optimizer.py 文件存在")
        return
    
    # 验证数据路径
    if not validate_data_paths():
        logger.error("数据文件验证失败")
        return
    
    # 准备贝叶斯优化参数
    bo_kwargs = {
        'model_type': args.model_type,
        'feature_type': args.feature_type,
        'validation_split': args.validation_split,
        'n_calls': args.bo_n_calls,
        'random_state': args.bo_random_state,
        'use_categorical': not getattr(args, 'no_categorical', False),
        'target_domain': args.target_domain
    }
    
    # 生成保存路径
    if args.output_dir:
        save_path = args.output_dir
    else:
        suffix = generate_experiment_suffix(args)
        save_path = f"./results_bayesian_optimization_real_experiment_{args.model_type}_{args.feature_type}_{args.target_domain}{suffix}"
    
    logger.info(f"贝叶斯优化结果将保存到: {save_path}")
    
    try:
        # 运行贝叶斯优化
        bo_results = run_bayesian_optimization(
            save_path=save_path,
            **bo_kwargs
        )
        
        logger.info("贝叶斯优化完成!")
        logger.info("优化结果:")
        
        # 打印优化结果
        if 'optimization_results' in bo_results:
            opt_results = bo_results['optimization_results']
            logger.info(f"最佳验证集AUC: {opt_results['best_validation_auc']:.4f}")
            logger.info(f"最佳参数: {opt_results['best_params']}")
            logger.info(f"总试验次数: {opt_results['total_trials']}")
            
            # 如果有优秀配置，也输出
            if 'good_configs' in opt_results and opt_results['good_configs']:
                logger.info(f"发现 {len(opt_results['good_configs'])} 个优秀配置 (测试集AUC > 0.7)")
        
        if 'final_results' in bo_results:
            final_results = bo_results['final_results']
            val_perf = final_results['validation_performance']
            holdout_perf = final_results['holdout_performance']
            
            logger.info(f"验证集最终AUC: {val_perf['auc']:.4f}")
            logger.info(f"保留测试集AUC: {holdout_perf['auc']:.4f}")
            
            # 计算泛化差距
            generalization_gap = val_perf['auc'] - holdout_perf['auc']
            logger.info(f"泛化差距: {generalization_gap:.4f}")
            
            if abs(generalization_gap) < 0.05:
                logger.info("✓ 泛化能力良好 (差距 < 0.05)")
            else:
                logger.warning("⚠ 可能存在过拟合风险 (差距 >= 0.05)")
        
        # 如果用户还想运行完整的MMD域适应实验，使用优化后的参数
        if hasattr(args, 'run_full_experiment_after_bo') and args.run_full_experiment_after_bo:
            logger.info("使用优化参数运行完整MMD域适应实验...")
            
            # 更新模型参数
            if 'optimization_results' in bo_results:
                best_params = bo_results['optimization_results']['best_params']
                
                # 将优化参数应用到args中
                for param, value in best_params.items():
                    if hasattr(args, param):
                        setattr(args, param, value)
                        logger.info(f"更新参数 {param} = {value}")
            
            # 运行标准跨域实验
            run_cross_domain_experiment_mode(args, logger)
        
    except Exception as e:
        logger.error(f"贝叶斯优化失败: {e}")
        import traceback
        logger.error(f"详细错误信息: {traceback.format_exc()}")
        raise

def run_bayesian_mmd_optimization_mode(args: argparse.Namespace, logger: logging.Logger):
    """运行贝叶斯MMD优化模式"""
    logger.info("运行贝叶斯MMD优化模式...")
    
    # 导入贝叶斯MMD优化模块
    try:
        from analytical_mmd_A2B_feature58.modeling.bayesian_mmd_optimizer import run_bayesian_mmd_optimization
    except ImportError as e:
        logger.error(f"无法导入贝叶斯MMD优化模块: {e}")
        logger.error("请确保 modeling/bayesian_mmd_optimizer.py 文件存在")
        return
    
    # 验证数据路径
    if not validate_data_paths():
        logger.error("数据文件验证失败")
        return
    
    # 准备贝叶斯MMD优化参数
    bo_mmd_kwargs = {
        'model_type': args.model_type,
        'feature_type': args.feature_type,
        'mmd_method': args.method,
        'use_class_conditional': args.use_class_conditional,
        'use_categorical': not getattr(args, 'no_categorical', False),
        'validation_split': args.validation_split,
        'n_calls': args.bo_n_calls,
        'random_state': args.bo_random_state,
        'target_domain': args.target_domain
    }
    
    # 生成保存路径
    if args.output_dir:
        save_path = args.output_dir
    else:
        suffix = generate_experiment_suffix(args)
        save_path = f"./results_bayesian_mmd_optimization_{args.model_type}_{args.method}_{args.feature_type}{suffix}"
    
    logger.info(f"贝叶斯MMD优化结果将保存到: {save_path}")
    
    try:
        # 运行贝叶斯MMD优化
        bo_mmd_results = run_bayesian_mmd_optimization(
            save_path=save_path,
            **bo_mmd_kwargs
        )
        
        logger.info("贝叶斯MMD优化完成!")
        logger.info("优化结果:")
        
        # 打印优化结果
        if 'optimization_results' in bo_mmd_results:
            opt_results = bo_mmd_results['optimization_results']
            logger.info(f"最佳验证集AUC: {opt_results['best_validation_auc']:.4f}")
            logger.info(f"最佳模型参数: {opt_results['best_model_params']}")
            logger.info(f"最佳MMD参数: {opt_results['best_mmd_params']}")
            logger.info(f"总试验次数: {opt_results['total_trials']}")
            
            # 如果有优秀配置，也输出
            if 'good_configs' in opt_results and opt_results['good_configs']:
                logger.info(f"发现 {len(opt_results['good_configs'])} 个优秀配置 (测试集AUC > 0.7)")
        
        if 'final_results' in bo_mmd_results:
            final_results = bo_mmd_results['final_results']
            val_perf = final_results['validation_performance']
            holdout_perf = final_results['holdout_performance']
            
            logger.info(f"验证集最终AUC: {val_perf['auc']:.4f}")
            logger.info(f"保留测试集AUC: {holdout_perf['auc']:.4f}")
            
            # 计算泛化差距
            generalization_gap = final_results['generalization_gap']['auc_gap']
            logger.info(f"泛化差距: {generalization_gap:.4f}")
            
            if abs(generalization_gap) < 0.05:
                logger.info("✓ 泛化能力良好 (差距 < 0.05)")
            else:
                logger.warning("⚠ 可能存在过拟合风险 (差距 >= 0.05)")
        
        # 如果用户选择自动运行完整的MMD域适应实验
        if args.auto_run_mmd_after_bo:
            logger.info("使用优化参数运行完整MMD域适应实验...")
            
            # 更新模型参数和MMD参数
            if 'optimization_results' in bo_mmd_results:
                best_model_params = bo_mmd_results['optimization_results']['best_model_params']
                best_mmd_params = bo_mmd_results['optimization_results']['best_mmd_params']
                
                # 将优化参数应用到args中
                for param, value in best_model_params.items():
                    if hasattr(args, param):
                        setattr(args, param, value)
                        logger.info(f"更新模型参数 {param} = {value}")
                
                # 将MMD参数应用到args中
                for param, value in best_mmd_params.items():
                    mmd_arg_name = f"{param}"  # 根据需要调整参数名映射
                    if hasattr(args, mmd_arg_name):
                        setattr(args, mmd_arg_name, value)
                        logger.info(f"更新MMD参数 {mmd_arg_name} = {value}")
            
            # 运行标准跨域实验
            run_cross_domain_experiment_mode(args, logger)
        
    except Exception as e:
        logger.error(f"贝叶斯MMD优化失败: {e}")
        import traceback
        logger.error(f"详细错误信息: {traceback.format_exc()}")
        raise

def main():
    """主函数"""
    args = parse_arguments()
    
    # 参数验证
    if (args.use_bayesian_optimization or args.use_bayesian_mmd_optimization) and args.data_split_strategy != 'three-way':
        print("错误: 贝叶斯优化只能在三分法模式下使用")
        print("请设置 --data-split-strategy three-way")
        return
    
    if args.validation_split <= 0 or args.validation_split >= 1:
        print("错误: 验证集比例必须在 (0, 1) 范围内")
        return
    
    # 设置日志
    logger = setup_experiment_logging(args.log_file)
    
    logger.info("=" * 60)
    logger.info("Analytical MMD Domain Adaptation Experiment")
    logger.info("=" * 60)
    
    # 打印实验配置
    logger.info("实验配置:")
    logger.info(f"  模型类型: {args.model_type}")
    logger.info(f"  特征类型: {args.feature_type}")
    logger.info(f"  MMD方法: {args.method}")
    logger.info(f"  数据划分策略: {args.data_split_strategy}")
    if args.data_split_strategy == 'three-way':
        logger.info(f"  验证集比例: {args.validation_split}")
    logger.info(f"  贝叶斯优化: {args.use_bayesian_optimization}")
    logger.info(f"  贝叶斯MMD优化: {args.use_bayesian_mmd_optimization}")
    if args.use_bayesian_optimization or args.use_bayesian_mmd_optimization:
        logger.info(f"  贝叶斯优化迭代次数: {args.bo_n_calls}")
    logger.info(f"  类条件MMD: {args.use_class_conditional}")
    logger.info(f"  阈值优化: {args.use_threshold_optimizer}")
    logger.info(f"  目标域: {args.target_domain}")
    
    # 检查可用模型
    available_models = get_available_models()
    logger.info(f"可用模型: {available_models}")
    
    if args.model_type not in available_models:
        logger.error(f"模型类型 {args.model_type} 不可用")
        logger.info("请安装相应的依赖包或选择其他模型类型")
        return
    
    try:
        if args.use_bayesian_mmd_optimization:
            # 使用贝叶斯MMD优化模式（优先级最高）
            logger.info("启动贝叶斯MMD优化模式...")
            run_bayesian_mmd_optimization_mode(args, logger)
        elif args.use_bayesian_optimization:
            # 使用贝叶斯优化模式
            logger.info("启动贝叶斯优化模式...")
            run_bayesian_optimization_mode(args, logger)
        elif args.compare_all:
            # 比较所有方法
            run_comparison_experiment(args, logger)
        else:
            # 标准跨域实验模式
            run_cross_domain_experiment_mode(args, logger)
            
    except KeyboardInterrupt:
        logger.info("实验被用户中断")
    except Exception as e:
        logger.error(f"实验失败: {e}")
        raise
    finally:
        logger.info("实验结束")

if __name__ == "__main__":
    main() 