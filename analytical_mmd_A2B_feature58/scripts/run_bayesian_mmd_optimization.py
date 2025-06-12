#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
贝叶斯MMD优化命令行脚本

同时优化模型超参数和MMD域适应参数，采用三分法数据划分：
- A域数据：训练集
- B/C域验证集：贝叶斯优化目标函数评估（包含MMD域适应）
- B/C域保留测试集：最终模型泛化能力评估
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from typing import Optional

# 获取脚本所在目录的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 计算 TabPFN 项目的根目录
tabpfn_root = os.path.dirname(os.path.dirname(script_dir))

# 将 TabPFN 项目根目录添加到 Python 路径
sys.path.insert(0, tabpfn_root)

try:
    from analytical_mmd_A2B_feature58.modeling.bayesian_mmd_optimizer import run_bayesian_mmd_optimization
    from analytical_mmd_A2B_feature58.utils.logging_setup import setup_logger
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
        description='运行贝叶斯MMD优化（同时优化模型参数和MMD参数）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本用法 - AutoTabPFN + Linear MMD
  python scripts/run_bayesian_mmd_optimization.py --model-type auto --mmd-method linear
  
  # 使用最佳7特征 + 类条件MMD
  python scripts/run_bayesian_mmd_optimization.py --model-type auto --feature-type best7 --mmd-method linear --use-class-conditional
  
  # 目标域C + KPCA MMD
  python scripts/run_bayesian_mmd_optimization.py --model-type auto --mmd-method kpca --target-domain C
  
  # 自定义验证集比例和优化次数
  python scripts/run_bayesian_mmd_optimization.py --model-type auto --mmd-method linear --validation-split 0.7 --n-calls 100
  
  # Random Forest + Mean-Std MMD
  python scripts/run_bayesian_mmd_optimization.py --model-type rf --mmd-method mean_std --n-calls 30
  
  # 完整配置示例
  python scripts/run_bayesian_mmd_optimization.py \\
    --model-type auto \\
    --feature-type best7 \\
    --mmd-method linear \\
    --use-class-conditional \\
    --target-domain B \\
    --validation-split 0.8 \\
    --n-calls 50 \\
    --auto-run-mmd-after-bo
        """
    )
    
    # 模型相关参数
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['auto', 'base', 'rf', 'tuned'],
        default='auto',
        help='模型类型: auto(AutoTabPFN), base(原生TabPFN), rf(Random Forest), tuned(TunedTabPFN)'
    )
    
    # 特征相关参数
    parser.add_argument(
        '--feature-type',
        type=str,
        choices=['all', 'best7'],
        default='best7',
        help='特征类型: all(58个特征) 或 best7(最佳7特征)'
    )
    
    parser.add_argument(
        '--no-categorical',
        action='store_true',
        help='不使用类别特征'
    )
    
    # MMD相关参数
    parser.add_argument(
        '--mmd-method',
        type=str,
        choices=['linear', 'mean_std'],
        default='linear',
        help='MMD方法 (已弃用，现在会自动搜索最佳方法)'
    )
    
    parser.add_argument(
        '--use-class-conditional',
        action='store_true',
        default=False,
        help='是否使用类条件MMD (已弃用，现在会自动搜索最佳选择)'
    )
    
    # 目标域选择
    parser.add_argument(
        '--target-domain',
        type=str,
        choices=['B', 'C'],
        default='B',
        help='目标域选择: B(河南癌症医院) 或 C(广州医科大学)'
    )
    
    # 优化相关参数
    parser.add_argument(
        '--validation-split',
        type=float,
        default=0.7,
        help='验证集比例 (默认: 0.7，即70%%用于验证，30%%用于holdout测试)'
    )
    
    parser.add_argument(
        '--n-calls',
        type=int,
        default=50,
        help='贝叶斯优化迭代次数 (默认: 50)'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='随机种子 (默认: 42)'
    )
    
    # 后续处理参数
    parser.add_argument(
        '--auto-run-mmd-after-bo',
        action='store_true',
        help='贝叶斯优化完成后自动运行完整的MMD域适应实验'
    )
    
    # 输出相关参数
    parser.add_argument(
        '--output-dir',
        type=str,
        help='结果保存目录 (默认: ./results_bayesian_mmd_optimization_<model_type>_<mmd_method>_<feature_type>)'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        help='日志文件路径 (默认: 在输出目录中自动生成)'
    )
    
    parser.add_argument(
        '--use-fixed-model-params',
        action='store_true',
        default=True,
        help='使用固定的最佳模型参数，只优化MMD参数 (推荐，默认: True)'
    )
    
    parser.add_argument(
        '--optimize-all-params',
        action='store_true',
        default=False,
        help='同时优化模型参数和MMD参数 (不推荐，会大幅增加搜索空间)'
    )
    
    parser.add_argument(
        '--evaluate-source-cv',
        action='store_true',
        default=False,
        help='评估A域交叉验证基准性能（可选功能）'
    )
    
    return parser.parse_args()

def setup_experiment_logging(log_file: Optional[str] = None):
    """设置实验日志"""
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"bayesian_mmd_optimization_{timestamp}.log"
    
    logger = setup_logger(
        log_level=logging.INFO,
        log_file=log_file,
        console_output=True
    )
    
    return logger

def main():
    """主函数"""
    args = parse_arguments()
    
    # 解析参数冲突
    if args.optimize_all_params:
        use_fixed_model_params = False
        logging.info("⚠️  启用联合优化模式：同时优化模型参数和MMD参数")
        logging.info("   这会大幅增加搜索空间，建议增加n_calls参数")
    else:
        use_fixed_model_params = args.use_fixed_model_params
        if use_fixed_model_params:
            logging.info("✅ 使用固定模型参数模式：只优化MMD参数（推荐）")
        else:
            logging.info("⚠️  使用联合优化模式：同时优化模型参数和MMD参数")
    
    # 生成输出目录
    if args.output_dir:
        save_path = args.output_dir
    else:
        categorical_suffix = "" if args.no_categorical else "_with_categorical"
        class_conditional_suffix = "_class_conditional" if args.use_class_conditional else ""
        target_suffix = f"_target_{args.target_domain}" if args.target_domain != 'B' else ""
        save_path = f"./results_bayesian_mmd_optimization_{args.model_type}_{args.mmd_method}_{args.feature_type}{categorical_suffix}{class_conditional_suffix}{target_suffix}"
    
    # 设置日志
    if args.log_file:
        log_file = args.log_file
    else:
        log_file = os.path.join(save_path, "bayesian_mmd_optimization.log")
    
    logger = setup_experiment_logging(log_file)
    
    logger.info("=" * 60)
    logger.info("贝叶斯MMD优化实验")
    logger.info("=" * 60)
    logger.info(f"模型类型: {args.model_type}")
    logger.info(f"特征类型: {args.feature_type}")
    logger.info(f"目标域: {args.target_domain}")
    logger.info(f"验证集比例: {args.validation_split}")
    logger.info(f"优化迭代次数: {args.n_calls}")
    logger.info(f"随机种子: {args.random_state}")
    logger.info("MMD配置: 自动搜索最佳方法和类条件选择")
    logger.info("  - 可选MMD方法: linear, mean_std")
    logger.info("  - 可选类条件: True/False")
    logger.info("=" * 60)
    
    # 打印实验配置
    logger.info("实验配置:")
    logger.info(f"  模型类型: {args.model_type}")
    logger.info(f"  特征类型: {args.feature_type}")
    logger.info(f"  MMD方法: {args.mmd_method}")
    logger.info(f"  类条件MMD: {args.use_class_conditional}")
    logger.info(f"  使用类别特征: {not args.no_categorical}")
    logger.info(f"  目标域: {args.target_domain}")
    logger.info(f"  验证集比例: {args.validation_split}")
    logger.info(f"  优化迭代次数: {args.n_calls}")
    logger.info(f"  随机种子: {args.random_state}")
    logger.info(f"  评估A域CV基准: {args.evaluate_source_cv}")
    logger.info(f"  自动运行MMD实验: {args.auto_run_mmd_after_bo}")
    logger.info(f"  结果保存路径: {save_path}")
    
    try:
        # 运行贝叶斯MMD优化
        results = run_bayesian_mmd_optimization(
            model_type=args.model_type,
            feature_type=args.feature_type,
            mmd_method=args.mmd_method,
            use_class_conditional=args.use_class_conditional,
            use_categorical=not args.no_categorical,
            validation_split=args.validation_split,
            n_calls=args.n_calls,
            target_domain=args.target_domain,
            random_state=args.random_state,
            save_path=save_path,
            use_fixed_model_params=use_fixed_model_params,
            evaluate_source_cv=args.evaluate_source_cv
        )
        
        # 打印主要结果
        logger.info("=" * 60)
        logger.info("贝叶斯MMD优化完成! 主要结果:")
        logger.info("=" * 60)
        
        # A域CV基准结果（如果有）
        if 'source_domain_cv' in results:
            source_cv = results['source_domain_cv']
            logger.info("A域交叉验证基准:")
            logger.info(f"  AUC: {source_cv['cv_results']['mean_scores']['auc']:.4f} ± {source_cv['cv_results']['std_scores']['auc']:.4f}")
            logger.info(f"  F1:  {source_cv['cv_results']['mean_scores']['f1']:.4f} ± {source_cv['cv_results']['std_scores']['f1']:.4f}")
            logger.info(f"  Acc: {source_cv['cv_results']['mean_scores']['accuracy']:.4f} ± {source_cv['cv_results']['std_scores']['accuracy']:.4f}")
            logger.info("")
        
        opt_results = results['optimization_results']
        final_results = results['final_results']
        
        logger.info(f"最佳验证集AUC: {opt_results['best_validation_auc']:.4f}")
        logger.info(f"总优化试验次数: {opt_results['total_trials']}")
        
        logger.info("\n最佳参数:")
        logger.info(f"模型参数: {opt_results['best_model_params']}")
        logger.info(f"MMD参数: {opt_results['best_mmd_params']}")
        
        logger.info("\n最终模型性能:")
        val_perf = final_results['validation_performance']
        holdout_perf = final_results['holdout_performance']
        
        logger.info(f"验证集性能:")
        logger.info(f"  AUC: {val_perf['auc']:.4f}")
        logger.info(f"  F1: {val_perf['f1']:.4f}")
        logger.info(f"  Accuracy: {val_perf['acc']:.4f}")
        
        logger.info(f"保留测试集性能:")
        logger.info(f"  AUC: {holdout_perf['auc']:.4f}")
        logger.info(f"  F1: {holdout_perf['f1']:.4f}")
        logger.info(f"  Accuracy: {holdout_perf['acc']:.4f}")
        
        # 跨域性能比较（如果有）
        if 'cross_domain_performance' in results:
            cross_perf = results['cross_domain_performance']
            logger.info(f"\n跨域性能比较:")
            logger.info(f"  A域CV基准 → B域验证集: {cross_perf['cross_domain_improvement_validation']:+.2f}%")
            logger.info(f"  A域CV基准 → B域测试集: {cross_perf['cross_domain_improvement_holdout']:+.2f}%")
            logger.info(f"  性能差距-验证集: {cross_perf['cross_domain_gap_validation']:+.4f}")
            logger.info(f"  性能差距-测试集: {cross_perf['cross_domain_gap_holdout']:+.4f}")
        
        # 计算泛化差距
        gen_gap = final_results['generalization_gap']
        auc_gap = gen_gap['auc_gap']
        f1_gap = gen_gap['f1_gap']
        acc_gap = gen_gap['acc_gap']
        
        logger.info(f"\n泛化差距 (验证集 - 保留测试集):")
        logger.info(f"  AUC差距: {auc_gap:+.4f}")
        logger.info(f"  F1差距: {f1_gap:+.4f}")
        logger.info(f"  Accuracy差距: {acc_gap:+.4f}")
        
        if abs(auc_gap) < 0.05:
            logger.info("✓ 模型泛化能力良好 (AUC差距 < 0.05)")
        else:
            logger.warning("⚠ 模型可能存在过拟合 (AUC差距 >= 0.05)")
        
        # 优秀配置汇总
        if opt_results.get('good_configs'):
            logger.info(f"\n发现 {len(opt_results['good_configs'])} 个优秀配置 (测试集AUC > 0.7)")
        
        logger.info(f"\n详细结果已保存到: {save_path}")
        if 'source_domain_cv' in results:
            logger.info("  - source_domain_cv_baseline.json: A域交叉验证基准")
        if 'cross_domain_performance' in results:
            logger.info("  - cross_domain_performance_comparison.json: 跨域性能比较")
        logger.info("  - bayesian_mmd_optimization_history.json: 优化历史")
        logger.info("  - final_mmd_evaluation.json: 最终评估结果")
        logger.info("  - experiment_config.json: 实验配置")
        
        # 如果需要，运行完整的MMD域适应实验
        if args.auto_run_mmd_after_bo:
            logger.info("\n" + "=" * 60)
            logger.info("使用优化参数运行完整MMD域适应实验...")
            logger.info("=" * 60)
            
            # 这里可以调用标准的MMD域适应实验
            # 使用优化后的参数
            logger.info("注意: 完整MMD域适应实验需要单独实现")
            logger.info("建议使用优化后的参数手动运行 run_analytical_mmd.py")
            logger.info(f"推荐命令:")
            
            cmd_parts = [
                "python scripts/run_analytical_mmd.py",
                f"--model-type {args.model_type}",
                f"--feature-type {args.feature_type}",
                f"--method {args.mmd_method}",
                f"--target-domain {args.target_domain}",
                "--data-split-strategy three-way",
                f"--validation-split {args.validation_split}"
            ]
            
            if args.use_class_conditional:
                cmd_parts.append("--use-class-conditional")
            if args.no_categorical:
                cmd_parts.append("--no-categorical")
            
            # 添加优化后的模型参数
            for param, value in opt_results['best_model_params'].items():
                cmd_parts.append(f"--{param.replace('_', '-')} {value}")
            
            # 添加优化后的MMD参数
            for param, value in opt_results['best_mmd_params'].items():
                cmd_parts.append(f"--{param.replace('_', '-')} {value}")
            
            logger.info(" \\\n  ".join(cmd_parts))
        
    except KeyboardInterrupt:
        logger.info("实验被用户中断")
    except Exception as e:
        logger.error(f"实验失败: {e}")
        import traceback
        logger.error(f"详细错误信息: {traceback.format_exc()}")
        raise
    finally:
        logger.info("实验结束")

if __name__ == "__main__":
    main() 