#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
贝叶斯优化命令行脚本

使用贝叶斯优化进行超参数调优，采用三分法数据划分：
- A域数据：训练集
- B域验证集：贝叶斯优化目标函数评估
- B域保留测试集：最终模型泛化能力评估
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# 获取脚本所在目录的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 计算 TabPFN 项目的根目录
tabpfn_root = os.path.dirname(os.path.dirname(script_dir))

# 将 TabPFN 项目根目录添加到 Python 路径
sys.path.insert(0, tabpfn_root)

try:
    from analytical_mmd_A2B_feature58.modeling.bayesian_optimizer import run_bayesian_optimization
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
        description='运行贝叶斯优化超参数调优',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本用法 - AutoTabPFN模型
  python scripts/run_bayesian_optimization.py --model-type auto
  
  # 使用最佳7特征
  python scripts/run_bayesian_optimization.py --model-type auto --feature-type best7
  
  # 不使用类别特征
  python scripts/run_bayesian_optimization.py --model-type auto --no-categorical
  
  # 自定义验证集比例和优化次数
  python scripts/run_bayesian_optimization.py --model-type auto --validation-split 0.7 --n-calls 100
  
  # Random Forest模型优化
  python scripts/run_bayesian_optimization.py --model-type rf --n-calls 30
  
  # 基础TabPFN模型
  python scripts/run_bayesian_optimization.py --model-type base --feature-type all
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
    
    # 优化相关参数
    parser.add_argument(
        '--validation-split',
        type=float,
        default=0.8,
        help='验证集比例 (默认: 0.8，即80%%用于验证，20%%用于holdout测试)'
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
    
    # 输出相关参数
    parser.add_argument(
        '--output-dir',
        type=str,
        help='结果保存目录 (默认: ./results_bayesian_optimization_<model_type>_<feature_type>)'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        help='日志文件路径 (默认: 在输出目录中自动生成)'
    )
    
    return parser.parse_args()

def setup_experiment_logging(log_file: str = None):
    """设置实验日志"""
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"bayesian_optimization_{timestamp}.log"
    
    logger = setup_logger(
        name="bayesian_optimization",
        log_file=log_file,
        level=logging.INFO
    )
    
    return logger

def main():
    """主函数"""
    args = parse_arguments()
    
    # 生成输出目录
    if args.output_dir:
        save_path = args.output_dir
    else:
        categorical_suffix = "" if args.no_categorical else "_with_categorical"
        save_path = f"./results_bayesian_optimization_{args.model_type}_{args.feature_type}{categorical_suffix}"
    
    # 设置日志
    if args.log_file:
        log_file = args.log_file
    else:
        log_file = os.path.join(save_path, "bayesian_optimization.log")
    
    logger = setup_experiment_logging(log_file)
    
    logger.info("=" * 60)
    logger.info("贝叶斯优化超参数调优实验")
    logger.info("=" * 60)
    
    # 打印实验配置
    logger.info("实验配置:")
    logger.info(f"  模型类型: {args.model_type}")
    logger.info(f"  特征类型: {args.feature_type}")
    logger.info(f"  使用类别特征: {not args.no_categorical}")
    logger.info(f"  验证集比例: {args.validation_split}")
    logger.info(f"  优化迭代次数: {args.n_calls}")
    logger.info(f"  随机种子: {args.random_state}")
    logger.info(f"  结果保存路径: {save_path}")
    
    try:
        # 运行贝叶斯优化
        results = run_bayesian_optimization(
            model_type=args.model_type,
            feature_type=args.feature_type,
            use_categorical=not args.no_categorical,
            validation_split=args.validation_split,
            n_calls=args.n_calls,
            random_state=args.random_state,
            save_path=save_path
        )
        
        # 打印主要结果
        logger.info("=" * 60)
        logger.info("贝叶斯优化完成! 主要结果:")
        logger.info("=" * 60)
        
        opt_results = results['optimization_results']
        final_results = results['final_results']
        
        logger.info(f"最佳验证集AUC: {opt_results['best_validation_auc']:.4f}")
        logger.info(f"最佳参数: {opt_results['best_params']}")
        logger.info(f"总优化试验次数: {opt_results['total_trials']}")
        
        logger.info("\n最终模型性能:")
        val_perf = final_results['validation_performance']
        holdout_perf = final_results['holdout_performance']
        
        logger.info(f"验证集性能:")
        logger.info(f"  AUC: {val_perf['auc']:.4f}")
        logger.info(f"  F1: {val_perf['f1']:.4f}")
        logger.info(f"  Accuracy: {val_perf['accuracy']:.4f}")
        
        logger.info(f"保留测试集性能:")
        logger.info(f"  AUC: {holdout_perf['auc']:.4f}")
        logger.info(f"  F1: {holdout_perf['f1']:.4f}")
        logger.info(f"  Accuracy: {holdout_perf['accuracy']:.4f}")
        
        # 计算泛化差距
        auc_gap = val_perf['auc'] - holdout_perf['auc']
        f1_gap = val_perf['f1'] - holdout_perf['f1']
        acc_gap = val_perf['accuracy'] - holdout_perf['accuracy']
        
        logger.info(f"\n泛化差距 (验证集 - 保留测试集):")
        logger.info(f"  AUC差距: {auc_gap:+.4f}")
        logger.info(f"  F1差距: {f1_gap:+.4f}")
        logger.info(f"  Accuracy差距: {acc_gap:+.4f}")
        
        if abs(auc_gap) < 0.05:
            logger.info("✓ 模型泛化能力良好 (AUC差距 < 0.05)")
        else:
            logger.warning("⚠ 模型可能存在过拟合 (AUC差距 >= 0.05)")
        
        logger.info(f"\n详细结果已保存到: {save_path}")
        logger.info("  - optimization_history.json: 优化历史")
        logger.info("  - final_evaluation.json: 最终评估结果")
        logger.info("  - confusion_matrices.png: 混淆矩阵图")
        
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