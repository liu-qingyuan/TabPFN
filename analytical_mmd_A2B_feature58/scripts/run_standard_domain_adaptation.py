#!/usr/bin/env python3
"""
标准域适应实验运行脚本

严格遵循域适应研究的实验设计原则：
1. 只使用源域数据进行模型训练和参数调优
2. 目标域数据仅用于最终评估
3. 通过源域内交叉验证选择最佳超参数

使用示例:
python scripts/run_standard_domain_adaptation.py --model-type auto --mmd-method linear --target-domain B
"""

import argparse
import logging
import os
import sys
from typing import Optional
from datetime import datetime

# 获取脚本所在目录的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 计算 TabPFN 项目的根目录
tabpfn_root = os.path.dirname(os.path.dirname(script_dir))

# 将 TabPFN 项目根目录添加到 Python 路径
sys.path.insert(0, tabpfn_root)

try:
    from analytical_mmd_A2B_feature58.modeling.standard_domain_adaptation_optimizer import run_standard_domain_adaptation
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
        description='标准域适应实验 - 严格遵循域适应研究的实验设计原则',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本使用
  python scripts/run_standard_domain_adaptation.py --model-type auto --mmd-method linear

  # 指定目标域为C
  python scripts/run_standard_domain_adaptation.py --model-type auto --mmd-method linear --target-domain C

  # 使用类条件MMD
  python scripts/run_standard_domain_adaptation.py --model-type auto --mmd-method linear --use-class-conditional

  # 禁用MMD参数调优（只调优模型参数）
  python scripts/run_standard_domain_adaptation.py --model-type auto --no-mmd-tuning

实验设计原则:
  1. 源域数据用于训练和参数调优
  2. 目标域数据仅用于最终评估
  3. 通过源域内交叉验证选择最佳参数
  4. 完全不使用目标域数据进行调参
        """
    )
    
    # 模型相关参数
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['auto', 'rf', 'base'],
        default='auto',
        help='模型类型 (默认: auto)'
    )
    
    parser.add_argument(
        '--feature-type',
        type=str,
        choices=['best7', 'all'],
        default='best7',
        help='特征类型 (默认: best7)'
    )
    
    parser.add_argument(
        '--no-categorical',
        action='store_true',
        help='禁用类别特征'
    )
    
    # 域适应相关参数
    parser.add_argument(
        '--mmd-method',
        type=str,
        choices=['linear', 'kpca', 'mean_std', 'coral', 'adaptive_coral', 'mean_variance', 'adaptive_mean_variance'],
        default='linear',
        help='域适应方法 (默认: linear)'
    )
    
    parser.add_argument(
        '--use-class-conditional',
        action='store_true',
        help='使用类条件MMD'
    )
    
    parser.add_argument(
        '--no-mmd-tuning',
        action='store_true',
        help='禁用MMD参数调优（只调优模型参数）'
    )
    
    # 实验设计参数
    parser.add_argument(
        '--target-domain',
        type=str,
        choices=['B', 'C'],
        default='B',
        help='目标域选择: B(河南癌症医院) 或 C(广州医科大学) (默认: B)'
    )
    
    parser.add_argument(
        '--source-val-split',
        type=float,
        default=0.2,
        help='源域验证集比例 (默认: 0.2，即20%%用于验证)'
    )
    
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='交叉验证折数 (默认: 5)'
    )
    
    # 优化相关参数
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
        help='结果保存目录 (默认: 自动生成)'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        help='日志文件路径 (默认: 在输出目录中自动生成)'
    )
    
    return parser.parse_args()


def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    """设置日志"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 清除现有的处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 创建格式器
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:  # 只有当目录不为空时才创建
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def generate_output_dir(args: argparse.Namespace) -> str:
    """生成输出目录名称"""
    components = [
        'results_standard_domain_adaptation',
        args.model_type,
        args.mmd_method,
        args.feature_type
    ]
    
    if args.use_class_conditional:
        components.append('class_conditional')
    
    if args.no_categorical:
        components.append('no_categorical')
    
    if args.no_mmd_tuning:
        components.append('no_mmd_tuning')
    
    components.append(f'target_{args.target_domain}')
    
    return '_'.join(components)


def main():
    """主函数"""
    args = parse_arguments()
    
    # 参数验证
    if args.source_val_split <= 0 or args.source_val_split >= 1:
        print("错误: 源域验证集比例必须在 (0, 1) 范围内")
        return
    
    if args.cv_folds < 2:
        print("错误: 交叉验证折数必须至少为2")
        return
    
    # 生成输出目录
    if args.output_dir:
        save_path = args.output_dir
    else:
        save_path = generate_output_dir(args)
    
    # 设置日志
    if args.log_file:
        log_file = args.log_file
    else:
        log_file = os.path.join(save_path, 'experiment.log')
    
    logger = setup_logging(log_file)
    
    logger.info("=" * 80)
    logger.info("标准域适应实验 - 严格遵循域适应研究的实验设计原则")
    logger.info("=" * 80)
    
    # 打印实验配置
    logger.info("实验配置:")
    logger.info(f"  模型类型: {args.model_type}")
    logger.info(f"  特征类型: {args.feature_type}")
    logger.info(f"  MMD方法: {args.mmd_method}")
    logger.info(f"  类条件MMD: {args.use_class_conditional}")
    logger.info(f"  使用类别特征: {not args.no_categorical}")
    logger.info(f"  MMD参数调优: {not args.no_mmd_tuning}")
    logger.info(f"  目标域: {args.target_domain}")
    logger.info(f"  源域验证集比例: {args.source_val_split}")
    logger.info(f"  交叉验证折数: {args.cv_folds}")
    logger.info(f"  贝叶斯优化迭代次数: {args.n_calls}")
    logger.info(f"  随机种子: {args.random_state}")
    logger.info(f"  结果保存路径: {save_path}")
    
    logger.info("\n实验设计原则:")
    logger.info("  1. 源域数据用于训练和参数调优")
    logger.info("  2. 目标域数据仅用于最终评估")
    logger.info("  3. 通过源域内交叉验证选择最佳参数")
    logger.info("  4. 完全不使用目标域数据进行调参")
    
    try:
        # 运行标准域适应实验
        results = run_standard_domain_adaptation(
            model_type=args.model_type,
            feature_type=args.feature_type,
            mmd_method=args.mmd_method,
            use_class_conditional=args.use_class_conditional,
            use_categorical=not args.no_categorical,
            source_val_split=args.source_val_split,
            cv_folds=args.cv_folds,
            n_calls=args.n_calls,
            target_domain=args.target_domain,
            random_state=args.random_state,
            save_path=save_path,
            use_source_cv_for_mmd_tuning=not args.no_mmd_tuning
        )
        
        # 打印主要结果
        logger.info("=" * 80)
        logger.info("标准域适应实验完成! 主要结果:")
        logger.info("=" * 80)
        
        opt_results = results['optimization']
        eval_results = results['evaluation']
        
        logger.info(f"最佳源域CV AUC: {opt_results['best_score']:.4f}")
        logger.info(f"总优化试验次数: {len(opt_results['all_trials'])}")
        
        logger.info("\n最佳参数:")
        logger.info(f"  模型参数: {opt_results['best_params']['model_params']}")
        logger.info(f"  MMD参数: {opt_results['best_params']['mmd_params']}")
        
        logger.info("\n最终模型性能:")
        logger.info(f"  源域验证集 AUC: {eval_results['source_validation']['auc']:.4f}")
        logger.info(f"  目标域直接预测 AUC: {eval_results['target_direct']['auc']:.4f}")
        
        if eval_results['target_adapted']:
            adapted_auc = eval_results['target_adapted']['auc']
            direct_auc = eval_results['target_direct']['auc']
            improvement = adapted_auc - direct_auc
            improvement_pct = improvement / direct_auc * 100
            
            logger.info(f"  目标域域适应后 AUC: {adapted_auc:.4f}")
            logger.info(f"  域适应改进: {improvement:.4f} ({improvement_pct:.1f}%)")
            
            if improvement > 0:
                logger.info("  ✓ 域适应有效提升了跨域性能")
            else:
                logger.info("  ✗ 域适应未能提升跨域性能")
        else:
            logger.info("  未进行MMD域适应（MMD参数调优被禁用）")
        
        # 性能分析
        source_auc = eval_results['source_validation']['auc']
        target_direct_auc = eval_results['target_direct']['auc']
        domain_gap = source_auc - target_direct_auc
        
        logger.info(f"\n域差距分析:")
        logger.info(f"  源域-目标域性能差距: {domain_gap:.4f} ({domain_gap/source_auc*100:.1f}%)")
        
        if domain_gap > 0.1:
            logger.info("  ⚠️  存在显著的域差距，域适应很有必要")
        elif domain_gap > 0.05:
            logger.info("  ⚠️  存在中等的域差距，域适应可能有帮助")
        else:
            logger.info("  ✓ 域差距较小，模型具有良好的跨域泛化能力")
        
        logger.info(f"\n实验结果已保存到: {save_path}")
        logger.info("包含以下文件:")
        logger.info("  - optimization_results.json: 贝叶斯优化详细结果")
        logger.info("  - evaluation_results.json: 最终模型评估结果")
        logger.info("  - experiment_config.json: 实验配置信息")
        logger.info("  - experiment.log: 完整实验日志")
        
    except KeyboardInterrupt:
        logger.info("实验被用户中断")
    except Exception as e:
        logger.error(f"实验失败: {e}")
        import traceback
        logger.error(f"详细错误信息: {traceback.format_exc()}")
        raise
    finally:
        logger.info("实验结束")


if __name__ == '__main__':
    main() 