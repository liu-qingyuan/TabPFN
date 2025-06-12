#!/usr/bin/env python3
"""
贝叶斯优化域适应实验脚本

使用贝叶斯优化自动搜索最佳的模型参数和域适应参数组合。
支持多种域适应方法和性能目标约束。

性能目标:
- 源域10折交叉验证平均AUC ≥ 80%
- 目标域直接预测AUC ≥ 68%  
- 目标域CORAL域适应后AUC ≥ 70%

使用示例:
python scripts/run_bayesian_optimization_domain_adaptation.py --target-domain B --domain-adapt-method coral
"""

import argparse
import logging
import os
import sys
import time
from typing import Optional, Dict, Any

# 获取脚本所在目录的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 计算 TabPFN 项目的根目录
tabpfn_root = os.path.dirname(os.path.dirname(script_dir))

# 将 TabPFN 项目根目录添加到 Python 路径
sys.path.insert(0, tabpfn_root)

try:
    from analytical_mmd_A2B_feature58.modeling.standard_domain_adaptation_optimizer import StandardDomainAdaptationOptimizer
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
        description='贝叶斯优化域适应实验 - 自动搜索最佳参数组合',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本使用（A→B，CORAL方法）
  python scripts/run_bayesian_optimization_domain_adaptation.py --target-domain B --domain-adapt-method coral

  # 指定目标域为C（A→C，MMD线性方法）
  python scripts/run_bayesian_optimization_domain_adaptation.py --target-domain C --domain-adapt-method linear

  # 使用类条件域适应
  python scripts/run_bayesian_optimization_domain_adaptation.py --target-domain B --domain-adapt-method coral --use-class-conditional

  # 禁用域适应（只优化模型参数）
  python scripts/run_bayesian_optimization_domain_adaptation.py --target-domain B --no-mmd
  
  # 包含基线模型对比和更多优化迭代
  python scripts/run_bayesian_optimization_domain_adaptation.py --target-domain B --domain-adapt-method coral --include-baselines --n-calls 100

性能目标:
  - 源域10折交叉验证平均AUC ≥ 80%
  - 目标域直接预测AUC ≥ 68%
  - 目标域CORAL域适应后AUC ≥ 70%
        """
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
        '--feature-type',
        type=str,
        choices=['best7', 'all'],
        default='best7',
        help='特征类型 (默认: best7)'
    )
    
    parser.add_argument(
        '--domain-adapt-method',
        type=str,
        choices=['linear', 'kpca', 'mean_std', 'coral', 'adaptive_coral', 'mean_variance', 'adaptive_mean_variance', 'tca', 'adaptive_tca', 'jda', 'adaptive_jda'],
        default='coral',
        help='域适应方法 (默认: coral)'
    )
    
    parser.add_argument(
        '--use-class-conditional',
        action='store_true',
        help='使用类条件域适应'
    )
    
    parser.add_argument(
        '--no-categorical',
        action='store_true',
        help='禁用类别特征'
    )
    
    parser.add_argument(
        '--no-mmd',
        action='store_true',
        help='禁用域适应（只优化模型参数）'
    )
    
    # 新增基线模型参数
    parser.add_argument(
        '--include-baselines',
        action='store_true',
        help='包含基线模型（PKUPH和Mayo）对比评估'
    )
    
    # 贝叶斯优化参数
    parser.add_argument(
        '--n-calls',
        type=int,
        default=50,
        help='贝叶斯优化迭代次数 (默认: 50)'
    )
    
    parser.add_argument(
        '--source-cv-folds',
        type=int,
        default=10,
        help='源域交叉验证折数 (默认: 10)'
    )
    
    # 数据划分参数
    parser.add_argument(
        '--source-val-split',
        type=float,
        default=0.2,
        help='源域验证集比例 (默认: 0.2)'
    )
    
    # 输出相关参数
    parser.add_argument(
        '--output-dir',
        type=str,
        help='结果保存目录 (默认: 自动生成)'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='随机种子 (默认: 42)'
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
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_performance_targets() -> Dict[str, float]:
    """获取性能目标约束"""
    return {
        'source_cv_auc_min': 0.80,  # 源域10折CV平均AUC ≥ 80%
        'target_direct_auc_min': 0.68,  # 目标域直接预测AUC ≥ 68%
        'target_adapted_auc_min': 0.70  # 目标域域适应后AUC ≥ 70%
    }


def check_performance_targets(results: Dict[str, Any]) -> Dict[str, Any]:
    """检查是否达到性能目标"""
    targets = get_performance_targets()
    evaluation = results.get('evaluation', {})
    
    # 检查源域CV性能
    source_cv = evaluation.get('source_cv', {})
    source_cv_auc = source_cv.get('auc', {}).get('mean', 0.0) if source_cv else 0.0
    source_cv_ok = source_cv_auc >= targets['source_cv_auc_min']
    
    # 检查目标域直接预测性能
    target_direct = evaluation.get('target_direct', {})
    target_direct_auc = target_direct.get('auc', 0.0)
    target_direct_ok = target_direct_auc >= targets['target_direct_auc_min']
    
    # 检查目标域域适应后性能
    target_adapted = evaluation.get('target_adapted', {})
    target_adapted_auc = target_adapted.get('auc', 0.0) if target_adapted else 0.0
    target_adapted_ok = target_adapted_auc >= targets['target_adapted_auc_min']
    
    return {
        'source_cv_target_met': source_cv_ok,
        'target_direct_target_met': target_direct_ok,
        'target_adapted_target_met': target_adapted_ok,
        'all_targets_met': source_cv_ok and target_direct_ok and target_adapted_ok,
        'metrics': {
            'source_cv_auc': source_cv_auc,
            'target_direct_auc': target_direct_auc,
            'target_adapted_auc': target_adapted_auc
        },
        'targets': targets
    }


def generate_output_dir(args: argparse.Namespace) -> str:
    """生成输出目录名称"""
    domain_method = getattr(args, 'domain_adapt_method', 'coral')
    components = [
        'results_bayesian_optimization',
        'auto',  # 固定使用auto模型
        domain_method,
        args.feature_type
    ]
    
    if args.use_class_conditional:
        components.append('class_conditional')
    
    if args.no_categorical:
        components.append('no_categorical')
    
    if args.no_mmd:
        components.append('no_domain_adaptation')
    
    components.append(f'target_{args.target_domain}')
    components.append(f'calls_{args.n_calls}')
    components.append(f'cv{args.source_cv_folds}')
    
    # 添加时间戳
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    components.append(timestamp)
    
    return '_'.join(components)


class BayesianOptimizationDomainAdaptation:
    """贝叶斯优化域适应实验类"""
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        
        # 生成保存路径
        if args.output_dir:
            self.save_path = args.output_dir
        else:
            self.save_path = generate_output_dir(args)
        
        # 创建优化器（进行真正的贝叶斯优化）
        self.optimizer = StandardDomainAdaptationOptimizer(
            model_type='auto',
            feature_type=args.feature_type,
            mmd_method=getattr(args, 'domain_adapt_method', 'coral'),
            use_class_conditional=args.use_class_conditional,
            use_categorical=not args.no_categorical,
            source_val_split=args.source_val_split,
            cv_folds=args.source_cv_folds,  # 使用指定的CV折数
            n_calls=args.n_calls,  # 使用指定的优化迭代次数
            random_state=args.random_state,
            target_domain=args.target_domain,
            save_path=self.save_path,
            use_source_cv_for_mmd_tuning=not args.no_mmd  # 如果启用域适应则调优域适应参数
        )
    
    def run_experiment(self) -> Dict[str, Any]:
        """运行贝叶斯优化实验"""
        logging.info("=" * 80)
        logging.info("贝叶斯优化域适应实验")
        logging.info("=" * 80)
        
        # 打印实验配置
        logging.info("实验配置:")
        logging.info(f"  模型类型: auto (AutoTabPFN)")
        logging.info(f"  特征类型: {self.args.feature_type}")
        domain_method = getattr(self.args, 'domain_adapt_method', 'coral')
        logging.info(f"  域适应方法: {domain_method}")
        logging.info(f"  类条件域适应: {self.args.use_class_conditional}")
        logging.info(f"  使用类别特征: {not self.args.no_categorical}")
        logging.info(f"  使用域适应: {not self.args.no_mmd}")
        logging.info(f"  包含基线模型: {self.args.include_baselines}")
        logging.info(f"  目标域: {self.args.target_domain}")
        logging.info("")
        logging.info("优化配置:")
        logging.info(f"  贝叶斯优化迭代次数: {self.args.n_calls}")
        logging.info(f"  源域交叉验证折数: {self.args.source_cv_folds}")
        logging.info(f"  域适应实验: 数据集A按{int((1-self.args.source_val_split)*100)}%/{int(self.args.source_val_split*100)}%划分为训练/验证集")
        logging.info(f"  随机种子: {self.args.random_state}")
        logging.info(f"  结果保存路径: {self.save_path}")
        
        # 显示性能目标
        targets = get_performance_targets()
        logging.info("\n性能目标:")
        logging.info(f"  源域{self.args.source_cv_folds}折CV平均AUC ≥ {targets['source_cv_auc_min']:.1%}")
        logging.info(f"  目标域直接预测AUC ≥ {targets['target_direct_auc_min']:.1%}")
        logging.info(f"  目标域域适应后AUC ≥ {targets['target_adapted_auc_min']:.1%}")
        
        try:
            # 1. 加载和准备数据
            self.optimizer.load_and_prepare_data()
            
            # 2. 评估基线模型（如果启用）
            baseline_results = None
            if self.args.include_baselines:
                logging.info("\n" + "=" * 50)
                logging.info("评估基线模型 (PKUPH & Mayo)")
                logging.info("=" * 50)
                baseline_results = self.optimizer.evaluate_baseline_models_performance()
            
            # 3. 运行贝叶斯优化
            logging.info("\n" + "=" * 50)
            logging.info("运行贝叶斯优化")
            logging.info("=" * 50)
            optimization_results = self.optimizer.run_optimization()
            
            # 4. 训练最终模型
            logging.info("\n" + "=" * 50)
            logging.info("训练最终模型")
            logging.info("=" * 50)
            self.optimizer.train_final_model()
            
            # 5. 评估AutoTabPFN源域性能（CV）
            autotabpfn_source_cv = None
            if self.args.source_cv_folds > 0:
                logging.info("\n" + "=" * 50)
                logging.info("评估AutoTabPFN源域交叉验证性能")
                logging.info("=" * 50)
                
                try:
                    autotabpfn_source_cv = self.optimizer.evaluate_autotabpfn_source_cv(cv_folds=self.args.source_cv_folds)
                except Exception as e:
                    logging.error(f"AutoTabPFN源域CV评估失败: {e}")
            
            # 6. 评估最终模型
            logging.info("\n" + "=" * 50)
            logging.info("评估最终模型")
            logging.info("=" * 50)
            evaluation_results = self.optimizer.evaluate_final_model()
            
            # 将源域CV结果添加到评估结果中
            if autotabpfn_source_cv is not None:
                evaluation_results['source_cv'] = autotabpfn_source_cv
            
            # 7. 检查性能目标
            complete_results = {
                'optimization': optimization_results,
                'evaluation': evaluation_results,
                'baseline_models': baseline_results,
                'config': {
                    'model_type': 'auto',
                    'feature_type': self.args.feature_type,
                    'domain_adapt_method': domain_method,
                    'target_domain': self.args.target_domain,
                    'include_baselines': self.args.include_baselines,
                    'n_calls': self.args.n_calls,
                    'source_cv_folds': self.args.source_cv_folds,
                    'experiment_type': 'bayesian_optimization'
                }
            }
            
            # 检查是否达到性能目标
            target_check = check_performance_targets(complete_results)
            complete_results['performance_targets'] = target_check
            
            # 8. 保存结果
            if baseline_results:
                self.optimizer.save_results(optimization_results, evaluation_results, baseline_results)
            else:
                self.optimizer.save_results(optimization_results, evaluation_results)
            
            # 9. 打印主要结果
            self._print_results(evaluation_results, baseline_results, target_check)
            
            return complete_results
            
        except Exception as e:
            logging.error(f"实验失败: {e}")
            import traceback
            logging.error(f"详细错误信息: {traceback.format_exc()}")
            raise
    
    def _print_results(self, evaluation_results: Dict[str, Any], baseline_results: Optional[Dict[str, Any]] = None, target_check: Optional[Dict[str, Any]] = None) -> None:
        """打印实验结果"""
        logging.info("\n" + "=" * 80)
        logging.info("贝叶斯优化域适应实验完成! 主要结果:")
        logging.info("=" * 80)
        
        # 打印基线模型结果
        if baseline_results and self.args.include_baselines:
            logging.info("\n🔍 基线模型性能对比:")
            logging.info("-" * 50)
            
            for model_name, results in baseline_results.items():
                if 'error' not in results:
                    source_cv = results['source_cv']
                    source_val = results['source_validation']
                    target_direct = results['target_direct']
                    
                    logging.info(f"\n{model_name.upper()} 模型:")
                    logging.info(f"  源域10折CV - AUC: {source_cv['auc']['mean']:.4f} ± {source_cv['auc']['std']:.4f}")
                    logging.info(f"  源域10折CV - ACC: {source_cv['accuracy']['mean']:.4f} ± {source_cv['accuracy']['std']:.4f}")
                    logging.info(f"  源域10折CV - F1:  {source_cv['f1']['mean']:.4f} ± {source_cv['f1']['std']:.4f}")
                    logging.info(f"  源域验证集 - AUC: {source_val['auc']:.4f}")
                    logging.info(f"  目标域测试 - AUC: {target_direct['auc']:.4f}")
                    
                    # 计算域差距 - 添加除零检查
                    domain_gap = source_val['auc'] - target_direct['auc']
                    if source_val['auc'] > 0:
                        logging.info(f"  域差距: {domain_gap:.4f} ({domain_gap/source_val['auc']*100:.1f}%)")
                    else:
                        logging.info(f"  域差距: {domain_gap:.4f} (源域AUC为0，无法计算百分比)")
                        logging.warning(f"  {model_name.upper()}模型在源域验证集上表现异常，AUC为0")
                else:
                    logging.error(f"{model_name.upper()} 模型评估失败: {results['error']}")
        
        # 打印AutoTabPFN结果
        logging.info("\n🚀 AutoTabPFN模型性能:")
        logging.info("-" * 50)
        
        source_metrics = evaluation_results['source_validation']
        direct_metrics = evaluation_results['target_direct']
        adapted_metrics = evaluation_results['target_adapted']
        source_cv_metrics = evaluation_results.get('source_cv')
        
        logging.info("最终模型性能:")
        
        # 显示源域评估结果（如果存在）
        if source_cv_metrics:
            if self.args.source_cv_folds > 0:
                logging.info(f"  源域{self.args.source_cv_folds}折CV (全部数据集A) - AUC: {source_cv_metrics['auc']['mean']:.4f} ± {source_cv_metrics['auc']['std']:.4f}")
                logging.info(f"  源域{self.args.source_cv_folds}折CV (全部数据集A) - ACC: {source_cv_metrics['accuracy']['mean']:.4f} ± {source_cv_metrics['accuracy']['std']:.4f}")
                logging.info(f"  源域{self.args.source_cv_folds}折CV (全部数据集A) - F1:  {source_cv_metrics['f1']['mean']:.4f} ± {source_cv_metrics['f1']['std']:.4f}")
            else:
                logging.info(f"  源域8:2划分 (全部数据集A) - AUC: {source_cv_metrics['auc']['mean']:.4f} ± {source_cv_metrics['auc']['std']:.4f}")
                logging.info(f"  源域8:2划分 (全部数据集A) - ACC: {source_cv_metrics['accuracy']['mean']:.4f} ± {source_cv_metrics['accuracy']['std']:.4f}")
                logging.info(f"  源域8:2划分 (全部数据集A) - F1:  {source_cv_metrics['f1']['mean']:.4f} ± {source_cv_metrics['f1']['std']:.4f}")
            logging.info("")
        
        logging.info(f"  源域验证集 (80%数据集A用于域适应) AUC: {source_metrics['auc']:.4f}")
        logging.info(f"  源域验证集 (80%数据集A用于域适应) ACC: {source_metrics['acc']:.4f}")
        logging.info(f"  源域验证集 (80%数据集A用于域适应) F1:  {source_metrics['f1']:.4f}")
        logging.info(f"  目标域直接预测 AUC: {direct_metrics['auc']:.4f}")
        logging.info(f"  目标域直接预测 ACC: {direct_metrics['acc']:.4f}")
        logging.info(f"  目标域直接预测 F1:  {direct_metrics['f1']:.4f}")
        
        if adapted_metrics and not self.args.no_mmd:
            adapted_auc = adapted_metrics['auc']
            direct_auc = direct_metrics['auc']
            improvement = adapted_auc - direct_auc
            improvement_pct = improvement / direct_auc * 100 if direct_auc > 0 else 0
            
            logging.info(f"  目标域域适应后 AUC: {adapted_auc:.4f}")
            logging.info(f"  目标域域适应后 ACC: {adapted_metrics['acc']:.4f}")
            logging.info(f"  目标域域适应后 F1:  {adapted_metrics['f1']:.4f}")
            if direct_auc > 0:
                logging.info(f"  域适应改进: {improvement:.4f} ({improvement_pct:.1f}%)")
            else:
                logging.info(f"  域适应改进: {improvement:.4f} (直接预测AUC为0，无法计算百分比)")
            
            if improvement > 0:
                logging.info("  ✓ 域适应有效提升了跨域性能")
            else:
                logging.info("  ✗ 域适应未能提升跨域性能")
        else:
            logging.info("  未进行域适应（域适应被禁用）")
        
        # 性能分析和对比
        logging.info("\n📊 模型性能对比分析:")
        logging.info("-" * 50)
        
        # AutoTabPFN域差距
        source_auc = source_metrics['auc']
        target_direct_auc = direct_metrics['auc']
        autotabpfn_domain_gap = source_auc - target_direct_auc
        
        if source_auc > 0:
            logging.info(f"AutoTabPFN 域差距: {autotabpfn_domain_gap:.4f} ({autotabpfn_domain_gap/source_auc*100:.1f}%)")
        else:
            logging.info(f"AutoTabPFN 域差距: {autotabpfn_domain_gap:.4f} (源域AUC为0，无法计算百分比)")
        
        # 与基线模型对比
        if baseline_results and self.args.include_baselines:
            logging.info("\n基线模型 vs AutoTabPFN (目标域AUC对比):")
            
            for model_name, results in baseline_results.items():
                if 'error' not in results:
                    baseline_target_auc = results['target_direct']['auc']
                    autotabpfn_auc = adapted_metrics['auc'] if adapted_metrics and not self.args.no_mmd else direct_metrics['auc']
                    
                    improvement_vs_baseline = autotabpfn_auc - baseline_target_auc
                    improvement_pct_vs_baseline = improvement_vs_baseline / baseline_target_auc * 100 if baseline_target_auc > 0 else 0
                    
                    if baseline_target_auc > 0:
                        logging.info(f"  AutoTabPFN vs {model_name.upper()}: {improvement_vs_baseline:+.4f} ({improvement_pct_vs_baseline:+.1f}%)")
                    else:
                        logging.info(f"  AutoTabPFN vs {model_name.upper()}: {improvement_vs_baseline:+.4f} (基线AUC为0，无法计算百分比)")
                    
                    if improvement_vs_baseline > 0:
                        logging.info(f"    ✓ AutoTabPFN优于{model_name.upper()}模型")
                    else:
                        logging.info(f"    ✗ AutoTabPFN未能超越{model_name.upper()}模型")
        
        if autotabpfn_domain_gap > 0.1:
            logging.info("\n⚠️  存在显著的域差距，域适应很有必要")
        elif autotabpfn_domain_gap > 0.05:
            logging.info("\n⚠️  存在中等的域差距，域适应可能有帮助")
        else:
            logging.info("\n✓ 域差距较小，模型具有良好的跨域泛化能力")
        
        # 显示性能目标达成情况
        if target_check:
            logging.info("\n🎯 性能目标达成情况:")
            logging.info("-" * 50)
            
            targets = target_check['targets']
            metrics = target_check['metrics']
            
            # 源域CV目标
            source_cv_met = target_check['source_cv_target_met']
            source_cv_auc = metrics['source_cv_auc']
            source_cv_target = targets['source_cv_auc_min']
            status_icon = "✅" if source_cv_met else "❌"
            logging.info(f"{status_icon} 源域{self.args.source_cv_folds}折CV平均AUC: {source_cv_auc:.1%} (目标: ≥{source_cv_target:.1%})")
            
            # 目标域直接预测目标
            target_direct_met = target_check['target_direct_target_met']
            target_direct_auc = metrics['target_direct_auc']
            target_direct_target = targets['target_direct_auc_min']
            status_icon = "✅" if target_direct_met else "❌"
            logging.info(f"{status_icon} 目标域直接预测AUC: {target_direct_auc:.1%} (目标: ≥{target_direct_target:.1%})")
            
            # 目标域域适应后目标
            target_adapted_met = target_check['target_adapted_target_met']
            target_adapted_auc = metrics['target_adapted_auc']
            target_adapted_target = targets['target_adapted_auc_min']
            status_icon = "✅" if target_adapted_met else "❌"
            if target_adapted_auc > 0:
                logging.info(f"{status_icon} 目标域域适应后AUC: {target_adapted_auc:.1%} (目标: ≥{target_adapted_target:.1%})")
            else:
                logging.info(f"❌ 目标域域适应后AUC: 未启用域适应 (目标: ≥{target_adapted_target:.1%})")
            
            # 总体目标达成
            all_targets_met = target_check['all_targets_met']
            if all_targets_met:
                logging.info("\n🎉 所有性能目标均已达成！")
            else:
                unmet_targets = []
                if not source_cv_met:
                    unmet_targets.append("源域CV性能")
                if not target_direct_met:
                    unmet_targets.append("目标域直接预测性能")
                if not target_adapted_met:
                    unmet_targets.append("目标域域适应性能")
                logging.info(f"\n⚠️  未达成目标: {', '.join(unmet_targets)}")
        
        logging.info(f"\n实验结果已保存到: {self.save_path}")
        logging.info("包含以下文件:")
        logging.info("  - optimization_results.json: 贝叶斯优化结果和最佳参数")
        logging.info("  - evaluation_results.json: AutoTabPFN模型评估结果")
        if baseline_results:
            logging.info("  - baseline_models_results.json: 基线模型评估结果")
        logging.info("  - experiment_config.json: 实验配置信息")
        logging.info("  - experiment.log: 完整实验日志")


def main():
    """主函数"""
    args = parse_arguments()
    
    # 参数验证
    if args.source_val_split <= 0 or args.source_val_split >= 1:
        print("错误: 源域验证集比例必须在 (0, 1) 范围内")
        return
    
    # 生成输出目录
    if args.output_dir:
        save_path = args.output_dir
    else:
        save_path = generate_output_dir(args)
    
    # 设置日志
    log_file = os.path.join(save_path, 'experiment.log')
    setup_logging(log_file)
    
    try:
        # 创建并运行实验
        experiment = BayesianOptimizationDomainAdaptation(args)
        experiment.run_experiment()
        
        logging.info("实验成功完成!")
        
    except KeyboardInterrupt:
        logging.info("实验被用户中断")
    except Exception as e:
        logging.error(f"实验失败: {e}")
        raise
    finally:
        logging.info("实验结束")


if __name__ == '__main__':
    main() 