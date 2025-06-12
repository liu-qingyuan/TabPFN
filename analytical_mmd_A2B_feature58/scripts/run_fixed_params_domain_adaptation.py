#!/usr/bin/env python3
"""
固定参数域适应实验脚本

使用预设的最佳参数，不进行贝叶斯优化，直接训练和评估模型。
适用于快速验证和对比实验。

使用示例:
python scripts/run_fixed_params_domain_adaptation.py --target-domain B
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
        description='固定参数域适应实验 - 使用预设的最佳参数',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本使用（A→B）
  python scripts/run_fixed_params_domain_adaptation.py --target-domain B

  # 指定目标域为C（A→C）
  python scripts/run_fixed_params_domain_adaptation.py --target-domain C

  # 使用类条件MMD
  python scripts/run_fixed_params_domain_adaptation.py --target-domain B --use-class-conditional

  # 禁用MMD域适应（只测试模型性能）
  python scripts/run_fixed_params_domain_adaptation.py --target-domain B --no-mmd
  
  # 包含基线模型对比
  python scripts/run_fixed_params_domain_adaptation.py --target-domain B --include-baselines

预设参数说明:
  使用经过调优的最佳参数组合，包括：
  - AutoTabPFN: max_time=30, preset=default, ges_scoring=f1
  - MMD: lr=0.01, n_epochs=200, batch_size=32
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
        help='使用类条件MMD'
    )
    
    parser.add_argument(
        '--no-categorical',
        action='store_true',
        help='禁用类别特征'
    )
    
    parser.add_argument(
        '--no-mmd',
        action='store_true',
        help='禁用MMD域适应（只测试模型性能）'
    )
    
    # 新增基线模型参数
    parser.add_argument(
        '--include-baselines',
        action='store_true',
        help='包含基线模型（PKUPH和Mayo）对比评估'
    )
    
    parser.add_argument(
        '--source-cv-folds',
        type=int,
        default=10,
        help='源域交叉验证折数 (设为0表示使用8:2划分而不是交叉验证, 默认: 10)'
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


def get_best_params(use_mmd: bool = True) -> Dict[str, Any]:
    """获取预设的最佳参数"""
    
    # 最佳模型参数 (根据调优结果)
    best_model_params = {
        'max_time': 30,
        # 'max_time': 60,
        'preset': 'default',
        'ges_scoring': 'f1',
        'max_models': 10, 
        'validation_method': 'cv',  # 修改为cv，更适合域适应
        'n_repeats': 1,
        'n_folds': 5,  # 修改为5折，与标准设置一致
        # 'n_folds': 10,  # 修改为5折，与标准设置一致
        'ges_n_iterations': 20,
        'ignore_limits': False
    }
    
    # MMD参数 
    best_mmd_params = {
        'lr': 0.01,
        'n_epochs': 100,
        'batch_size': 32,
        'lambda_reg': 1e-3,
        'gamma': 1.0,  
        'staged_training': False,
        'dynamic_gamma': False  
    } if use_mmd else {}
    
    return {
        'model_params': best_model_params,
        'mmd_params': best_mmd_params
    }


def generate_output_dir(args: argparse.Namespace) -> str:
    """生成输出目录名称"""
    domain_method = getattr(args, 'domain_adapt_method', getattr(args, 'mmd_method', 'linear'))
    components = [
        'results_fixed_params',
        'auto',  # 固定使用auto模型
        domain_method,
        args.feature_type
    ]
    
    if args.use_class_conditional:
        components.append('class_conditional')
    
    if args.no_categorical:
        components.append('no_categorical')
    
    if args.no_mmd:
        components.append('no_mmd')
    
    components.append(f'target_{args.target_domain}')
    
    # 添加时间戳
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    components.append(timestamp)
    
    return '_'.join(components)


class FixedParamsDomainAdaptation:
    """固定参数域适应实验类"""
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.best_params = get_best_params(use_mmd=not args.no_mmd)
        
        # 生成保存路径
        if args.output_dir:
            self.save_path = args.output_dir
        else:
            self.save_path = generate_output_dir(args)
        
        # 创建优化器（但不进行优化）
        self.optimizer = StandardDomainAdaptationOptimizer(
            model_type='auto',
            feature_type=args.feature_type,
            mmd_method=getattr(args, 'domain_adapt_method', args.mmd_method if hasattr(args, 'mmd_method') else 'linear'),
            use_class_conditional=args.use_class_conditional,
            use_categorical=not args.no_categorical,
            source_val_split=args.source_val_split,
            cv_folds=5,  # 固定为5折
            n_calls=1,  # 不进行优化，只运行1次
            random_state=args.random_state,
            target_domain=args.target_domain,
            save_path=self.save_path,
            use_source_cv_for_mmd_tuning=False  # 不调优MMD参数
        )
    
    def run_experiment(self) -> Dict[str, Any]:
        """运行固定参数实验"""
        logging.info("=" * 80)
        logging.info("固定参数域适应实验")
        logging.info("=" * 80)
        
        # 打印实验配置
        logging.info("实验配置:")
        logging.info(f"  模型类型: auto (AutoTabPFN)")
        logging.info(f"  特征类型: {self.args.feature_type}")
        domain_method = getattr(self.args, 'domain_adapt_method', getattr(self.args, 'mmd_method', 'linear'))
        logging.info(f"  域适应方法: {domain_method}")
        logging.info(f"  类条件MMD: {self.args.use_class_conditional}")
        logging.info(f"  使用类别特征: {not self.args.no_categorical}")
        logging.info(f"  使用MMD域适应: {not self.args.no_mmd}")
        logging.info(f"  包含基线模型: {self.args.include_baselines}")
        logging.info(f"  目标域: {self.args.target_domain}")
        logging.info("")
        logging.info("数据划分策略:")
        if self.args.source_cv_folds > 0:
            logging.info(f"  源域CV评估: 使用全部数据集A进行{self.args.source_cv_folds}折交叉验证")
        else:
            logging.info(f"  源域评估: 使用数据集A的8:2划分进行评估")
        logging.info(f"  域适应实验: 数据集A按{int((1-self.args.source_val_split)*100)}%/{int(self.args.source_val_split*100)}%划分为训练/验证集")
        logging.info(f"  随机种子: {self.args.random_state}")
        logging.info(f"  结果保存路径: {self.save_path}")
        
        logging.info("\n预设参数:")
        logging.info(f"  模型参数: {self.best_params['model_params']}")
        if self.best_params['mmd_params']:
            logging.info(f"  域适应参数: {self.best_params['mmd_params']}")
        else:
            logging.info("  域适应参数: 禁用域适应")
        
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
            
            # 3. 直接设置最佳参数（跳过优化）
            self.optimizer.best_params = self.best_params
            self.optimizer.best_score = 0.0  # 占位符，因为没有进行优化
            
            logging.info("\n" + "=" * 50)
            logging.info("训练AutoTabPFN模型")
            logging.info("=" * 50)
            logging.info("跳过贝叶斯优化，直接使用预设的最佳参数")
            
            # 4. 训练最终模型
            self.optimizer.train_final_model()
            
            # 4.5. 评估AutoTabPFN源域性能（8:2划分或CV）
            autotabpfn_source_cv = None
            if self.args.source_cv_folds > 0:
                logging.info("\n" + "=" * 50)
                logging.info("评估AutoTabPFN源域交叉验证性能")
                logging.info("=" * 50)
                
                try:
                    autotabpfn_source_cv = self.optimizer.evaluate_autotabpfn_source_cv(cv_folds=self.args.source_cv_folds)
                except Exception as e:
                    logging.error(f"AutoTabPFN源域CV评估失败: {e}")
            else:
                logging.info("\n" + "=" * 50)
                logging.info("评估AutoTabPFN源域8:2划分性能")
                logging.info("=" * 50)
                logging.info("使用与域适应相同的8:2数据划分进行评估")
            
            # 5. 评估最终模型
            evaluation_results = self.optimizer.evaluate_final_model()
            
            # 将源域CV结果添加到评估结果中
            if autotabpfn_source_cv is not None:
                evaluation_results['source_cv'] = autotabpfn_source_cv
            
            # 6. 保存结果
            optimization_results = {
                'best_params': self.best_params,
                'best_score': 0.0,
                'all_trials': [],
                'note': 'Fixed parameters experiment - no optimization performed'
            }
            
            # 更新save_results调用以包含baseline_results
            if baseline_results:
                self.optimizer.save_results(optimization_results, evaluation_results, baseline_results)
            else:
                self.optimizer.save_results(optimization_results, evaluation_results)
            
            # 7. 打印主要结果
            self._print_results(evaluation_results, baseline_results)
            
            # 8. 返回完整结果
            complete_results = {
                'optimization': optimization_results,
                'evaluation': evaluation_results,
                'baseline_models': baseline_results,
                'config': {
                    'model_type': 'auto',
                    'feature_type': self.args.feature_type,
                    'domain_adapt_method': getattr(self.args, 'domain_adapt_method', getattr(self.args, 'mmd_method', 'linear')),
                    'target_domain': self.args.target_domain,
                    'include_baselines': self.args.include_baselines,
                    'best_params': self.best_params,
                    'experiment_type': 'fixed_params'
                }
            }
            
            return complete_results
            
        except Exception as e:
            logging.error(f"实验失败: {e}")
            import traceback
            logging.error(f"详细错误信息: {traceback.format_exc()}")
            raise
    
    def _print_results(self, evaluation_results: Dict[str, Any], baseline_results: Optional[Dict[str, Any]] = None) -> None:
        """打印实验结果"""
        logging.info("\n" + "=" * 80)
        logging.info("固定参数域适应实验完成! 主要结果:")
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
        
        logging.info(f"\n实验结果已保存到: {self.save_path}")
        logging.info("包含以下文件:")
        logging.info("  - optimization_results.json: 实验配置和参数")
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
        experiment = FixedParamsDomainAdaptation(args)
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