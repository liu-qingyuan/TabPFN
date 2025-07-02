#!/usr/bin/env python3
"""
使用真实医疗数据集A和B的UDA可视化示例

这个脚本使用真实的医疗数据集进行完整的UDA可视化分析：
- 数据集A (AI4health) 作为源域
- 数据集B (HenanCancerHospital) 作为目标域
- 使用best8特征进行域适应
- 生成完整的可视化分析报告

运行示例: python examples/real_data_visualization.py
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def load_real_medical_data():
    """加载真实医疗数据集A和B"""
    try:
        from data.loader import MedicalDataLoader
        
        # 使用MedicalDataLoader加载真实数据
        loader = MedicalDataLoader()
        
        # 加载数据集A和B，使用best8特征
        print("📊 加载真实医疗数据...")
        data_A = loader.load_dataset('A', feature_type='best8')
        data_B = loader.load_dataset('B', feature_type='best8')
        
        # 提取特征和标签
        X_A = pd.DataFrame(data_A['X'], columns=data_A['feature_names'])
        y_A = pd.Series(data_A['y'], name='label')
        X_B = pd.DataFrame(data_B['X'], columns=data_B['feature_names'])
        y_B = pd.Series(data_B['y'], name='label')
        
        # 确保A和B数据集使用相同的特征列（特征对齐）
        common_features = list(set(X_A.columns) & set(X_B.columns))
        if len(common_features) != len(X_A.columns) or len(common_features) != len(X_B.columns):
            print(f"⚠ 警告: A和B数据集特征不完全一致")
            print(f"  A特征: {list(X_A.columns)}")
            print(f"  B特征: {list(X_B.columns)}")
            print(f"  共同特征: {common_features}")
            # 使用共同特征
            X_A = X_A[common_features]
            X_B = X_B[common_features]
        
        print(f"✅ 成功加载真实医疗数据:")
        print(f"  数据集A (AI4health): {X_A.shape}")
        print(f"  数据集B (HenanCancerHospital): {X_B.shape}")
        print(f"  特征列表: {list(X_A.columns)}")
        print(f"  A类别分布: {dict(y_A.value_counts().sort_index())}")
        print(f"  B类别分布: {dict(y_B.value_counts().sort_index())}")
        
        return X_A.values, y_A.values, X_B.values, y_B.values, list(X_A.columns)
        
    except Exception as e:
        print(f"❌ 加载真实数据失败: {e}")
        print("请确保数据文件存在且data.loader模块可用")
        raise


def analyze_data_distribution(X_A, y_A, X_B, y_B, feature_names):
    """分析数据分布特征"""
    print(f"\n📈 数据分布分析:")
    
    # 基本统计
    print(f"源域 (A) 统计:")
    print(f"  样本数: {len(X_A)}")
    print(f"  正类比例: {np.mean(y_A):.3f}")
    print(f"  特征均值: {np.mean(X_A, axis=0)[:3]} ...")
    print(f"  特征标准差: {np.std(X_A, axis=0)[:3]} ...")
    
    print(f"目标域 (B) 统计:")
    print(f"  样本数: {len(X_B)}")
    print(f"  正类比例: {np.mean(y_B):.3f}")
    print(f"  特征均值: {np.mean(X_B, axis=0)[:3]} ...")
    print(f"  特征标准差: {np.std(X_B, axis=0)[:3]} ...")
    
    # 计算域间差异
    mean_diff = np.mean(np.abs(np.mean(X_A, axis=0) - np.mean(X_B, axis=0)))
    std_diff = np.mean(np.abs(np.std(X_A, axis=0) - np.std(X_B, axis=0)))
    
    print(f"域间差异:")
    print(f"  平均特征均值差异: {mean_diff:.4f}")
    print(f"  平均特征标准差差异: {std_diff:.4f}")
    
    return {
        'mean_difference': mean_diff,
        'std_difference': std_diff,
        'source_positive_rate': np.mean(y_A),
        'target_positive_rate': np.mean(y_B)
    }


def run_uda_with_visualization(X_source, y_source, X_target, y_target, 
                              feature_names, method_name='TCA'):
    """运行UDA方法并生成可视化"""
    
    from preprocessing.uda_processor import create_uda_processor
    from preprocessing.uda_visualizer import create_uda_visualizer
    
    # 导入TabPFN
    try:
        from tabpfn import TabPFNClassifier
        base_estimator = TabPFNClassifier(n_estimators=32)
        print("✅ 使用TabPFN作为基础估计器")
    except ImportError:
        from sklearn.linear_model import LogisticRegression
        base_estimator = LogisticRegression(penalty=None, random_state=42, max_iter=1000)
        print("⚠ TabPFN不可用，使用LogisticRegression作为fallback")
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/real_data_uda_{method_name}_{timestamp}"
    
    print(f"\n🔬 运行UDA方法: {method_name}")
    print(f"📁 输出目录: {output_dir}")
    
    # 创建UDA处理器，针对医疗数据优化参数
    processor = create_uda_processor(
        method_name=method_name,
        base_estimator=base_estimator,
        save_results=True,
        output_dir=output_dir
    )
    
    # 针对不同方法优化参数
    if method_name == 'TCA':
        # TCA参数优化：针对医疗数据的小样本、高维特征
        processor.config.method_params.update({
            'n_components': None,  # 自动选择最佳组件数
            'mu': 0.1,  # 较小的mu值，减少正则化，适合小样本
            'kernel': 'linear'  # 线性核，适合医疗特征
        })
        print(f"  TCA参数优化: n_components={min(6, len(feature_names)-1)}, mu=0.1, kernel=linear")
    elif method_name == 'SA':
        # SA参数优化
        processor.config.method_params.update({
            'n_components': None  # 自动选择最佳组件数
        })
        print(f"  SA参数优化: n_components=auto")
    
    # 拟合UDA方法
    print("🔧 拟合UDA方法...")
    uda_method, uda_results = processor.fit_transform(
        X_source, y_source, X_target, y_target
    )
    
    print(f"✅ UDA方法拟合完成")
    print(f"性能指标:")
    for metric in ['accuracy', 'auc', 'f1', 'precision', 'recall']:
        if metric in uda_results:
            print(f"  {metric.upper()}: {uda_results[metric]:.4f}")
    
    # 创建可视化器
    visualizer = create_uda_visualizer(
        figsize=(16, 12),
        save_plots=True,
        output_dir=output_dir
    )
    
    # 完整可视化分析
    print(f"\n🎨 生成完整可视化分析...")
    viz_results = visualizer.visualize_domain_adaptation_complete(
        X_source, y_source, X_target, y_target,
        uda_method=uda_method,
        method_name=f"{method_name}_RealData"
    )
    
    return uda_results, viz_results, output_dir


def compare_multiple_uda_methods(X_source, y_source, X_target, y_target, feature_names):
    """对比多种UDA方法在真实数据上的表现"""
    
    print(f"\n🔍 多种UDA方法对比分析")
    print("=" * 50)
    
    methods_to_test = ['TCA', 'SA', 'CORAL']  # TCA优先，因为在医疗数据上表现最佳
    results_summary = {}
    
    for method_name in methods_to_test:
        print(f"\n--- 测试方法: {method_name} ---")
        
        try:
            uda_results, viz_results, output_dir = run_uda_with_visualization(
                X_source, y_source, X_target, y_target, 
                feature_names, method_name
            )
            
            # 提取关键指标
            summary = {
                'method': method_name,
                'accuracy': uda_results.get('accuracy', 0),
                'auc': uda_results.get('auc', 0),
                'f1': uda_results.get('f1', 0),
                'output_dir': output_dir
            }
            
            # 添加域距离改进（优先使用ADAPT指标）
            if 'domain_distances' in viz_results:
                distances = viz_results['domain_distances']
                
                # 使用标准化指标作为主要改进度量
                summary['cov_improvement'] = distances.get('cov_distance_improvement', 0)
                summary['norm_linear_improvement'] = distances.get('normalized_linear_discrepancy_improvement', 0)
                summary['norm_frechet_improvement'] = distances.get('normalized_frechet_distance_improvement', 0)
                
                # 标准化备用指标（优先使用）
                summary['norm_kl_improvement'] = distances.get('normalized_kl_divergence_improvement', 0)
                summary['norm_wasserstein_improvement'] = distances.get('normalized_wasserstein_improvement', 0)
                
                # 原始备用指标
                summary['kl_improvement'] = distances.get('kl_divergence_improvement', 0)
                summary['wasserstein_improvement'] = distances.get('wasserstein_improvement', 0)
                summary['mmd_improvement'] = distances.get('mmd_improvement', 0)
            
            results_summary[method_name] = summary
            
            print(f"✅ {method_name} 完成:")
            print(f"   AUC: {summary['auc']:.4f}")
            print(f"   Accuracy: {summary['accuracy']:.4f}")
            print(f"   F1: {summary['f1']:.4f}")
            
        except Exception as e:
            print(f"❌ {method_name} 失败: {e}")
            results_summary[method_name] = {'method': method_name, 'error': str(e)}
    
    # 生成对比报告
    generate_comparison_report(results_summary)
    
    return results_summary


def generate_comparison_report(results_summary):
    """生成方法对比报告"""
    
    print(f"\n📊 UDA方法对比报告")
    print("=" * 60)
    
    # 过滤成功的方法
    successful_methods = {k: v for k, v in results_summary.items() if 'error' not in v}
    
    if not successful_methods:
        print("❌ 没有成功的方法可供对比")
        return
    
    # 性能对比表格
    print(f"{'方法':<8} {'AUC':<8} {'Accuracy':<10} {'F1':<8} {'协方差改进':<10} {'线性差异改进':<12} {'Frechet改进':<12}")
    print("-" * 78)
    
    best_auc = 0
    best_method = ""
    
    for method, results in successful_methods.items():
        auc = results.get('auc', 0)
        acc = results.get('accuracy', 0)
        f1 = results.get('f1', 0)
        
        # 优先使用ADAPT指标
        cov_imp = results.get('cov_improvement', 0)
        norm_linear_imp = results.get('norm_linear_improvement', 0)
        norm_frechet_imp = results.get('norm_frechet_improvement', 0)
        
        # 如果ADAPT指标不可用，使用标准化备用指标
        if cov_imp == 0 and norm_linear_imp == 0 and norm_frechet_imp == 0:
            # 优先使用标准化版本
            norm_kl_imp = results.get('norm_kl_improvement', 0)
            norm_ws_imp = results.get('norm_wasserstein_improvement', 0)
            
            if norm_kl_imp != 0 or norm_ws_imp != 0:
                # 使用标准化备用指标
                mmd_imp = results.get('mmd_improvement', 0)
                print(f"{method:<8} {auc:<8.4f} {acc:<10.4f} {f1:<8.4f} {norm_kl_imp:<10.4f} {norm_ws_imp:<12.4f} {mmd_imp:<12.4f}")
            else:
                # 使用原始备用指标
                kl_imp = results.get('kl_improvement', 0)
                ws_imp = results.get('wasserstein_improvement', 0)
                mmd_imp = results.get('mmd_improvement', 0)
                print(f"{method:<8} {auc:<8.4f} {acc:<10.4f} {f1:<8.4f} {kl_imp:<10.4f} {ws_imp:<12.4f} {mmd_imp:<12.4f}")
        else:
            print(f"{method:<8} {auc:<8.4f} {acc:<10.4f} {f1:<8.4f} {cov_imp:<10.4f} {norm_linear_imp:<12.4f} {norm_frechet_imp:<12.4f}")
        
        if auc > best_auc:
            best_auc = auc
            best_method = method
    
    print("\n🏆 最佳方法:")
    print(f"   {best_method} (AUC: {best_auc:.4f})")
    
    # 保存对比结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_file = f"results/uda_methods_comparison_{timestamp}.json"
    
    import json
    os.makedirs("results", exist_ok=True)
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"📁 对比结果已保存: {comparison_file}")


def main():
    """主函数"""
    print("🏥 真实医疗数据UDA可视化分析")
    print("=" * 50)
    
    try:
        # 检查环境
        from uda.adapt_methods import is_adapt_available
        
        if not is_adapt_available():
            print("❌ Adapt库不可用，请安装: pip install adapt-python")
            return
        
        print("✅ 环境检查通过")
        
        # 1. 加载真实医疗数据
        X_source, y_source, X_target, y_target, feature_names = load_real_medical_data()
        
        # 2. 分析数据分布
        data_stats = analyze_data_distribution(X_source, y_source, X_target, y_target, feature_names)
        
        # 3. 运行单个最佳方法的完整分析
        print(f"\n🎯 使用最佳方法 (TCA) 进行详细分析")
        uda_results, viz_results, output_dir = run_uda_with_visualization(
            X_source, y_source, X_target, y_target, 
            feature_names, method_name='TCA'
        )
        
        # 输出详细结果
        if 'domain_distances' in viz_results:
            distances = viz_results['domain_distances']
            print(f"\n📊 域适应效果分析:")
            
            # 检查可用的改进指标
            improvement_metrics = []
            
            # 优先使用ADAPT库的指标
            if 'cov_distance_improvement' in distances:
                print(f"  协方差距离改进: {distances['cov_distance_improvement']:.4f}")
                improvement_metrics.append(distances['cov_distance_improvement'])
            
            if 'normalized_linear_discrepancy_improvement' in distances:
                print(f"  标准化线性差异改进: {distances['normalized_linear_discrepancy_improvement']:.4f}")
                improvement_metrics.append(distances['normalized_linear_discrepancy_improvement'])
            
            if 'normalized_frechet_distance_improvement' in distances:
                print(f"  标准化Frechet距离改进: {distances['normalized_frechet_distance_improvement']:.4f}")
                improvement_metrics.append(distances['normalized_frechet_distance_improvement'])
            
            # 标准化备用指标（优先使用）
            if 'normalized_kl_divergence_improvement' in distances:
                print(f"  标准化KL散度改进: {distances['normalized_kl_divergence_improvement']:.4f}")
                improvement_metrics.append(distances['normalized_kl_divergence_improvement'])
            
            if 'normalized_wasserstein_improvement' in distances:
                print(f"  标准化Wasserstein距离改进: {distances['normalized_wasserstein_improvement']:.4f}")
                improvement_metrics.append(distances['normalized_wasserstein_improvement'])
            
            # 原始备用指标（如果标准化版本不可用）
            if 'kl_divergence_improvement' in distances:
                print(f"  KL散度改进: {distances['kl_divergence_improvement']:.4f}")
                improvement_metrics.append(distances['kl_divergence_improvement'])
            
            if 'wasserstein_improvement' in distances:
                print(f"  Wasserstein距离改进: {distances['wasserstein_improvement']:.4f}")
                improvement_metrics.append(distances['wasserstein_improvement'])
            
            if 'mmd_improvement' in distances:
                print(f"  MMD改进: {distances['mmd_improvement']:.4f}")
                improvement_metrics.append(distances['mmd_improvement'])
            
            # 评估改进效果
            if improvement_metrics:
                # 过滤掉NaN值
                valid_improvements = [imp for imp in improvement_metrics if not np.isnan(imp)]
                if valid_improvements:
                    avg_improvement = np.mean(valid_improvements)
                    
                    if avg_improvement > 0:
                        print(f"  ✅ 域适应效果: 良好 (平均改进: {avg_improvement:.4f})")
                    else:
                        print(f"  ⚠ 域适应效果: 有限 (平均改进: {avg_improvement:.4f})")
                else:
                    print(f"  ⚠ 域适应效果: 无法评估 (所有改进指标为NaN)")
            else:
                print(f"  ⚠ 域适应效果: 无可用的改进指标")
        
        print(f"\n📁 详细结果已保存到: {output_dir}")
        
        # 4. 多方法对比（自动运行）
        print(f"\n🔍 自动运行多方法对比分析...")
        comparison_results = compare_multiple_uda_methods(
            X_source, y_source, X_target, y_target, feature_names
        )
        
        print(f"\n✅ 真实医疗数据UDA可视化分析完成！")
        print(f"📁 查看 results/ 目录下的所有可视化结果")
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 