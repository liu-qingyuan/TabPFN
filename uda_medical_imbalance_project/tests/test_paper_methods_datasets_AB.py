"""
UDA Medical Imbalance Project - 论文LR方法在数据集A和B上的独立测试

分别在数据集A (AI4health) 和数据集B (HenanCancerHospital) 上独立测试论文LR方法。
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Dict, Tuple, Any

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modeling.paper_methods import PaperLRModel
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')


def load_datasets() -> Dict[str, pd.DataFrame]:
    """加载数据集A和B"""
    print("🔄 加载数据集...")
    
    # 数据集路径（基于项目根目录）
    data_dir = project_root.parent / "data"
    
    dataset_paths = {
        'A': data_dir / "AI4healthcare.xlsx",
        'B': data_dir / "HenanCancerHospital_features63_58.xlsx"
    }
    
    datasets = {}
    
    for name, path in dataset_paths.items():
        if path.exists():
            print(f"  📂 加载数据集{name}: {path.name}")
            df = pd.read_excel(path)
            datasets[name] = df
            print(f"     样本数: {len(df)}, 特征数: {len(df.columns)-1}")
            print(f"     标签分布: {df['Label'].value_counts().to_dict()}")
        else:
            print(f"  ❌ 数据集{name}文件不存在: {path}")
    
    return datasets


def analyze_feature_coverage(datasets: Dict[str, pd.DataFrame]) -> Tuple[list, Dict[str, Dict]]:
    """分析论文方法特征在各数据集中的覆盖情况"""
    print("\n🔍 分析特征兼容性...")
    
    # 论文方法需要的特征
    paper_features = [
        'Feature2', 'Feature5', 'Feature48', 'Feature49', 'Feature50', 
        'Feature52', 'Feature56', 'Feature57', 'Feature61', 'Feature42', 'Feature43'
    ]
    
    print(f"  📋 论文LR方法需要的特征: {paper_features}")
    
    # 检查每个数据集的特征覆盖情况
    feature_coverage = {}
    for name, df in datasets.items():
        available_features = [f for f in paper_features if f in df.columns]
        missing_features = [f for f in paper_features if f not in df.columns]
        
        feature_coverage[name] = {
            'available': available_features,
            'missing': missing_features,
            'coverage_rate': len(available_features) / len(paper_features)
        }
        
        print(f"  📊 数据集{name}:")
        print(f"     可用特征: {len(available_features)}/{len(paper_features)} ({feature_coverage[name]['coverage_rate']:.1%})")
        if missing_features:
            print(f"     缺失特征: {missing_features}")
    
    return paper_features, feature_coverage


def test_paper_lr_on_dataset(dataset_name: str, df: pd.DataFrame, 
                            features: list) -> Dict[str, Any]:
    """在单个数据集上测试论文LR方法"""
    print(f"\n🧪 在数据集{dataset_name}上测试论文LR方法...")
    
    # 准备数据 - 只使用可用的特征
    available_features = [f for f in features if f in df.columns]
    X = df[available_features].copy()
    y = df['Label'].copy()
    
    print(f"  📊 使用特征数: {len(available_features)}/{len(features)}")
    print(f"  📊 样本数: {len(X)}")
    print(f"  📊 正负样本比: {y.value_counts().to_dict()}")
    
    # 创建论文LR模型（预定义系数，不需要训练）
    model = PaperLRModel()
    model.fit(X, y)  # 只是标记为已拟合
    
    # 在全部数据上预测
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    risk_scores = model.get_risk_scores(X)
    
    # 计算性能指标
    metrics = {
        'dataset': dataset_name,
        'accuracy': accuracy_score(y, predictions),
        'auc': roc_auc_score(y, probabilities[:, 1]),
        'f1': f1_score(y, predictions, zero_division=0),
        'precision': precision_score(y, predictions, zero_division=0),
        'recall': recall_score(y, predictions, zero_division=0),
        'risk_score_mean': risk_scores.mean(),
        'risk_score_std': risk_scores.std(),
        'risk_score_min': risk_scores.min(),
        'risk_score_max': risk_scores.max(),
        'features_used': len(available_features),
        'features_missing': len(features) - len(available_features),
        'sample_count': len(X),
        'positive_samples': (y == 1).sum(),
        'negative_samples': (y == 0).sum()
    }
    
    # 计算分类别准确率
    if (y == 0).sum() > 0:
        acc_0 = accuracy_score(y[y == 0], predictions[y == 0])
    else:
        acc_0 = 0.0
    
    if (y == 1).sum() > 0:
        acc_1 = accuracy_score(y[y == 1], predictions[y == 1])
    else:
        acc_1 = 0.0
    
    metrics['acc_negative'] = acc_0
    metrics['acc_positive'] = acc_1
    
    # 显示详细结果
    print(f"  📈 性能指标:")
    print(f"     准确率: {metrics['accuracy']:.4f}")
    print(f"     AUC: {metrics['auc']:.4f}")
    print(f"     F1分数: {metrics['f1']:.4f}")
    print(f"     精确率: {metrics['precision']:.4f}")
    print(f"     召回率: {metrics['recall']:.4f}")
    print(f"     负类准确率: {metrics['acc_negative']:.4f}")
    print(f"     正类准确率: {metrics['acc_positive']:.4f}")
    print(f"  📊 风险评分统计:")
    print(f"     均值: {metrics['risk_score_mean']:.3f}")
    print(f"     标准差: {metrics['risk_score_std']:.3f}")
    print(f"     范围: [{metrics['risk_score_min']:.3f}, {metrics['risk_score_max']:.3f}]")
    
    return metrics


def print_comparison_summary(results: Dict[str, Dict[str, Any]]) -> None:
    """打印数据集A和B的对比摘要"""
    print("\n" + "="*80)
    print("📊 论文LR方法在数据集A和B上的测试结果对比")
    print("="*80)
    
    # 创建对比表格
    if len(results) >= 2:
        comparison_data = []
        
        for dataset_name, metrics in results.items():
            comparison_data.append({
                '数据集': dataset_name,
                '准确率': f"{metrics['accuracy']:.4f}",
                'AUC': f"{metrics['auc']:.4f}", 
                'F1分数': f"{metrics['f1']:.4f}",
                '精确率': f"{metrics['precision']:.4f}",
                '召回率': f"{metrics['recall']:.4f}",
                '负类准确率': f"{metrics['acc_negative']:.4f}",
                '正类准确率': f"{metrics['acc_positive']:.4f}",
                '风险评分均值': f"{metrics['risk_score_mean']:.3f}",
                '使用特征数': f"{metrics['features_used']}/{metrics['features_used'] + metrics['features_missing']}",
                '总样本数': metrics['sample_count']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        # 性能对比分析
        print(f"\n📈 性能对比分析:")
        
        dataset_names = list(results.keys())
        if len(dataset_names) == 2:
            dataset_A, dataset_B = dataset_names[0], dataset_names[1]
            metrics_A, metrics_B = results[dataset_A], results[dataset_B]
            
            auc_diff = metrics_A['auc'] - metrics_B['auc']
            acc_diff = metrics_A['accuracy'] - metrics_B['accuracy']
            f1_diff = metrics_A['f1'] - metrics_B['f1']
            
            print(f"  🎯 {dataset_A} vs {dataset_B}:")
            print(f"     AUC差异: {auc_diff:+.4f} ({'A更好' if auc_diff > 0 else 'B更好' if auc_diff < 0 else '相当'})")
            print(f"     准确率差异: {acc_diff:+.4f} ({'A更好' if acc_diff > 0 else 'B更好' if acc_diff < 0 else '相当'})")
            print(f"     F1差异: {f1_diff:+.4f} ({'A更好' if f1_diff > 0 else 'B更好' if f1_diff < 0 else '相当'})")
            
            # 特征可用性对比
            features_A = metrics_A['features_used']
            features_B = metrics_B['features_used']
            print(f"  🔧 特征可用性:")
            print(f"     数据集A可用特征: {features_A}/11")
            print(f"     数据集B可用特征: {features_B}/11")
            
            # 样本分布对比
            print(f"  📊 样本分布:")
            print(f"     数据集A: 正样本{metrics_A['positive_samples']}, 负样本{metrics_A['negative_samples']}")
            print(f"     数据集B: 正样本{metrics_B['positive_samples']}, 负样本{metrics_B['negative_samples']}")


def save_results(results: Dict[str, Dict[str, Any]], output_dir: str) -> None:
    """保存测试结果"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 保存结果到: {output_path}")
    
    # 保存详细结果表格
    summary_data = []
    for dataset_name, metrics in results.items():
        summary_data.append({
            'dataset': dataset_name,
            **metrics
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = output_path / "paper_lr_results_datasets_AB.csv"
    summary_df.to_csv(summary_path, index=False, encoding='utf-8')
    print(f"  📄 详细结果保存至: {summary_path}")
    
    # 保存JSON格式的结果
    import json
    json_path = output_path / "paper_lr_results_datasets_AB.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"  📄 JSON结果保存至: {json_path}")


def main() -> Dict[str, Dict[str, Any]]:
    """主函数"""
    print("🚀 论文LR方法在数据集A和B上的独立测试")
    print("="*80)
    
    # 加载数据集
    datasets = load_datasets()
    
    if not datasets:
        print("❌ 未找到可用的数据集文件")
        return {}
    
    # 分析特征覆盖情况
    paper_features, feature_coverage = analyze_feature_coverage(datasets)
    
    results = {}
    
    # 在各数据集上独立测试论文LR方法
    for name, df in datasets.items():
        if feature_coverage[name]['coverage_rate'] > 0:
            metrics = test_paper_lr_on_dataset(name, df, paper_features)
            results[f'数据集{name}'] = metrics
        else:
            print(f"⚠️  数据集{name}缺失所有必要特征，跳过测试")
    
    # 显示对比摘要
    if results:
        print_comparison_summary(results)
        
        # 保存结果
        save_results(results, "tests/results_paper_lr_AB")
        
        print("\n✅ 测试完成!")
    else:
        print("\n❌ 无可用测试结果")
    
    return results


if __name__ == "__main__":
    results = main()
    
    # 为方便调试，输出关键指标
    if results:
        print(f"\n🎯 关键指标摘要:")
        for name, metrics in results.items():
            print(f"  {name}: AUC={metrics['auc']:.4f}, 准确率={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}") 