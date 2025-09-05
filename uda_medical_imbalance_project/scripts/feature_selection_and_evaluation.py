#!/usr/bin/env python3
"""
统一RFE特征选择和性能评估脚本 (使用AB交集58个特征)

这个脚本结合了以下功能：
1. predict_healthcare_RFE.py - 使用TabPFN进行RFE特征选择
2. evaluate_feature_numbers.py - 跨不同特征数量的性能评估

特征集说明：
- 使用A数据集（AI4healthcare.xlsx）
- 仅使用AB交集的58个特征（移除Feature12, Feature33, Feature34, Feature36, Feature40）
- 评估范围：3-58个特征（生成56行结果数据）

运行示例: python scripts/feature_selection_and_evaluation.py

Author: Generated for UDA Medical Imbalance Project
Date: 2024
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入项目配置
from config.settings import get_features_by_type, SELECTED_58_FEATURES

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance
from tabpfn import TabPFNClassifier
from types import SimpleNamespace
from tqdm import tqdm
import torch


class TabPFNWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper class to make TabPFN compatible with sklearn's RFE
    """
    _estimator_type = "classifier"
    
    def __sklearn_tags__(self):
        return SimpleNamespace(
            estimator_type="classifier",
            binary_only=True,
            classifier_tags=SimpleNamespace(poor_score=False),
            regressor_tags=SimpleNamespace(poor_score=False),
            input_tags=SimpleNamespace(sparse=False, allow_nan=True),
            target_tags=SimpleNamespace(required=True)
        )

    def __init__(self, device='cuda', n_estimators=32, softmax_temperature=0.9,
                 balance_probabilities=False, average_before_softmax=False,
                 ignore_pretraining_limits=True, random_state=42,
                 n_repeats=5):
        self.device = device
        self.n_estimators = n_estimators
        self.softmax_temperature = softmax_temperature
        self.balance_probabilities = balance_probabilities
        self.average_before_softmax = average_before_softmax
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.random_state = random_state
        self.n_repeats = n_repeats

    def fit(self, X, y):
        # Set class information for scoring functions
        self.classes_ = np.unique(y)
        
        # Initialize TabPFN model
        self.model_ = TabPFNClassifier(
            device=self.device,
            n_estimators=self.n_estimators,
            softmax_temperature=self.softmax_temperature,
            balance_probabilities=self.balance_probabilities,
            average_before_softmax=self.average_before_softmax,
            ignore_pretraining_limits=self.ignore_pretraining_limits,
            random_state=self.random_state
        )
        self.model_.fit(X, y)
        
        # Calculate feature importance using permutation importance
        result = permutation_importance(
            self, X, y, 
            scoring='roc_auc',
            n_repeats=self.n_repeats,
            random_state=self.random_state
        )
        self.feature_importances_ = result.importances_mean
        self.feature_importances_std_ = result.importances_std
        
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)
    
    def score(self, X, y):
        y_proba = self.predict_proba(X)[:, 1]
        return roc_auc_score(y, y_proba)
    
    def get_feature_importance(self):
        """Return feature importance scores"""
        return self.feature_importances_

    def get_feature_importance_scores(self):
        """Return feature importance scores and standard deviations"""
        return {
            'mean': self.feature_importances_,
            'std': self.feature_importances_std_
        }


def select_features_rfe(X, y, n_features=3):
    """
    Use RFE with TabPFN as base estimator for feature selection
    """
    n_features_total = X.shape[1]
    n_iterations = n_features_total - n_features
    
    # Initialize TabPFN wrapper
    base_model = TabPFNWrapper(
        device='cuda',
        n_estimators=32,
        softmax_temperature=0.9,
        balance_probabilities=False,
        average_before_softmax=False,
        ignore_pretraining_limits=True,
        random_state=42
    )
    
    # Initialize RFE
    rfe = RFE(
        estimator=base_model,
        n_features_to_select=n_features,
        step=1,
        verbose=2
    )
    
    # Create progress bar and fit RFE
    print("Fitting RFE with TabPFN as base model...")
    with tqdm(total=n_iterations, desc='Eliminating features') as pbar:
        rfe.fit(X, y)
        pbar.update(n_iterations)
    
    # Get selected features
    selected_features = X.columns[rfe.support_].tolist()
    
    # Get feature importance ranking
    feature_ranking = pd.DataFrame({
        'Feature': X.columns,
        'Rank': rfe.ranking_
    }).sort_values('Rank')
    
    return selected_features, feature_ranking


def evaluate_feature_performance(X, y, feature_ranking, results_dir):
    """
    Evaluate model performance across different numbers of features
    Using the complete RFE ranking from most to least important features
    """
    print("\n" + "="*60)
    print("🔄 第二阶段：跨特征数量性能评估")
    print("="*60)
    
    # Get ranked features (sorted by RFE rank: 1=most important, 63=least important)
    ranked_features = feature_ranking.sort_values('Rank')['Feature'].tolist()
    
    print(f"📋 使用RFE排序: {ranked_features[:3]}...{ranked_features[-3:]}")
    print(f"📊 可用特征总数: {len(ranked_features)}")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define feature numbers to test (3 to 58, AB交集特征)
    feature_numbers = list(range(3, len(ranked_features) + 1))
    print(f"🎯 将评估特征数: {feature_numbers[0]} 到 {feature_numbers[-1]} (共{len(feature_numbers)}次评估)")
    print(f"⏰ 预计总用时: {len(feature_numbers) * 2:.1f}-{len(feature_numbers) * 4:.1f}分钟 (相比63特征约节省15%时间)")
    print("📝 使用10折交叉验证评估每个特征组合...")
    print("📊 生成结果：3-58特征性能对比数据")
    
    
    # Store all results
    all_results = []
    
    # Evaluate each feature count
    for n_features in tqdm(feature_numbers, desc=f"Evaluating features (3-{len(ranked_features)})"):
        # Select top n features according to RFE ranking (rank 1 = most important)
        selected_features = ranked_features[:n_features]
        X_selected = X[selected_features]
        
        # 只在关键里程碑打印详细信息以减少输出冗余
        if n_features <= 10 or n_features % 10 == 0:
            print(f"\n🔍 评估 {n_features} 个特征...")
            print(f"前{min(5, len(selected_features))}个特征: {selected_features[:5]}")
        
        # 10-fold cross validation
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        fold_scores = []
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X_selected), 1):
            X_train, X_test = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train model
            start_time = time.time()
            clf = TabPFNClassifier(
                device='cuda',
                n_estimators=32,
                softmax_temperature=0.9,
                balance_probabilities=False,
                average_before_softmax=False,
                ignore_pretraining_limits=True,
                random_state=42
            )
            
            # Ensure reproducible results
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)
                
            clf.fit(X_train, y_train)
            
            # Make predictions
            y_pred = clf.predict(X_test)
            y_pred_proba = clf.predict_proba(X_test)
            fold_time = time.time() - start_time
            
            # Calculate metrics
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            f1 = f1_score(y_test, y_pred)
            
            # Calculate per-class accuracy
            conf_matrix = confusion_matrix(y_test, y_pred)
            acc_0 = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
            acc_1 = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
            
            fold_scores.append({
                'fold': fold,
                'accuracy': acc,
                'auc': auc,
                'f1': f1,
                'acc_0': acc_0,
                'acc_1': acc_1,
                'time': fold_time
            })
        
        # Calculate mean and std
        metrics = ['accuracy', 'auc', 'f1', 'acc_0', 'acc_1', 'time']
        mean_scores = {f'mean_{m}': np.mean([s[m] for s in fold_scores]) for m in metrics}
        std_scores = {f'std_{m}': np.std([s[m] for s in fold_scores]) for m in metrics}
        
        # Store result
        result = {
            'n_features': n_features,
            'features': ', '.join(selected_features),
            **mean_scores,
            **std_scores
        }
        all_results.append(result)
        
        # 打印当前结果 (简化输出)
        if n_features <= 10 or n_features % 10 == 0:
            print(f"✅ {n_features}个特征结果:")
            print(f"   AUC: {mean_scores['mean_auc']:.4f}±{std_scores['std_auc']:.4f}")
            print(f"   准确率: {mean_scores['mean_accuracy']:.4f}±{std_scores['std_accuracy']:.4f}")
            print(f"   F1: {mean_scores['mean_f1']:.4f}±{std_scores['std_f1']:.4f}")
        else:
            # 中间结果的快速摘要
            print(f"⚡ N={n_features}: AUC={mean_scores['mean_auc']:.3f}")
    
    # 保存结果到CSV
    results_df = pd.DataFrame(all_results)
    csv_path = results_dir / "feature_number_comparison.csv"
    results_df.to_csv(csv_path, index=False)
    
    print(f"\n📊 性能评估结果已保存: {csv_path}")
    
    # 创建可视化
    create_performance_visualization(all_results, feature_numbers, results_dir)
    
    return results_df


def create_performance_visualization(all_results, feature_numbers, results_dir):
    """
    Create performance comparison visualization
    """
    plt.figure(figsize=(15, 10))
    metrics = ['auc', 'accuracy', 'f1']
    colors = ['blue', 'green', 'red']
    
    for metric, color in zip(metrics, colors):
        mean_values = [result[f'mean_{metric}'] for result in all_results]
        std_values = [result[f'std_{metric}'] for result in all_results]
        
        plt.plot(feature_numbers, mean_values, marker='o', color=color, label=metric.upper())
        plt.fill_between(
            feature_numbers,
            [m - s for m, s in zip(mean_values, std_values)],
            [m + s for m, s in zip(mean_values, std_values)],
            color=color,
            alpha=0.2
        )
    
    plt.xlabel('Number of Features')
    plt.ylabel('Score')
    plt.title('Model Performance vs Number of Features')
    plt.legend()
    plt.grid(True)
    
    # 保存图表
    plot_path = results_dir / "performance_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 可视化图表已保存: {plot_path}")


def main():
    """主函数"""
    print("🧬 统一RFE特征选择和性能评估")
    print("=" * 60)
    
    # 创建时间戳输出目录 (标注使用58个特征)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = project_root / "results" / f"feature_selection_evaluation_58features_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 数据路径配置 (基于loader.py的路径设置)
    data_path = "/home/24052432g/TabPFN/data/AI4healthcare.xlsx"
    
    # 加载数据
    print(f"\n📂 加载数据...")
    print(f"数据路径: {data_path}")
    
    try:
        df = pd.read_excel(data_path)
        
        # 使用AB交集的58个特征 (移除Feature12, Feature33, Feature34, Feature36, Feature40)
        required_features = get_features_by_type('selected58')
        
        # 验证A数据集是否包含所需的58个特征
        available_features = [f for f in required_features if f in df.columns]
        missing_features = [f for f in required_features if f not in df.columns]
        
        if missing_features:
            print(f"⚠️ 警告：A数据集中缺失以下特征: {missing_features}")
            print(f"将使用可用的{len(available_features)}个特征进行分析")
        else:
            print(f"✅ A数据集包含所有58个AB交集特征")
            
        X = df[available_features].copy()
        y = df["Label"].copy()
        
        print(f"✅ 数据加载成功")
        print(f"   样本数: {X.shape[0]}")
        print(f"   特征数: {X.shape[1]} (AB交集特征)")
        print(f"   特征范围: {available_features[0]} 到 {available_features[-1]}")
        print(f"   标签分布: {y.value_counts().to_dict()}")
        print(f"   移除的特征: Feature12, Feature33, Feature34, Feature36, Feature40")
        
    except FileNotFoundError:
        print(f"❌ 数据文件未找到: {data_path}")
        print("请确保数据文件路径正确")
        return None, None
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return None, None
    
    # Phase 1: RFE特征选择
    print("\n" + "="*60)
    print("🔬 第一阶段：基于TabPFN的RFE特征选择")
    print("="*60)
    
    print("🧠 使用TabPFN执行递归特征消除(RFE)...")
    print("📋 这将生成AB交集58个特征的完整重要性排序")
    print("⏰ 预计用时：4-8分钟 (取决于GPU性能，相比63特征略快)")
    
    try:
        # 执行RFE特征选择，选择3个最优特征但获得完整排序
        selected_features, feature_ranking = select_features_rfe(X, y, n_features=3)
        
        # 保存特征排序结果 (58个特征)
        ranking_path = results_dir / "RFE_feature_ranking_58features.csv"
        feature_ranking.to_csv(ranking_path, index=False)
        
        print(f"✅ RFE特征选择完成")
        
    except Exception as e:
        print(f"❌ RFE特征选择失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
    print(f"📁 完整特征排序已保存: {ranking_path}")
    print(f"📊 RFE处理: 从AB交集{X.shape[1]}个特征开始，逐步消除到3个特征")
    print(f"📋 排序说明: Rank 1 = 最重要 (最后保留), Rank {X.shape[1]} = 最不重要 (最先消除)")
    print(f"🗑️ 已排除的特征: Feature12, Feature33, Feature34, Feature36, Feature40")
    
    print("\n🏆 Top 10 最重要特征 (Rank 1-10):")
    print(feature_ranking.head(10).to_string(index=False))
    print("\n🗑️ Bottom 10 最不重要特征:")
    print(feature_ranking.tail(10).to_string(index=False))
    
    # 验证RFE逻辑: 选中的3个特征应该rank为1,2,3
    print(f"\n✅ 验证 - 选中的3个特征及其排序:")
    selected_feature_ranks = feature_ranking[feature_ranking['Feature'].isin(selected_features)].sort_values('Rank')
    print(selected_feature_ranks.to_string(index=False))
    
    # Phase 2: 性能评估
    print(f"\n🔄 第二阶段：使用RFE排序进行3-58特征性能评估...")
    try:
        results_df = evaluate_feature_performance(X, y, feature_ranking, results_dir)
    except Exception as e:
        print(f"❌ 性能评估失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
    # 总结
    print("\n" + "="*60)
    print("🎉 执行完成")
    print("="*60)
    
    print("\n📁 生成的文件:")
    print(f"1. 特征排序(58特征): {ranking_path}")
    print(f"2. 性能结果(3-58特征): {results_dir / 'feature_number_comparison.csv'}")
    print(f"3. 性能图表: {results_dir / 'performance_comparison.png'}")
    print(f"4. 结果数据行数: {len(results_df)} 行 (从3个特征到58个特征)")
    
    # 找到最佳性能的特征数量
    best_auc_idx = results_df['mean_auc'].idxmax()
    best_result = results_df.iloc[best_auc_idx]
    
    print(f"\n🏆 最佳性能摘要:")
    print(f"最佳特征数量: {best_result['n_features']}")
    print(f"最佳AUC: {best_result['mean_auc']:.4f} ± {best_result['std_auc']:.4f}")
    print(f"最佳准确率: {best_result['mean_accuracy']:.4f} ± {best_result['std_accuracy']:.4f}")
    print(f"最佳F1分数: {best_result['mean_f1']:.4f} ± {best_result['std_f1']:.4f}")
    
    print(f"\n📂 结果目录: {results_dir}")
    
    return results_df, feature_ranking


if __name__ == "__main__":
    results_df, feature_ranking = main()