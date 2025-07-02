import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import logging
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.feature_selection import RFE
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import KMeansSMOTE, SMOTE
from tabpfn import TabPFNClassifier
from types import SimpleNamespace

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('rfe_analysis_B.log')
    ]
)

class TabPFNWrapper(BaseEstimator, ClassifierMixin):
    """TabPFN包装器，使其兼容sklearn的RFE接口"""
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
                 ignore_pretraining_limits=True, random_state=42, n_repeats=3):
        self.device = device
        self.n_estimators = n_estimators
        self.softmax_temperature = softmax_temperature
        self.balance_probabilities = balance_probabilities
        self.average_before_softmax = average_before_softmax
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.random_state = random_state
        self.n_repeats = n_repeats

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        
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
        
        # 使用置换重要性计算特征重要性
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

def evaluate_rfe_features_by_number(
    feature_range=range(3, 21),  # 测试3-20个特征
    device='cuda',
    n_estimators=32,
    random_state=42,
    cv_folds=10,  
    base_path='./results'
):
    """
    分析不同特征数量下RFE特征选择的性能（在河南数据集B上）
    """
    # 创建结果目录
    results_dir = f'{base_path}/rfe_feature_number_analysis_B'
    os.makedirs(results_dir, exist_ok=True)
    
    logging.info("="*60)
    logging.info("RFE特征选择 - 河南数据集(B) 不同特征数量性能分析")
    logging.info("="*60)
    
    # ==============================
    # 1. 数据加载
    # ==============================
    logging.info("加载河南癌症数据集...")
    df = pd.read_excel("data/HenanCancerHospital_translated_english.xlsx")
    
    # 获取所有特征列
    features = [col for col in df.columns if col.startswith('Feature')]
    X = df[features].copy()
    y = df["Label"].copy()
    
    logging.info(f"数据形状: {X.shape}")
    logging.info(f"标签分布: 阳性={y.sum()}, 阴性={len(y)-y.sum()}")
    
    # ==============================
    # 2. 为不同特征数量进行RFE评估
    # ==============================
    results_summary = []
    
    for n_features in feature_range:
        logging.info(f"\n{'='*40}")
        logging.info(f"RFE评估 {n_features} 个特征的性能")
        logging.info(f"{'='*40}")
        
        # 使用RFE进行特征选择
        base_model = TabPFNWrapper(
            device=device,
            n_estimators=n_estimators,
            random_state=random_state,
            n_repeats=2  # 减少计算时间
        )
        
        # RFE选择器
        rfe_selector = RFE(
            estimator=base_model,
            n_features_to_select=n_features,
            step=1,
            verbose=0
        )
        
        logging.info(f"执行RFE选择 {n_features} 个特征...")
        rfe_selector.fit(X.values, y.values)
        
        # 获取选中的特征
        selected_features = [features[i] for i in range(len(features)) if rfe_selector.support_[i]]
        feature_rankings = rfe_selector.ranking_
        
        # 获取选择特征的数据
        X_selected = X[selected_features].values
        
        logging.info(f"RFE选择的特征: {selected_features}")
        logging.info(f"特征重要性排名前3: {[features[i] for i in np.argsort(feature_rankings)[:3]]}")
        
        # 交叉验证评估
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X_selected), 1):
            # 划分训练测试集
            X_train_fold = X_selected[train_idx]
            X_test_fold = X_selected[test_idx]
            y_train_fold = y.iloc[train_idx]
            y_test_fold = y.iloc[test_idx]
            
            # 数据平衡处理
            try:
                # 先尝试KMeans-SMOTE
                smote = KMeansSMOTE(
                    k_neighbors=min(4, min(y_train_fold.sum(), len(y_train_fold)-y_train_fold.sum())-1),
                    cluster_balance_threshold='auto',
                    random_state=random_state,
                    n_jobs=1
                )
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train_fold, y_train_fold)
            except:
                # 降级到SMOTE
                try:
                    smote = SMOTE(
                        k_neighbors=min(4, min(y_train_fold.sum(), len(y_train_fold)-y_train_fold.sum())-1),
                        random_state=random_state
                    )
                    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_fold, y_train_fold)
                except:
                    # 如果SMOTE也失败，使用原始数据
                    X_train_resampled, y_train_resampled = X_train_fold, y_train_fold
            
            # 训练TabPFN模型
            clf = TabPFNClassifier(
                device=device,
                n_estimators=n_estimators,
                random_state=random_state
            )
            clf.fit(X_train_resampled, y_train_resampled)
            
            # 预测
            y_pred = clf.predict(X_test_fold)
            y_pred_proba = clf.predict_proba(X_test_fold)
            
            # 计算指标
            acc = accuracy_score(y_test_fold, y_pred)
            auc = roc_auc_score(y_test_fold, y_pred_proba[:, 1])
            f1 = f1_score(y_test_fold, y_pred)
            
            fold_results.append({
                'fold': fold,
                'accuracy': acc,
                'auc': auc,
                'f1': f1
            })
        
        # 计算平均性能
        fold_df = pd.DataFrame(fold_results)
        mean_auc = fold_df['auc'].mean()
        std_auc = fold_df['auc'].std()
        mean_acc = fold_df['accuracy'].mean()
        std_acc = fold_df['accuracy'].std()
        mean_f1 = fold_df['f1'].mean()
        std_f1 = fold_df['f1'].std()
        
        logging.info(f"平均性能: AUC={mean_auc:.4f}±{std_auc:.4f}, ACC={mean_acc:.4f}±{std_acc:.4f}, F1={mean_f1:.4f}±{std_f1:.4f}")
        
        # 保存结果
        results_summary.append({
            'n_features': n_features,
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'mean_f1': mean_f1,
            'std_f1': std_f1,
            'avg_ranking': np.mean(feature_rankings[:n_features]),
            'top3_features': ', '.join(selected_features[:3])
        })
    
    # ==============================
    # 3. 结果分析和可视化
    # ==============================
    results_df = pd.DataFrame(results_summary)
    
    # 保存详细结果
    results_df.to_csv(f'{results_dir}/fscore_feature_number_results.csv', index=False)
    
    # 找到最佳性能
    best_auc_idx = results_df['mean_auc'].idxmax()
    best_acc_idx = results_df['mean_accuracy'].idxmax()
    best_f1_idx = results_df['mean_f1'].idxmax()
    
    logging.info(f"\n{'='*60}")
    logging.info("性能分析总结")
    logging.info(f"{'='*60}")
    logging.info(f"最佳AUC: {results_df.iloc[best_auc_idx]['mean_auc']:.4f} (特征数: {results_df.iloc[best_auc_idx]['n_features']})")
    logging.info(f"最佳ACC: {results_df.iloc[best_acc_idx]['mean_accuracy']:.4f} (特征数: {results_df.iloc[best_acc_idx]['n_features']})")
    logging.info(f"最佳F1: {results_df.iloc[best_f1_idx]['mean_f1']:.4f} (特征数: {results_df.iloc[best_f1_idx]['n_features']})")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 子图1: AUC vs 特征数量
    axes[0,0].errorbar(results_df['n_features'], results_df['mean_auc'], 
                       yerr=results_df['std_auc'], marker='o', capsize=5)
    axes[0,0].set_title('AUC vs Number of Features (F-Score Selection)')
    axes[0,0].set_xlabel('Number of Features')
    axes[0,0].set_ylabel('AUC')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].axvline(x=results_df.iloc[best_auc_idx]['n_features'], color='r', linestyle='--', alpha=0.7, label='Best AUC')
    axes[0,0].legend()
    
    # 子图2: Accuracy vs 特征数量
    axes[0,1].errorbar(results_df['n_features'], results_df['mean_accuracy'], 
                       yerr=results_df['std_accuracy'], marker='s', capsize=5, color='orange')
    axes[0,1].set_title('Accuracy vs Number of Features (F-Score Selection)')
    axes[0,1].set_xlabel('Number of Features')
    axes[0,1].set_ylabel('Accuracy')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].axvline(x=results_df.iloc[best_acc_idx]['n_features'], color='r', linestyle='--', alpha=0.7, label='Best ACC')
    axes[0,1].legend()
    
    # 子图3: F1 vs 特征数量
    axes[1,0].errorbar(results_df['n_features'], results_df['mean_f1'], 
                       yerr=results_df['std_f1'], marker='^', capsize=5, color='green')
    axes[1,0].set_title('F1-Score vs Number of Features (F-Score Selection)')
    axes[1,0].set_xlabel('Number of Features')
    axes[1,0].set_ylabel('F1-Score')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].axvline(x=results_df.iloc[best_f1_idx]['n_features'], color='r', linestyle='--', alpha=0.7, label='Best F1')
    axes[1,0].legend()
    
    # 子图4: 三个指标对比
    axes[1,1].plot(results_df['n_features'], results_df['mean_auc'], 'o-', label='AUC', linewidth=2)
    axes[1,1].plot(results_df['n_features'], results_df['mean_accuracy'], 's-', label='Accuracy', linewidth=2)
    axes[1,1].plot(results_df['n_features'], results_df['mean_f1'], '^-', label='F1-Score', linewidth=2)
    axes[1,1].set_title('All Metrics vs Number of Features')
    axes[1,1].set_xlabel('Number of Features')
    axes[1,1].set_ylabel('Score')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/fscore_feature_number_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ==============================
    # 4. 与RFE方法对比
    # ==============================
    logging.info(f"\n{'='*60}")
    logging.info("与RFE预筛选特征对比")
    logging.info(f"{'='*60}")
    
    # RFE的10个特征
    rfe_features = ['Feature63', 'Feature2', 'Feature46', 'Feature61', 
                    'Feature56', 'Feature42', 'Feature39', 'Feature43', 'Feature48', 'Feature5']
    
    # 河南数据集B上的RFE选择的10个特征
    rfe_b_selector = RFE(
        estimator=TabPFNWrapper(device='cuda', random_state=42, n_repeats=2),
        n_features_to_select=10,
        step=1,
        verbose=0
    )
    rfe_b_selector.fit(X.values, y.values)
    rfe_b_features = [features[i] for i in range(len(features)) if rfe_b_selector.support_[i]]
    
    logging.info(f"A数据集RFE预筛选10特征: {rfe_features}")
    logging.info(f"B数据集RFE选择10特征: {rfe_b_features}")
    
    # 计算特征重叠
    overlap_features = set(rfe_features) & set(rfe_b_features)
    overlap_ratio = len(overlap_features) / 10
    
    logging.info(f"\n特征重叠分析:")
    logging.info(f"重叠特征: {sorted(list(overlap_features))}")
    logging.info(f"重叠比例: {overlap_ratio:.1%} ({len(overlap_features)}/10)")
    logging.info(f"B数据集RFE独有: {sorted(list(set(rfe_b_features) - set(rfe_features)))}")
    logging.info(f"A数据集RFE独有: {sorted(list(set(rfe_features) - set(rfe_b_features)))}")
    
    # 找到RFE方法的10特征性能
    rfe_10_result = results_df[results_df['n_features'] == 10].iloc[0]
    logging.info(f"\nB数据集RFE 10特征性能: AUC={rfe_10_result['mean_auc']:.4f}±{rfe_10_result['std_auc']:.4f}")
    logging.info(f"B数据集最佳RFE性能: AUC={results_df.iloc[best_auc_idx]['mean_auc']:.4f} (特征数: {results_df.iloc[best_auc_idx]['n_features']})")
    
    return results_df

# ==============================
# 运行分析
# ==============================
if __name__ == "__main__":
    logging.info("开始河南数据集B的RFE特征数量分析...")
    
    # 运行分析
    results = evaluate_rfe_features_by_number(
        feature_range=range(3, 16),  # 测试3-15个特征（RFE计算较慢）
        cv_folds=5,  # 5折交叉验证（加快计算）
        base_path='./results'
    )
    
    logging.info("\n分析完成！结果已保存到 ./results/rfe_feature_number_analysis_B/") 