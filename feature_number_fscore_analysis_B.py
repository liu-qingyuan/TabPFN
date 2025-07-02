import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import KMeansSMOTE, SMOTE
from tabpfn import TabPFNClassifier

def evaluate_fscore_features_by_number(
    feature_range=range(3, 21),  # 测试3-20个特征
    device='cuda',
    n_estimators=32,
    random_state=42,
    cv_folds=10,  
    base_path='./results'
):
    """
    分析不同特征数量下F-score特征选择的性能
    """
    # 创建结果目录
    results_dir = f'{base_path}/fscore_feature_number_analysis'
    os.makedirs(results_dir, exist_ok=True)
    
    print("="*60)
    print("F-score特征选择 - 不同特征数量性能分析")
    print("="*60)
    
    # ==============================
    # 1. 数据加载
    # ==============================
    print("加载河南癌症数据集...")
    df = pd.read_excel("data/HenanCancerHospital_translated_english.xlsx")
    
    # 获取所有特征列
    features = [col for col in df.columns if col.startswith('Feature')]
    X = df[features].copy()
    y = df["Label"].copy()
    
    print(f"数据形状: {X.shape}")
    print(f"标签分布: 阳性={y.sum()}, 阴性={len(y)-y.sum()}")
    
    # ==============================
    # 2. 为不同特征数量进行评估
    # ==============================
    results_summary = []
    
    for n_features in feature_range:
        print(f"\n{'='*40}")
        print(f"评估 {n_features} 个特征的性能")
        print(f"{'='*40}")
        
        # F-score特征选择
        selector = SelectKBest(score_func=f_classif, k=n_features)
        X_selected = selector.fit_transform(X, y)
        
        # 获取选中的特征
        selected_features = [features[i] for i in selector.get_support(indices=True)]
        feature_scores = selector.scores_[selector.get_support(indices=True)]
        
        print(f"选择的特征: {selected_features[:]}")
        print(f"平均F-score: {np.mean(feature_scores):.2f}")
        
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
        
        print(f"平均性能: AUC={mean_auc:.4f}±{std_auc:.4f}, ACC={mean_acc:.4f}±{std_acc:.4f}, F1={mean_f1:.4f}±{std_f1:.4f}")
        
        # 保存结果
        results_summary.append({
            'n_features': n_features,
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'mean_f1': mean_f1,
            'std_f1': std_f1,
            'avg_fscore': np.mean(feature_scores),
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
    
    print(f"\n{'='*60}")
    print("性能分析总结")
    print(f"{'='*60}")
    print(f"最佳AUC: {results_df.iloc[best_auc_idx]['mean_auc']:.4f} (特征数: {results_df.iloc[best_auc_idx]['n_features']})")
    print(f"最佳ACC: {results_df.iloc[best_acc_idx]['mean_accuracy']:.4f} (特征数: {results_df.iloc[best_acc_idx]['n_features']})")
    print(f"最佳F1: {results_df.iloc[best_f1_idx]['mean_f1']:.4f} (特征数: {results_df.iloc[best_f1_idx]['n_features']})")
    
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
    print(f"\n{'='*60}")
    print("与RFE预筛选特征对比")
    print(f"{'='*60}")
    
    # RFE的10个特征
    rfe_features = ['Feature63', 'Feature2', 'Feature46', 'Feature61', 
                    'Feature56', 'Feature42', 'Feature39', 'Feature43', 'Feature48', 'Feature5']
    
    # F-score选择的10个特征
    fscore_10_selector = SelectKBest(score_func=f_classif, k=10)
    fscore_10_selector.fit(X, y)
    fscore_10_features = [features[i] for i in fscore_10_selector.get_support(indices=True)]
    
    print(f"RFE预筛选10特征: {rfe_features}")
    print(f"F-score选择10特征: {fscore_10_features}")
    
    # 计算特征重叠
    overlap_features = set(rfe_features) & set(fscore_10_features)
    overlap_ratio = len(overlap_features) / 10
    
    print(f"\n特征重叠分析:")
    print(f"重叠特征: {sorted(list(overlap_features))}")
    print(f"重叠比例: {overlap_ratio:.1%} ({len(overlap_features)}/10)")
    print(f"F-score独有: {sorted(list(set(fscore_10_features) - set(rfe_features)))}")
    print(f"RFE独有: {sorted(list(set(rfe_features) - set(fscore_10_features)))}")
    
    # 找到F-score方法的10特征性能
    fscore_10_result = results_df[results_df['n_features'] == 10].iloc[0]
    print(f"\nF-score 10特征性能: AUC={fscore_10_result['mean_auc']:.4f}±{fscore_10_result['std_auc']:.4f}")
    print(f"最佳F-score性能: AUC={results_df.iloc[best_auc_idx]['mean_auc']:.4f} (特征数: {results_df.iloc[best_auc_idx]['n_features']})")
    
    return results_df

# ==============================
# 运行分析
# ==============================
if __name__ == "__main__":
    print("开始F-score特征数量分析...")
    
    # 运行分析
    results = evaluate_fscore_features_by_number(
        feature_range=range(3, 21),  # 测试3-20个特征
        cv_folds=10,  # 10折交叉验证
        base_path='./results'
    )
    
    print("\n分析完成！结果已保存到 ./results/fscore_feature_number_analysis/") 