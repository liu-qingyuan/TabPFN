import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Tuple, Dict, Any
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import KMeansSMOTE, SMOTE
from tabpfn import TabPFNClassifier

def analyze_with_kmeans_smote_and_fscore_selection(
    device: str = 'cuda',
    n_estimators: int = 32,
    softmax_temperature: float = 0.9,
    balance_probabilities: bool = False,
    average_before_softmax: bool = False,
    ignore_pretraining_limits: bool = True,
    random_state: int = 42,
    k_best_features: int = 10,
    base_path: str = './results'
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    使用F-score特征选择 + KMeans-SMOTE + TabPFN进行癌症预测分析
    
    Args:
        device: 设备类型 ('cuda' 或 'cpu')
        n_estimators: TabPFN估计器数量
        softmax_temperature: Softmax温度参数
        balance_probabilities: 是否平衡概率
        average_before_softmax: 是否在softmax前平均
        ignore_pretraining_limits: 是否忽略预训练限制
        random_state: 随机种子
        k_best_features: 选择的最佳特征数量
        base_path: 结果保存路径
        
    Returns:
        交叉验证结果DataFrame和分析数据字典
    """
    # 创建结果目录
    os.makedirs(base_path, exist_ok=True)
    
    # ==============================
    # 1. 数据加载和预处理
    # ==============================
    print("加载河南癌症数据集...")
    df = pd.read_excel("data/HenanCancerHospital_translated_english.xlsx")
    
    # 获取所有特征列
    features = [col for col in df.columns if col.startswith('Feature')]
    X = df[features].copy()
    y = df["Label"].copy()
    
    print(f"原始数据形状: {X.shape}")
    print(f"原始标签分布: 阳性(癌症)={y.sum()}, 阴性(健康)={len(y)-y.sum()}")
    print(f"原始阳性比例: {y.mean():.1%}")
    
    # ==============================
    # 2. F-score特征选择 - 自动选择最佳K个特征
    # ==============================
    print(f"\n{'='*50}")
    print(f"F-score特征选择 - 自动选择最佳 {k_best_features} 个特征")
    print(f"{'='*50}")
    
    # 使用 F-score 进行特征选择
    selector = SelectKBest(score_func=f_classif, k=k_best_features)
    X_selected = selector.fit_transform(X, y)
    
    # 获取选中的特征名称和分数
    selected_features = [features[i] for i in selector.get_support(indices=True)]
    feature_scores = selector.scores_[selector.get_support(indices=True)]
    
    print(f"F-score自动选择的 {k_best_features} 个最佳特征:")
    for i, (feature, score) in enumerate(zip(selected_features, feature_scores)):
        print(f"  {i+1:2d}. {feature}: F-score={score:.2f}")
    
    # ==============================
    # 3. 10折交叉验证 (在每折训练集上单独应用KMeans-SMOTE)
    # ==============================
    print(f"\n{'='*50}")
    print("10折交叉验证 (每折训练集单独应用KMeans-SMOTE)")
    print(f"{'='*50}")
    
    kf = KFold(n_splits=10, shuffle=True, random_state=random_state)
    
    # 存储每个样本的预测结果
    all_predictions = []
    all_probabilities = []
    all_true_labels = []
    all_indices = []
    
    overall_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_selected), 1):
        print(f"处理 Fold {fold}...")
        
        # 使用选择的特征数据进行划分
        X_train_fold, X_test_fold = X_selected[train_idx], X_selected[test_idx]
        y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
        
        print(f"  原始训练集: 阳性={y_train_fold.sum()}, 阴性={len(y_train_fold)-y_train_fold.sum()}")
        
        # 仅在训练集上应用KMeans-SMOTE，调整参数以处理小样本问题
        try:
            kmeans_smote_fold = KMeansSMOTE(
                k_neighbors=4,  # 邻居数量
                cluster_balance_threshold='auto',  # 自动调整聚类平衡阈值
                random_state=random_state,
                n_jobs=1  # 单线程避免并发问题
            )
        except Exception as e:
            print(f"  ⚠️  KMeans-SMOTE失败，改用SMOTE: {e}")
            kmeans_smote_fold = SMOTE(
                k_neighbors=4,
                random_state=random_state
            )
        
        X_train_resampled, y_train_resampled = kmeans_smote_fold.fit_resample(X_train_fold, y_train_fold)
        X_train_resampled = np.array(X_train_resampled)
        y_train_resampled = np.array(y_train_resampled)
        
        print(f"  平衡后训练集: 阳性={y_train_resampled.sum()}, 阴性={len(y_train_resampled)-y_train_resampled.sum()}")
        
        # 训练模型 (使用平衡后的训练集)
        clf = TabPFNClassifier(
            device=device,
            n_estimators=n_estimators,
            softmax_temperature=softmax_temperature,
            balance_probabilities=balance_probabilities,
            average_before_softmax=average_before_softmax,
            ignore_pretraining_limits=ignore_pretraining_limits,
            random_state=random_state
        )
        clf.fit(X_train_resampled, y_train_resampled)
        
        # 在原始测试集上预测 (重要：不使用平衡后的测试集)
        y_pred = clf.predict(X_test_fold)
        y_pred_proba = clf.predict_proba(X_test_fold)
        
        # 保存预测结果
        all_predictions.extend(y_pred)
        all_probabilities.extend(y_pred_proba[:, 1])
        all_true_labels.extend(y_test_fold.values)
        all_indices.extend(test_idx)
        
        # 计算fold性能
        acc = accuracy_score(y_test_fold, y_pred)
        auc = roc_auc_score(y_test_fold, y_pred_proba[:, 1])
        f1 = f1_score(y_test_fold, y_pred)
        
        overall_scores.append({
            'fold': fold,
            'accuracy': acc,
            'auc': auc,
            'f1': f1
        })
        
        print(f"  测试集性能 - AUC: {auc:.4f}, ACC: {acc:.4f}, F1: {f1:.4f}")
    
    # ==============================
    # 4. 重建完整的预测结果
    # ==============================
    print(f"\n{'='*50}")
    print("重建完整预测结果")
    print(f"{'='*50}")
    
    # 创建完整的预测结果DataFrame (按原始索引顺序)
    prediction_results = pd.DataFrame({
        'Original_Index': all_indices,
        'True_Label': all_true_labels,
        'Predicted_Label': all_predictions,
        'Malignant_Probability': all_probabilities
    })
    
    # 按原始索引排序
    prediction_results = prediction_results.sort_values('Original_Index').reset_index(drop=True)
    
    # 计算整体性能
    overall_acc = accuracy_score(prediction_results['True_Label'], prediction_results['Predicted_Label'])
    overall_auc = roc_auc_score(prediction_results['True_Label'], prediction_results['Malignant_Probability'])
    overall_f1 = f1_score(prediction_results['True_Label'], prediction_results['Predicted_Label'])
    
    print(f"交叉验证整体性能 (F-score特征选择):")
    print(f"  AUC: {overall_auc:.4f}")
    print(f"  ACC: {overall_acc:.4f}")
    print(f"  F1: {overall_f1:.4f}")
    
    # ==============================
    # 5. 癌症类型分析
    # ==============================
    print(f"\n{'='*50}")
    print("癌症类型检出率分析 (基于F-score特征选择)")
    print(f"{'='*50}")
    
    # 创建分析结果DataFrame
    analysis_results = pd.DataFrame({
        'True_Label': prediction_results['True_Label'].values,
        'Predicted_Label': prediction_results['Predicted_Label'].values,
        'Malignant_Probability': prediction_results['Malignant_Probability'].values,
        'Cancer_Type': df['Type_Raw_English'].values,
        'T_Stage': df['T_stage'].values,
        'Stage_Raw': df['Stage_Raw'].values
    })
    
    analysis_results['Correct_Prediction'] = (analysis_results['True_Label'] == analysis_results['Predicted_Label'])
    
    # 只分析真实的阳性样本
    positive_results = analysis_results[analysis_results['True_Label'] == 1].copy()
    
    cancer_analysis = []
    cancer_counts = positive_results['Cancer_Type'].value_counts()
    all_cancer_types = cancer_counts.index.tolist()
    
    for cancer_type in all_cancer_types:
        subset = positive_results[positive_results['Cancer_Type'] == cancer_type]
        
        total_samples = len(subset)
        correct_predictions = subset['Correct_Prediction'].sum()
        wrong_predictions = total_samples - correct_predictions
        accuracy = correct_predictions / total_samples
        avg_malignant_prob = subset['Malignant_Probability'].mean()
        
        cancer_analysis.append({
            'Cancer_Type': cancer_type,
            'Total_Samples': total_samples,
            'Correctly_Predicted_as_Positive': correct_predictions,
            'Wrongly_Predicted_as_Negative': wrong_predictions,
            'Detection_Rate': accuracy,
            'Avg_Malignant_Probability': avg_malignant_prob
        })
        
        print(f"\n{cancer_type}:")
        print(f"  总样本: {total_samples}")
        print(f"  正确识别为癌症: {correct_predictions} ({accuracy:.3f})")
        print(f"  错误判断为健康: {wrong_predictions}")
        print(f"  平均恶性概率: {avg_malignant_prob:.3f}")
        if accuracy < 0.8:
            print(f"  ⚠️  该癌症类型检出率较低！")
        elif accuracy > 0.95:
            print(f"  ✅ 该癌症类型检出率很高")
    
    # ==============================
    # 6. 性能对比分析
    # ==============================
    print(f"\n{'='*50}")
    print("F-score特征选择效果分析")
    print(f"{'='*50}")
    
    # 混淆矩阵
    conf_matrix = confusion_matrix(prediction_results['True_Label'], prediction_results['Predicted_Label'])
    tn, fp, fn, tp = conf_matrix.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"混淆矩阵:")
    print(f"真阴性(TN): {tn:3d} | 假阳性(FP): {fp:3d}")
    print(f"假阴性(FN): {fn:3d} | 真阳性(TP): {tp:3d}")
    print(f"\n敏感度(Sensitivity): {sensitivity:.3f}")
    print(f"特异度(Specificity): {specificity:.3f}")
    
    # ==============================
    # 7. 保存结果
    # ==============================
    
    overall_df = pd.DataFrame(overall_scores)
    
    # 保存F-score特征选择结果
    feature_selection_df = pd.DataFrame({
        'Feature': selected_features,
        'F_Score': feature_scores
    }).sort_values('F_Score', ascending=False)
    feature_selection_df.to_csv(f'{base_path}/FScore_Selected_Features_Top{k_best_features}.csv', index=False)
    
    # 保存交叉验证结果
    overall_df.to_csv(f'{base_path}/KMeans_SMOTE_FScore_CV_Results.csv', index=False)
    
    # 保存交叉验证预测结果
    analysis_results.to_csv(f'{base_path}/KMeans_SMOTE_FScore_CrossValidation_Predictions.csv', index=False)
    
    # 保存癌症类型分析
    cancer_analysis_df = pd.DataFrame(cancer_analysis)
    cancer_analysis_df.to_csv(f'{base_path}/KMeans_SMOTE_FScore_Cancer_Type_Analysis.csv', index=False)
    
    # ==============================
    # 8. 可视化对比
    # ==============================
    
    if len(cancer_analysis) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 子图1: F-score特征重要性
        axes[0,0].bar(range(len(selected_features)), feature_scores)
        axes[0,0].set_title(f'Top {k_best_features} F-Score Selected Features')
        axes[0,0].set_ylabel('F-Score Value')
        axes[0,0].set_xticks(range(len(selected_features)))
        axes[0,0].set_xticklabels([f[:8] for f in selected_features], rotation=45)
        
        # 子图2: 癌症类型检出率
        cancer_analysis_df = pd.DataFrame(cancer_analysis)
        axes[0,1].bar(range(len(cancer_analysis_df)), cancer_analysis_df['Detection_Rate'])
        axes[0,1].set_title('Cancer Detection Rate by Type (F-Score)')
        axes[0,1].set_ylabel('Detection Rate')
        axes[0,1].set_xticks(range(len(cancer_analysis_df)))
        axes[0,1].set_xticklabels([ct[:10] + '...' if len(ct) > 10 else ct for ct in cancer_analysis_df['Cancer_Type']], rotation=45)
        axes[0,1].axhline(y=0.8, color='r', linestyle='--', alpha=0.5)
        
        # 子图3: 交叉验证性能
        metrics = ['accuracy', 'auc', 'f1']
        metric_means = [overall_df[metric].mean() for metric in metrics]
        metric_stds = [overall_df[metric].std() for metric in metrics]
        
        axes[1,0].bar(metrics, metric_means, yerr=metric_stds, capsize=5)
        axes[1,0].set_title('Cross-Validation Performance (F-Score)')
        axes[1,0].set_ylabel('Score')
        
        # 子图4: 混淆矩阵
        axes[1,1].imshow(conf_matrix, cmap='Blues', alpha=0.7)
        for i in range(2):
            for j in range(2):
                axes[1,1].text(j, i, conf_matrix[i, j], ha='center', va='center', fontsize=20)
        axes[1,1].set_title('Confusion Matrix (F-Score Method)')
        axes[1,1].set_xlabel('Predicted')
        axes[1,1].set_ylabel('Actual')
        axes[1,1].set_xticks([0, 1])
        axes[1,1].set_yticks([0, 1])
        axes[1,1].set_xticklabels(['Negative', 'Positive'])
        axes[1,1].set_yticklabels(['Negative', 'Positive'])
        
        plt.tight_layout()
        plt.savefig(f'{base_path}/KMeans_SMOTE_FScore_Analysis_Plots.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return overall_df, {
        'selected_features': selected_features,
        'feature_scores': feature_scores,
        'cv_performance': {
            'auc': overall_auc,
            'accuracy': overall_acc,
            'f1': overall_f1
        },
        'cancer_analysis': cancer_analysis,
        'confusion_matrix': conf_matrix,
        'selected_features_shape': X_selected.shape
    }

# ==============================
# 运行分析
# ==============================
if __name__ == "__main__":
    print("TabPFN + KMeans-SMOTE + F-score特征选择 癌症预测分析")
    print("="*60)
    
    # 设置结果路径
    results_path = './results/KMeans_SMOTE_FScore_Best10_Analysis'
    
    # 运行分析
    results, analysis_data = analyze_with_kmeans_smote_and_fscore_selection(
        device='cuda',
        n_estimators=32,
        softmax_temperature=0.9,
        balance_probabilities=False,
        average_before_softmax=False,
        ignore_pretraining_limits=True,
        random_state=42,
        k_best_features=10,
        base_path=results_path
    )
    
    # 打印最终摘要
    print(f"\n{'='*60}")
    print("F-score特征选择方法 - 最终分析摘要")
    print(f"{'='*60}")
    
    print(f"选择的特征数量: {len(analysis_data['selected_features'])}")
    print(f"选择的特征: {', '.join(analysis_data['selected_features'])}")
    
    print(f"\n交叉验证性能:")
    print(f"  平均AUC: {analysis_data['cv_performance']['auc']:.4f}")
    print(f"  平均准确率: {analysis_data['cv_performance']['accuracy']:.4f}")
    print(f"  平均F1分数: {analysis_data['cv_performance']['f1']:.4f}")
    
    print(f"\n结果已保存到: {results_path}")
    print("F-score特征选择分析完成！") 