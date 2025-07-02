import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from tabpfn import TabPFNClassifier
from typing import Dict, Tuple, Any
import numpy as np

# 导入imbalanced-learn中的KMeans-SMOTE
try:
    from imblearn.over_sampling import KMeansSMOTE
except ImportError:
    print("正在安装 imbalanced-learn...")
    os.system("pip install imbalanced-learn")
    from imblearn.over_sampling import KMeansSMOTE

def analyze_with_kmeans_smote_and_feature_selection(
    device: str = 'cuda',
    n_estimators: int = 32,
    softmax_temperature: float = 0.9,
    balance_probabilities: bool = False,
    average_before_softmax: bool = False,
    ignore_pretraining_limits: bool = True,
    random_state: int = 42,
    k_best_features: int = 8,
    base_path: str = './results'
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    使用KMeans-SMOTE数据平衡和特征选择的TabPFN癌症预测分析
    """
    # 创建结果目录
    os.makedirs(base_path, exist_ok=True)
    
    # ==============================
    # 1. 读取数据
    # ==============================
    print("加载河南癌症医院数据...")
    df = pd.read_excel("data/HenanCancerHospital_translated_english.xlsx")
    
    # 选择特征列
    features = [c for c in df.columns if c.startswith("Feature")]
    X = df[features].copy()
    y = df["Label"].copy()
    
    print(f"原始数据形状: {X.shape}")
    print(f"原始标签分布: 阳性(癌症)={y.sum()}, 阴性(健康)={len(y)-y.sum()}")
    print(f"原始阳性比例: {y.mean():.1%}")
    
    # ==============================
    # 2. 使用预定义的最佳8特征
    # ==============================
    print(f"\n{'='*50}")
    print("使用预定义的最佳8特征 (基于RFE预筛选结果)")
    print(f"{'='*50}")
    
    # 预定义的最佳8特征
    BEST_8_FEATURES = [
        'Feature63', 'Feature2', 'Feature46', 'Feature61', 
        'Feature56', 'Feature42', 'Feature39', 'Feature43'
    ]
    
    # 检查这些特征是否存在于数据中
    available_features = [f for f in BEST_8_FEATURES if f in features]
    missing_features = [f for f in BEST_8_FEATURES if f not in features]
    
    if missing_features:
        print(f"⚠️  缺失特征: {missing_features}")
        print(f"✅ 可用特征: {available_features}")
    else:
        print(f"✅ 所有预定义特征均可用")
    
    # 选择可用的特征
    selected_features = available_features
    feature_indices = [features.index(f) for f in selected_features]
    X_selected = X.iloc[:, feature_indices].values
    
    # 计算这些特征的F-score (用于后续分析)
    from sklearn.feature_selection import f_classif
    feature_scores, _ = f_classif(X_selected, y)
    
    print(f"使用的 {len(selected_features)} 个特征及其F-score:")
    for i, (feature, score) in enumerate(zip(selected_features, feature_scores)):
        print(f"  {i+1:2d}. {feature}: F-score={score:.2f}")
    
    # 不进行数据标准化，直接使用原始特征数据
    
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
                k_neighbors=4,  # 减少邻居数量
                cluster_balance_threshold='auto',  # 自动调整聚类平衡阈值
                random_state=random_state,
                n_jobs=1  # 单线程避免并发问题
            )
        except Exception as e:
            print(f"  ⚠️  KMeans-SMOTE失败，改用SMOTE: {e}")
            from imblearn.over_sampling import SMOTE
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
    # 6. 重建完整的预测结果
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
    
    print(f"交叉验证整体性能:")
    print(f"  AUC: {overall_auc:.4f}")
    print(f"  ACC: {overall_acc:.4f}")
    print(f"  F1: {overall_f1:.4f}")
    
    # ==============================
    # 7. 癌症类型分析
    # ==============================
    print(f"\n{'='*50}")
    print("癌症类型检出率分析 (基于交叉验证预测)")
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
    # 8. 对比分析
    # ==============================
    print(f"\n{'='*50}")
    print("改进效果对比")
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
    # 9. 保存结果
    # ==============================
    
    overall_df = pd.DataFrame(overall_scores)
    
    # 保存特征选择结果
    feature_selection_df = pd.DataFrame({
        'Feature': selected_features,
        'F_Score': feature_scores
    }).sort_values('F_Score', ascending=False)
    feature_selection_df.to_csv(f'{base_path}/Selected_Features_Top{k_best_features}.csv', index=False)
    
    # 保存交叉验证结果
    overall_df.to_csv(f'{base_path}/KMeans_SMOTE_CV_Results.csv', index=False)
    
    # 保存交叉验证预测结果
    analysis_results.to_csv(f'{base_path}/KMeans_SMOTE_CrossValidation_Predictions.csv', index=False)
    
    # 保存癌症类型分析
    cancer_analysis_df = pd.DataFrame(cancer_analysis)
    cancer_analysis_df.to_csv(f'{base_path}/KMeans_SMOTE_Cancer_Type_Analysis.csv', index=False)
    
    # ==============================
    # 10. 可视化
    # ==============================
    
    if len(cancer_analysis) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 子图1: 特征重要性
        axes[0,0].bar(range(len(selected_features)), feature_scores)
        axes[0,0].set_title(f'Top {k_best_features} Feature Scores')
        axes[0,0].set_ylabel('F-Score')
        axes[0,0].set_xticks(range(len(selected_features)))
        axes[0,0].set_xticklabels([f[:8] for f in selected_features], rotation=45)
        
        # 子图2: 数据平衡前后对比
        categories = ['Before SMOTE', 'After SMOTE (avg per fold)']
        # 使用理论上的平衡数据进行比较（每个类别样本数相等）
        original_pos = y.sum()
        original_neg = len(y) - y.sum()
        # 估算平衡后的样本数（取较大的类别数量）
        balanced_count = max(original_pos, original_neg)
        positive_counts = [original_pos, balanced_count]
        negative_counts = [original_neg, balanced_count]
        
        x = np.arange(len(categories))
        width = 0.35
        axes[0,1].bar(x - width/2, positive_counts, width, label='Positive', alpha=0.8)
        axes[0,1].bar(x + width/2, negative_counts, width, label='Negative', alpha=0.8)
        axes[0,1].set_title('Data Balance Comparison')
        axes[0,1].set_ylabel('Sample Count')
        axes[0,1].set_xticks(x)
        axes[0,1].set_xticklabels(categories)
        axes[0,1].legend()
        
        # 子图3: 癌症类型检出率
        cancer_analysis_df = pd.DataFrame(cancer_analysis)
        axes[0,2].bar(range(len(cancer_analysis_df)), cancer_analysis_df['Detection_Rate'])
        axes[0,2].set_title('Cancer Detection Rate by Type')
        axes[0,2].set_ylabel('Detection Rate')
        axes[0,2].set_xticks(range(len(cancer_analysis_df)))
        axes[0,2].set_xticklabels([ct[:10] + '...' if len(ct) > 10 else ct for ct in cancer_analysis_df['Cancer_Type']], rotation=45)
        axes[0,2].axhline(y=0.8, color='r', linestyle='--', alpha=0.5)
        
        # 子图4: 交叉验证性能
        metrics = ['accuracy', 'auc', 'f1']
        metric_means = [overall_df[metric].mean() for metric in metrics]
        metric_stds = [overall_df[metric].std() for metric in metrics]
        
        axes[1,0].bar(metrics, metric_means, yerr=metric_stds, capsize=5)
        axes[1,0].set_title('Cross-Validation Performance')
        axes[1,0].set_ylabel('Score')
        
        # 子图5: 混淆矩阵
        axes[1,1].imshow(conf_matrix, cmap='Blues', alpha=0.7)
        for i in range(2):
            for j in range(2):
                axes[1,1].text(j, i, conf_matrix[i, j], ha='center', va='center', fontsize=20)
        axes[1,1].set_title('Confusion Matrix (Original Data)')
        axes[1,1].set_xlabel('Predicted')
        axes[1,1].set_ylabel('Actual')
        axes[1,1].set_xticks([0, 1])
        axes[1,1].set_yticks([0, 1])
        axes[1,1].set_xticklabels(['Negative', 'Positive'])
        axes[1,1].set_yticklabels(['Negative', 'Positive'])
        
        # 子图6: 预测概率分布
        axes[1,2].hist(analysis_results[analysis_results['True_Label']==0]['Malignant_Probability'], 
                      bins=20, alpha=0.5, label='Healthy', color='blue')
        axes[1,2].hist(analysis_results[analysis_results['True_Label']==1]['Malignant_Probability'], 
                      bins=20, alpha=0.5, label='Cancer', color='red')
        axes[1,2].axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='Threshold')
        axes[1,2].set_title('Probability Distribution')
        axes[1,2].set_xlabel('Malignant Probability')
        axes[1,2].set_ylabel('Count')
        axes[1,2].legend()
        
        plt.tight_layout()
        plt.savefig(f'{base_path}/KMeans_SMOTE_Analysis_Plots.png', dpi=300, bbox_inches='tight')
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
    print("TabPFN + KMeans-SMOTE + 特征选择 癌症预测分析")
    print("="*60)
    
    # 运行分析
    results, analysis_data = analyze_with_kmeans_smote_and_feature_selection(
        device='cuda',
        n_estimators=32,
        softmax_temperature=0.9,
        balance_probabilities=False,
        average_before_softmax=False,
        ignore_pretraining_limits=True,
        random_state=42,
        k_best_features=8,
        base_path='./results/KMeans_SMOTE_Best8_Analysis'
    )
    
    print(f"\n分析完成！结果已保存到 ./results/KMeans_SMOTE_Best8_Analysis/")
    
    # 输出总结
    print("\n" + "="*60)
    print("改进效果总结")
    print("="*60)
    
    print(f"选择的最佳8个特征:")
    for i, (feature, score) in enumerate(zip(analysis_data['selected_features'], analysis_data['feature_scores'])):
        print(f"  {i+1:2d}. {feature}: {score:.2f}")
    
    print(f"\n选择的特征形状: {analysis_data['selected_features_shape']}")
    
    cv_perf = analysis_data['cv_performance']
    print(f"\n交叉验证最终性能:")
    print(f"  AUC: {cv_perf['auc']:.4f}")
    print(f"  ACC: {cv_perf['accuracy']:.4f}")
    print(f"  F1: {cv_perf['f1']:.4f}")
    
    cv_results = results
    print(f"\n交叉验证平均性能:")
    print(f"  AUC: {cv_results['auc'].mean():.4f} ± {cv_results['auc'].std():.4f}")
    print(f"  ACC: {cv_results['accuracy'].mean():.4f} ± {cv_results['accuracy'].std():.4f}")
    print(f"  F1: {cv_results['f1'].mean():.4f} ± {cv_results['f1'].std():.4f}")
    
    cancer_analysis = analysis_data['cancer_analysis']
    if cancer_analysis:
        print(f"\n癌症类型检出率:")
        for analysis in cancer_analysis:
            status = "⚠️ 较低" if analysis['Detection_Rate'] < 0.8 else "✅ 良好" if analysis['Detection_Rate'] > 0.95 else "🔶 中等"
            print(f"  {analysis['Cancer_Type']}: {analysis['Detection_Rate']:.3f} {status}")
        
        total_cancer_samples = sum([a['Total_Samples'] for a in cancer_analysis])
        total_detected = sum([a['Correctly_Predicted_as_Positive'] for a in cancer_analysis])
        overall_detection_rate = total_detected / total_cancer_samples
        print(f"\n整体癌症检出率: {overall_detection_rate:.3f} ({total_detected}/{total_cancer_samples})") 