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

def analyze_cancer_prediction_performance(
    device: str = 'cuda',
    n_estimators: int = 32,
    softmax_temperature: float = 0.9,
    balance_probabilities: bool = False,
    average_before_softmax: bool = False,
    ignore_pretraining_limits: bool = True,
    random_state: int = 42,
    base_path: str = './results'
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    分析TabPFN对不同癌症类型的预测准确性
    注意：只有Label=1的样本才有癌症类型信息，Label=0的样本代表健康/阴性
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
    
    print(f"数据形状: {X.shape}")
    print(f"标签分布: 阳性(癌症)={y.sum()}, 阴性(健康)={len(y)-y.sum()}")
    
    # 只有Label=1的样本才有癌症类型信息
    positive_samples = df[df['Label'] == 1]
    print(f"阳性样本癌症类型分布:\n{positive_samples['Type_Raw_English'].value_counts()}")
    
    # ==============================
    # 2. 10折交叉验证获取预测结果
    # ==============================
    print("\n" + "="*50)
    print("10折交叉验证获取预测结果")
    print("="*50)
    
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    # 存储每个样本的预测结果
    all_predictions = []
    all_probabilities = []
    all_true_labels = []
    all_indices = []
    
    overall_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        print(f"处理 Fold {fold}...")
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # 训练模型
        clf = TabPFNClassifier(
            device=device,
            n_estimators=n_estimators,
            softmax_temperature=softmax_temperature,
            balance_probabilities=balance_probabilities,
            average_before_softmax=average_before_softmax,
            ignore_pretraining_limits=ignore_pretraining_limits,
            random_state=random_state
        )
        clf.fit(X_train, y_train)
        
        # 预测
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)
        
        # 保存预测结果
        all_predictions.extend(y_pred)
        all_probabilities.extend(y_pred_proba[:, 1])  # 恶性概率
        all_true_labels.extend(y_test.values)
        all_indices.extend(test_idx)
        
        # 计算fold性能
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        f1 = f1_score(y_test, y_pred)
        
        overall_scores.append({
            'fold': fold,
            'accuracy': acc,
            'auc': auc,
            'f1': f1
        })
        
        print(f"  AUC: {auc:.4f}, ACC: {acc:.4f}, F1: {f1:.4f}")
    
    # ==============================
    # 3. 创建完整的预测结果DataFrame
    # ==============================
    print("\n创建预测结果分析...")
    
    # 重新排序以匹配原始数据顺序
    prediction_results = pd.DataFrame({
        'Original_Index': all_indices,
        'True_Label': all_true_labels,
        'Predicted_Label': all_predictions,
        'Malignant_Probability': all_probabilities
    })
    
    # 按原始索引排序
    prediction_results = prediction_results.sort_values('Original_Index').reset_index(drop=True)
    
    # 添加癌症相关信息（只对Label=1的样本有效）
    prediction_results['Cancer_Type'] = df.loc[prediction_results['Original_Index'], 'Type_Raw_English'].values
    prediction_results['T_Stage'] = df.loc[prediction_results['Original_Index'], 'T_stage'].values
    prediction_results['N_Stage'] = df.loc[prediction_results['Original_Index'], 'N_stage'].values
    prediction_results['Stage_Raw'] = df.loc[prediction_results['Original_Index'], 'Stage_Raw'].values
    
    # 计算预测准确性
    prediction_results['Correct_Prediction'] = (prediction_results['True_Label'] == prediction_results['Predicted_Label'])
    
    # ==============================
    # 4. 整体预测性能分析
    # ==============================
    print("\n" + "="*50)
    print("整体预测性能分析")
    print("="*50)
    
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(prediction_results['True_Label'], prediction_results['Predicted_Label'])
    
    # 真实阳性样本数量
    true_positive_samples = prediction_results[prediction_results['True_Label'] == 1]
    true_negative_samples = prediction_results[prediction_results['True_Label'] == 0]
    
    # 预测结果统计
    print(f"真实阳性样本: {len(true_positive_samples)}")
    print(f"真实阴性样本: {len(true_negative_samples)}")
    print(f"预测为阳性的样本: {prediction_results['Predicted_Label'].sum()}")
    print(f"预测为阴性的样本: {len(prediction_results) - prediction_results['Predicted_Label'].sum()}")
    
    print(f"\n混淆矩阵:")
    print(f"真阴性(TN): {conf_matrix[0,0]}, 假阳性(FP): {conf_matrix[0,1]}")
    print(f"假阴性(FN): {conf_matrix[1,0]}, 真阳性(TP): {conf_matrix[1,1]}")
    
    # 计算各类准确率
    tn, fp, fn, tp = conf_matrix.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # 真阳性率（敏感度）
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # 真阴性率（特异度）
    
    print(f"\n敏感度(Sensitivity): {sensitivity:.3f} - 正确识别癌症的比例")
    print(f"特异度(Specificity): {specificity:.3f} - 正确识别健康的比例")
    
    # ==============================
    # 5. 分析真实阳性样本中不同癌症类型的预测准确性
    # ==============================
    print("\n" + "="*50)
    print("不同癌症类型预测准确性分析（仅真实阳性样本）")
    print("="*50)
    
    cancer_analysis = []
    
    # 只分析真实的阳性样本（有癌症类型信息）
    positive_results = prediction_results[prediction_results['True_Label'] == 1].copy()
    
    # 获取所有癌症类型
    cancer_counts = positive_results['Cancer_Type'].value_counts()
    all_cancer_types = cancer_counts.index.tolist()
    
    for cancer_type in all_cancer_types:
        subset = positive_results[positive_results['Cancer_Type'] == cancer_type]
        
        total_samples = len(subset)
        correct_predictions = subset['Correct_Prediction'].sum()  # 被正确预测为阳性的
        wrong_predictions = total_samples - correct_predictions   # 被错误预测为阴性的
        accuracy = correct_predictions / total_samples
        
        # 获取该癌症类型样本的平均预测概率
        avg_malignant_prob = subset['Malignant_Probability'].mean()
        
        cancer_analysis.append({
            'Cancer_Type': cancer_type,
            'Total_Samples': total_samples,
            'Correctly_Predicted_as_Positive': correct_predictions,
            'Wrongly_Predicted_as_Negative': wrong_predictions,
            'Detection_Rate': accuracy,  # 检出率
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
    # 6. 分析不同分期的预测准确性
    # ==============================
    print("\n" + "="*50)
    print("不同分期预测准确性分析（仅真实阳性样本）")
    print("="*50)
    
    # T分期分析
    print("\nT分期检出率:")
    t_stage_analysis = positive_results.groupby('T_Stage').agg({
        'Correct_Prediction': ['count', 'sum', 'mean'],
        'Malignant_Probability': 'mean'
    }).round(3)
    t_stage_analysis.columns = ['Total_Count', 'Detected_Count', 'Detection_Rate', 'Avg_Probability']
    print(t_stage_analysis)
    
    # 综合分期分析
    print("\n主要综合分期检出率:")
    stage_counts = positive_results['Stage_Raw'].value_counts()
    major_stages = stage_counts[stage_counts >= 3].index.tolist()  # 至少3个样本
    
    stage_analysis = []
    for stage in major_stages:
        subset = positive_results[positive_results['Stage_Raw'] == stage]
        detection_rate = subset['Correct_Prediction'].mean()
        avg_prob = subset['Malignant_Probability'].mean()
        stage_analysis.append({
            'Stage': stage,
            'Count': len(subset),
            'Detection_Rate': detection_rate,
            'Avg_Probability': avg_prob
        })
        print(f"  {stage}: {len(subset)}个样本, 检出率: {detection_rate:.3f}, 平均概率: {avg_prob:.3f}")
    
    # ==============================
    # 7. 分析被错误预测的癌症样本
    # ==============================
    print("\n" + "="*50)
    print("被错误预测为健康的癌症样本分析")
    print("="*50)
    
    false_negative_samples = positive_results[positive_results['Correct_Prediction'] == False]
    
    if len(false_negative_samples) > 0:
        print(f"总共有 {len(false_negative_samples)} 个癌症样本被错误预测为健康:")
        
        fn_cancer_types = false_negative_samples['Cancer_Type'].value_counts()
        for cancer_type, count in fn_cancer_types.items():
            total_of_this_type = len(positive_results[positive_results['Cancer_Type'] == cancer_type])
            miss_rate = count / total_of_this_type
            print(f"  {cancer_type}: {count}个 (占该类型的{miss_rate:.1%})")
        
        print(f"\n这些样本的平均恶性概率: {false_negative_samples['Malignant_Probability'].mean():.3f}")
        print(f"最低恶性概率: {false_negative_samples['Malignant_Probability'].min():.3f}")
        print(f"最高恶性概率: {false_negative_samples['Malignant_Probability'].max():.3f}")
    else:
        print("所有癌症样本都被正确识别！")
    
    # ==============================
    # 8. 保存结果
    # ==============================
    
    # 保存整体CV结果
    overall_df = pd.DataFrame(overall_scores)
    print(f"\n整体CV结果:")
    print(f"AUC: {overall_df['auc'].mean():.4f} ± {overall_df['auc'].std():.4f}")
    print(f"ACC: {overall_df['accuracy'].mean():.4f} ± {overall_df['accuracy'].std():.4f}")
    print(f"F1: {overall_df['f1'].mean():.4f} ± {overall_df['f1'].std():.4f}")
    
    overall_df.to_csv(f'{base_path}/HenanCancer_Overall_CV_Results.csv', index=False)
    
    # 保存详细预测结果
    prediction_results.to_csv(f'{base_path}/HenanCancer_Detailed_Predictions.csv', index=False)
    
    # 保存癌症类型分析
    cancer_analysis_df = pd.DataFrame(cancer_analysis)
    cancer_analysis_df.to_csv(f'{base_path}/HenanCancer_CancerType_Detection_Analysis.csv', index=False)
    
    # 保存假阴性样本分析
    if len(false_negative_samples) > 0:
        false_negative_samples.to_csv(f'{base_path}/HenanCancer_False_Negative_Samples.csv', index=False)
    
    # 保存分期分析
    t_stage_analysis.to_csv(f'{base_path}/HenanCancer_T_Stage_Detection.csv')
    stage_analysis_df = pd.DataFrame(stage_analysis)
    stage_analysis_df.to_csv(f'{base_path}/HenanCancer_Combined_Stage_Detection.csv', index=False)
    
    # ==============================
    # 9. 创建可视化
    # ==============================
    
    if len(cancer_analysis) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        cancer_analysis_df = pd.DataFrame(cancer_analysis)
        
        # 子图1: 不同癌症类型的检出率
        axes[0,0].bar(range(len(cancer_analysis_df)), cancer_analysis_df['Detection_Rate'])
        axes[0,0].set_title('Cancer Detection Rate by Type')
        axes[0,0].set_ylabel('Detection Rate')
        axes[0,0].set_xticks(range(len(cancer_analysis_df)))
        axes[0,0].set_xticklabels([ct[:15] + '...' if len(ct) > 15 else ct for ct in cancer_analysis_df['Cancer_Type']], rotation=45)
        axes[0,0].axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='80% threshold')
        axes[0,0].legend()
        
        # 子图2: 平均恶性概率
        axes[0,1].bar(range(len(cancer_analysis_df)), cancer_analysis_df['Avg_Malignant_Probability'])
        axes[0,1].set_title('Average Malignant Probability by Cancer Type')
        axes[0,1].set_ylabel('Probability')
        axes[0,1].set_xticks(range(len(cancer_analysis_df)))
        axes[0,1].set_xticklabels([ct[:10] + '...' if len(ct) > 10 else ct for ct in cancer_analysis_df['Cancer_Type']], rotation=45)
        axes[0,1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Decision threshold')
        axes[0,1].legend()
        
        # 子图3: 样本数量分布
        axes[0,2].bar(range(len(cancer_analysis_df)), cancer_analysis_df['Total_Samples'])
        axes[0,2].set_title('Sample Count by Cancer Type')
        axes[0,2].set_ylabel('Sample Count')
        axes[0,2].set_xticks(range(len(cancer_analysis_df)))
        axes[0,2].set_xticklabels([ct[:10] + '...' if len(ct) > 10 else ct for ct in cancer_analysis_df['Cancer_Type']], rotation=45)
        
        # 子图4: 正确vs错误预测
        correct_counts = cancer_analysis_df['Correctly_Predicted_as_Positive']
        wrong_counts = cancer_analysis_df['Wrongly_Predicted_as_Negative']
        
        x = np.arange(len(cancer_analysis_df))
        width = 0.35
        axes[1,0].bar(x - width/2, correct_counts, width, label='Correctly Detected', alpha=0.8, color='green')
        axes[1,0].bar(x + width/2, wrong_counts, width, label='Missed (False Negative)', alpha=0.8, color='red')
        axes[1,0].set_title('Detection Results by Cancer Type')
        axes[1,0].set_ylabel('Sample Count')
        axes[1,0].set_xticks(x)
        axes[1,0].set_xticklabels([ct[:10] + '...' if len(ct) > 10 else ct for ct in cancer_analysis_df['Cancer_Type']], rotation=45)
        axes[1,0].legend()
        
        # 子图5: T分期检出率
        if len(t_stage_analysis) > 0:
            axes[1,1].bar(range(len(t_stage_analysis)), t_stage_analysis['Detection_Rate'])
            axes[1,1].set_title('Detection Rate by T Stage')
            axes[1,1].set_ylabel('Detection Rate')
            axes[1,1].set_xticks(range(len(t_stage_analysis)))
            axes[1,1].set_xticklabels(t_stage_analysis.index, rotation=45)
            axes[1,1].axhline(y=0.8, color='r', linestyle='--', alpha=0.5)
        
        # 子图6: 整体混淆矩阵
        axes[1,2].imshow(conf_matrix, cmap='Blues', alpha=0.7)
        for i in range(2):
            for j in range(2):
                axes[1,2].text(j, i, conf_matrix[i, j], ha='center', va='center', fontsize=20)
        axes[1,2].set_title('Confusion Matrix')
        axes[1,2].set_xlabel('Predicted')
        axes[1,2].set_ylabel('Actual')
        axes[1,2].set_xticks([0, 1])
        axes[1,2].set_yticks([0, 1])
        axes[1,2].set_xticklabels(['Negative', 'Positive'])
        axes[1,2].set_yticklabels(['Negative', 'Positive'])
        
        plt.tight_layout()
        plt.savefig(f'{base_path}/HenanCancer_Detection_Analysis_Plots.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return overall_df, {
        'cancer_analysis': cancer_analysis,
        'prediction_results': prediction_results,
        'stage_analysis': stage_analysis,
        'false_negative_samples': false_negative_samples,
        'confusion_matrix': conf_matrix
    }

# ==============================
# 运行分析
# ==============================
if __name__ == "__main__":
    print("TabPFN河南癌症医院预测准确性分析")
    print("="*50)
    
    # 运行分析
    results, analysis_data = analyze_cancer_prediction_performance(
        device='cuda',
        n_estimators=32,
        softmax_temperature=0.9,
        balance_probabilities=False,
        average_before_softmax=False,
        ignore_pretraining_limits=True,
        random_state=42,
        base_path='./results/HenanCancer_Detection_Analysis'
    )
    
    print("\n分析完成！结果已保存到 ./results/HenanCancer_Detection_Analysis/")
    
    # 输出总结
    cancer_analysis = analysis_data['cancer_analysis']
    if cancer_analysis:
        print("\n" + "="*50)
        print("癌症类型检出率总结")
        print("="*50)
        for analysis in cancer_analysis:
            status = "⚠️ 较低" if analysis['Detection_Rate'] < 0.8 else "✅ 良好" if analysis['Detection_Rate'] > 0.95 else "🔶 中等"
            print(f"{analysis['Cancer_Type']}: {analysis['Detection_Rate']:.3f} ({analysis['Total_Samples']}个样本) {status}")
        
        # 总体统计
        total_cancer_samples = sum([a['Total_Samples'] for a in cancer_analysis])
        total_detected = sum([a['Correctly_Predicted_as_Positive'] for a in cancer_analysis])
        overall_detection_rate = total_detected / total_cancer_samples
        print(f"\n整体癌症检出率: {overall_detection_rate:.3f} ({total_detected}/{total_cancer_samples})") 