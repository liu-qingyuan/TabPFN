import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# 导入AutoTabPFN分类器
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier
# 导入SHAP分析模块
from tabpfn_extensions import interpretability

# 设置日志
logging.disable(logging.INFO)

def evaluate_metrics(y_true, y_pred, y_pred_proba):
    """计算所有评估指标"""
    conf_matrix = confusion_matrix(y_true, y_pred)
    return {
        'acc': accuracy_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_pred_proba),
        'f1': f1_score(y_true, y_pred),
        'acc_0': conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1]),
        'acc_1': conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
    }

def print_metrics(dataset_name, metrics):
    """打印评估指标"""
    print(f"{dataset_name} Accuracy: {metrics['acc']:.4f}")
    print(f"{dataset_name} AUC: {metrics['auc']:.4f}")
    print(f"{dataset_name} F1 Score: {metrics['f1']:.4f}")
    print(f"{dataset_name} Class 0 Accuracy: {metrics['acc_0']:.4f}")
    print(f"{dataset_name} Class 1 Accuracy: {metrics['acc_1']:.4f}")

def analyze_with_shap(
    clf,
    X_train,
    y_train,
    X_test,
    y_test,
    y_pred,
    feature_names,
    dataset_name,
    n_samples=50,
    base_path='./results'
):
    """
    使用SHAP分析模型
    """
    print(f"\n===== {dataset_name} Dataset SHAP Analysis =====")
    
    # 创建保存目录
    shap_path = f"{base_path}/shap_analysis"
    os.makedirs(shap_path, exist_ok=True)
    
    # 创建样本级别的SHAP分析保存目录
    sample_shap_path = f"{shap_path}/sample_level"
    os.makedirs(sample_shap_path, exist_ok=True)
    
    # 获取四种预测情况的样本索引
    tp_indices = np.where((y_test == 1) & (y_pred == 1))[0]  # True Positive
    fn_indices = np.where((y_test == 1) & (y_pred == 0))[0]  # False Negative
    fp_indices = np.where((y_test == 0) & (y_pred == 1))[0]  # False Positive
    tn_indices = np.where((y_test == 0) & (y_pred == 0))[0]  # True Negative
    
    # 选择每种情况的前10个样本进行分析（或者如果样本量少于10个，则使用所有样本）
    sample_indices = {
        'TP': tp_indices[:10] if len(tp_indices) >= 10 else tp_indices,
        'FN': fn_indices[:10] if len(fn_indices) >= 10 else fn_indices,
        'FP': fp_indices[:10] if len(fp_indices) >= 10 else fp_indices,
        'TN': tn_indices[:10] if len(tn_indices) >= 10 else tn_indices
    }
    
    print("\nSelected samples for analysis:")
    for case, indices in sample_indices.items():
        print(f"{case}: {len(indices)} samples")
    
    # 计算所有选定样本的SHAP值
    all_samples = []
    for case_indices in sample_indices.values():
        all_samples.extend(case_indices)
    
    # 确保我们有样本进行分析
    if len(all_samples) == 0:
        print("No samples available for SHAP analysis")
        return
    
    print("\nCalculating SHAP values...")
    X_explain = X_test[all_samples]
    
    # 计算SHAP值
    shap_values = interpretability.shap.get_shap_values(
        estimator=clf,
        test_x=X_explain,
        attribute_names=feature_names,
        algorithm="permutation",
    )
    
    # 创建总体SHAP可视化
    fig = interpretability.shap.plot_shap(shap_values)
    plt.savefig(f"{shap_path}/{dataset_name}_shap_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 计算特征重要性排名
    print("\nCalculating feature importance ranking...")
    # 检查SHAP值的结构并相应地提取数据
    print("SHAP values structure:", type(shap_values))
    
    try:
        # 尝试获取SHAP值
        if hasattr(shap_values, 'values'):
            # 有些版本的SHAP库会返回带有values属性的对象
            shap_array = shap_values.values
            print("Extracting SHAP values from .values attribute")
        elif hasattr(shap_values, 'shap_values'):
            # 有些版本会返回带有shap_values属性的对象
            shap_array = shap_values.shap_values
            print("Extracting SHAP values from .shap_values attribute")
        else:
            # 如果是直接返回数组
            shap_array = shap_values
            print("Using SHAP values directly")
        
        print("SHAP array shape:", shap_array.shape)
        
        # 根据形状处理SHAP值
        if len(shap_array.shape) == 2:
            # 二维数组 [样本数, 特征数]
            abs_shap_values = np.abs(shap_array)
            mean_abs_shap = np.mean(abs_shap_values, axis=0)
        elif len(shap_array.shape) == 3:
            # 三维数组处理
            if shap_array.shape == (len(all_samples), len(feature_names), 2):
                # 形状是 [样本数, 特征数, 类别数]
                print("Detected shape [样本数, 特征数, 类别数]")
                # 提取第1个类别的SHAP值 (正类)
                abs_shap_values = np.abs(shap_array[:, :, 1])
                # 对样本维度取平均
                mean_abs_shap = np.mean(abs_shap_values, axis=0)
                print(f"Feature dimension size: {mean_abs_shap.shape}")
                
                # 保存每个样本的原始SHAP值（非绝对值）以便后续可视化
                raw_shap_values = shap_array[:, :, 1]
            elif shap_array.shape[1] == len(feature_names):
                # 如果第二个维度匹配特征数量
                print(f"Second dimension matches feature count: {shap_array.shape[1]}")
                # 取绝对值并对第一个维度取平均
                abs_shap_values = np.abs(shap_array)
                # 对第一个维度（样本）取平均
                mean_abs_shap = np.mean(abs_shap_values, axis=0)
                # 如果还有第三个维度，取第一个类别
                if len(mean_abs_shap.shape) > 1 and mean_abs_shap.shape[1] == 2:
                    print("Selecting SHAP values for positive class")
                    mean_abs_shap = mean_abs_shap[:, 1]
                
                # 保存每个样本的原始SHAP值
                if len(shap_array.shape) == 3 and shap_array.shape[2] == 2:
                    raw_shap_values = shap_array[:, :, 1]
                else:
                    raw_shap_values = shap_array
            else:
                # 其他情况，尝试常见的三维形状处理
                print("Attempting common 3D shape handling")
                if shap_array.shape[0] == 2:
                    # 形状可能是 [类别数, 样本数, 特征数]
                    abs_shap_values = np.abs(shap_array[1])  # 选择正类
                    raw_shap_values = shap_array[1]  # 原始SHAP值
                elif shap_array.shape[2] == 2:
                    # 形状可能是 [样本数, 特征数, 类别数]
                    abs_shap_values = np.abs(shap_array[:, :, 1])  # 选择正类
                    raw_shap_values = shap_array[:, :, 1]  # 原始SHAP值
                else:
                    # 未知形状，尝试直接使用
                    abs_shap_values = np.abs(shap_array)
                    raw_shap_values = shap_array
                
                # 对样本维度取平均
                if len(abs_shap_values.shape) > 1:
                    mean_abs_shap = np.mean(abs_shap_values, axis=0)
                else:
                    mean_abs_shap = abs_shap_values
        else:
            raise ValueError(f"Unable to handle shape {shap_array.shape} of SHAP values")
        
        print(f"Calculated feature importance shape: {mean_abs_shap.shape}")
        
        # 确保特征名称和SHAP值数量匹配
        if len(mean_abs_shap) != len(feature_names):
            print(f"Warning: Feature count ({len(feature_names)}) does not match SHAP values count ({len(mean_abs_shap)})")
            # 如果数量不匹配，使用索引作为特征名称
            feature_names_adj = [f"Feature_{i}" for i in range(len(mean_abs_shap))]
        else:
            feature_names_adj = feature_names
            
        # 创建特征重要性DataFrame
        feature_importance = pd.DataFrame({
            'Feature': feature_names_adj,
            'Mean_ABS_SHAP': mean_abs_shap
        })
        
        # 按重要性排序
        feature_importance = feature_importance.sort_values('Mean_ABS_SHAP', ascending=False)
        
        # 保存特征重要性排名
        feature_importance.to_csv(f"{shap_path}/{dataset_name}_feature_importance.csv", index=False)
        
        # 打印所有特征的排名
        print("\nFeature Importance Ranking:")
        print(feature_importance.to_string(index=False))
        
        # 可视化特征重要性
        plt.figure(figsize=(12, max(8, len(feature_names_adj) * 0.3)))
        plt.barh(feature_importance['Feature'], feature_importance['Mean_ABS_SHAP'])
        plt.xlabel('Mean |SHAP Value|')
        plt.title(f'{dataset_name} - Feature Importance Ranking')
        plt.tight_layout()
        plt.savefig(f"{shap_path}/{dataset_name}_feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 为每个样本创建单独的SHAP可视化（显示原始特征影响，不取绝对值）
        print("\nCreating visualizations for individual samples...")
        current_idx = 0
        
        # 为每个类别的样本创建单独的可视化
        for case, indices in sample_indices.items():
            if len(indices) == 0:
                continue
                
            # 获取当前案例的样本数量
            n_case_samples = len(indices)
            print(f"Creating visualizations for {n_case_samples} {case} samples...")
            
            # 创建该类别的目录
            case_dir = f"{sample_shap_path}/{case}"
            os.makedirs(case_dir, exist_ok=True)
            
            # 处理每个样本
            for i in range(n_case_samples):
                sample_idx = current_idx + i
                if sample_idx >= len(raw_shap_values):
                    print(f"Warning: Sample index {sample_idx} exceeds SHAP values range")
                    continue
                
                # 获取原始样本数据
                original_idx = all_samples[sample_idx]
                sample_x = X_test[original_idx]
                sample_y = y_test[original_idx]
                sample_pred = y_pred[original_idx]
                sample_label = f"{case}_{i+1}"
                
                # 1. 创建waterfall图 - 显示每个特征对预测的贡献
                plt.figure(figsize=(12, max(8, len(feature_names_adj) * 0.25)))
                
                # 获取当前样本的SHAP值
                if len(raw_shap_values.shape) == 2:
                    # 如果是二维数组 [样本数, 特征数]
                    sample_shap = raw_shap_values[sample_idx]
                else:
                    print(f"Unable to extract sample SHAP values, shape is {raw_shap_values.shape}")
                    continue
                
                # 排序特征，按SHAP值的绝对值大小
                sorted_indices = np.argsort(-np.abs(sample_shap))
                top_features = sorted_indices[:20]  # 只显示前20个最重要的特征
                
                # 提取顶部特征的名称和SHAP值
                top_feature_names = [feature_names_adj[i] for i in top_features]
                top_shap_values = [sample_shap[i] for i in top_features]
                
                # 确定颜色：正值为红色，负值为蓝色
                colors = ['red' if val > 0 else 'blue' for val in top_shap_values]
                
                # 绘制条形图
                plt.barh(range(len(top_feature_names)), top_shap_values, color=colors)
                plt.yticks(range(len(top_feature_names)), top_feature_names)
                plt.xlabel('SHAP Value (Red = Increase Probability, Blue = Decrease Probability)')
                plt.title(f'{case} Sample {i+1} (Index {original_idx}) - True Label: {sample_y}, Predicted Label: {sample_pred}')
                plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
                plt.grid(axis='x', linestyle='--', alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"{case_dir}/{sample_label}_top_features.png", dpi=300, bbox_inches='tight')
                plt.close()
                
                # 2. 创建表格形式的SHAP值文件
                sample_df = pd.DataFrame({
                    'Feature': feature_names_adj,
                    'SHAP_Value': sample_shap,
                    'Abs_SHAP_Value': np.abs(sample_shap)
                })
                sample_df = sample_df.sort_values('Abs_SHAP_Value', ascending=False)
                sample_df.to_csv(f"{case_dir}/{sample_label}_shap_values.csv", index=False)
            
            # 更新当前索引
            current_idx += n_case_samples
        
    except Exception as e:
        print(f"Error calculating feature importance: {str(e)}")
        import traceback
        traceback.print_exc()
        print("Will continue with other analyses...")
    
    print(f"\n{dataset_name} Dataset SHAP Analysis Completed. Results saved in: {shap_path}/")

def run_experiment(
    X_train,
    y_train,
    X_test,
    y_test,
    feature_names,
    dataset_name,
    device='cuda',
    max_time=10,  # 添加最大优化时间参数
    random_state=42,
    base_path='./results'
):
    """运行实验"""
    # 创建结果目录
    os.makedirs(base_path, exist_ok=True)
    
    # 生成实验名称
    exp_name = f"AutoTabPFN-{dataset_name}-R{random_state}"
    
    print("Training Data Shape:", X_train.shape)
    print("Test Data Shape:", X_test.shape)
    print("Training Label Distribution:\n", pd.Series(y_train).value_counts())
    print("Test Label Distribution:\n", pd.Series(y_test).value_counts())
    
    # 检查特征名称中是否包含标签列
    if 'Label' in feature_names:
        print("Warning: 'Label' found in feature names. Removing it...")
        feature_names = [f for f in feature_names if f != 'Label']
        print(f"Updated feature names: {feature_names}")
    
    # 转换为NumPy数组
    X_train_values = X_train.values.astype(np.float32)
    X_test_values = X_test.values.astype(np.float32)
    y_train_values = y_train.values.astype(np.int32)
    y_test_values = y_test.values.astype(np.int32)
    
    # 数据标准化
    print("\nApplying data standardization...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_values)
    X_test_scaled = scaler.transform(X_test_values)
    
    # 初始化并训练模型
    print("\nStarting model training...")
    start_time = time.time()
    clf = AutoTabPFNClassifier(
        device=device,
        max_time=max_time,  # 添加最大优化时间参数
        random_state=random_state
    )
    clf.fit(X_train_scaled, y_train_values)
    
    # 预测
    y_pred_proba = clf.predict_proba(X_test_scaled)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # 打印原始预测概率
    print("\nExample of raw prediction probabilities:")
    
    # 获取四种预测情况的样本索引
    tp_indices = np.where((y_test_values == 1) & (y_pred == 1))[0]  # True Positive
    fn_indices = np.where((y_test_values == 1) & (y_pred == 0))[0]  # False Negative
    fp_indices = np.where((y_test_values == 0) & (y_pred == 1))[0]  # False Positive
    tn_indices = np.where((y_test_values == 0) & (y_pred == 0))[0]  # True Negative
    
    # 选择每种情况的前2个样本展示
    sample_indices = {
        'TP': tp_indices[:2] if len(tp_indices) >= 2 else tp_indices,
        'FN': fn_indices[:2] if len(fn_indices) >= 2 else fn_indices,
        'FP': fp_indices[:2] if len(fp_indices) >= 2 else fp_indices,
        'TN': tn_indices[:2] if len(tn_indices) >= 2 else tn_indices
    }
    
    for case, indices in sample_indices.items():
        print(f"\n{case} Sample prediction probabilities:")
        for i, idx in enumerate(indices):
            pos_prob = y_pred_proba[idx, 1]  # 正类的概率
            neg_prob = y_pred_proba[idx, 0]  # 负类的概率
            print(f"{case} Sample {i+1} (Index {idx}):")
            print(f"   Predicted Probabilities: [Negative: {neg_prob:.6f}, Positive: {pos_prob:.6f}]")
            print(f"   Predicted Label: {y_pred[idx]}")
            print(f"   True Label: {y_test_values[idx]}")
    
    train_time = time.time() - start_time
    print(f"Training and prediction completed, time: {train_time:.2f} seconds")
    
    # 计算指标
    metrics = evaluate_metrics(y_test_values, y_pred, y_pred_proba[:, 1])
    metrics['time'] = train_time
    
    # 打印结果
    print_metrics(dataset_name, metrics)
    
    # 保存结果
    results_df = pd.DataFrame([metrics])
    results_df.to_csv(f'{base_path}/{exp_name}-results.csv', index=False)
    
    # 混淆矩阵可视化
    plt.figure(figsize=(8, 6))
    conf_matrix = confusion_matrix(y_test_values, y_pred)
    plt.imshow(conf_matrix, cmap='Blues')
    plt.colorbar()
    plt.xticks([0, 1], ['Predicted 0', 'Predicted 1'])
    plt.yticks([0, 1], ['Actual 0', 'Actual 1'])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(conf_matrix[i, j]), 
                    ha="center", va="center", color="red")
    
    plt.title(f'{dataset_name} Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'{base_path}/{exp_name}-confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # SHAP分析
    print("\nStarting SHAP Analysis...")
    analyze_with_shap(
        clf=clf,
        X_train=X_train_scaled,  # 使用标准化后的数据
        y_train=y_train_values,
        X_test=X_test_scaled,    # 使用标准化后的数据
        y_test=y_test_values,
        y_pred=y_pred,
        feature_names=feature_names,
        dataset_name=dataset_name,
        base_path=base_path
    )
    
    return metrics

if __name__ == "__main__":
    # 加载数据集
    print("\nLoading datasets...")
    print("1. Loading AI4healthcare.xlsx...")
    df_ai4health = pd.read_excel("data/AI4healthcare.xlsx")
    
    print("2. Loading HenanCancerHospital_features63_58.xlsx...")
    df_henan = pd.read_excel("data/HenanCancerHospital_features63_58.xlsx")
    
    # 获取两个数据集的特征列表（明确排除Label列）
    features_ai4health = [col for col in df_ai4health.columns if col != 'Label']
    features_henan = [col for col in df_henan.columns if col != 'Label']
    
    # 找出共有的特征
    common_features = list(set(features_ai4health) & set(features_henan))
    print(f"\nNumber of common features: {len(common_features)}")
    print("Common features:", common_features)
    
    # 准备训练数据（B数据集）
    X_train = df_henan[common_features].copy()
    y_train = df_henan["Label"].copy()
    
    # 准备测试数据（A数据集）
    X_test = df_ai4health[common_features].copy()
    y_test = df_ai4health["Label"].copy()
    
    # 打印数据形状以确认
    print("\nData shapes after preparation:")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # 设置结果目录
    base_path = './results_B_train_A_test_auto'
    os.makedirs(base_path, exist_ok=True)
    
    # 运行实验
    print("\n\n=== Cross-Dataset Analysis (B->A) with AutoTabPFN ===")
    metrics = run_experiment(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=common_features,
        dataset_name="B_train_A_test",
        device='cuda',
        max_time=30,  # 设置最大优化时间为30秒
        random_state=42,
        base_path=base_path
    )
    
    # 保存数据集信息
    datasets_info = pd.DataFrame({
        'Dataset': ['B (Training)', 'A (Testing)'],
        'Samples': [len(X_train), len(X_test)],
        'Features': [len(common_features), len(common_features)],
        'Positive_Samples': [sum(y_train), sum(y_test)],
        'Negative_Samples': [len(y_train)-sum(y_train), len(y_test)-sum(y_test)]
    })
    
    print("\nDataset Information:")
    print(datasets_info.to_string(index=False))
    datasets_info.to_csv(f'{base_path}/datasets_info.csv', index=False)
    
    # 保存结果
    results_df = pd.DataFrame([metrics])
    results_df.to_csv(f'{base_path}/summary_results.csv', index=False)
    print("\nResults saved to:", base_path)