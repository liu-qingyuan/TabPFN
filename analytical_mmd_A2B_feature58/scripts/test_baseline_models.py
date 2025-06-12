#!/usr/bin/env python3
"""
基线模型测试脚本

快速验证PKUPH和Mayo模型的实现
"""

import sys
import os
import pandas as pd
import numpy as np

# 添加项目路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

try:
    from analytical_mmd_A2B_feature58.modeling.baseline_models import PKUPHModel, MayoModel, evaluate_baseline_models
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保所有依赖项都已安装")
    sys.exit(1)

def test_baseline_models():
    """测试基线模型"""
    print("🧪 测试基线模型...")
    
    # 创建测试数据
    n_samples = 100
    np.random.seed(42)
    
    # PKUPH模型所需特征
    pkuph_features = ['Feature2', 'Feature48', 'Feature49', 'Feature4', 'Feature50', 'Feature53']
    pkuph_data = pd.DataFrame({
        feature: np.random.normal(0, 1, n_samples) for feature in pkuph_features
    })
    
    # Mayo模型所需特征
    mayo_features = ['Feature2', 'Feature3', 'Feature5', 'Feature48', 'Feature49', 'Feature63']
    mayo_data = pd.DataFrame({
        feature: np.random.normal(0, 1, n_samples) for feature in mayo_features
    })
    
    # 创建虚拟标签
    y = np.random.randint(0, 2, n_samples)
    
    print("\n📊 测试PKUPH模型:")
    print("-" * 30)
    
    # 测试PKUPH模型
    pkuph_model = PKUPHModel()
    print(f"模型特征: {pkuph_model.get_feature_names()}")
    
    # 训练（实际上只是标记为已拟合）
    pkuph_model.fit(pkuph_data, y)
    
    # 预测
    pkuph_pred = pkuph_model.predict(pkuph_data)
    pkuph_proba = pkuph_model.predict_proba(pkuph_data)
    
    print(f"预测形状: {pkuph_pred.shape}")
    print(f"概率形状: {pkuph_proba.shape}")
    print(f"预测值范围: {pkuph_pred.min()} - {pkuph_pred.max()}")
    print(f"概率值范围: {pkuph_proba.min():.4f} - {pkuph_proba.max():.4f}")
    print(f"正类概率均值: {pkuph_proba[:, 1].mean():.4f}")
    
    print("\n📊 测试Mayo模型:")
    print("-" * 30)
    
    # 测试Mayo模型
    mayo_model = MayoModel()
    print(f"模型特征: {mayo_model.get_feature_names()}")
    
    # 训练（实际上只是标记为已拟合）
    mayo_model.fit(mayo_data, y)
    
    # 预测
    mayo_pred = mayo_model.predict(mayo_data)
    mayo_proba = mayo_model.predict_proba(mayo_data)
    
    print(f"预测形状: {mayo_pred.shape}")
    print(f"概率形状: {mayo_proba.shape}")
    print(f"预测值范围: {mayo_pred.min()} - {mayo_pred.max()}")
    print(f"概率值范围: {mayo_proba.min():.4f} - {mayo_proba.max():.4f}")
    print(f"正类概率均值: {mayo_proba[:, 1].mean():.4f}")
    
    print("\n🔄 测试评估函数:")
    print("-" * 30)
    
    # 创建包含所有特征的测试数据
    all_features = list(set(pkuph_features + mayo_features))
    test_data = pd.DataFrame({
        feature: np.random.normal(0, 1, n_samples) for feature in all_features
    })
    
    # 评估两个模型
    results = evaluate_baseline_models(
        test_data, y, test_data, y, 
        models=['pkuph', 'mayo']
    )
    
    for model_name, result in results.items():
        if 'error' not in result:
            metrics = result['metrics']
            print(f"\n{model_name.upper()} 模型指标:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")
        else:
            print(f"{model_name.upper()} 模型出错: {result['error']}")
    
    print("\n✅ 基线模型测试完成!")

if __name__ == '__main__':
    test_baseline_models() 