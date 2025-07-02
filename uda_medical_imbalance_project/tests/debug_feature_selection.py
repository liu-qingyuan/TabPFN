"""
调试特征选择功能
验证TABPFN模型是否正确使用了指定的特征集
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.loader import MedicalDataLoader
from config.settings import BEST_7_FEATURES, SELECTED_FEATURES
from evaluation.cross_validation import CrossValidationEvaluator
import pandas as pd

def debug_feature_selection():
    """调试特征选择功能"""
    print("="*80)
    print("调试特征选择功能")
    print("="*80)
    
    # 1. 检查配置的特征
    print(f"BEST_7_FEATURES ({len(BEST_7_FEATURES)}个): {BEST_7_FEATURES}")
    print(f"SELECTED_FEATURES ({len(SELECTED_FEATURES)}个): {SELECTED_FEATURES[:10]}...")
    print()
    
    # 2. 加载数据
    print("加载数据集A...")
    data_loader = MedicalDataLoader()
    dataset_A = data_loader.load_dataset('A', 'all')  # 加载所有特征
    
    X = pd.DataFrame(dataset_A['X'], columns=dataset_A['feature_names'])
    y = pd.Series(dataset_A['y'])
    
    print(f"原始数据形状: {X.shape}")
    print(f"原始特征名称 (前10个): {X.columns[:10].tolist()}")
    print()
    
    # 3. 测试CrossValidationEvaluator的特征选择
    print("测试CrossValidationEvaluator的特征选择:")
    
    # 创建TabPFN评估器，使用best7特征
    cv_evaluator = CrossValidationEvaluator(
        model_type='tabpfn',
        feature_set='best7',
        scaler_type='standard',
        imbalance_method='smote',
        cv_folds=2,  # 只用2折快速测试
        random_state=42,
        verbose=False
    )
    
    print(f"评估器配置:")
    print(f"  模型类型: {cv_evaluator.model_type}")
    print(f"  特征集: {cv_evaluator.feature_set}")
    print(f"  选择的特征: {cv_evaluator.features}")
    print(f"  特征数量: {len(cv_evaluator.features)}")
    print()
    
    # 4. 验证特征选择是否正确
    print("验证特征选择:")
    
    # 检查评估器是否选择了正确的特征
    if cv_evaluator.features == BEST_7_FEATURES:
        print("✓ 特征选择正确: 使用了BEST_7_FEATURES")
    else:
        print("✗ 特征选择错误!")
        print(f"  期望: {BEST_7_FEATURES}")
        print(f"  实际: {cv_evaluator.features}")
    
    # 检查数据中是否包含所需特征
    missing_features = [f for f in BEST_7_FEATURES if f not in X.columns]
    if missing_features:
        print(f"✗ 数据中缺少特征: {missing_features}")
    else:
        print("✓ 数据中包含所有BEST_7特征")
    
    print()
    
    # 5. 模拟特征选择过程
    print("模拟特征选择过程:")
    X_selected = X[BEST_7_FEATURES].copy()
    print(f"选择特征后数据形状: {X_selected.shape}")
    print(f"选择的特征列: {X_selected.columns.tolist()}")
    
    # 6. 验证其他模型的特征选择
    print("\n验证其他模型的特征选择:")
    
    model_configs = [
        ('pkuph', 'all'),
        ('mayo', 'all'), 
        ('paper_lr', 'all'),
        ('tabpfn', 'best8'),
        ('tabpfn', 'all')
    ]
    
    for model_type, feature_set in model_configs:
        evaluator = CrossValidationEvaluator(
            model_type=model_type,
            feature_set=feature_set,
            verbose=False
        )
        print(f"  {model_type} + {feature_set}: {len(evaluator.features)}个特征")
        if len(evaluator.features) <= 10:
            print(f"    特征: {evaluator.features}")
        else:
            print(f"    特征 (前5个): {evaluator.features[:5]}")

if __name__ == "__main__":
    debug_feature_selection() 