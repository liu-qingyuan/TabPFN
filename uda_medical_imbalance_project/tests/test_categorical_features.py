#!/usr/bin/env python3
"""
测试不同特征集对应的类别特征

基于settings.py中的定义，验证best7和best10特征集中包含的类别特征
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data.loader import MedicalDataLoader

def test_categorical_features():
    """测试不同特征集的类别特征"""
    
    print("=" * 80)
    print("测试不同特征集对应的类别特征")
    print("=" * 80)
    
    # 创建数据加载器
    loader = MedicalDataLoader()
    
    # 定义测试的特征集类型（RFE预筛选）
    feature_types = ['best7', 'best8', 'best9', 'best10', 'all']
    
    for feature_type in feature_types:
        print(f"\n{feature_type.upper()} 特征集:")
        print("-" * 50)
        
        # 获取特征列表
        features = loader._get_features_by_type(feature_type)
        print(f"特征数量: {len(features)}")
        print(f"特征列表: {features}")
        
        # 获取类别特征索引
        categorical_indices = loader._get_categorical_indices(features)
        print(f"类别特征数量: {len(categorical_indices)}")
        print(f"类别特征索引: {categorical_indices}")
        
        # 显示具体的类别特征名称
        categorical_features = [features[i] for i in categorical_indices]
        print(f"类别特征名称: {categorical_features}")
        
        # 显示详细映射
        print("详细映射:")
        for i, feature in enumerate(features):
            is_categorical = i in categorical_indices
            status = "类别特征" if is_categorical else "数值特征"
            print(f"  索引{i:2d}: {feature:10s} -> {status}")
    
    print("\n" + "=" * 80)
    print("类别特征定义 (基于settings.py):")
    print("=" * 80)
    print("所有类别特征:")
    for i, cat_feature in enumerate(loader.CAT_FEATURE_NAMES):
        print(f"  {i+1:2d}. {cat_feature}")
    
    print(f"\n总计: {len(loader.CAT_FEATURE_NAMES)} 个类别特征")


def test_feature_intersection():
    """测试特征集与类别特征的交集"""
    
    print("\n" + "=" * 80)
    print("特征集与类别特征的交集分析")
    print("=" * 80)
    
    loader = MedicalDataLoader()
    
    feature_types = ['best7', 'best8', 'best9', 'best10', 'all']
    
    for feature_type in feature_types:
        features = loader._get_features_by_type(feature_type)
        
        # 计算交集
        categorical_in_set = set(features) & set(loader.CAT_FEATURE_NAMES)
        numerical_in_set = set(features) - set(loader.CAT_FEATURE_NAMES)
        
        print(f"\n{feature_type.upper()} 特征集分析:")
        print(f"  总特征数: {len(features)}")
        print(f"  类别特征数: {len(categorical_in_set)}")
        print(f"  数值特征数: {len(numerical_in_set)}")
        print(f"  类别特征比例: {len(categorical_in_set)/len(features)*100:.1f}%")
        
        print(f"  类别特征: {sorted(categorical_in_set)}")
        print(f"  数值特征: {sorted(numerical_in_set)}")


def test_specific_features():
    """测试特定特征集的详细信息"""
    
    print("\n" + "=" * 80)
    print("特定特征集详细信息")
    print("=" * 80)
    
    loader = MedicalDataLoader()
    
    # 基于settings.py的定义
    print("基于settings.py的特征定义:")
    print(f"BEST_7_FEATURES: {loader.BEST_7_FEATURES}")
    print(f"BEST_10_FEATURES: {loader.BEST_10_FEATURES}")
    
    # 验证best7中的类别特征
    print(f"\nBEST_7中的类别特征:")
    for feature in loader.BEST_7_FEATURES:
        is_categorical = feature in loader.CAT_FEATURE_NAMES
        print(f"  {feature}: {'类别特征' if is_categorical else '数值特征'}")
    
    # 验证best10中的类别特征
    print(f"\nBEST_10中的类别特征:")
    for feature in loader.BEST_10_FEATURES:
        is_categorical = feature in loader.CAT_FEATURE_NAMES
        print(f"  {feature}: {'类别特征' if is_categorical else '数值特征'}")


if __name__ == "__main__":
    try:
        test_categorical_features()
        test_feature_intersection()
        test_specific_features()
        
        print("\n" + "=" * 80)
        print("测试完成!")
        print("=" * 80)
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc() 