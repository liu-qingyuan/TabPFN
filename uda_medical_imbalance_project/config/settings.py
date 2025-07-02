"""
UDA Medical Imbalance Project - 配置设置

包含预定义的特征集、类别特征和数据路径配置
基于RFE预筛选的最优特征组合
"""

import logging
from typing import List, Dict, Any

# 全部63个特征（Feature1到Feature63）
ALL_63_FEATURES = [
    'Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Feature6', 'Feature7', 'Feature8', 'Feature9', 'Feature10',
    'Feature11', 'Feature12', 'Feature13', 'Feature14', 'Feature15', 'Feature16', 'Feature17', 'Feature18', 'Feature19', 'Feature20',
    'Feature21', 'Feature22', 'Feature23', 'Feature24', 'Feature25', 'Feature26', 'Feature27', 'Feature28', 'Feature29', 'Feature30',
    'Feature31', 'Feature32', 'Feature33', 'Feature34', 'Feature35', 'Feature36', 'Feature37', 'Feature38', 'Feature39', 'Feature40',
    'Feature41', 'Feature42', 'Feature43', 'Feature44', 'Feature45', 'Feature46', 'Feature47', 'Feature48', 'Feature49', 'Feature50',
    'Feature51', 'Feature52', 'Feature53', 'Feature54', 'Feature55', 'Feature56', 'Feature57', 'Feature58', 'Feature59', 'Feature60',
    'Feature61', 'Feature62', 'Feature63'
]

# 经过RFE筛选的58个特征（移除了Feature12, Feature33, Feature34, Feature36, Feature40）
SELECTED_58_FEATURES = [
    'Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Feature6', 'Feature7', 'Feature8', 'Feature9', 'Feature10',
    'Feature11', 'Feature13', 'Feature14', 'Feature15', 'Feature16', 'Feature17', 'Feature18', 'Feature19', 'Feature20', 'Feature21',
    'Feature22', 'Feature23', 'Feature24', 'Feature25', 'Feature26', 'Feature27', 'Feature28', 'Feature29', 'Feature30', 'Feature31',
    'Feature32', 'Feature35', 'Feature37', 'Feature38', 'Feature39', 'Feature41', 'Feature42', 'Feature43', 'Feature44', 'Feature45',
    'Feature46', 'Feature47', 'Feature48', 'Feature49', 'Feature50', 'Feature51', 'Feature52', 'Feature53', 'Feature54', 'Feature55',
    'Feature56', 'Feature57', 'Feature58', 'Feature59', 'Feature60', 'Feature61', 'Feature62', 'Feature63'
]

# 为了向后兼容，保留原名称
SELECTED_FEATURES = SELECTED_58_FEATURES

# 最佳7特征配置 (基于RFE预筛选结果)
BEST_7_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 
    'Feature56', 'Feature42', 'Feature39', 'Feature43'
]

# 最佳8特征配置 (基于RFE预筛选结果)
BEST_8_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature61',
    'Feature56', 'Feature42', 'Feature39', 'Feature43'
]

# 最佳9特征配置 (基于RFE预筛选结果)
BEST_9_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature61',
    'Feature56', 'Feature42', 'Feature39', 'Feature43', 'Feature48'
]

# 最佳10特征配置 (基于RFE预筛选结果)
BEST_10_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature61', 
    'Feature56', 'Feature42', 'Feature39', 'Feature43', 'Feature48', 'Feature5'
]

# 类别特征名称 (20个类别特征)
CAT_FEATURE_NAMES = [
    'Feature1', 'Feature3', 'Feature4', 'Feature5', 'Feature6', 'Feature7', 'Feature8', 'Feature9', 'Feature10', 'Feature11',
    'Feature45', 'Feature46', 'Feature49', 'Feature50', 'Feature51', 'Feature52', 'Feature53', 'Feature54', 'Feature55', 'Feature63'
]

# 数据文件路径
DATA_PATHS = {
    'A': "/home/24052432g/TabPFN/data/AI4healthcare.xlsx",
    'B': "/home/24052432g/TabPFN/data/HenanCancerHospital_features63_58.xlsx",
    'C': "/home/24052432g/TabPFN/data/GuangzhouMedicalHospital_features23_no_nan_new_fixed.xlsx"
}

# 标签列名
LABEL_COL = "Label"

# 数据集映射
DATASET_MAPPING = {
    'A': 'AI4health',
    'B': 'HenanCancerHospital', 
    'C': 'GuangzhouMedicalHospital'
}

def get_features_by_type(feature_type: str = 'best10') -> List[str]:
    """
    根据类型获取特征列表
    
    Args:
        feature_type: 特征集类型 ('all63', 'selected58', 'best7', 'best8', 'best9', 'best10')
        
    Returns:
        特征名称列表
    """
    if feature_type == 'all63':
        return ALL_63_FEATURES.copy()
    elif feature_type == 'selected58':
        return SELECTED_58_FEATURES.copy()
    elif feature_type == 'best7':
        return BEST_7_FEATURES.copy()
    elif feature_type == 'best8':
        return BEST_8_FEATURES.copy()
    elif feature_type == 'best9':
        return BEST_9_FEATURES.copy()
    elif feature_type == 'best10':
        return BEST_10_FEATURES.copy()
    else:
        raise ValueError(f"不支持的特征类型: {feature_type}. 支持的类型: ['all63', 'selected58', 'best7', 'best8', 'best9', 'best10']")

def get_categorical_features(feature_type: str = 'best10') -> List[str]:
    """
    获取指定特征集中的类别特征
    
    Args:
        feature_type: 特征集类型
        
    Returns:
        类别特征名称列表
    """
    selected_features = get_features_by_type(feature_type)
    categorical_features = []
    
    for feature in selected_features:
        if feature in CAT_FEATURE_NAMES:
            categorical_features.append(feature)
    
    return categorical_features

def get_categorical_indices(feature_type: str = 'best10') -> List[int]:
    """
    获取类别特征在选定特征中的索引
    
    Args:
        feature_type: 特征集类型
        
    Returns:
        类别特征索引列表
    """
    selected_features = get_features_by_type(feature_type)
    categorical_indices = []
    
    for i, feature in enumerate(selected_features):
        if feature in CAT_FEATURE_NAMES:
            categorical_indices.append(i)
    
    return categorical_indices

def get_feature_set_info(feature_type: str = 'best10') -> Dict[str, Any]:
    """
    获取特征集的详细信息
    
    Args:
        feature_type: 特征集类型
        
    Returns:
        包含特征集信息的字典
    """
    features = get_features_by_type(feature_type)
    categorical_features = get_categorical_features(feature_type)
    categorical_indices = get_categorical_indices(feature_type)
    
    return {
        'feature_type': feature_type,
        'total_features': len(features),
        'feature_names': features,
        'categorical_features': categorical_features,
        'categorical_indices': categorical_indices,
        'categorical_count': len(categorical_features),
        'numerical_count': len(features) - len(categorical_features),
        'categorical_ratio': len(categorical_features) / len(features) * 100
    }

# 日志级别
LOG_LEVEL = logging.INFO

# 实验配置
EXPERIMENT_CONFIG = {
    'random_state': 42,
    'cv_folds': 10,
    'test_size': 0.2,
    'validation_size': 0.2
}

# 可视化配置
VISUALIZATION_CONFIG = {
    'figure_size': (10, 8),
    'dpi': 300,
    'style': 'seaborn-v0_8',
    'color_palette': 'viridis',
    'font_size': 12
}

# 支持的不平衡处理方法
IMBALANCE_METHODS = [
    'none',
    'smote',
    'smotenc',
    'borderline_smote',
    'kmeans_smote',
    'svm_smote',
    'adasyn',
    'smote_tomek',
    'smote_enn',
    'random_under',
    'edited_nn'
]

# 支持的标准化方法
SCALING_METHODS = [
    'none',
    'standard',
    'robust'
]

# 支持的UDA方法
UDA_METHODS = {
    'covariate_shift': ['DM'],
    'linear_kernel': ['SA', 'TCA', 'JDA', 'CORAL'],
    'deep_learning': ['DANN', 'ADDA', 'WDGRL', 'DeepCORAL', 'MCD', 'MDD', 'CDAN'],
    'optimal_transport': ['POT']
}

if __name__ == "__main__":
    # 测试特征集配置
    print("=" * 60)
    print("特征集配置测试")
    print("=" * 60)
    
    for feature_type in ['all63', 'selected58', 'best7', 'best8', 'best9', 'best10']:
        info = get_feature_set_info(feature_type)
        print(f"\n{feature_type.upper()}特征集:")
        print(f"  总特征数: {info['total_features']}")
        print(f"  类别特征数: {info['categorical_count']}")
        print(f"  数值特征数: {info['numerical_count']}")
        print(f"  类别特征比例: {info['categorical_ratio']:.1f}%")
        print(f"  类别特征: {info['categorical_features']}") 