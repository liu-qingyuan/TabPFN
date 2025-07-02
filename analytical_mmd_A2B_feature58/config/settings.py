import logging
from typing import Optional, List

# 全局常量 - 58个选定特征
SELECTED_FEATURES = [
    'Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Feature6', 'Feature7', 'Feature8', 'Feature9', 'Feature10',
    'Feature11', 'Feature13', 'Feature14', 'Feature15', 'Feature16', 'Feature17', 'Feature18', 'Feature19', 'Feature20', 'Feature21',
    'Feature22', 'Feature23', 'Feature24', 'Feature25', 'Feature26', 'Feature27', 'Feature28', 'Feature29', 'Feature30', 'Feature31',
    'Feature32', 'Feature35', 'Feature37', 'Feature38', 'Feature39', 'Feature41', 'Feature42', 'Feature43', 'Feature44', 'Feature45',
    'Feature46', 'Feature47', 'Feature48', 'Feature49', 'Feature50', 'Feature51', 'Feature52', 'Feature53', 'Feature54', 'Feature55',
    'Feature56', 'Feature57', 'Feature58', 'Feature59', 'Feature60', 'Feature61', 'Feature62', 'Feature63'
]

# 最佳7特征配置 (基于实验结果)
BEST_7_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 
    'Feature56', 'Feature42', 'Feature39', 'Feature43'
]

# 最佳11特征配置 (用户指定的特征)
BEST_10_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature61', 
    'Feature56', 'Feature42', 'Feature39', 'Feature43', 'Feature48', 'Feature5'
]

# 类别特征名称 (与CORAL版本保持一致)
CAT_FEATURE_NAMES = [
    'Feature1', 'Feature3', 'Feature4', 'Feature5', 'Feature6', 'Feature7', 'Feature8', 'Feature9', 'Feature10', 'Feature11',
    'Feature45', 'Feature46', 'Feature49', 'Feature50', 'Feature51', 'Feature52', 'Feature53', 'Feature54', 'Feature55', 'Feature63'
]

# 最佳7特征中的类别特征
BEST_7_CAT_FEATURES = ['Feature63', 'Feature46']

# 最佳11特征中的类别特征
BEST_11_CAT_FEATURES = ['Feature63', 'Feature46', 'Feature5']

# 类别特征索引 (在58个特征中的位置)
CAT_IDX = [SELECTED_FEATURES.index(f) for f in CAT_FEATURE_NAMES if f in SELECTED_FEATURES]

# 最佳7特征中的类别特征索引
BEST_7_CAT_IDX = [BEST_7_FEATURES.index(f) for f in BEST_7_CAT_FEATURES if f in BEST_7_FEATURES]

# 最佳10特征对应的类别特征（从BEST_10_FEATURES中筛选出的类别特征）
BEST_10_CAT_FEATURES = [f for f in BEST_10_FEATURES if f in ['Feature63', 'Feature46', 'Feature61']]

# 最佳10特征中的类别特征索引
BEST_10_CAT_IDX = [BEST_10_FEATURES.index(f) for f in BEST_10_CAT_FEATURES if f in BEST_10_FEATURES]

# 多模型配置
MODEL_CONFIGS = {
    'auto': {
        # 根据 best_params.json 的最佳参数配置
        'max_time': 30,
        'preset': 'default',
        'ges_scoring_string': 'accuracy',  # best_params.json中的ges_scoring
        'device': 'cuda',
        'random_state': 42,
        'ignore_pretraining_limits': False,  # best_params.json中的ignore_limits: false
        'categorical_feature_indices': None,  # 将在模型创建时动态设置
        'phe_init_args': {
            'max_models': 15,  # best_params.json中的max_models
            'validation_method': 'cv',  # best_params.json中的validation_method
            'n_repeats': 100,  # best_params.json中的n_repeats
            'n_folds': 10,  # best_params.json中的n_folds
            'holdout_fraction': 0.4,  # best_params.json中的holdout_fraction
            'ges_n_iterations': 20  # best_params.json中的ges_n_iterations
        }
    },
    'tuned': {
        'random_state': 42
    },
    'base': {
        'device': 'cuda',
        'random_state': 42
    },
    'rf': {
        'n_estimators': 10,
        'max_depth': None,
        'random_state': 42,
        'n_jobs': -1
    }
}

# TabPFN模型参数 (保持向后兼容)
TABPFN_PARAMS = MODEL_CONFIGS['auto']

# 域适应方法配置 - 包含MMD、CORAL和Mean-Variance Alignment
MMD_METHODS = {
    'linear': {
        'n_epochs': 200,
        'lr': 3e-4,  # 使用测试中验证的更小的学习率
        'batch_size': 64,
        'lambda_reg': 1e-3,  # 正则化
        'staged_training': True,  # 使用测试中验证的分阶段训练
        'dynamic_gamma': True,  # 启用动态gamma搜索
        'gamma_search_values': [0.01, 0.05, 0.1],  # 您指定的gamma搜索范围
        'standardize_features': True,  # 标准化输入特征
        'use_gradient_clipping': True,  # 开启梯度裁剪
        'max_grad_norm': 1.0,
        'monitor_gradients': True  # 监控梯度范数
    },
    'kpca': {
        'kernel': 'rbf',
        'gamma': 0.05,  # 使用测试中验证的gamma值
        'n_components': 10,  # 限制组件数避免过拟合
        'use_inverse_transform': False,
        'standardize': True
    },
    'mean_std': {
        # 简单的均值标准差对齐，无需额外参数
    },
    'coral': {
        'regularization': 1e-6,  # CORAL正则化参数
        'standardize': True  # 是否标准化特征
    },
    'adaptive_coral': {
        'regularization_range': (1e-8, 1e-3),  # 正则化参数搜索范围
        'n_trials': 10,  # 搜索试验次数
        'standardize': True
    },
    'mean_variance': {
        'align_mean': True,  # 是否对齐均值
        'align_variance': True,  # 是否对齐方差
        'standardize': False  # 是否预先标准化特征
    },
    'adaptive_mean_variance': {
        'standardize': False  # 自适应均值-方差对齐
    },
    'tca': {
        'subspace_dim': 10,  # 子空间维度
        'gamma': None,  # RBF核参数（None表示使用中值启发式）
        'regularization': 1e-3,  # 正则化参数
        'standardize': True,  # 是否标准化特征
        'center_kernel': True,  # 是否中心化核矩阵
        'use_sparse_solver': True  # 是否使用稀疏特征值求解器
    },
    'adaptive_tca': {
        'subspace_dim_range': (5, 20),  # 子空间维度搜索范围
        'gamma_range': (0.1, 10.0),  # gamma参数搜索范围
        'n_trials': 10,  # 搜索试验次数
        'standardize': True,  # 是否标准化特征
        'center_kernel': True,  # 是否中心化核矩阵
        'use_sparse_solver': True  # 是否使用稀疏特征值求解器
    },
    'jda': {
        'subspace_dim': 10,  # 子空间维度（推荐范围：5-50）
        'gamma': None,  # RBF核参数（None表示使用中值启发式，推荐范围：0.001-10）
        'mu': 0.5,  # 边缘分布和条件分布的权重平衡参数（推荐范围：0.2-0.8）
        'max_iterations': 5,  # 最大迭代次数（推荐不超过10次）
        'regularization': 1e-3,  # 正则化参数（推荐较小值）
        'standardize': True,  # 是否标准化特征
        'center_kernel': True,  # 是否中心化核矩阵
        'use_sparse_solver': True,  # 是否使用稀疏特征值求解器
        'confidence_threshold': 0.7  # 伪标签置信度阈值（推荐范围：0.6-0.9）
    },
    'adaptive_jda': {
        'subspace_dim_range': (5, 30),  # 子空间维度搜索范围
        'gamma_range': (0.001, 10.0),  # gamma参数搜索范围（扩展范围）
        'mu_range': (0.2, 0.8),  # mu参数搜索范围（聚焦稳定区间）
        'n_trials': 12,  # 搜索试验次数
        'max_iterations': 3,  # 每次试验的最大迭代次数（减少避免过拟合）
        'standardize': True,  # 是否标准化特征
        'center_kernel': True,  # 是否中心化核矩阵
        'use_sparse_solver': True,  # 是否使用稀疏特征值求解器
        'confidence_threshold': 0.7  # 伪标签置信度阈值
    }
}

# 结果输出路径
BASE_PATH = 'results_analytical_mmd_A2B_feature58'

# 数据文件路径 (使用绝对路径)
DATA_PATHS = {
    'A': "/home/24052432g/TabPFN/data/AI4healthcare.xlsx",
    'B': "/home/24052432g/TabPFN/data/HenanCancerHospital_features63_58.xlsx",
    'C': "/home/24052432g/TabPFN/data/GuangzhouMedicalHospital_features23_no_nan_new_fixed.xlsx"  # 更新后的广州医科大学数据集
}

# 保持向后兼容
DATA_PATH_A = DATA_PATHS['A']
DATA_PATH_B = DATA_PATHS['B']
DATA_PATH_C = DATA_PATHS['C']  # 添加C数据集的快捷路径

# 标签列名
LABEL_COL = "Label"

# 可视化配置
VISUALIZATION_CONFIG = {
    'tsne_perplexity': 30,
    'tsne_n_iter': 1000,
    'histogram_bins': 30,
    'figure_dpi': 300,
    'n_features_to_plot': 12  # 在直方图中显示的特征数量
}

# 日志级别
LOG_LEVEL = logging.INFO

# 实验配置
EXPERIMENT_CONFIG = {
    'test_size': 0.2,  # 源域验证集比例
    'random_state': 42,
    'optimize_threshold': True,  # 是否优化决策阈值
    'cross_validation': True,  # 是否使用交叉验证
    'save_visualizations': True,  # 是否保存可视化结果
    'save_models': False  # 是否保存训练好的模型
}

# 跨域实验配置
CROSS_DOMAIN_CONFIG = {
    'use_best_7_features': True,  # 是否使用最佳7特征
    'enable_cross_validation': True,  # 是否启用交叉验证
    'cv_folds': 10,  # 交叉验证折数
    'evaluate_all_datasets': True,  # 是否评估所有数据集
    'save_detailed_results': True  # 是否保存详细结果
}

# 模型预设配置 - 简化版本
MODEL_PRESETS = {
    'default': {
        'auto': {},
        'tuned': {},
        'base': {},
        'rf': {}
    }
}

def get_features_by_type(feature_type: str = 'all'):
    """
    根据类型获取特征列表
    
    参数:
    - feature_type: 特征类型 ('all', 'best7', 'best11', 'categorical')
    
    返回:
    - list: 特征名称列表
    """
    if feature_type == 'all':
        return SELECTED_FEATURES
    elif feature_type == 'best7':
        return BEST_7_FEATURES
    elif feature_type == 'best10':
        return BEST_10_FEATURES
    elif feature_type == 'categorical':
        return CAT_FEATURE_NAMES
    else:
        raise ValueError(f"不支持的特征类型: {feature_type}")

def get_categorical_indices(feature_type: str = 'all'):
    """
    根据特征类型获取类别特征索引
    
    参数:
    - feature_type: 特征类型 ('all', 'best7', 'best11')
    
    返回:
    - list: 类别特征索引列表
    """
    if feature_type == 'all':
        return CAT_IDX
    elif feature_type == 'best7':
        return BEST_7_CAT_IDX
    elif feature_type == 'best10':
        return BEST_10_CAT_IDX
    else:
        raise ValueError(f"不支持的特征类型: {feature_type}")

def get_model_config(model_type: str, preset: str = 'balanced', categorical_feature_indices: Optional[List[int]] = None):
    """
    获取模型配置
    
    参数:
    - model_type: 模型类型 ('auto', 'base', 'rf')
    - preset: 预设类型 ('fast', 'balanced', 'accurate')
    - categorical_feature_indices: 类别特征索引列表
    
    返回:
    - dict: 模型配置字典
    """
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    base_config = MODEL_CONFIGS[model_type].copy()
    
    if preset in MODEL_PRESETS and model_type in MODEL_PRESETS[preset]:
        base_config.update(MODEL_PRESETS[preset][model_type])
    
    # 如果提供了categorical_feature_indices，添加到配置中
    if categorical_feature_indices is not None:
        base_config['categorical_feature_indices'] = categorical_feature_indices
    
    return base_config 