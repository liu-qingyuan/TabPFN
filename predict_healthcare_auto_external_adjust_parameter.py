import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import argparse
import json
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

from typing import List, Dict, Any, Tuple
from functools import partial

import logging
import sys

# 先移除 root logger 里所有的 handler
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# 创建两个 handler：
stdout_handler = logging.StreamHandler(sys.stdout)  # 处理 INFO 及以上的日志
stderr_handler = logging.StreamHandler(sys.stderr)  # 处理 WARNING 及以上的日志

# 设置不同的日志级别：
stdout_handler.setLevel(logging.INFO)   # 只处理 INFO及以上
stderr_handler.setLevel(logging.WARNING)  # 只处理 WARNING 及以上


# 设置日志格式
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
stdout_handler.setFormatter(formatter)
stderr_handler.setFormatter(formatter)

# 把 handler 添加到 root logger
logging.root.addHandler(stdout_handler)
logging.root.addHandler(stderr_handler)
logging.root.setLevel(logging.INFO)  # 让 root logger 处理 INFO 及以上的日志

# 测试日志输出
logging.debug("This is a debug message")     # 不会输出
logging.info("This is an info message")      # 只进入 output.log
logging.warning("This is a warning message")  # 进入output.log以及error.log
logging.error("This is an error message")    # 进入output.log以及error.log
logging.critical("This is a critical message")  # 进入output.log以及error.log





# 尝试导入贝叶斯优化库
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    logging.warning("警告: scikit-optimize未安装，将使用随机搜索。安装方法: pip install scikit-optimize")
    SKOPT_AVAILABLE = False

# Create results directory
os.makedirs('./results/best_7_features', exist_ok=True)

# Define the best 7 features
best_features = [
    'Feature63', 'Feature2', 'Feature46', 
    'Feature56', 'Feature42', 'Feature39', 'Feature43'
]

logging.info("\nEvaluating Best 7 Features:")
logging.info(best_features)

# Load data
logging.info("\nLoading datasets...")
train_df = pd.read_excel("data/AI4healthcare.xlsx")
external_df = pd.read_excel("data/HenanCancerHospital_features63_58.xlsx")

# Prepare data
X_train = train_df[best_features].copy()
y_train = train_df["Label"].copy()
X_external = external_df[best_features].copy()
y_external = external_df["Label"].copy()

logging.info("\nData Information:")
logging.info(f"Training Data Shape: {X_train.shape}")
logging.info(f"External Data Shape: {X_external.shape}")
logging.info(f"\nTraining Label Distribution:\n {y_train.value_counts()}")
logging.info(f"External Label Distribution:\n {y_external.value_counts()}")

# Apply standard scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_external_scaled = scaler.transform(X_external)

# ==============================
# 1. 10-fold Cross Validation on Training Data
# ==============================
logging.info("\n10-fold Cross Validation Results:")
logging.info("=" * 50)

kf = KFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled), 1):
    X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # Train model
    model = AutoTabPFNClassifier(device='cuda', max_time=2, random_state=42)
    model.fit(X_fold_train, y_fold_train)
    
    # Evaluate
    y_val_pred = model.predict(X_fold_val)
    y_val_proba = model.predict_proba(X_fold_val)
    
    # Calculate metrics
    fold_acc = accuracy_score(y_fold_val, y_val_pred)
    fold_auc = roc_auc_score(y_fold_val, y_val_proba[:, 1])
    fold_f1 = f1_score(y_fold_val, y_val_pred)
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_fold_val, y_val_pred)
    fold_acc_0 = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
    fold_acc_1 = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
    
    cv_scores.append({
        'fold': fold,
        'accuracy': fold_acc,
        'auc': fold_auc,
        'f1': fold_f1,
        'acc_0': fold_acc_0,
        'acc_1': fold_acc_1
    })
    
    logging.info(f"\nFold {fold}:")
    logging.info(f"Accuracy: {fold_acc:.4f}")
    logging.info(f"AUC: {fold_auc:.4f}")
    logging.info(f"F1: {fold_f1:.4f}")
    logging.info(f"Class 0 Accuracy: {fold_acc_0:.4f}")
    logging.info(f"Class 1 Accuracy: {fold_acc_1:.4f}")

# Calculate and print average cross-validation scores
cv_df = pd.DataFrame(cv_scores)
logging.info("\nAverage Cross-validation Results:")
logging.info("=" * 50)
logging.info(f"Accuracy: {cv_df['accuracy'].mean():.4f} ± {cv_df['accuracy'].std():.4f}")
logging.info(f"AUC: {cv_df['auc'].mean():.4f} ± {cv_df['auc'].std():.4f}")
logging.info(f"F1: {cv_df['f1'].mean():.4f} ± {cv_df['f1'].std():.4f}")
logging.info(f"Class 0 Accuracy: {cv_df['acc_0'].mean():.4f} ± {cv_df['acc_0'].std():.4f}")
logging.info(f"Class 1 Accuracy: {cv_df['acc_1'].mean():.4f} ± {cv_df['acc_1'].std():.4f}")

# ==============================
# 2. External Validation
# ==============================
logging.info("\nExternal Validation Results:")
logging.info("=" * 50)

# Train final model on full training data
final_model = AutoTabPFNClassifier(device='cuda', max_time=2, random_state=42)
final_model.fit(X_train_scaled, y_train)

# Evaluate on external data
y_external_pred = final_model.predict(X_external_scaled)
y_external_proba = final_model.predict_proba(X_external_scaled)

# Calculate metrics
external_acc = accuracy_score(y_external, y_external_pred)
external_auc = roc_auc_score(y_external, y_external_proba[:, 1])
external_f1 = f1_score(y_external, y_external_pred)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_external, y_external_pred)
external_acc_0 = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
external_acc_1 = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])

logging.info("\nExternal Validation Metrics:")
logging.info(f"Accuracy: {external_acc:.4f}")
logging.info(f"AUC: {external_auc:.4f}")
logging.info(f"F1: {external_f1:.4f}")
logging.info(f"Class 0 Accuracy: {external_acc_0:.4f}")
logging.info(f"Class 1 Accuracy: {external_acc_1:.4f}")
logging.info(f"\nConfusion Matrix:\n{conf_matrix}")

# Save results
results = {
    'cross_validation': {
        'accuracy': f"{cv_df['accuracy'].mean():.4f} ± {cv_df['accuracy'].std():.4f}",
        'auc': f"{cv_df['auc'].mean():.4f} ± {cv_df['auc'].std():.4f}",
        'f1': f"{cv_df['f1'].mean():.4f} ± {cv_df['f1'].std():.4f}",
        'acc_0': f"{cv_df['acc_0'].mean():.4f} ± {cv_df['acc_0'].std():.4f}",
        'acc_1': f"{cv_df['acc_1'].mean():.4f} ± {cv_df['acc_1'].std():.4f}"
    },
    'external_validation': {
        'accuracy': f"{external_acc:.4f}",
        'auc': f"{external_auc:.4f}",
        'f1': f"{external_f1:.4f}",
        'acc_0': f"{external_acc_0:.4f}",
        'acc_1': f"{external_acc_1:.4f}",
        'confusion_matrix': conf_matrix.tolist()
    }
}

# Save to file
with open('./results/best_7_features/results.txt', 'w') as f:
    f.write("Best 7 Features Evaluation Results\n")
    f.write("=" * 50 + "\n\n")
    
    f.write("Features:\n")
    for i, feature in enumerate(best_features, 1):
        f.write(f"{i}. {feature}\n")
    
    f.write("\nCross-validation Results:\n")
    f.write("-" * 30 + "\n")
    for metric, value in results['cross_validation'].items():
        f.write(f"{metric}: {value}\n")
    
    f.write("\nExternal Validation Results:\n")
    f.write("-" * 30 + "\n")
    for metric, value in results['external_validation'].items():
        if metric != 'confusion_matrix':
            f.write(f"{metric}: {value}\n")
    f.write(f"\nConfusion Matrix:\n{conf_matrix}")

logging.info("\nResults have been saved to: ./results/best_7_features/results.txt")

# 添加命令行参数解析
def parse_args():
    parser = argparse.ArgumentParser(description='AutoTabPFN Healthcare Prediction with Parameter Tuning')
    
    # 基本参数
    parser.add_argument('--output_dir', type=str, default='./results_hyperopt_best7',
                        help='输出结果的目录')
    parser.add_argument('--features', type=str, nargs='+', 
                        default=['Feature63', 'Feature2', 'Feature46', 
                                'Feature56', 'Feature42', 'Feature39', 'Feature43'],
                        help='用于预测的特征列表')
    parser.add_argument('--categorical_features', type=str, nargs='+',
                        default=['Feature63', 'Feature46'],
                        help='可选的分类特征列表')
    parser.add_argument('--device', type=str, default='cuda', 
                        choices=['cpu', 'cuda'],
                        help='计算设备')
    parser.add_argument('--random_state', type=int, default=42,
                        help='随机种子，用于可重复性')
    parser.add_argument('--n_trials', type=int, default=300,
                        help='超参数组合尝试次数')
    parser.add_argument('--n_cv_folds', type=int, default=10,
                        help='交叉验证折数')
    parser.add_argument('--use_bayesian', action='store_true',
                        help='是否使用贝叶斯优化(需要scikit-optimize)')
    
    return parser.parse_args()

def get_categorical_indices(all_features: List[str], categorical_features: List[str]) -> List[int]:
    """获取分类特征在特征列表中的索引位置
    
    Args:
        all_features: 所有特征的列表
        categorical_features: 分类特征的列表
        
    Returns:
        分类特征的索引列表
    """
    indices = []
    for cat_feature in categorical_features:
        if cat_feature in all_features:
            indices.append(all_features.index(cat_feature))
    return indices

def get_parameter_space_bayesian() -> Dict[str, Any]:
    """定义贝叶斯优化的参数搜索空间，根据之前的实验结果进行调整"""
    param_space = {
        'use_categorical': Categorical([True, False]),
        'max_time': Integer(15, 60),  # 根据最佳结果30秒，缩小范围到15-60秒
        'preset': Categorical(['default', 'avoid_overfitting']),
        'ges_scoring': Categorical(['f1', 'roc', 'accuracy']),  # 根据最佳结果使用f1，调整顺序
        'max_models': Categorical([5, 10, 15, 20, 25]),  # 根据最佳结果10，缩小范围
        'validation_method': Categorical(['holdout', 'cv']),
        'n_repeats': Integer(100, 200),  # 根据最佳结果150，调整范围
        'n_folds': Categorical([5, 10]),  # 根据最佳结果10，移除None选项
        'holdout_fraction': Real(0.3, 0.5),  # 根据最佳结果0.5，调整范围
        'ges_n_iterations': Integer(15, 25),  # 根据最佳结果20，调整范围
        'ignore_limits': Categorical([True, False]),
    }
    return param_space

def get_parameter_grid() -> Dict[str, List[Any]]:
    """定义扩展的参数网格搜索空间，根据之前的实验结果进行调整"""
    param_grid = {
        'max_time': [15, 30, 45, 60],  # 根据最佳结果30秒，调整范围
        'preset': ['default', 'avoid_overfitting'],
        'ges_scoring': ['f1', 'roc', 'accuracy'],  # 根据最佳结果使用f1，调整顺序
        'max_models': [5, 10, 15, 20, 25],  # 根据最佳结果10，缩小范围
        'validation_method': ['holdout', 'cv'],
        'n_repeats': [100, 150, 200],  # 根据最佳结果150，调整范围
        'n_folds': [5, 10],  # 根据最佳结果10，移除None选项
        'holdout_fraction': [0.3, 0.4, 0.5],  # 根据最佳结果0.5，调整范围
        'ges_n_iterations': [15, 20, 25],  # 根据最佳结果20，调整范围
        'ignore_limits': [True, False],
    }
    return param_grid

def evaluate_model_on_external(
    model: Any, 
    X_external: np.ndarray, 
    y_external: pd.Series, 
    n_folds: int = 10,
    random_state: int = 42
) -> Dict[str, Any]:
    """使用K折交叉验证评估模型在外部数据集上的性能
    
    Args:
        model: 训练好的模型
        X_external: 外部测试集特征
        y_external: 外部测试集标签
        n_folds: 交叉验证折数
        random_state: 随机种子
        
    Returns:
        包含各种性能指标的字典
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    fold_results = []
    all_preds = []
    all_probs = []
    all_true = []
    
    for fold, (_, test_idx) in enumerate(kf.split(X_external), 1):
        X_test_fold = X_external[test_idx]
        y_test_fold = y_external.iloc[test_idx]
        
        # 预测
        y_pred = model.predict(X_test_fold)
        y_proba = model.predict_proba(X_test_fold)
        
        # 保存预测结果
        all_preds.extend(y_pred)
        all_probs.extend(y_proba[:, 1])
        all_true.extend(y_test_fold)
        
        # 计算指标
        fold_acc = accuracy_score(y_test_fold, y_pred)
        try:
            fold_auc = roc_auc_score(y_test_fold, y_proba[:, 1])
        except:
            fold_auc = 0.5  # 如果出现错误，设置为默认值
        fold_f1 = f1_score(y_test_fold, y_pred)
        
        # 计算混淆矩阵
        conf_matrix = confusion_matrix(y_test_fold, y_pred)
        # 避免除以零错误
        if conf_matrix[0, 0] + conf_matrix[0, 1] > 0:
            fold_acc_0 = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
        else:
            fold_acc_0 = 0
            
        if conf_matrix[1, 0] + conf_matrix[1, 1] > 0:
            fold_acc_1 = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
        else:
            fold_acc_1 = 0
        
        fold_results.append({
            'fold': fold,
            'accuracy': fold_acc,
            'auc': fold_auc,
            'f1': fold_f1,
            'acc_0': fold_acc_0,
            'acc_1': fold_acc_1
        })
    
    # 计算整体指标
    all_true = np.array(all_true)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    overall_acc = accuracy_score(all_true, all_preds)
    overall_auc = roc_auc_score(all_true, all_probs)
    overall_f1 = f1_score(all_true, all_preds)
    
    # 计算整体混淆矩阵
    overall_cm = confusion_matrix(all_true, all_preds)
    overall_acc_0 = overall_cm[0, 0] / (overall_cm[0, 0] + overall_cm[0, 1]) if (overall_cm[0, 0] + overall_cm[0, 1]) > 0 else 0
    overall_acc_1 = overall_cm[1, 1] / (overall_cm[1, 0] + overall_cm[1, 1]) if (overall_cm[1, 0] + overall_cm[1, 1]) > 0 else 0
    
    # 计算平均和标准差
    metrics_df = pd.DataFrame(fold_results)
    
    return {
        'fold_results': fold_results,
        'overall': {
            'accuracy': overall_acc,
            'auc': overall_auc,
            'f1': overall_f1,
            'acc_0': overall_acc_0,
            'acc_1': overall_acc_1,
            'confusion_matrix': overall_cm.tolist()
        },
        'means': {
            'accuracy': metrics_df['accuracy'].mean(),
            'auc': metrics_df['auc'].mean(),
            'f1': metrics_df['f1'].mean(),
            'acc_0': metrics_df['acc_0'].mean(),
            'acc_1': metrics_df['acc_1'].mean(),
        },
        'stds': {
            'accuracy': metrics_df['accuracy'].std(),
            'auc': metrics_df['auc'].std(),
            'f1': metrics_df['f1'].std(),
            'acc_0': metrics_df['acc_0'].std(),
            'acc_1': metrics_df['acc_1'].std(),
        }
    }

def train_and_evaluate_model(
    X_train: np.ndarray, 
    y_train: pd.Series, 
    X_external: np.ndarray, 
    y_external: pd.Series,
    params: Dict[str, Any],
    args: Any,
    use_categorical: bool = True
) -> Tuple[Dict[str, Any], float]:
    """根据给定参数训练模型并在训练集和外部数据集上评估"""
    # 获取分类特征索引
    categorical_indices = get_categorical_indices(args.features, args.categorical_features) if use_categorical else []
    
    # 创建模型参数配置
    phe_init_args = {}
    if params.get('max_models') is not None:
        phe_init_args['max_models'] = params['max_models']
    if params.get('validation_method'):
        phe_init_args['validation_method'] = params['validation_method']
    if params.get('n_repeats'):
        phe_init_args['n_repeats'] = params['n_repeats']
    if params.get('n_folds'):
        phe_init_args['n_folds'] = params['n_folds']
    if params.get('holdout_fraction'):
        phe_init_args['holdout_fraction'] = params['holdout_fraction']
    if params.get('ges_n_iterations'):
        phe_init_args['ges_n_iterations'] = params['ges_n_iterations']
    
    # 创建并训练模型
    model = AutoTabPFNClassifier(
        max_time=params.get('max_time', 60),
        preset=params.get('preset', 'default'),
        ges_scoring_string=params.get('ges_scoring', 'roc'),
        device=args.device,
        random_state=args.random_state,
        ignore_pretraining_limits=params.get('ignore_limits', False),
        categorical_feature_indices=categorical_indices if categorical_indices else None,
        phe_init_args=phe_init_args
    )
    
    # 训练和记录时间
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # 在训练集上评估
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)
    
    # 计算训练集指标
    train_acc = accuracy_score(y_train, y_train_pred)
    train_auc = roc_auc_score(y_train, y_train_proba[:, 1])
    train_f1 = f1_score(y_train, y_train_pred)
    
    # 计算训练集混淆矩阵
    train_conf_matrix = confusion_matrix(y_train, y_train_pred)
    train_acc_0 = train_conf_matrix[0, 0] / (train_conf_matrix[0, 0] + train_conf_matrix[0, 1])
    train_acc_1 = train_conf_matrix[1, 1] / (train_conf_matrix[1, 0] + train_conf_matrix[1, 1])
    
    # 使用交叉验证评估外部数据集
    evaluation_results = evaluate_model_on_external(
        model, 
        X_external, 
        y_external, 
        n_folds=args.n_cv_folds, 
        random_state=args.random_state
    )
    
    # 添加训练集结果和训练时间
    evaluation_results['train_metrics'] = {
        'accuracy': train_acc,
        'auc': train_auc,
        'f1': train_f1,
        'acc_0': train_acc_0,
        'acc_1': train_acc_1,
        'confusion_matrix': train_conf_matrix.tolist()
    }
    evaluation_results['train_time'] = train_time
    
    # 返回评估结果和AUC得分
    return evaluation_results, evaluation_results['overall']['auc']

def optimize_with_bayesian(
    X_train: np.ndarray, 
    y_train: pd.Series, 
    X_external: np.ndarray, 
    y_external: pd.Series,
    args: Any
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """使用贝叶斯优化来寻找最佳参数
    
    Args:
        X_train: 训练集特征
        y_train: 训练集标签
        X_external: 外部测试集特征
        y_external: 外部测试集标签
        args: 程序参数
        
    Returns:
        所有试验结果，最佳参数和最佳结果
    """
    # 定义参数空间
    param_space = get_parameter_space_bayesian()
    dimensions = list(param_space.values())
    dimension_names = list(param_space.keys())
    
    # 定义需要优化的目标函数
    @use_named_args(dimensions=dimensions)
    def objective(**params):
        # 提取是否使用分类特征的布尔值并从参数中移除
        use_categorical = params.pop('use_categorical', False)
        
        # 训练和评估模型
        logging.info(f"\n尝试参数: {params}")
        logging.info(f"使用分类特征: {args.categorical_features if use_categorical else []}")
        
        try:
            _, auc = train_and_evaluate_model(
                X_train, y_train, X_external, y_external, 
                params, args, use_categorical
            )
            
            # 优化器尝试最小化目标函数，所以我们返回负AUC
            logging.info(f"得到AUC: {auc:.4f}")
            return -auc
        except Exception as e:
            logging.error(f"评估时出错: {str(e)}")
            return 0.0  # 错误情况下返回很差的分数
    
    # 运行贝叶斯优化
    logging.info("\n开始贝叶斯优化...")
    result = gp_minimize(
        objective,
        dimensions=dimensions,
        n_calls=args.n_trials,
        random_state=args.random_state,
        verbose=True,
        n_jobs=1
    )
    
    # 提取最佳参数
    best_params_raw = dict(zip(dimension_names, result.x))
    
    # 从参数中提取是否使用分类特征的布尔值
    use_categorical = best_params_raw.pop('use_categorical', False)
    
    # 使用最佳参数再次训练和评估
    logging.info("\n使用最佳参数重新评估...")
    best_results, _ = train_and_evaluate_model(
        X_train, y_train, X_external, y_external, 
        best_params_raw, args, use_categorical
    )
    
    # 收集所有试验
    all_trials = []
    for i, (params_raw, value) in enumerate(zip(result.x_iters, result.func_vals)):
        param_dict = dict(zip(dimension_names, params_raw))
        
        # 提取是否使用分类特征的布尔值
        trial_use_categorical = param_dict.pop('use_categorical', False)
        
        all_trials.append({
            'trial_id': i,
            'params': param_dict,
            'use_categorical': trial_use_categorical,
            'auc': -value if value < 0 else None  # 如果是错误情况，值为0，则auc为None
        })
    
    best_info = {
        'model_params': best_params_raw,
        'use_categorical': use_categorical
    }
    
    return all_trials, best_info, best_results

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 创建结果目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 打印特征信息
    logging.info("\n开始特征评估和参数优化:")
    logging.info(f"特征: {args.features}")
    logging.info(f"可选的分类特征: {args.categorical_features}")
    
    # 加载数据
    logging.info("\n加载数据集...")
    train_df = pd.read_excel("data/AI4healthcare.xlsx")
    external_df = pd.read_excel("data/HenanCancerHospital_features63_58.xlsx")
    
    # 准备数据
    X_train_df = train_df[args.features].copy()
    y_train = train_df["Label"].copy()
    X_external_df = external_df[args.features].copy()
    y_external = external_df["Label"].copy()
    
    logging.info("\n数据集信息:")
    logging.info(f"训练集形状: {X_train_df.shape}")
    logging.info(f"外部测试集形状: {X_external_df.shape}")
    logging.info(f"训练集标签分布:\n {y_train.value_counts()}")
    logging.info(f"外部数据集标签分布:\n {y_external.value_counts()}")
    
    # 应用标准缩放到所有数据 - 分类特征的处理将在每次模型评估时进行
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_external_scaled = scaler.transform(X_external_df)
    
    results = []
    best_auc = 0
    best_params = None
    best_result = None
    best_use_categorical = False
    
    # 如果可用且用户选择，使用贝叶斯优化
    if SKOPT_AVAILABLE and args.use_bayesian:
        all_trials, best_info, best_result = optimize_with_bayesian(
            X_train_scaled, y_train, X_external_scaled, y_external, args
        )
        
        results = all_trials
        best_use_categorical = best_info['use_categorical']
        best_params = best_info['model_params']
        best_auc = best_result['overall']['auc']
        
    else:
        # 获取参数搜索空间
        param_grid = get_parameter_grid()
        
        # 生成参数组合
        logging.info("\n使用随机搜索进行参数优化...")
        
        # 添加默认参数组合
        default_params = {
            'max_time': 60,
            'preset': 'default',
            'ges_scoring': 'roc',
            'max_models': 10,
            'validation_method': 'holdout',
            'n_repeats': 50,
            'n_folds': 5,
            'holdout_fraction': 0.33,
            'ges_n_iterations': 25,
            'ignore_limits': False
        }
        
        # 准备两种类别特征使用方式
        categorical_options = [True, False]  # True表示使用类别特征，False表示不使用
        
        # 生成随机参数组合
        import random
        random.seed(args.random_state)
        
        param_combinations = []
        
        # 首先添加默认参数与两种类别特征处理方式的组合
        for use_categorical in categorical_options:
            param_combinations.append((default_params, use_categorical))
        
        # 对于随机搜索，随机选择参数和类别特征处理方式
        n_random_trials = args.n_trials - len(param_combinations)
        for _ in range(n_random_trials):
            # 随机选择模型参数
            params = {}
            for key, values in param_grid.items():
                params[key] = random.choice(values)
            
            # 随机决定是否使用类别特征
            use_categorical = random.choice(categorical_options)
            
            param_combinations.append((params, use_categorical))
        
        logging.info(f"\n将尝试 {len(param_combinations)} 种参数组合")
        
        # 对每个参数组合进行评估
        for trial_idx, (params, use_categorical) in enumerate(param_combinations, 1):
            logging.info(f"\n\n试验 {trial_idx}/{len(param_combinations)}")
            logging.info(f"参数: {params}")
            logging.info(f"使用分类特征: {args.categorical_features if use_categorical else []}")
            
            try:
                # 训练和评估模型
                evaluation_results, auc = train_and_evaluate_model(
                    X_train_scaled, y_train, X_external_scaled, y_external, 
                    params, args, use_categorical
                )
                
                trial_result = {
                    'trial_id': trial_idx,
                    'params': params,
                    'use_categorical': use_categorical,
                    'evaluation': evaluation_results,
                    'auc': auc
                }
                
                results.append(trial_result)
                
                # 打印评估结果摘要
                logging.info("\n外部测试集评估结果:")
                logging.info(f"AUC: {auc:.4f}")
                logging.info(f"准确率: {evaluation_results['overall']['accuracy']:.4f}")
                logging.info(f"F1分数: {evaluation_results['overall']['f1']:.4f}")
                logging.info(f"训练时间: {evaluation_results['train_time']:.2f} 秒")
                
                # 根据AUC更新最佳模型
                if auc > best_auc:
                    best_auc = auc
                    best_params = params
                    best_result = evaluation_results
                    best_use_categorical = use_categorical
                    logging.info(f"找到新的最佳模型! AUC: {best_auc:.4f}")
            
            except Exception as e:
                logging.error(f"评估时出错: {str(e)}")
                continue
    
    # 保存所有结果
    all_results_file = os.path.join(args.output_dir, 'all_trial_results.json')
    with open(all_results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 保存最佳结果
    best_params_file = os.path.join(args.output_dir, 'best_params.json')
    with open(best_params_file, 'w') as f:
        json.dump({
            'model_params': best_params,
            'use_categorical': best_use_categorical
        }, f, indent=2)
    
    best_result_file = os.path.join(args.output_dir, 'best_result.json')
    with open(best_result_file, 'w') as f:
        json.dump(best_result, f, indent=2)
    
    # 创建最佳模型的结果文本文件
    best_result_txt = os.path.join(args.output_dir, 'best_model_summary.txt')
    with open(best_result_txt, 'w') as f:
        f.write("最佳模型参数和结果 (基于外部测试集AUC)\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("数据集特征:\n")
        for i, feature in enumerate(args.features, 1):
            cat_suffix = " (分类)" if best_use_categorical and feature in args.categorical_features else ""
            f.write(f"{i}. {feature}{cat_suffix}\n")
        
        f.write("\n最佳参数:\n")
        f.write("-" * 30 + "\n")
        if best_params:
            for param, value in best_params.items():
                f.write(f"{param}: {value}\n")
        
        f.write("\n使用分类特征处理:\n")
        if best_use_categorical:
            f.write(f"是 - 使用以下特征作为分类特征: {args.categorical_features}\n")
        else:
            f.write("否 - 不使用任何分类特征\n")
        
        f.write("\n训练集性能指标:\n")
        f.write("-" * 30 + "\n")
        if best_result and 'train_metrics' in best_result:
            f.write(f"准确率: {best_result['train_metrics']['accuracy']:.4f}\n")
            f.write(f"AUC: {best_result['train_metrics']['auc']:.4f}\n")
            f.write(f"F1分数: {best_result['train_metrics']['f1']:.4f}\n")
            f.write(f"类别0准确率: {best_result['train_metrics']['acc_0']:.4f}\n")
            f.write(f"类别1准确率: {best_result['train_metrics']['acc_1']:.4f}\n")
            f.write(f"训练时间: {best_result['train_time']:.2f} 秒\n")
            f.write("\n训练集混淆矩阵:\n")
            f.write(f"{np.array(best_result['train_metrics']['confusion_matrix'])}\n")
        
        f.write("\n外部测试集性能指标:\n")
        f.write("-" * 30 + "\n")
        if best_result:
            f.write(f"准确率: {best_result['overall']['accuracy']:.4f} (±{best_result['stds']['accuracy']:.4f})\n")
            f.write(f"AUC: {best_result['overall']['auc']:.4f} (±{best_result['stds']['auc']:.4f})\n")
            f.write(f"F1分数: {best_result['overall']['f1']:.4f} (±{best_result['stds']['f1']:.4f})\n")
            f.write(f"类别0准确率: {best_result['overall']['acc_0']:.4f} (±{best_result['stds']['acc_0']:.4f})\n")
            f.write(f"类别1准确率: {best_result['overall']['acc_1']:.4f} (±{best_result['stds']['acc_1']:.4f})\n")
            
            f.write("\n外部测试集混淆矩阵:\n")
            conf_matrix = np.array(best_result['overall']['confusion_matrix'])
            f.write(f"{conf_matrix}\n")
    
    # 绘制最佳模型的混淆矩阵
    if best_result is not None:
        plt.figure(figsize=(8, 6))
        conf_matrix = np.array(best_result['overall']['confusion_matrix'])
        plt.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
        plt.title('最佳模型的外部测试集混淆矩阵')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['类别 0', '类别 1'])
        plt.yticks(tick_marks, ['类别 0', '类别 1'])
        
        # 在混淆矩阵中显示数字
        thresh = conf_matrix.max() / 2.
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                plt.text(j, i, format(conf_matrix[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if conf_matrix[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'best_model_confusion_matrix.png'), dpi=300)
    
    # 最终总结
    logging.info("\n\n" + "=" * 50)
    logging.info("参数调优完成!")
    if best_params is not None:
        logging.info(f"最佳AUC: {best_auc:.4f}")
        logging.info("最佳参数:")
        for param, value in best_params.items():
            logging.info(f"  {param}: {value}")
        logging.info("\n分类特征处理:")
        if best_use_categorical:
            logging.info(f"  使用以下特征作为分类特征: {args.categorical_features}")
        else:
            logging.info("  不使用任何分类特征")
    else:
        logging.info("未找到有效的参数组合")
    logging.info(f"\n所有结果保存在: {args.output_dir}")
    logging.info("=" * 50)

if __name__ == "__main__":
    main() 