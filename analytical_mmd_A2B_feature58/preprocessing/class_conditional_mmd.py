import numpy as np
import logging
from typing import Dict, Tuple, Optional, Any
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit

from .mmd import mmd_transform, compute_mmd


def generate_pseudo_labels(X_source: np.ndarray, y_source: np.ndarray, 
                          X_target: np.ndarray, method: str = 'knn', 
                          **kwargs) -> np.ndarray:
    """
    为目标域生成伪标签
    
    参数:
    - X_source: 源域特征
    - y_source: 源域标签
    - X_target: 目标域特征
    - method: 伪标签生成方法，默认'knn'
    - **kwargs: 其他参数
    
    返回:
    - pseudo_labels: 目标域伪标签
    """
    if method == 'knn':
        n_neighbors = kwargs.get('n_neighbors', 5)
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_source, y_source)
        pseudo_labels = knn.predict(X_target)
        
        logging.info(f"KNN伪标签生成完成，分布: {np.bincount(pseudo_labels)}")
        return pseudo_labels
    
    else:
        raise ValueError(f"不支持的伪标签生成方法: {method}")


def create_partial_labels(y_target: np.ndarray, label_ratio: float = 0.1, 
                         random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    创建部分标签（分层采样）
    
    参数:
    - y_target: 目标域真实标签
    - label_ratio: 使用的标签比例
    - random_state: 随机种子
    
    返回:
    - partial_labels: 部分标签数组（-1表示未标记）
    - labeled_indices: 已标记样本的索引
    """
    # 分层采样确保每个类别都有样本
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1-label_ratio, random_state=random_state)
    
    for labeled_idx, _ in sss.split(np.arange(len(y_target)), y_target):
        break
    
    # 创建部分标签数组
    partial_labels = np.full_like(y_target, -1)  # -1表示未标记
    partial_labels[labeled_idx] = y_target[labeled_idx]
    
    logging.info(f"创建部分标签: 总样本{len(y_target)}, 已标记{len(labeled_idx)} ({label_ratio:.1%})")
    logging.info(f"已标记样本分布: {np.bincount(partial_labels[partial_labels >= 0])}")
    
    return partial_labels, labeled_idx


def class_conditional_mmd_transform(X_source: np.ndarray, y_source: np.ndarray, 
                                   X_target: np.ndarray, 
                                   target_labels: Optional[np.ndarray] = None,
                                   use_partial_labels: bool = False,
                                   label_ratio: float = 0.1,
                                   method: str = 'linear', 
                                   cat_idx: Optional[list] = None,
                                   random_state: int = 42,
                                   **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    类条件MMD变换，为每个类别分别进行域适应
    
    参数:
    - X_source: 源域特征 [n_samples_source, n_features]
    - y_source: 源域标签 [n_samples_source]
    - X_target: 目标域特征 [n_samples_target, n_features]
    - target_labels: 目标域标签（可选，用于生成伪标签或部分标签）
    - use_partial_labels: 是否使用部分真实标签
    - label_ratio: 使用的真实标签比例（仅当use_partial_labels=True时有效）
    - method: MMD对齐方法，可选 'linear', 'kpca', 'mean_std'
    - cat_idx: 类别特征索引
    - random_state: 随机种子
    - **kwargs: 传递给mmd_transform的其他参数
    
    返回:
    - X_target_aligned: 类条件对齐后的目标域特征
    - mmd_info: MMD相关信息
    """
    if cat_idx is None:
        raise ValueError("cat_idx must be provided for class_conditional_mmd_transform")
    
    # 准备目标域标签
    if target_labels is not None and use_partial_labels:
        # 使用部分真实标签
        partial_labels, labeled_indices = create_partial_labels(
            target_labels, label_ratio, random_state
        )
        
        # 对未标记部分先用标准MMD对齐，然后预测
        unlabeled_mask = (partial_labels == -1)
        if np.any(unlabeled_mask):
            X_target_unlabeled = X_target[unlabeled_mask]
            X_target_unlabeled_aligned, _ = mmd_transform(
                X_source, X_target_unlabeled, method=method, cat_idx=cat_idx, **kwargs
            )
            
            # 使用KNN预测未标记部分
            pseudo_labels_unlabeled = generate_pseudo_labels(
                X_source, y_source, X_target_unlabeled_aligned
            )
            
            # 合并标签
            yt_pseudo = partial_labels.copy()
            yt_pseudo[unlabeled_mask] = pseudo_labels_unlabeled
        else:
            yt_pseudo = partial_labels.copy()
        
        logging.info(f"部分真实标签 + 部分伪标签分布: {np.bincount(yt_pseudo)}")
        
    elif target_labels is not None:
        # 使用全部真实标签（主要用于测试）
        yt_pseudo = target_labels.copy()
        logging.info(f"使用全部真实标签分布: {np.bincount(yt_pseudo)}")
        
    else:
        # 生成伪标签
        yt_pseudo = generate_pseudo_labels(X_source, y_source, X_target)
    
    # 初始化对齐后的目标域特征矩阵
    X_target_aligned = np.copy(X_target)
    
    # 获取唯一类别
    classes = np.unique(y_source)
    
    # 类别特定的MMD信息
    class_specific_info = {}
    
    # 计算整体初始MMD
    all_idx = list(range(X_source.shape[1]))
    cont_idx = [i for i in all_idx if i not in cat_idx]
    X_source_cont = X_source[:, cont_idx]
    X_target_cont = X_target[:, cont_idx]
    overall_initial_mmd = compute_mmd(X_source_cont, X_target_cont, kernel='rbf', gamma=1.0)
    
    logging.info(f"类条件MMD - 整体初始MMD: {overall_initial_mmd:.6f}")
    
    # 为每个类别分别对齐
    for c in classes:
        # 获取属于类别c的源域样本
        source_mask = (y_source == c)
        X_source_c = X_source[source_mask]
        
        # 获取属于类别c的目标域样本（基于伪标签）
        target_mask = (yt_pseudo == c)
        X_target_c = X_target[target_mask]
        
        if len(X_target_c) == 0:
            logging.warning(f"目标域中没有伪标签为{c}的样本，跳过此类别")
            class_specific_info[c] = {
                'initial_mmd': None,
                'final_mmd': None,
                'reduction': None,
                'count': 0
            }
            continue
        
        logging.info(f"类别{c}: 源域样本 = {len(X_source_c)}, 目标域样本 = {len(X_target_c)}")
        
        # 对此类别使用MMD对齐
        try:
            X_target_c_aligned, mmd_c_info = mmd_transform(
                X_source_c, X_target_c, method=method, cat_idx=cat_idx, **kwargs
            )
            
            # 将变换后的样本放回对齐的目标域特征矩阵
            X_target_aligned[target_mask] = X_target_c_aligned
            
            # 保存类别特定信息
            class_specific_info[c] = {
                'initial_mmd': mmd_c_info['initial_mmd'],
                'final_mmd': mmd_c_info['final_mmd'],
                'reduction': mmd_c_info['reduction'],
                'count': len(X_target_c)
            }
            
        except Exception as e:
            logging.error(f"类别{c}的MMD对齐失败: {str(e)}")
            class_specific_info[c] = {
                'initial_mmd': None,
                'final_mmd': None,
                'reduction': None,
                'count': len(X_target_c),
                'error': str(e)
            }
    
    # 计算整体最终MMD
    X_target_aligned_cont = X_target_aligned[:, cont_idx]
    overall_final_mmd = compute_mmd(X_source_cont, X_target_aligned_cont, kernel='rbf', gamma=1.0)
    overall_reduction = (overall_initial_mmd - overall_final_mmd) / overall_initial_mmd * 100
    
    logging.info(f"类条件MMD - 整体最终MMD: {overall_final_mmd:.6f}")
    logging.info(f"类条件MMD - 整体MMD减少: {overall_reduction:.2f}%")
    
    # 编译MMD信息
    mmd_info = {
        'method': f'class_conditional_{method}',
        'use_partial_labels': use_partial_labels,
        'label_ratio': label_ratio if use_partial_labels else None,
        'overall_initial_mmd': overall_initial_mmd,
        'overall_final_mmd': overall_final_mmd,
        'overall_reduction': overall_reduction,
        'class_specific': class_specific_info,
        'pseudo_labels': yt_pseudo
    }
    
    return X_target_aligned, mmd_info


def run_class_conditional_mmd_experiment(X_source: np.ndarray, y_source: np.ndarray,
                                        X_target: np.ndarray, y_target: np.ndarray,
                                        tabpfn_model,
                                        method: str = 'linear',
                                        use_partial_labels: bool = False,
                                        label_ratio: float = 0.1,
                                        cat_idx: Optional[list] = None,
                                        random_state: int = 42,
                                        **kwargs) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    运行类条件MMD实验的便捷函数
    
    参数:
    - X_source: 源域特征
    - y_source: 源域标签
    - X_target: 目标域特征
    - y_target: 目标域标签
    - tabpfn_model: 已训练的TabPFN模型
    - method: MMD方法
    - use_partial_labels: 是否使用部分真实标签
    - label_ratio: 真实标签比例
    - cat_idx: 类别特征索引
    - random_state: 随机种子
    - **kwargs: 其他参数
    
    返回:
    - y_pred: 预测标签
    - y_proba: 预测概率
    - mmd_info: MMD信息
    """
    # 应用类条件MMD变换
    X_target_aligned, mmd_info = class_conditional_mmd_transform(
        X_source, y_source, X_target,
        target_labels=y_target if use_partial_labels else None,
        use_partial_labels=use_partial_labels,
        label_ratio=label_ratio,
        method=method,
        cat_idx=cat_idx,
        random_state=random_state,
        **kwargs
    )
    
    # 在对齐后的目标域上预测
    y_pred = tabpfn_model.predict(X_target_aligned)
    y_proba = tabpfn_model.predict_proba(X_target_aligned)
    
    return y_pred, y_proba, mmd_info 