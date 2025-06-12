import numpy as np
import scipy.linalg
import logging
from sklearn.neighbors import KNeighborsClassifier # For pseudo-labeling in class_conditional_coral_transform

# 修改后的CORAL变换函数，区分类别特征和连续特征
def coral_transform(Xs: np.ndarray, Xt: np.ndarray, cat_idx: list) -> np.ndarray:
    """
    解析版CORAL变换，直接计算协方差变换矩阵
    将目标域特征对齐到源域协方差框架下
    仅对连续特征进行变换，保留类别特征不变
    
    参数:
    - Xs: 源域特征 [n_samples_source, n_features]
    - Xt: 目标域特征 [n_samples_target, n_features]
    - cat_idx: 类别特征的索引列表
    
    返回:
    - Xt_aligned: 对齐后的目标域特征
    """
    all_idx = list(range(Xs.shape[1]))
    cont_idx = [i for i in all_idx if i not in cat_idx]
    
    logging.info(f"类别特征索引: {cat_idx}")
    logging.info(f"连续特征索引: {cont_idx}")
    
    # 提取连续特征
    Xs_cont = Xs[:, cont_idx]
    Xt_cont = Xt[:, cont_idx]
    
    # 对齐均值（可选但推荐）
    mu_s = np.mean(Xs_cont, axis=0)
    mu_t = np.mean(Xt_cont, axis=0)
    Xt_cont_centered = Xt_cont - mu_t + mu_s
    
    # 计算连续特征的协方差矩阵，添加小的对角线正则化以确保矩阵可逆
    Cs = np.cov(Xs_cont, rowvar=False) + 1e-5*np.eye(len(cont_idx))
    Ct = np.cov(Xt_cont_centered, rowvar=False) + 1e-5*np.eye(len(cont_idx))
    
    # 矩阵平方根 - 目标域到源域的变换
    Ct_inv_sqrt = scipy.linalg.fractional_matrix_power(Ct, -0.5)
    Cs_sqrt = scipy.linalg.fractional_matrix_power(Cs, 0.5)
    
    # 计算转换矩阵 - 先漂白目标域，再上色为源域（仅应用于连续特征）
    A = np.dot(Ct_inv_sqrt, Cs_sqrt)  # 线性映射矩阵
    Xt_cont_aligned = np.dot(Xt_cont_centered, A)
    
    # 将变换后的连续特征与原始类别特征合并
    Xt_aligned = Xt.copy()
    Xt_aligned[:, cont_idx] = Xt_cont_aligned
    
    # 记录特征分布变化
    logging.info(f"连续特征变换前均值差异: {np.mean(np.abs(np.mean(Xs_cont, axis=0) - np.mean(Xt_cont, axis=0))):.6f}")
    logging.info(f"连续特征变换后均值差异: {np.mean(np.abs(np.mean(Xs_cont, axis=0) - np.mean(Xt_cont_aligned, axis=0))):.6f}")
    logging.info(f"连续特征变换前标准差差异: {np.mean(np.abs(np.std(Xs_cont, axis=0) - np.std(Xt_cont, axis=0))):.6f}")
    logging.info(f"连续特征变换后标准差差异: {np.mean(np.abs(np.std(Xs_cont, axis=0) - np.std(Xt_cont_aligned, axis=0))):.6f}")
    
    # 检查类别特征是否保持不变
    if not np.array_equal(Xt[:, cat_idx], Xt_aligned[:, cat_idx]):
        logging.error("错误：类别特征在变换过程中被改变")
    else:
        logging.info("验证成功：类别特征在CORAL变换过程中保持不变")
    
    return Xt_aligned

# 添加类条件CORAL变换函数
def class_conditional_coral_transform(Xs: np.ndarray, ys: np.ndarray, Xt: np.ndarray, yt_pseudo: np.ndarray = None, cat_idx: list = None, alpha: float = 0.1) -> np.ndarray:
    """
    类条件CORAL变换，对每个类别分别进行协方差对齐
    
    参数:
    - Xs: 源域特征 [n_samples_source, n_features]
    - ys: 源域标签 [n_samples_source]
    - Xt: 目标域特征 [n_samples_target, n_features]
    - yt_pseudo: 目标域伪标签，如果没有则使用源域模型预测 [n_samples_target]
    - cat_idx: 类别特征的索引，如果为None则自动使用TabPFN默认值
    - alpha: 正则化参数，用于平滑类别协方差矩阵
    
    返回:
    - Xt_aligned: 类条件对齐后的目标域特征
    """
    # 如果没有指定类别特征索引, cat_idx must be provided.
    if cat_idx is None:
        # This was a default, but now it must be passed.
        # Consider raising an error or using a globally defined one if that's the design.
        raise ValueError("cat_idx must be provided for class_conditional_coral_transform")

    all_idx = list(range(Xs.shape[1]))
    cont_idx = [i for i in all_idx if i not in cat_idx]
    
    # 获取不同的类别
    classes = np.unique(ys)
    n_classes = len(classes)
    
    logging.info(f"执行类条件CORAL对齐，共有{n_classes}个类别")
    
    # 如果目标域没有伪标签，则使用普通CORAL先对齐，然后用源域模型预测
    if yt_pseudo is None:
        logging.info("目标域没有提供伪标签，先使用普通CORAL进行对齐，再用源域模型预测伪标签")
        Xt_temp = coral_transform(Xs, Xt, cat_idx)  # Pass cat_idx
        
        # 需要一个模型来预测伪标签，这里假设我们已经有了一个训练好的TabPFN模型
        # 为简化流程，我们在这里使用scikit-learn的最近邻分类器
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(Xs, ys)
        yt_pseudo = knn.predict(Xt_temp)
        
        logging.info(f"生成的伪标签分布: {np.bincount(yt_pseudo)}")
    
    # 初始化目标域对齐后的特征矩阵
    Xt_aligned = Xt.copy()
    
    # 对每个类别分别执行CORAL
    for c in classes:
        # 获取源域中属于类别c的样本
        Xs_c = Xs[ys == c]
        
        # 获取目标域中属于类别c的样本（根据伪标签）
        class_mask = (yt_pseudo == c)
        Xt_c = Xt[class_mask]
        
        if len(Xt_c) < 2:  # 需要至少2个样本才能计算协方差
            logging.warning(f"类别{c}在目标域中样本数量过少({len(Xt_c)}个)，无法进行类条件CORAL对齐，跳过")
            # Xt_aligned[class_mask] = Xt_c # Keep original if not enough samples for this class
            continue
            
        if len(Xs_c) < 2:  # 需要至少2个样本才能计算协方差
            logging.warning(f"类别{c}在源域中样本数量过少({len(Xs_c)}个)，无法进行类条件CORAL对齐，跳过")
            # For target samples of this class, we can't align. What to do?
            # One option: use the overall CORAL transformation for these.
            # Another: leave them as is. For now, leave as is (by continuing).
            continue
        
        logging.info(f"对类别{c}进行CORAL对齐：源域{len(Xs_c)}个样本，目标域{len(Xt_c)}个样本")
        
        # 只对连续特征执行CORAL，类别特征保持不变
        Xs_c_cont = Xs_c[:, cont_idx]
        Xt_c_cont = Xt_c[:, cont_idx]
        
        # 对齐均值
        mu_s_c = np.mean(Xs_c_cont, axis=0)
        mu_t_c = np.mean(Xt_c_cont, axis=0)
        Xt_c_cont_centered = Xt_c_cont - mu_t_c + mu_s_c
        
        # 计算类内协方差矩阵
        # 添加正则化项防止矩阵奇异，alpha控制正则化强度
        Cs_c = np.cov(Xs_c_cont, rowvar=False) + alpha * np.eye(len(cont_idx))
        Ct_c = np.cov(Xt_c_cont_centered, rowvar=False) + alpha * np.eye(len(cont_idx))
        
        # 计算变换矩阵 - 先漂白目标域类内协方差，再上色为源域类内协方差
        Ct_c_inv_sqrt = scipy.linalg.fractional_matrix_power(Ct_c, -0.5)
        Cs_c_sqrt = scipy.linalg.fractional_matrix_power(Cs_c, 0.5)
        A_c = np.dot(Ct_c_inv_sqrt, Cs_c_sqrt)
        
        # 应用变换到目标域的类别c样本的连续特征
        Xt_c_cont_aligned = np.dot(Xt_c_cont_centered, A_c)
        
        # 更新对齐后的目标域特征 - 修复索引错误
        # 原代码: Xt_aligned[yt_pseudo == c, cont_idx] = Xt_c_cont_aligned
        # 修复后的代码:
        for i, feat_idx in enumerate(cont_idx):
            Xt_aligned[class_mask, feat_idx] = Xt_c_cont_aligned[:, i]
    
    # 验证类别特征是否保持不变
    if not np.array_equal(Xt[:, cat_idx], Xt_aligned[:, cat_idx]):
        logging.error("错误：类别特征在类条件CORAL变换过程中被改变")
    else:
        logging.info("验证成功：类别特征在类条件CORAL变换过程中保持不变")
    
    return Xt_aligned


def generate_pseudo_labels_for_coral(Xs_scaled: np.ndarray, ys: np.ndarray, Xt_scaled: np.ndarray, cat_idx: list) -> np.ndarray:
    """使用普通CORAL对齐目标域，然后用源域模型预测伪标签"""
    logging.info("Generating pseudo-labels: Applying initial standard CORAL alignment to target domain.")
    Xt_temp_aligned = coral_transform(Xs_scaled, Xt_scaled, cat_idx)
    
    logging.info("Generating pseudo-labels: Training KNN classifier on source domain to predict pseudo-labels for target.")
    # 使用scikit-learn的最近邻分类器作为示例，实际应用中可能是TabPFN或其他模型
    # 这里我们不能直接用TabPFN，因为它可能还未在主流程中训练
    knn = KNeighborsClassifier(n_neighbors=5) # Or another simple model
    knn.fit(Xs_scaled, ys)
    yt_pseudo = knn.predict(Xt_temp_aligned)
    
    logging.info(f"Generated pseudo-label distribution: {np.bincount(yt_pseudo).tolist() if len(yt_pseudo) > 0 else '[]'}")
    return yt_pseudo 