import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
from sklearn.metrics.pairwise import pairwise_kernels, pairwise_distances
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Optional, Any
from sklearn.linear_model import Ridge

def compute_mmd(X_s: np.ndarray, X_t: np.ndarray, kernel: str = 'rbf', gamma: float = 1.0) -> float:
    """
    计算Maximum Mean Discrepancy (MMD) - 使用无偏估计
    
    参数:
    - X_s: 源域特征 [n_samples_source, n_features]
    - X_t: 目标域特征 [n_samples_target, n_features]
    - kernel: 核函数类型，默认'rbf'
    - gamma: 核函数参数，默认1.0
    
    返回:
    - mmd_value: MMD距离值
    """
    n_s = X_s.shape[0]
    n_t = X_t.shape[0]
    
    # 根据核函数类型决定是否传递gamma参数
    if kernel in ['linear', 'polynomial', 'poly', 'sigmoid', 'cosine']:
        # 这些核函数不使用gamma参数或有不同的参数名
        if kernel == 'linear':
            K_ss = pairwise_kernels(X_s, X_s, metric=kernel)
            K_tt = pairwise_kernels(X_t, X_t, metric=kernel)
            K_st = pairwise_kernels(X_s, X_t, metric=kernel)
        else:
            K_ss = pairwise_kernels(X_s, X_s, metric=kernel, gamma=gamma)
            K_tt = pairwise_kernels(X_t, X_t, metric=kernel, gamma=gamma)
            K_st = pairwise_kernels(X_s, X_t, metric=kernel, gamma=gamma)
    else:
        # rbf, laplacian等核函数使用gamma参数
        K_ss = pairwise_kernels(X_s, X_s, metric=kernel, gamma=gamma)
        K_tt = pairwise_kernels(X_t, X_t, metric=kernel, gamma=gamma)
        K_st = pairwise_kernels(X_s, X_t, metric=kernel, gamma=gamma)
    
    # 使用无偏估计：去除对角线元素
    # E[k(x,x')] where x != x'
    K_ss_sum = K_ss.sum() - np.trace(K_ss)  # 去除对角线
    K_tt_sum = K_tt.sum() - np.trace(K_tt)  # 去除对角线
    
    # 计算无偏MMD^2
    if n_s > 1 and n_t > 1:
        mmd_squared = (K_ss_sum / (n_s * (n_s - 1)) + 
                      K_tt_sum / (n_t * (n_t - 1)) - 
                      2 * K_st.mean())
    else:
        # 如果样本数太少，使用有偏估计
        mmd_squared = K_ss.mean() + K_tt.mean() - 2 * K_st.mean()
    
    # 确保MMD非负，返回MMD（开方）
    return max(0, mmd_squared) ** 0.5

def median_heuristic_gamma(X: np.ndarray, sample_size: int = 1000) -> float:
    """
    使用中值启发式计算合适的gamma参数
    
    参数:
    - X: 输入数据
    - sample_size: 用于计算的样本数量（避免大数据集计算过慢）
    
    返回:
    - gamma: 推荐的gamma值
    """
    if X.shape[0] > sample_size:
        # 随机采样以加速计算
        indices = np.random.choice(X.shape[0], sample_size, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X
    
    # 计算成对距离
    distances = pairwise_distances(X_sample, X_sample)
    # 取上三角部分（去除对角线和重复）
    upper_tri_distances = distances[np.triu_indices_from(distances, k=1)]
    
    if len(upper_tri_distances) > 0:
        median_dist = np.median(upper_tri_distances)
        gamma = 1.0 / (2 * median_dist ** 2) if median_dist > 0 else 1.0
    else:
        gamma = 1.0
    
    logging.info(f"中值启发式计算的gamma: {gamma:.6f}")
    return gamma

def compute_multiple_kernels_mmd(X_s: np.ndarray, X_t: np.ndarray, 
                                kernels: Optional[list] = None, 
                                gammas: Optional[list] = None) -> Dict[str, float]:
    """
    使用多个内核计算MMD
    
    参数:
    - X_s: 源域特征
    - X_t: 目标域特征
    - kernels: 核函数列表，如果为None则使用['rbf', 'laplacian', 'polynomial']
    - gammas: gamma参数列表，如果为None则使用[0.1, 1.0, 10.0]
    
    返回:
    - mmd_values: 各个内核下的MMD值字典
    """
    if kernels is None:
        kernels = ['rbf', 'laplacian', 'polynomial']
    if gammas is None:
        gammas = [0.1, 1.0, 10.0]
    
    mmd_values = {}
    
    for kernel in kernels:
        for gamma in gammas:
            key = f"{kernel}_gamma{gamma}"
            try:
                mmd_values[key] = compute_mmd(X_s, X_t, kernel=kernel, gamma=gamma)
            except Exception as e:
                logging.warning(f"无法计算内核 {key} 的MMD: {str(e)}")
                mmd_values[key] = float('nan')
    
    # 找出最小MMD值及其对应的内核
    valid_mmds = {k: v for k, v in mmd_values.items() if not np.isnan(v)}
    if valid_mmds:
        best_kernel = min(valid_mmds.items(), key=lambda x: x[1])[0]
        mmd_values['best_kernel'] = best_kernel
        mmd_values['min_mmd'] = valid_mmds[best_kernel]
    
    return mmd_values

class MMDLinearTransform:
    """使用线性变换最小化MMD的特征对齐器 - 改进版本"""
    
    def __init__(self, input_dim: int, gamma: float = 1.0, lr: float = 0.01, 
                 n_epochs: int = 300, batch_size: int = 64, lambda_reg: float = 1e-4,
                 staged_training: bool = True, dynamic_gamma: bool = True,
                 gamma_search_values: Optional[list] = None,
                 use_gradient_clipping: bool = False, max_grad_norm: float = 1.0,
                 standardize_features: bool = False, monitor_gradients: bool = False):
        self.input_dim = input_dim
        self.gamma = gamma
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lambda_reg = lambda_reg
        self.staged_training = staged_training  # 分阶段训练
        self.dynamic_gamma = dynamic_gamma  # 动态gamma调整
        self.gamma_search_values = gamma_search_values or [0.1, 0.5, 1.0]
        self.use_gradient_clipping = use_gradient_clipping  # 梯度裁剪
        self.max_grad_norm = max_grad_norm  # 梯度裁剪阈值
        self.standardize_features = standardize_features  # 特征标准化
        self.monitor_gradients = monitor_gradients  # 监控梯度范数
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = nn.Linear(input_dim, input_dim, bias=True).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        self.best_model_state = None
        self.best_mmd = float('inf')
        self.original_mmd = None
        self.best_gamma = gamma
        
        # 训练历史记录
        self.training_history = {
            'epoch': [],
            'batch_mmd': [],
            'full_mmd': [],
            'loss': [],
            'gamma_used': []
        }
        
        logging.info(f"初始化改进版MMDLinearTransform: input_dim={input_dim}, staged_training={staged_training}, dynamic_gamma={dynamic_gamma}")
        
    def rbf_kernel(self, X: torch.Tensor, Y: torch.Tensor, gamma: Optional[float] = None) -> torch.Tensor:
        """计算RBF内核矩阵，支持动态gamma"""
        if gamma is None:
            gamma = self.gamma
        X_norm = torch.sum(X ** 2, dim=1, keepdim=True)
        Y_norm = torch.sum(Y ** 2, dim=1, keepdim=True)
        XY = torch.mm(X, Y.t())
        dist = X_norm + Y_norm.t() - 2 * XY
        return torch.exp(-gamma * dist)
    
    def compute_mmd_torch(self, X_s: torch.Tensor, X_t: torch.Tensor, gamma: Optional[float] = None) -> torch.Tensor:
        """计算PyTorch版本的MMD（开方版本，与compute_mmd保持一致）"""
        K_XX = self.rbf_kernel(X_s, X_s, gamma)
        K_YY = self.rbf_kernel(X_t, X_t, gamma)
        K_XY = self.rbf_kernel(X_s, X_t, gamma)
        
        n_s = X_s.size(0)
        n_t = X_t.size(0)
        
        # 使用无偏估计
        if n_s > 1 and n_t > 1:
            K_XX_sum = K_XX.sum() - torch.trace(K_XX)
            K_YY_sum = K_YY.sum() - torch.trace(K_YY)
            mmd_squared = (K_XX_sum / (n_s * (n_s - 1)) + 
                          K_YY_sum / (n_t * (n_t - 1)) - 
                          2 * K_XY.mean())
        else:
            mmd_squared = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
        
        # 确保非负并开方，与compute_mmd保持一致
        return torch.sqrt(torch.clamp(mmd_squared, min=0))
    
    def find_optimal_gamma(self, X_s: np.ndarray, X_t: np.ndarray) -> float:
        """使用网格搜索找到最优gamma参数"""
        logging.info("开始gamma参数网格搜索...")
        
        best_gamma = self.gamma
        best_mmd = float('inf')
        
        for gamma_candidate in self.gamma_search_values:
            mmd_value = compute_mmd(X_s, X_t, kernel='rbf', gamma=gamma_candidate)
            logging.info(f"  gamma={gamma_candidate:.3f}: MMD={mmd_value:.6f}")
            
            if mmd_value < best_mmd:
                best_mmd = mmd_value
                best_gamma = gamma_candidate
        
        logging.info(f"最优gamma: {best_gamma:.3f} (MMD: {best_mmd:.6f})")
        return best_gamma
    
    def compute_full_dataset_mmd(self, X_s_tensor: torch.Tensor, X_t_tensor: torch.Tensor) -> float:
        """计算整个数据集上的MMD（用于监控训练进度）"""
        self.model.eval()
        with torch.no_grad():
            X_t_transformed = self.model(X_t_tensor)
            mmd_value = self.compute_mmd_torch(X_s_tensor, X_t_transformed, self.best_gamma)
        self.model.train()
        return mmd_value.item()
    
    def fit(self, X_s: np.ndarray, X_t: np.ndarray) -> 'MMDLinearTransform':
        """训练线性变换以最小化MMD - 改进版本"""
        # 特征标准化（如果启用）
        if self.standardize_features:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            X_s = self.scaler.fit_transform(X_s)
            X_t = self.scaler.transform(X_t)
            logging.info("已对特征进行标准化")
        else:
            self.scaler = None
        
        # 确保输入维度匹配
        if X_s.shape[1] != self.input_dim:
            logging.warning(f"输入维度不匹配: X_s.shape[1]={X_s.shape[1]}, self.input_dim={self.input_dim}")
            self.input_dim = X_s.shape[1]
            self.model = nn.Linear(self.input_dim, self.input_dim, bias=True).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
            logging.info(f"已调整模型输入维度为 {self.input_dim}")
        
        # 动态gamma搜索
        if self.dynamic_gamma:
            self.best_gamma = self.find_optimal_gamma(X_s, X_t)
            self.gamma = self.best_gamma
        
        # 记录原始MMD
        self.original_mmd = compute_mmd(X_s, X_t, kernel='rbf', gamma=self.best_gamma)
        logging.info(f"原始MMD (gamma={self.best_gamma:.3f}): {self.original_mmd:.6f}")
        
        # 转换为PyTorch张量
        X_s_tensor = torch.FloatTensor(X_s).to(self.device)
        X_t_tensor = torch.FloatTensor(X_t).to(self.device)
        
        # 增大批次大小以减少批次MMD与全量MMD的差异
        effective_batch_size = min(self.batch_size, X_s.shape[0] // 4, X_t.shape[0] // 4)
        effective_batch_size = max(effective_batch_size, 32)  # 最小批次大小
        
        if effective_batch_size != self.batch_size:
            logging.info(f"调整批次大小: {self.batch_size} -> {effective_batch_size}")
            self.batch_size = effective_batch_size
        
        # 创建数据加载器
        source_dataset = TensorDataset(X_s_tensor)
        target_dataset = TensorDataset(X_t_tensor)
        source_loader = DataLoader(source_dataset, batch_size=self.batch_size, shuffle=True)
        target_loader = DataLoader(target_dataset, batch_size=self.batch_size, shuffle=True)
        
        # 分阶段训练策略
        if self.staged_training:
            # 第一阶段：只优化MMD，不加正则化
            stage1_epochs = self.n_epochs // 3
            stage2_epochs = self.n_epochs - stage1_epochs
            
            logging.info(f"分阶段训练: 阶段1({stage1_epochs}轮,无正则) + 阶段2({stage2_epochs}轮,有正则)")
            
            # 阶段1：纯MMD优化
            self._train_stage(X_s_tensor, X_t_tensor, source_loader, target_loader, 
                            stage1_epochs, lambda_reg=0.0, stage_name="阶段1(纯MMD)")
            
            # 阶段2：MMD + 正则化
            self._train_stage(X_s_tensor, X_t_tensor, source_loader, target_loader, 
                            stage2_epochs, lambda_reg=self.lambda_reg, stage_name="阶段2(MMD+正则)")
        else:
            # 传统训练：从开始就加正则化
            self._train_stage(X_s_tensor, X_t_tensor, source_loader, target_loader, 
                            self.n_epochs, lambda_reg=self.lambda_reg, stage_name="标准训练")
        
        # 加载最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            
        logging.info(f"训练完成。原始MMD: {self.original_mmd:.6f}, 最终MMD: {self.best_mmd:.6f}")
        # 修复除零错误：当原始MMD为0时，设置减少百分比为0
        if self.original_mmd > 1e-10:
            mmd_reduction_percent = (self.original_mmd - self.best_mmd) / self.original_mmd * 100
        else:
            mmd_reduction_percent = 0.0
        logging.info(f"MMD减少: {mmd_reduction_percent:.2f}%")
        
        return self
    
    def _train_stage(self, X_s_tensor: torch.Tensor, X_t_tensor: torch.Tensor, 
                    source_loader: DataLoader, target_loader: DataLoader,
                    n_epochs: int, lambda_reg: float, stage_name: str):
        """训练的一个阶段"""
        logging.info(f"开始{stage_name} (lambda_reg={lambda_reg:.6f})")
        
        for epoch in range(n_epochs):
            self.model.train()
            total_loss = 0
            batch_count = 0
            batch_mmd_sum = 0
            
            # 重新创建源数据和目标数据的迭代器
            source_iter = iter(source_loader)
            target_iter = iter(target_loader)
            
            while True:
                try:
                    source_batch = next(source_iter)[0]
                except StopIteration:
                    break
                    
                try:
                    target_batch = next(target_iter)[0]
                except StopIteration:
                    target_iter = iter(target_loader)
                    target_batch = next(target_iter)[0]
                
                self.optimizer.zero_grad()
                
                # 对目标域应用变换
                target_transformed = self.model(target_batch)
                
                # 计算MMD损失（开方版本）
                mmd_loss = self.compute_mmd_torch(source_batch, target_transformed, self.best_gamma)
                
                # 添加正则化项
                loss = mmd_loss
                if lambda_reg > 0:
                    weights = next(self.model.parameters())
                    identity_reg = lambda_reg * torch.norm(weights - torch.eye(weights.shape[0], weights.shape[1]).to(self.device))
                    loss = mmd_loss + identity_reg
                
                loss.backward()
                
                # 监控梯度范数（如果启用）
                if self.monitor_gradients:
                    total_norm = 0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5
                    if batch_count % 10 == 0:  # 每10个batch打印一次
                        logging.debug(f"Gradient norm: {total_norm:.6f}")
                
                # 梯度裁剪（如果启用）
                if self.use_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                batch_mmd_sum += mmd_loss.item()
                batch_count += 1
            
            # 每隔一定轮数计算全量MMD
            if (epoch + 1) % 20 == 0 or epoch == 0 or epoch == n_epochs - 1:
                full_mmd = self.compute_full_dataset_mmd(X_s_tensor, X_t_tensor)
                
                # 保存最佳模型（基于全量MMD）
                if full_mmd < self.best_mmd:
                    self.best_mmd = full_mmd
                    self.best_model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
                
                # 记录训练历史
                avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
                avg_batch_mmd = batch_mmd_sum / batch_count if batch_count > 0 else float('inf')
                
                self.training_history['epoch'].append(len(self.training_history['epoch']) + 1)
                self.training_history['batch_mmd'].append(avg_batch_mmd)
                self.training_history['full_mmd'].append(full_mmd)
                self.training_history['loss'].append(avg_loss)
                self.training_history['gamma_used'].append(self.best_gamma)
                
                logging.info(f"{stage_name} Epoch {epoch+1}/{n_epochs}: "
                           f"Loss={avg_loss:.6f}, BatchMMD={avg_batch_mmd:.6f}, FullMMD={full_mmd:.6f}")
    
    def get_training_history(self) -> Dict[str, list]:
        """获取训练历史记录"""
        return self.training_history.copy()
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """对输入数据应用学习到的变换"""
        # 如果训练时使用了标准化，先标准化输入
        if self.standardize_features and hasattr(self, 'scaler') and self.scaler is not None:
            X = self.scaler.transform(X)
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        self.model.eval()
        with torch.no_grad():
            X_transformed = self.model(X_tensor).cpu().numpy()
        
        # 如果训练时使用了标准化，逆标准化输出
        if self.standardize_features and hasattr(self, 'scaler') and self.scaler is not None:
            X_transformed = self.scaler.inverse_transform(X_transformed)
        
        return X_transformed

def mmd_kernel_pca_transform(X_s: np.ndarray, X_t: np.ndarray, cat_idx: Optional[list] = None, 
                            n_components: Optional[int] = None, kernel: str = 'rbf', 
                            gamma: Optional[float] = None, 
                            use_inverse_transform: bool = False,
                            standardize: bool = True) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    改进的核PCA MMD对齐方法
    
    参数:
    - X_s: 源域特征
    - X_t: 目标域特征
    - cat_idx: 类别特征索引，默认使用TabPFN默认值
    - n_components: PCA组件数，默认None（使用所有组件）
    - kernel: 核函数类型
    - gamma: 核函数参数，如果为None则使用中值启发式
    - use_inverse_transform: 是否使用逆变换回原始空间（默认False，在核空间完成所有操作）
    - standardize: 是否在核PCA前标准化特征
    
    返回:
    - X_t_aligned: 对齐后的目标域特征
    - mmd_info: MMD相关信息
    """
    # 分离类别特征和连续特征
    if cat_idx is None:
        raise ValueError("cat_idx must be provided for mmd_transform")
    all_idx = list(range(X_s.shape[1]))
    cont_idx = [i for i in all_idx if i not in cat_idx]
    
    # 仅对连续特征应用变换
    X_s_cont = X_s[:, cont_idx]
    X_t_cont = X_t[:, cont_idx]
    
    # 标准化连续特征（推荐）
    if standardize:
        scaler = StandardScaler()
        X_s_cont = scaler.fit_transform(X_s_cont)
        X_t_cont = scaler.transform(X_t_cont)
        logging.info("已对连续特征进行标准化")
    
    # 使用中值启发式计算gamma（如果未提供）
    if gamma is None:
        combined_data_cont = np.vstack([X_s_cont, X_t_cont])
        gamma = median_heuristic_gamma(combined_data_cont)
    
    # 计算连续特征的初始MMD
    initial_mmd = compute_mmd(X_s_cont, X_t_cont, kernel=kernel, gamma=gamma)
    logging.info(f"改进KPCA方法 - 初始MMD (连续特征): {initial_mmd:.6f}")
    
    # 设置组件数
    if n_components is None:
        n_components = min(X_s_cont.shape[1], X_s_cont.shape[0], X_t_cont.shape[0])
        # 保留足够的信息，但避免过拟合
        n_components = min(n_components, max(10, X_s_cont.shape[1] // 2))
    
    logging.info(f"使用 {n_components} 个核PCA组件")
    
    # 在合并数据上训练核PCA
    combined_data_cont = np.vstack((X_s_cont, X_t_cont))
    
    # 根据是否需要逆变换来设置fit_inverse_transform
    if kernel == 'rbf':
        kpca = KernelPCA(
            n_components=n_components, 
            kernel='rbf', 
            gamma=gamma, 
            fit_inverse_transform=use_inverse_transform
        )
    elif kernel == 'linear':
        kpca = KernelPCA(
            n_components=n_components, 
            kernel='linear', 
            fit_inverse_transform=use_inverse_transform
        )
    elif kernel == 'poly':
        kpca = KernelPCA(
            n_components=n_components, 
            kernel='poly', 
            gamma=gamma, 
            fit_inverse_transform=use_inverse_transform
        )
    else:  # 默认使用rbf
        kpca = KernelPCA(
            n_components=n_components, 
            kernel='rbf', 
            gamma=gamma, 
            fit_inverse_transform=use_inverse_transform
        )
    
    kpca.fit(combined_data_cont)
    
    # 变换数据到核PCA空间
    X_s_kpca = kpca.transform(X_s_cont)
    X_t_kpca = kpca.transform(X_t_cont)
    
    # 在核PCA空间计算MMD（使用线性核，因为已经在特征映射后的空间）
    mmd_in_kpca_before = compute_mmd(X_s_kpca, X_t_kpca, kernel='linear')
    logging.info(f"MMD (核PCA空间，对齐前): {mmd_in_kpca_before:.6f}")
    
    # 在核PCA空间进行分布对齐
    X_s_kpca_mean = np.mean(X_s_kpca, axis=0)
    X_t_kpca_mean = np.mean(X_t_kpca, axis=0)
    
    X_s_kpca_std = np.std(X_s_kpca, axis=0) + 1e-8
    X_t_kpca_std = np.std(X_t_kpca, axis=0) + 1e-8
    
    X_t_kpca_aligned = ((X_t_kpca - X_t_kpca_mean) / X_t_kpca_std) * X_s_kpca_std + X_s_kpca_mean
    
    # 在对齐后的核PCA空间计算MMD
    mmd_in_kpca_after = compute_mmd(X_s_kpca, X_t_kpca_aligned, kernel='linear')
    logging.info(f"MMD (核PCA空间，对齐后): {mmd_in_kpca_after:.6f}")
    
    # 计算核PCA空间的MMD减少
    kpca_reduction = (mmd_in_kpca_before - mmd_in_kpca_after) / mmd_in_kpca_before * 100 if mmd_in_kpca_before > 0 else 0
    logging.info(f"核PCA空间MMD减少: {kpca_reduction:.2f}%")
    
    if use_inverse_transform:
        # 使用逆变换回原始空间（可能有较大误差）
        logging.info("使用逆变换回原始空间（可能有重构误差）")
        
        try:
            # 检查重构误差
            X_s_cont_reconstructed = kpca.inverse_transform(X_s_kpca)
            X_t_cont_reconstructed = kpca.inverse_transform(X_t_kpca)
            
            X_s_cont_norm = np.linalg.norm(X_s_cont.astype(np.float64))
            if X_s_cont_norm > 1e-9:
                reconstruction_error_source = np.linalg.norm((X_s_cont - X_s_cont_reconstructed).astype(np.float64)) / X_s_cont_norm
                logging.info(f"源域重构相对误差: {reconstruction_error_source:.4f}")
                
                if reconstruction_error_source > 0.1:  # 10%以上误差
                    logging.warning(f"源域重构误差过大 ({reconstruction_error_source:.1%})，逆变换可能不可靠")
            
            if np.linalg.norm(X_t_cont) > 1e-9:
                reconstruction_error_target = np.linalg.norm(X_t_cont - X_t_cont_reconstructed) / np.linalg.norm(X_t_cont)
                logging.info(f"目标域重构相对误差: {reconstruction_error_target:.4f}")
                
                if reconstruction_error_target > 0.1:  # 10%以上误差
                    logging.warning(f"目标域重构误差过大 ({reconstruction_error_target:.1%})，逆变换可能不可靠")
            
            # 逆变换对齐后的数据
            X_t_cont_aligned = kpca.inverse_transform(X_t_kpca_aligned)
            
            # 如果标准化了，需要逆标准化
            if standardize:
                X_t_cont_aligned = scaler.inverse_transform(X_t_cont_aligned)
            
            # 在原始空间计算最终MMD
            final_mmd = compute_mmd(X_s[:, cont_idx], X_t_cont_aligned, kernel=kernel, gamma=gamma)
            logging.info(f"最终MMD (逆变换到原始空间): {final_mmd:.6f}")
            
            # 构建完整的对齐特征矩阵
            X_t_aligned = X_t.copy()
            X_t_aligned[:, cont_idx] = X_t_cont_aligned
            
        except Exception as e:
            logging.error(f"逆变换失败: {str(e)}，回退到简单对齐")
            # 回退到简单的均值方差对齐
            X_s_cont_mean = np.mean(X_s_cont, axis=0)
            X_s_cont_std = np.std(X_s_cont, axis=0) + 1e-8
            X_t_cont_mean = np.mean(X_t_cont, axis=0)
            X_t_cont_std = np.std(X_t_cont, axis=0) + 1e-8
            X_t_cont_aligned = ((X_t_cont - X_t_cont_mean) / X_t_cont_std) * X_s_cont_std + X_s_cont_mean
            
            X_t_aligned = X_t.copy()
            X_t_aligned[:, cont_idx] = X_t_cont_aligned
            final_mmd = compute_mmd(X_s[:, cont_idx], X_t_cont_aligned, kernel=kernel, gamma=gamma)
    
    else:
        # 推荐方式：不使用逆变换，在核PCA空间完成所有操作
        logging.info("在核PCA空间完成对齐，不使用逆变换")
        
        # 重要修改：使用核PCA空间的对齐特征，而不是回退到简单对齐
        # 为了兼容现有接口，我们需要将核PCA空间的特征映射回原始维度
        
        # 方法1：使用核PCA空间的对齐特征重构原始空间特征 + CORAL对齐
        # 通过最小二乘法找到从核PCA空间到原始空间的映射，然后在原始空间再做CORAL对齐
        try:
            # 训练一个Ridge回归模型，从核PCA空间映射到原始空间
            ridge = Ridge(alpha=1e-2)
            ridge.fit(X_s_kpca, X_s_cont)
            
            # 使用训练好的模型将对齐后的核PCA特征映射回原始空间
            X_t_back = ridge.predict(X_t_kpca_aligned)
            
            logging.info("使用Ridge回归将核PCA空间的对齐特征映射回原始空间")
            
            # 在原始空间再做一次CORAL对齐，确保KL散度和Wasserstein距离也得到改善
            eps = 1e-8
            
            # 计算协方差矩阵
            X_s_cont_centered = X_s_cont - np.mean(X_s_cont, axis=0)
            X_t_back_centered = X_t_back - np.mean(X_t_back, axis=0)
            
            Xs_cov = np.cov(X_s_cont_centered, rowvar=False) + eps * np.eye(X_s_cont.shape[1])
            Xt_cov = np.cov(X_t_back_centered, rowvar=False) + eps * np.eye(X_t_back.shape[1])
            
            # CORAL变换：先白化目标域，再用源域协方差重新着色
            try:
                # 目标域白化矩阵
                E_t, V_t = np.linalg.eigh(Xt_cov)
                E_t = np.maximum(E_t, eps)  # 确保特征值为正
                A = V_t @ np.diag(1.0 / np.sqrt(E_t)) @ V_t.T
                
                # 源域着色矩阵（使用Cholesky分解）
                try:
                    C = np.linalg.cholesky(Xs_cov)
                except np.linalg.LinAlgError:
                    # 如果Cholesky分解失败，使用特征值分解
                    E_s, V_s = np.linalg.eigh(Xs_cov)
                    E_s = np.maximum(E_s, eps)
                    C = V_s @ np.diag(np.sqrt(E_s)) @ V_s.T
                
                # 应用CORAL变换
                X_t_cont_aligned = X_t_back_centered @ A.T @ C.T + np.mean(X_s_cont, axis=0)
                
                logging.info("在原始空间应用CORAL对齐以改善KL散度和Wasserstein距离")
                
            except np.linalg.LinAlgError as coral_error:
                logging.warning(f"CORAL对齐失败: {coral_error}，使用简单的均值-方差对齐")
                # 回退到简单的均值-方差对齐
                X_s_mean = np.mean(X_s_cont, axis=0)
                X_s_std = np.std(X_s_cont, axis=0) + eps
                X_t_back_mean = np.mean(X_t_back, axis=0)
                X_t_back_std = np.std(X_t_back, axis=0) + eps
                X_t_cont_aligned = ((X_t_back - X_t_back_mean) / X_t_back_std) * X_s_std + X_s_mean
            
        except Exception as e:
            logging.warning(f"Ridge回归映射失败: {str(e)}，使用加权平均方法")
            
            # 方法2：使用加权平均的方式结合核PCA信息和简单对齐
            # 简单对齐作为基础
            X_s_cont_mean = np.mean(X_s_cont, axis=0)
            X_s_cont_std = np.std(X_s_cont, axis=0) + 1e-8
            X_t_cont_mean = np.mean(X_t_cont, axis=0)
            X_t_cont_std = np.std(X_t_cont, axis=0) + 1e-8
            X_t_cont_simple = ((X_t_cont - X_t_cont_mean) / X_t_cont_std) * X_s_cont_std + X_s_cont_mean
            
            # 使用核PCA空间的对齐信息调整简单对齐结果
            # 计算核PCA空间中的调整向量
            kpca_adjustment = X_t_kpca_aligned - X_t_kpca
            
            # 将调整向量的影响传播到原始空间（简化版本）
            # 使用核PCA组件的方差权重
            if hasattr(kpca, 'eigenvalues_') and kpca.eigenvalues_ is not None:
                weights = kpca.eigenvalues_ / np.sum(kpca.eigenvalues_)
            else:
                weights = np.ones(n_components) / n_components
            
            # 计算加权调整
            weighted_adjustment = np.zeros_like(X_t_cont_simple)
            for i in range(min(n_components, len(weights))):
                if i < kpca_adjustment.shape[1]:
                    # 简单的线性组合（这是一个近似）
                    feature_adjustment = kpca_adjustment[:, i:i+1] * weights[i]
                    if feature_adjustment.shape[1] == 1 and X_t_cont_simple.shape[1] > 1:
                        # 广播调整到所有特征
                        feature_adjustment = np.tile(feature_adjustment, (1, X_t_cont_simple.shape[1]))
                    elif feature_adjustment.shape[1] > X_t_cont_simple.shape[1]:
                        # 截断到匹配维度
                        feature_adjustment = feature_adjustment[:, :X_t_cont_simple.shape[1]]
                    
                    if feature_adjustment.shape == weighted_adjustment.shape:
                        weighted_adjustment += feature_adjustment
            
            # 结合简单对齐和核PCA调整
            alpha = 0.3  # 核PCA调整的权重
            X_t_cont_aligned = (1 - alpha) * X_t_cont_simple + alpha * (X_t_cont_simple + weighted_adjustment)
            
            logging.info(f"使用加权平均方法，核PCA调整权重: {alpha}")
        
        # 如果标准化了，需要逆标准化
        if standardize:
            X_t_cont_aligned = scaler.inverse_transform(X_t_cont_aligned)
        
        X_t_aligned = X_t.copy()
        X_t_aligned[:, cont_idx] = X_t_cont_aligned
        
        # 计算原始空间的最终MMD
        final_mmd_original = compute_mmd(X_s[:, cont_idx], X_t_cont_aligned, kernel=kernel, gamma=gamma)
        
        # 使用核PCA空间的MMD作为主要评估指标，但也报告原始空间的MMD
        final_mmd = mmd_in_kpca_after
        logging.info(f"最终MMD (核PCA空间): {final_mmd:.6f}")
        logging.info(f"最终MMD (原始空间): {final_mmd_original:.6f}")
    
    # 验证类别特征是否保持不变
    if not np.array_equal(X_t[:, cat_idx], X_t_aligned[:, cat_idx]):
        logging.error("错误：类别特征在核PCA变换过程中被改变")
    else:
        logging.info("验证成功：类别特征在核PCA变换过程中保持不变")
    
    mmd_info = {
        "initial_mmd": initial_mmd,
        "mmd_in_kpca_space_before_align": mmd_in_kpca_before, 
        "mmd_in_kpca_space_after_align": mmd_in_kpca_after, 
        "final_mmd": final_mmd,
        "gamma_used": gamma,
        "n_components_used": n_components,
        "use_inverse_transform": use_inverse_transform,
        "standardized": standardize
    }
    
    return X_t_aligned, mmd_info

def mmd_transform(X_s: np.ndarray, X_t: np.ndarray, method: str = 'linear', 
                 cat_idx: Optional[list] = None, **kwargs: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    使用MMD进行特征对齐的主函数
    
    参数:
    - X_s: 源域特征 [n_samples_source, n_features]
    - X_t: 目标域特征 [n_samples_target, n_features]
    - method: 对齐方法，可选 'linear', 'kpca', 'mean_std'
    - cat_idx: 类别特征索引，必须提供
    - **kwargs: 传递给具体对齐方法的其他参数
    
    返回:
    - X_t_aligned: 对齐后的目标域特征
    - mmd_info: MMD相关信息
    """
    if cat_idx is None:
        raise ValueError("cat_idx must be provided for mmd_transform")
    
    all_idx = list(range(X_s.shape[1]))
    cont_idx = [i for i in all_idx if i not in cat_idx]
    
    # 分离连续特征
    X_s_cont = X_s[:, cont_idx]
    X_t_cont = X_t[:, cont_idx]
    
    # 首先确定要使用的gamma参数（保证评估和训练一致）
    gamma = kwargs.get('gamma', None)
    gamma_search_values = kwargs.get('gamma_search_values', None)
    dynamic_gamma = kwargs.get('dynamic_gamma', True)
    
    # 如果设置了gamma搜索值且启用了动态gamma，不使用中值启发式
    if gamma is None and not (dynamic_gamma and gamma_search_values):
        # 只有在没有设置gamma搜索值或禁用动态gamma时才使用中值启发式
        gamma = median_heuristic_gamma(np.vstack((X_s_cont, X_t_cont)))
        logging.info(f"使用中值启发式计算的gamma: {gamma:.6f}")
    elif gamma is None and dynamic_gamma and gamma_search_values:
        # 如果启用了动态gamma且有搜索值，使用搜索值中的第一个作为初始值
        gamma = gamma_search_values[0]
        logging.info(f"使用gamma搜索值中的初始值: {gamma:.6f}")
    elif gamma is not None:
        logging.info(f"使用明确指定的gamma: {gamma:.6f}")
    
    # 统一计算初始MMD基线（使用与训练相同的gamma）
    initial_mmd = compute_mmd(X_s_cont, X_t_cont, kernel='rbf', gamma=gamma)
    logging.info(f"统一基线MMD (连续特征，gamma={gamma:.6f}): {initial_mmd:.6f}")
    
    # 选择对齐方法
    if method == 'linear':
        # 使用线性变换对齐
        
        lr = kwargs.get('lr', 0.01)
        n_epochs = kwargs.get('n_epochs', 300)
        batch_size = kwargs.get('batch_size', 64)
        lambda_reg = kwargs.get('lambda_reg', 1e-4)
        staged_training = kwargs.get('staged_training', True)
        dynamic_gamma = kwargs.get('dynamic_gamma', True)
        gamma_search_values = kwargs.get('gamma_search_values', None)
        use_gradient_clipping = kwargs.get('use_gradient_clipping', False)
        max_grad_norm = kwargs.get('max_grad_norm', 1.0)
        standardize_features = kwargs.get('standardize_features', False)
        monitor_gradients = kwargs.get('monitor_gradients', False)
        
        # 初始化改进版MMDLinearTransform，使用连续特征的维度
        mmd_linear = MMDLinearTransform(
            input_dim=len(cont_idx),
            gamma=gamma,
            lr=lr,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lambda_reg=lambda_reg,
            staged_training=staged_training,
            dynamic_gamma=dynamic_gamma,
            gamma_search_values=gamma_search_values,
            use_gradient_clipping=use_gradient_clipping,
            max_grad_norm=max_grad_norm,
            standardize_features=standardize_features,
            monitor_gradients=monitor_gradients
        )
        
        # 训练时只使用连续特征
        mmd_linear.fit(X_s_cont, X_t_cont)
        
        # 获取训练时实际使用的gamma值（可能通过动态搜索更新了）
        actual_gamma_used = mmd_linear.best_gamma
        
        # 创建结果数组
        X_t_aligned = X_t.copy()
        
        # 提取连续特征，变换后再放回
        X_t_cont_aligned = mmd_linear.transform(X_t_cont)
        X_t_aligned[:, cont_idx] = X_t_cont_aligned
        
        # 重要修复：使用训练时相同的gamma计算最终MMD
        final_mmd = compute_mmd(X_s_cont, X_t_cont_aligned, kernel='rbf', gamma=actual_gamma_used)
        
        # 重新计算初始MMD（使用相同的gamma确保公平比较）
        initial_mmd_corrected = compute_mmd(X_s_cont, X_t_cont, kernel='rbf', gamma=actual_gamma_used)
        
        logging.info(f"Gamma一致性修复:")
        logging.info(f"  训练使用的gamma: {actual_gamma_used:.6f}")
        logging.info(f"  修正后初始MMD: {initial_mmd_corrected:.6f}")
        logging.info(f"  修正后最终MMD: {final_mmd:.6f}")
        
        align_info = {
            'method': 'linear',
            'n_epochs': n_epochs,
            'lambda_reg': lambda_reg,
            'gamma_used': actual_gamma_used,
            'initial_mmd_corrected': initial_mmd_corrected
        }
        
    elif method == 'kpca':
        # 使用改进的核PCA对齐
        kernel_param = kwargs.get('kernel', 'rbf')
        gamma_param = kwargs.get('gamma', None)  # 如果为None，将使用中值启发式
        n_components_param = kwargs.get('n_components', None)
        use_inverse_transform = kwargs.get('use_inverse_transform', False)  # 默认不使用逆变换
        standardize = kwargs.get('standardize', True)  # 默认标准化
        
        X_t_aligned, kpca_internal_info = mmd_kernel_pca_transform(
            X_s, X_t, cat_idx, n_components_param, kernel_param, gamma_param,
            use_inverse_transform, standardize
        )
        
        final_mmd = kpca_internal_info['final_mmd']
        
        align_info = {
            'method': 'kpca_improved',
            'kernel': kernel_param,
            'gamma_used': kpca_internal_info['gamma_used'],
            'n_components_used': kpca_internal_info['n_components_used'],
            'use_inverse_transform': use_inverse_transform,
            'standardized': standardize,
            'mmd_in_kpca_space_before_align': kpca_internal_info['mmd_in_kpca_space_before_align'],
            'mmd_in_kpca_space_after_align': kpca_internal_info['mmd_in_kpca_space_after_align']
        }
        
    elif method == 'mean_std':
        # 简化的均值-标准差对齐方法
        X_s_mean = np.mean(X_s_cont, axis=0)
        X_s_std = np.std(X_s_cont, axis=0) + 1e-8
        X_t_mean = np.mean(X_t_cont, axis=0)
        X_t_std = np.std(X_t_cont, axis=0) + 1e-8
        
        # 单步对齐：标准化目标域数据，然后重新缩放到源域分布
        X_t_cont_aligned = ((X_t_cont - X_t_mean) / X_t_std) * X_s_std + X_s_mean
        
        # 重构完整特征矩阵
        X_t_aligned = np.copy(X_t)
        X_t_aligned[:, cont_idx] = X_t_cont_aligned
        
        final_mmd = compute_mmd(X_s_cont, X_t_cont_aligned, kernel='rbf', gamma=gamma)
        
        align_info = {
            'method': 'mean_std_simple',
            'description': '单步均值-方差对齐'
        }
        
    else:
        raise ValueError(f"未知的对齐方法: {method}")
    
    # 对于linear方法，使用修正后的初始MMD计算减少百分比
    if method == 'linear' and 'initial_mmd_corrected' in align_info:
        corrected_initial_mmd = align_info['initial_mmd_corrected']
        # 修复除零错误：当初始MMD为0时，设置减少百分比为0
        if corrected_initial_mmd > 1e-10:  # 使用小的阈值避免数值误差
            mmd_reduction = (corrected_initial_mmd - final_mmd) / corrected_initial_mmd * 100
        else:
            mmd_reduction = 0.0
            logging.info("初始MMD接近0，无需域适应")
        
        logging.info(f"最终MMD: {final_mmd:.6f}")
        logging.info(f"MMD减少 (gamma一致性修正): {mmd_reduction:.2f}%")
        
        # 返回对齐特征和MMD信息
        mmd_info = {
            'method': method,
            'initial_mmd': corrected_initial_mmd,  # 使用修正后的初始MMD
            'final_mmd': final_mmd,
            'reduction': mmd_reduction,
            'align_info': align_info,
            'gamma_consistency_fixed': True
        }
    else:
        # 其他方法使用原来的逻辑，同样修复除零错误
        if initial_mmd > 1e-10:  # 使用小的阈值避免数值误差
            mmd_reduction = (initial_mmd - final_mmd) / initial_mmd * 100
        else:
            mmd_reduction = 0.0
            logging.info("初始MMD接近0，无需域适应")
        
        logging.info(f"最终MMD: {final_mmd:.6f}")
        logging.info(f"MMD减少: {mmd_reduction:.2f}%")
        
        # 返回对齐特征和MMD信息
        mmd_info = {
            'method': method,
            'initial_mmd': initial_mmd,  # 统一基线
            'final_mmd': final_mmd,
            'reduction': mmd_reduction,
            'align_info': align_info
        }
    
    return X_t_aligned, mmd_info
