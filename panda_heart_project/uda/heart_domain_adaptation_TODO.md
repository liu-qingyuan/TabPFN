# 域适应方法 TODO

## 模块概述
**文件**: `uda/heart_domain_adaptation.py`
**功能**: 心脏病跨医院域适应算法实现
**负责人**: [待分配]
**预计工时**: 20小时

---

## 📋 详细任务清单

### TASK-013: TCA域适应实现
**优先级**: 🔥 High | **预计工时**: 8小时 | **截止**: Week 4

#### 子任务
- [ ] **TASK-013-1**: TCA核心算法
  - **核函数选择**: 线性核、RBF核、多项式核
  - **参数优化**: mu值网格搜索和交叉验证
  - **维度选择**: 自动维度确定策略
  - **收敛优化**: 迭代收敛和数值稳定性

- [ ] **TASK-013-2**: 医学数据适配
  - **缺失值处理**: TCA对缺失值的鲁棒性
  - **特征权重**: 不同临床特征的TCA权重
  - **正则化**: L2正则化和平滑约束
  - **批量处理**: 大数据集分块TCA处理

- [ ] **TASK-013-3**: 跨医院优化
  - **多源TCA**: 多个源域的TCA融合
  - **渐进适应**: 源域到目标域的渐进TCA
  - **参数共享**: 不同医院对的参数共享策略

#### 验收标准
- [ ] TCA算法实现正确且收敛稳定
- [ ] 医学数据处理鲁棒性强
- [ ] 跨医院适应效果显著

#### 技术要求
```python
# 伪代码示例
class HeartDiseaseTCA:
    """心脏病专用TCA域适应"""

    def __init__(self, kernel_type='linear', mu=0.1, n_components=None,
                 kernel_params=None, tol=1e-5, max_iter=1000):
        self.kernel_type = kernel_type
        self.mu = mu
        self.n_components = n_components
        self.kernel_params = kernel_params or {}
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X_source, y_source, X_target):
        """训练TCA变换"""
        n_source = X_source.shape[0]
        n_total = n_source + X_target.shape[0]

        # 构建核矩阵
        K = self._compute_kernel_matrix(np.vstack([X_source, X_target]))

        # 构建MMD矩阵
        L = self._compute_mmd_matrix(n_source, n_total)

        # 构建单位矩阵约束
        H = np.eye(n_total) - np.ones((n_total, n_total)) / n_total

        # 求解广义特征值问题
        self._solve_eigen_problem(K, L, H, n_total)

        return self

    def transform(self, X_source, X_target):
        """应用TCA变换"""
        K_source = self._compute_kernel_matrix(X_source, self.X_train_)
        K_target = self._compute_kernel_matrix(X_target, self.X_train_)

        X_source_tca = K_source @ self.components_.T
        X_target_tca = K_target @ self.components_.T

        return X_source_tca, X_target_tca

    def _compute_kernel_matrix(self, X1, X2=None):
        """计算核矩阵"""
        if X2 is None:
            X2 = X1

        if self.kernel_type == 'linear':
            return X1 @ X2.T
        elif self.kernel_type == 'rbf':
            gamma = self.kernel_params.get('gamma', 1.0)
            return np.exp(-gamma * np.linalg.norm(X1[:, None] - X2[None, :], axis=2)**2)
        elif self.kernel_type == 'polynomial':
            degree = self.kernel_params.get('degree', 3)
            gamma = self.kernel_params.get('gamma', 1.0)
            coef0 = self.kernel_params.get('coef0', 1)
            return (gamma * X1 @ X2.T + coef0) ** degree
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
```

---

### TASK-014: CORAL域适应实现
**优先级**: 🔥 High | **预计工时**: 6小时 | **截止**: Week 4

#### 子任务
- [ ] **TASK-014-1**: CORAL核心算法
  - **协方差对齐**: 源域和目标域协方差对齐
  - **白化变换**: 数据白化和色彩变换
  - **正则化**: 协方差矩阵正则化
  - **数值稳定**: 奇异值分解优化

- [ ] **TASK-014-2**: 医学数据优化
  - **特征标准化**: CORAL前的数据标准化
  - **缺失值处理**: 协方差估计的缺失值处理
  - **批次效应**: 医院批次效应的CORAL校正

- [ ] **TASK-014-3**: 参数调优
  - **正则化强度**: CORAL正则化参数优化
  - **特征权重**: 不同特征类型的CORAL权重
  - **集成策略**: CORAL与其他方法的集成

#### 验收标准
- [ ] CORAL算法实现正确
- [ ] 医学数据预处理完善
- [ ] 参数调优策略有效

#### 技术要求
```python
# 伪代码示例
class HeartDiseaseCORAL:
    """心脏病专用CORAL域适应"""

    def __init__(self, reg_param=1e-3, adaptive_weights=False):
        self.reg_param = reg_param
        self.adaptive_weights = adaptive_weights

    def fit(self, X_source, y_source, X_target):
        """训练CORAL变换"""
        # 数据标准化
        X_source_std = self._standardize_data(X_source)
        X_target_std = self._standardize_data(X_target)

        # 计算协方差矩阵
        cov_source = self._compute_covariance(X_source_std)
        cov_target = self._compute_covariance(X_target_std)

        # CORAL变换
        self.transformation_matrix_ = self._compute_coral_transform(
            cov_source, cov_target
        )

        return self

    def transform(self, X):
        """应用CORAL变换"""
        X_std = self._standardize_data(X)
        return X_std @ self.transformation_matrix_

    def _compute_coral_transform(self, cov_source, cov_target):
        """计算CORAL变换矩阵"""
        # 正则化
        cov_source_reg = cov_source + self.reg_param * np.eye(cov_source.shape[0])
        cov_target_reg = cov_target + self.reg_param * np.eye(cov_target.shape[0])

        # Cholesky分解
        L_source = np.linalg.cholesky(cov_source_reg)
        L_target = np.linalg.cholesky(cov_target_reg)

        # CORAL变换
        coral_transform = np.linalg.inv(L_source) @ L_target

        return coral_transform
```

---

### TASK-015: SA域适应实现
**优先级**: 🔥 Medium | **预计工时**: 6小时 | **截止**: Week 5

#### 子任务
- [ ] **TASK-015-1**: 子空间对齐算法
  - **PCA降维**: 源域和目标域PCA降维
  - **子空间对齐**: 子空间基向量对齐
  - **最优子空间**: 子空间维度自动选择
  - **对齐度量**: 子空间对齐程度评估

- [ ] **TASK-015-2**: 医学数据适配
  - **特征预处理**: SA前的特征处理
  - **方差保留**: 医学重要特征的方差保留
  - **鲁棒性**: SA对噪声和缺失值的鲁棒性

#### 验收标准
- [ ] SA算法实现正确
- [ ] 医学特征保留良好
- [ ] 子空间对齐效果显著

#### 技术要求
```python
# 伪代码示例
class HeartDiseaseSA:
    """心脏病专用子空间对齐域适应"""

    def __init__(self, n_components=0.95, alignment_method='procrustes'):
        self.n_components = n_components
        self.alignment_method = alignment_method

    def fit(self, X_source, y_source, X_target):
        """训练子空间对齐"""
        # PCA降维
        self.pca_source_, self.pca_target_ = self._fit_pca(X_source, X_target)

        # 提取子空间基
        basis_source = self.pca_source_.components_
        basis_target = self.pca_target_.components_

        # 子空间对齐
        self.alignment_matrix_ = self._align_subspaces(basis_source, basis_target)

        return self

    def transform(self, X):
        """应用子空间对齐变换"""
        # 先PCA变换
        X_pca = self.pca_source_.transform(X)
        # 再对齐变换
        return X_pca @ self.alignment_matrix_

    def _align_subspaces(self, basis_source, basis_target):
        """对齐子空间基向量"""
        if self.alignment_method == 'procrustes':
            # Procrustes分析
            M = basis_target.T @ basis_source
            U, _, Vt = np.linalg.svd(M)
            return Vt.T @ U.T
        else:
            # 直接最小二乘
            M = basis_target.T @ basis_source
            return np.linalg.lstsq(basis_source, basis_target, rcond=None)[0]
```

---

## 🔧 实现细节

### 域适应方法集成
```python
@dataclass
class DomainAdaptationConfig:
    """域适应配置"""
    methods: List[str] = field(default_factory=lambda: ['tca', 'coral', 'sa'])
    parameter_grids: Dict[str, Dict] = field(default_factory=lambda: {
        'tca': {'mu': [0.01, 0.1, 1.0, 10.0], 'kernel_type': ['linear', 'rbf']},
        'coral': {'reg_param': [1e-4, 1e-3, 1e-2, 1e-1]},
        'sa': {'n_components': [0.8, 0.9, 0.95, 0.99]}
    })
    cv_folds: int = 5
    scoring: str = 'roc_auc'

class HeartDiseaseDomainAdapter:
    """心脏病域适应主类"""

    def __init__(self, config: DomainAdaptationConfig):
        self.config = config
        self.adapters = {}
        self.best_params = {}

    def fit_adapters(self, X_source, y_source, X_target):
        """训练所有域适应方法"""
        for method in self.config.methods:
            adapter = self._create_adapter(method)

            # 参数搜索
            best_adapter, best_params = self._hyperparameter_search(
                adapter, X_source, y_source, X_target, method
            )

            # 训练最优适配器
            best_adapter.fit(X_source, y_source, X_target)

            self.adapters[method] = best_adapter
            self.best_params[method] = best_params

    def transform_data(self, X_source, X_target, method='tca'):
        """应用域适应变换"""
        if method not in self.adapters:
            raise ValueError(f"Adapter {method} not fitted")

        adapter = self.adapters[method]
        return adapter.transform(X_source), adapter.transform(X_target)

    def evaluate_adaptation(self, X_source, y_source, X_target, y_target,
                           model, metrics=['auc_roc', 'accuracy']):
        """评估域适应效果"""
        results = {}

        # 原始性能（无适应）
        model.fit(X_source, y_source)
        y_pred_no_uda = model.predict(X_target)
        y_prob_no_uda = model.predict_proba(X_target)[:, 1]

        no_uda_metrics = self._compute_metrics(y_target, y_pred_no_uda, y_prob_no_uda, metrics)

        # 各种域适应方法的性能
        for method, adapter in self.adapters.items():
            # 域适应变换
            X_source_ada, X_target_ada = adapter.transform(X_source, X_target)

            # 训练和预测
            model.fit(X_source_ada, y_source)
            y_pred_ada = model.predict(X_target_ada)
            y_prob_ada = model.predict_proba(X_target_ada)[:, 1]

            # 计算指标
            ada_metrics = self._compute_metrics(y_target, y_pred_ada, y_prob_ada, metrics)

            # 计算适应增益
            adaptation_gain = {}
            for metric in metrics:
                adaptation_gain[metric] = ada_metrics[metric] - no_uda_metrics[metric]

            results[method] = {
                'metrics': ada_metrics,
                'adaptation_gain': adaptation_gain,
                'relative_improvement': {
                    metric: adaptation_gain[metric] / no_uda_metrics[metric]
                    for metric in metrics if no_uda_metrics[metric] > 0
                }
            }

        return results
```

---

## 🧪 测试计划

### 单元测试
- [ ] **TCA算法**: 核函数计算、特征值求解验证
- [ ] **CORAL算法**: 协方差对齐、数值稳定性验证
- [ ] **SA算法**: PCA降维、子空间对齐验证
- [ ] **参数优化**: 网格搜索、交叉验证验证

### 集成测试
- [ ] **多方法对比**: 不同域适应方法的性能对比
- [ ] **端到端流程**: 完整域适应流程验证
- [ ] **参数敏感性**: 参数变化对性能的影响

### 医学验证
- [ ] **跨医院验证**: 不同医院对的适应效果
- [ ] **临床合理性**: 域适应结果的医学解释
- [ ] **稳定性分析**: 域适应方法的稳定性

---

## 📊 预期输出

### 域适应结果
- `uda/tca_results.json` - TCA域适应结果
- `uda/coral_results.json` - CORAL域适应结果
- `uda/sa_results.json` - SA域适应结果
- `uda/method_comparison.json` - 方法对比结果

### 可视化输出
- `uda/domain_adaptation_performance.png` - 域适应性能对比
- `uda/tca_visualization.png` - TCA降维可视化
- `uda/coral_alignment.png` - CORAL对齐效果

---

## 🚨 风险与缓解

### 风险识别
1. **数值不稳定** (矩阵求逆、特征值分解)
2. **过拟合** (参数过多、复杂度过高)
3. **医学不合理** (域适应破坏医学特征)

### 缓解策略
1. **正则化 + 数值优化**
2. **交叉验证 + 简单性约束**
3. **医学约束 + 特征保护**

---

## 📞 联系信息
**负责人**: [待分配]
**算法顾问**: [域适应专家]
**医学支持**: [心脏病学专家]

*最后更新: 2025-11-18*