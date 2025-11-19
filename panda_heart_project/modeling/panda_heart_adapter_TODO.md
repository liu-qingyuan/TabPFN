# PANDA适配器模块 TODO

## 模块概述
**文件**: `models/panda_heart_adapter.py`
**功能**: 将PANDA框架适配到心脏病多中心诊断任务
**负责人**: [待分配]
**预计工时**: 26小时

---

## 📋 详细任务清单

### TASK-004: PANDA框架心脏病数据适配
**优先级**: 🔥 High | **预计工时**: 12小时 | **截止**: Week 3

#### 子任务
- [ ] **TASK-004-1**: TabPFN 32成员集成配置
  - **基础架构**: 基于TabPFN预训练Transformer
  - **集成策略**: 特征旋转 + 输入变换 + 种子变化
  - **心脏病适配**:
    - 生命体征专用变换
    - 临床特征约束
    - 类别不平衡权重
    - TabPFN特征子集多样性

- [ ] **TASK-004-2**: 心脏病专用特征变换
  - **特征分组**: 人口统计、风险因素、症状、诊断
  - **变换策略**:
    - 年龄相关特征: 年龄调整变换
    - 生命体征: 临床范围标准化
    - 心电图特征: 医学约束变换
  - **变换验证**: 医学合理性检查

- [ ] **TASK-004-3**: 类别不平衡处理
  - **策略**: 加权损失 + 重采样
  - **权重计算**: 基于医学重要性
  - **阈值优化**: Youden指数最大化
  - **评估**: 临床可接受性

#### 验收标准
- [ ] 32成员集成成功构建
- [ ] 心脏病专用特征变换医学合理
- [ ] 不平衡处理提升敏感性>90%

#### 技术要求
```python
# 伪代码示例
class PANDAHeartAdapter:
    """PANDA心脏病适配器"""

    def __init__(self, config):
        self.ensemble_size = 32
        self.feature_groups = {
            'demographics': ['age', 'sex'],
            'vital_signs': ['trestbps', 'chol', 'fbs'],
            'symptoms': ['cp', 'exang', 'oldpeak'],
            'diagnostic': ['restecg', 'ca', 'thal', 'slope', 'thalach']
        }

    def fit(self, X, y):
        """训练PANDA模型"""
        pass

    def predict_proba(self, X):
        """预测概率"""
        pass

    def get_feature_importance(self):
        """获取特征重要性"""
        pass
```

---

### TASK-005: 不确定性量化实现
**优先级**: 🟡 Medium | **预计工时**: 6小时 | **截止**: Week 3

#### 子任务
- [ ] **TASK-005-1**: 预测不确定性估计
  - **方法**: 深度集成 + MC Dropout
  - **类型**: 认知不确定性 + 偶然不确定性
  - **计算**: 集成方差 + Dropout采样方差

- [ ] **TASK-005-2**: 置信区间计算
  - **方法**: Bootstrap + 分位数估计
  - **置信水平**: 95%临床标准
  - **医学解释**: 风险分层决策支持

- [ ] **TASK-005-3**: 可靠性评分
  - **指标**: Expected Calibration Error (ECE)
  - **可视化**: 可靠性图 + 置信区间图
  - **临床应用**: 信任度评估

#### 验收标准
- [ ] 不确定性量化方法完整实现
- [ ] 置信区间覆盖概率≈95%
- [ ] 可靠性评分ECE < 0.05

#### 技术要求
```python
# 伪代码示例
class UncertaintyQuantifier:
    """不确定性量化器"""

    def __init__(self, model, n_samples=1000):
        self.model = model
        self.n_samples = n_samples

    def predict_with_uncertainty(self, X):
        """带不确定性预测"""
        # 深度集成预测
        ensemble_preds = self._ensemble_predict(X)
        # MC Dropout预测
        dropout_preds = self._dropout_predict(X)

        mean_pred = np.mean(ensemble_preds, axis=0)
        uncertainty = np.var(ensemble_preds, axis=0)

        return mean_pred, uncertainty

    def compute_confidence_interval(self, X, confidence=0.95):
        """计算置信区间"""
        pass
```

---

### TASK-006: 临床约束优化
**优先级**: 🟡 Medium | **预计工时**: 8小时 | **截止**: Week 4

#### 子任务
- [ ] **TASK-006-1**: 特征重要性约束
  - **临床先验**: 心脏病学专家知识
  - **约束方法**: 正则化 + 硬约束
  - **验证**: 特征重要性医学解释性

- [ ] **TASK-006-2**: 生理范围限制
  - **约束项**: 年龄-心率关系
  - **合理性**: 预测符合生理学
  - **检测**: 异常预测识别

- [ ] **TASK-006-3**: 医学知识融入
  - **知识图谱**: 心脏病诊断流程
  - **规则约束**: 医学诊断规则
  - **学习**: 知识引导的特征学习

#### 验收标准
- [ ] 特征重要性符合医学直觉
- [ ] 生理约束有效实施
- [ ] 医学知识成功融入

#### 技术要求
```python
# 伪代码示例
class ClinicalConstraints:
    """临床约束管理器"""

    def __init__(self):
        self.physiological_rules = {
            'max_heart_rate': lambda age: 220 - age,
            'blood_pressure_range': (60, 250),
            'cholesterol_range': (100, 600)
        }

    def apply_constraints(self, model, X):
        """应用临床约束"""
        # 特征重要性约束
        feature_importance_loss = self._compute_importance_loss(model, X)

        # 生理约束
        physiological_loss = self._compute_physiological_loss(model, X)

        # 总约束损失
        total_loss = feature_importance_loss + physiological_loss

        return total_loss
```

---

## 🔧 实现细节

### 核心架构设计
```python
@dataclass
class PANDAHeartConfig:
    """PANDA心脏病配置"""
    ensemble_size: int = 32
    feature_rotations: bool = True
    clinical_constraints: bool = True
    uncertainty_quantification: bool = True

    # 特征编码配置
    feature_encoding: Dict[str, str] = field(default_factory=lambda: {
        'age': 'clinical_normalized',
        'sex': 'binary',
        'cp': 'ordinal',
        'trestbps': 'clinical_normalized',
        'chol': 'clinical_normalized',
        'thalach': 'age_adjusted',
        'oldpeak': 'robust_scaling',
        'ca': 'numeric',
        'thal': 'one_hot'
    })

    # 集成多样性配置
    diversity_config: Dict[str, Any] = field(default_factory=lambda: {
        'rotation_variance': 0.1,
        'noise_level': 0.05,
        'subset_ratio': 0.8
    })

class PANDAHeartAdapter:
    """PANDA心脏病适配器主类"""

    def __init__(self, config: PANDAHeartConfig):
        self.config = config
        self.ensemble_models = []
        self.feature_encoder = None
        self.uncertainty_quantifier = None
        self.clinical_constraints = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> 'PANDAHeartAdapter':
        """训练PANDA模型"""
        # 特征编码
        X_encoded = self._encode_features(X_train)

        # 构建集成模型
        self._build_ensemble(X_encoded, y_train)

        # 训练集成成员
        self._train_ensemble(X_encoded, y_train)

        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """预测概率"""
        X_encoded = self._encode_features(X_test)
        return self._ensemble_predict_proba(X_encoded)

    def predict_with_uncertainty(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """带不确定性预测"""
        if self.config.uncertainty_quantification:
            return self.uncertainty_quantifier.predict_with_uncertainty(X_test)
        else:
            # 简单集成不确定性
            return self._simple_uncertainty_predict(X_test)
```

### 集成成员构建
```python
def _build_ensemble(self, X: np.ndarray, y: np.ndarray):
    """构建集成成员"""
    for i in range(self.config.ensemble_size):
        # 特征旋转
        X_rotated = self._apply_feature_rotation(X, i)

        # 数据子集采样
        subset_idx = self._sample_subset(len(X), i)

        # 噪声注入
        X_noisy = self._inject_noise(X_rotated[subset_idx], i)

        # 创建TabPFN实例
        model = self._create_tabpfn_instance(i)

        self.ensemble_models.append({
            'model': model,
            'subset_idx': subset_idx,
            'rotation_matrix': self._get_rotation_matrix(i),
            'noise_params': self._get_noise_params(i)
        })
```

---

## 🧪 测试计划

### 单元测试
- [ ] **集成构建测试**: 验证32成员正确创建
- [ ] **特征变换测试**: 验证编码正确性
- [ ] **不确定性测试**: 验证量化方法
- [ ] **约束测试**: 验证临床约束

### 集成测试
- [ ] **端到端训练**: 从数据到模型
- [ ] **预测一致性**: 多次预测稳定性
- [ ] **性能基准**: 与基线模型对比

### 医学验证
- [ ] **特征重要性**: 医学专家验证
- [ ] **预测合理性**: 临床案例验证
- [ ] **不确定性解释**: 临床实用性

---

## 📊 预期输出

### 模型文件
- `models/panda_heart_model.pkl` - 完整PANDA模型
- `models/feature_encoders.pkl` - 特征编码器
- `models/uncertainty_models.pkl` - 不确定性模型
- `models/clinical_constraints.pkl` - 约束参数

### 性能报告
- 集成多样性分析
- 不确定性量化效果
- 临床约束影响
- 与基线对比结果

---

## 🚨 风险与缓解

### 风险识别
1. **计算复杂度过高** (32成员训练)
2. **内存占用过大** (GPU限制)
3. **临床约束过度限制** (性能下降)

### 缓解策略
1. **并行训练 + 梯度检查点**
2. **模型分片 + 内存优化**
3. **约束权重调优 + 敏感性分析**

---

## 📞 联系信息
**负责人**: [待分配]
**医学顾问**: [心脏病学专家]
**技术支持**: [AI工程师]

*最后更新: 2025-11-18*