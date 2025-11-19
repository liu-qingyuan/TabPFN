# 特征工程模块 TODO

## 模块概述
**文件**: `features/clinical_feature_engineering.py`
**功能**: 心脏病数据特征工程和临床特征处理
**负责人**: [待分配]
**预计工时**: 16小时

---

## 📋 详细任务清单

### TASK-010: 临床特征标准化
**优先级**: 🔥 High | **预计工时**: 6小时 | **截止**: Week 3

#### 子任务
- [ ] **TASK-010-1**: 基础特征编码
  - **年龄**: 临床标准化 (z-score)
  - **性别**: 二进制编码 (0=女, 1=男)
  - **胸痛类型**: 序数编码 (1-4级)
  - **空腹血糖**: 二进制编码 (0/1)

- [ ] **TASK-010-2**: 生命体征特征处理
  - **血压**: 年龄调整的标准化
  - **胆固醇**: 临床范围验证
  - **最大心率**: 年龄预测心率对比
  - **ST段压低**: 鲁棒缩放处理

- [ ] **TASK-010-3**: 心电图特征编码
  - **静息心电图**: 序数编码 (0-2级)
  - **ST段斜率**: 序数编码 (1-3级)
  - **地中海贫血**: 独热编码 (3/6/7)

#### 验收标准
- [ ] 14个临床特征全部正确编码
- [ ] 临床合理性验证通过
- [ ] 缺失值处理策略明确

#### 技术要求
```python
# 伪代码示例
class ClinicalFeatureEncoder:
    """心脏病临床特征编码器"""

    def __init__(self):
        self.feature_config = {
            # 人口统计学特征
            'age': {'type': 'continuous', 'method': 'clinical_standardized'},
            'sex': {'type': 'binary', 'method': 'binary_encoding'},

            # 症状特征
            'cp': {'type': 'categorical', 'method': 'ordinal_encoding', 'categories': [1,2,3,4]},
            'exang': {'type': 'binary', 'method': 'binary_encoding'},

            # 生命体征
            'trestbps': {'type': 'continuous', 'method': 'age_adjusted_scaling'},
            'chol': {'type': 'continuous', 'method': 'clinical_range_validation'},
            'thalach': {'type': 'continuous', 'method': 'age_predicted_comparison'},
            'fbs': {'type': 'binary', 'method': 'binary_encoding'},

            # 心电图特征
            'restecg': {'type': 'categorical', 'method': 'ordinal_encoding', 'categories': [0,1,2]},
            'oldpeak': {'type': 'continuous', 'method': 'robust_scaling'},
            'slope': {'type': 'categorical', 'method': 'ordinal_encoding', 'categories': [1,2,3]},

            # 诊断特征
            'ca': {'type': 'categorical', 'method': 'numeric_encoding', 'categories': [0,1,2,3]},
            'thal': {'type': 'categorical', 'method': 'one_hot_encoding', 'categories': [3,6,7]}
        }

    def fit_transform(self, X, y=None):
        """训练并转换特征"""
        self._validate_clinical_ranges(X)
        return self._encode_features(X)

    def transform(self, X):
        """转换新数据"""
        return self._encode_features(X)
```

---

### TASK-011: 特征选择和优化
**优先级**: 🔥 High | **预计工时**: 6小时 | **截止**: Week 4

#### 子任务
- [ ] **TASK-011-1**: 医学特征选择
  - **临床相关性**: 基于医学文献的特征重要性
  - **统计显著性**: 单变量特征选择
  - **多重共线性**: VIF分析和特征去冗余
  - **缺失值容忍**: 高缺失率特征处理

- [ ] **TASK-011-2**: RFE特征选择
  - **TabPFN特征重要性**: 基于模型的特征排序
  - **递归特征消除**: 逐步特征筛选
  - **最优特征集**: best7-best10特征组合
  - **交叉验证**: 特征稳定性验证

- [ ] **TASK-011-3**: 跨域特征对齐
  - **特征分布对比**: 各中心特征差异分析
  - **域不变特征**: 跨中心稳定特征识别
  - **特征适配**: 域适应专用特征工程

#### 验收标准
- [ ] 特征选择结果医学合理
- [ ] RFE特征组合性能最优
- [ ] 跨域特征对齐有效

#### 技术要求
```python
# 伪代码示例
class HeartDiseaseFeatureSelector:
    """心脏病特征选择器"""

    def __init__(self):
        self.clinical_feature_sets = {
            'best7': ['age', 'sex', 'cp', 'thalach', 'exang', 'oldpeak', 'ca'],
            'best8': ['age', 'sex', 'cp', 'trestbps', 'thalach', 'exang', 'oldpeak', 'ca'],
            'best9': ['age', 'sex', 'cp', 'trestbps', 'chol', 'thalach', 'exang', 'oldpeak', 'ca'],
            'best10': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak', 'ca', 'thal']
        }

    def rfe_feature_selection(self, X, y, estimator, feature_names):
        """递归特征消除"""
        rfe = RFE(
            estimator=estimator,
            n_features_to_select=8,
            step=1,
            importance_getter='auto'
        )

        rfe.fit(X, y)
        selected_features = [feature_names[i] for i in range(len(feature_names)) if rfe.support_[i]]

        return {
            'selected_features': selected_features,
            'feature_ranking': rfe.ranking_,
            'feature_importance': self._compute_feature_importance(rfe, estimator, X, y)
        }

    def cross_domain_feature_analysis(self, X_dict, centers):
        """跨中心特征分析"""
        feature_stats = {}

        for center, X_center in X_dict.items():
            feature_stats[center] = {
                'mean': X_center.mean(),
                'std': X_center.std(),
                'missing_rate': X_center.isnull().mean(),
                'distribution': self._analyze_distribution(X_center)
            }

        # 计算特征稳定性
        stability_scores = self._compute_feature_stability(feature_stats)

        return {
            'feature_statistics': feature_stats,
            'stability_scores': stability_scores,
            'domain_invariant_features': self._select_domain_invariant_features(stability_scores)
        }
```

---

### TASK-012: 临床特征工程
**优先级**: 🔥 Medium | **预计工时**: 4小时 | **截止**: Week 4

#### 子任务
- [ ] **TASK-012-1**: 医学衍生特征
  - **年龄调整指标**: 最大心率/预测最大心率
  - **血压比值**: 收缩压/舒张压（如有）
  - **心血管风险**: 基于年龄和性别的基础风险评分

- [ ] **TASK-012-2**: 交互特征生成
  - **年龄×症状**: 年龄与胸痛类型的交互
  - **性别×风险**: 性别与其他风险因子交互
  - **多症状组合**: 复合症状指标

- [ ] **TASK-012-3**: 临床验证
  - **医学合理性**: 衍生特征的医学解释
  - **统计显著性**: 新特征的预测能力
  - **过拟合防范**: 特征复杂性控制

#### 验收标准
- [ ] 衍生特征具有医学意义
- [ ] 交互特征提升模型性能
- [ ] 特征工程无过拟合风险

#### 技术要求
```python
# 伪代码示例
class ClinicalFeatureEngineer:
    """临床特征工程"""

    def __init__(self):
        self.medical_reference = {
            'max_heart_rate_formula': '220 - age',  # 简化公式
            'bp_normal_ranges': {'systolic': (90, 120), 'diastolic': (60, 80)},
            'cholesterol_ranges': {'normal': (0, 200), 'borderline': (200, 240), 'high': (240, 500)}
        }

    def create_derived_features(self, X):
        """创建医学衍生特征"""
        X_derived = X.copy()

        # 年龄调整最大心率
        X_derived['heart_rate_reserve'] = X['thalach'] / (220 - X['age'])
        X_derived['heart_rate_achievement'] = X['thalach'] / (220 - X['age'])

        # 血压相关特征
        X_derived['bp_age_risk'] = self._compute_bp_age_risk(X['trestbps'], X['age'])

        # 胆固醇年龄风险
        X_derived['cholesterol_age_risk'] = self._compute_chol_age_risk(X['chol'], X['age'])

        return X_derived

    def create_interaction_features(self, X):
        """创建交互特征"""
        X_interaction = X.copy()

        # 年龄与胸痛类型交互
        X_interaction['age_cp_severe'] = X['age'] * (X['cp'] == 4).astype(int)

        # 性别与运动诱发心绞痛
        X_interaction['male_exang'] = X['sex'] * X['exang']

        # 综合风险评分
        X_interaction['combined_risk_score'] = (
            X['age'] / 100 +  # 年龄权重
            X['sex'] * 0.3 +  # 性别权重
            (X['cp'] - 1) * 0.2 +  # 胸痛权重
            X['exang'] * 0.3  # 运动心绞痛权重
        )

        return X_interaction
```

---

## 🔧 实现细节

### 特征工程配置
```python
@dataclass
class FeatureEngineeringConfig:
    """特征工程配置"""
    clinical_validation: bool = True
    missing_threshold: float = 0.3
    correlation_threshold: float = 0.8
    feature_sets: List[str] = field(default_factory=lambda: ['best7', 'best8', 'best9', 'best10'])
    cross_validation_folds: int = 10
    random_state: int = 42

class HeartDiseaseFeaturePipeline:
    """心脏病特征工程主流程"""

    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config
        self.encoder = ClinicalFeatureEncoder()
        self.selector = HeartDiseaseFeatureSelector()
        self.engineer = ClinicalFeatureEngineer()
        self.feature_sets = {}

    def fit_transform_pipeline(self, X_raw, y=None, center_info=None):
        """完整的特征工程流程"""
        # 1. 数据验证和清洗
        X_cleaned = self._validate_and_clean(X_raw)

        # 2. 基础特征编码
        X_encoded = self.encoder.fit_transform(X_cleaned, y)

        # 3. 衍生特征生成
        X_derived = self.engineer.create_derived_features(X_encoded)
        X_enhanced = self.engineer.create_interaction_features(X_derived)

        # 4. 特征选择
        if y is not None:
            self.feature_sets = self._perform_feature_selection(X_enhanced, y)

        return X_enhanced

    def get_feature_sets(self):
        """获取不同特征组合"""
        return self.feature_sets
```

---

## 🧪 测试计划

### 单元测试
- [ ] **特征编码**: 验证14个临床特征编码正确性
- [ ] **特征选择**: 验证RFE和统计选择方法
- [ ] **衍生特征**: 验证医学衍生特征计算
- [ ] **缺失值处理**: 验证缺失值插补策略

### 集成测试
- [ ] **完整流程**: 端到端特征工程验证
- [ ] **跨中心一致性**: 不同数据中心处理一致性
- [ ] **性能提升**: 特征工程对模型性能提升

### 临床验证
- [ ] **医学专家评审**: 特征工程医学合理性
- [ ] **临床标准符合**: 衍生特征临床解释性
- [ ] **风险评分验证**: 衍生风险指标准确性

---

## 📊 预期输出

### 特征工程结果
- `features/encoded_features.json` - 编码后特征数据
- `features/feature_selection_results.json` - 特征选择结果
- `features/feature_importance_ranking.json` - 特征重要性排序
- `features/cross_domain_analysis.json` - 跨中心特征分析

### 可视化输出
- `features/feature_correlation_matrix.png` - 特征相关性热图
- `features/feature_importance_plot.png` - 特征重要性图
- `features/cross_center_distribution.png` - 跨中心特征分布

---

## 🚨 风险与缓解

### 风险识别
1. **特征编码错误** (数据质量问题)
2. **过拟合特征** (泛化能力下降)
3. **医学不合理** (临床解释性问题)

### 缓解策略
1. **多重验证 + 医学专家评审**
2. **交叉验证 + 正则化控制**
3. **医学文献支持 + 临床专家验证**

---

## 📞 联系信息
**负责人**: [待分配]
**医学顾问**: [心脏病学专家]
**技术支持**: [特征工程工程师]

*最后更新: 2025-11-18*