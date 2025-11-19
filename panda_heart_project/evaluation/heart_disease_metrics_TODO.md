# 评估工具模块 TODO

## 模块概述
**文件**: `evaluation/heart_disease_metrics.py`
**功能**: 医学专用评估指标和跨域适应效果评估
**负责人**: [待分配]
**预计工时**: 18小时

---

## 📋 详细任务清单

### TASK-007: 医学专用评估指标
**优先级**: 🔥 High | **预计工时**: 6小时 | **截止**: Week 4

#### 子任务
- [ ] **TASK-007-1**: 核心诊断性能指标
  - **AUC-ROC**: 受试者工作特征曲线下面积
  - **AUC-PR**: 精确率-召回率曲线下面积
  - **敏感性**: 真阳性率 (漏诊最小化)
  - **特异性**: 真阴性率
  - **NPV**: 阴性预测值 (筛查能力)

- [ ] **TASK-007-2**: 校准性能指标
  - **Brier Score**: 概率预测准确度
  - **Calibration Slope**: 校准斜率
  - **Hosmer-Lemeshow**: 拟合优度检验
  - **ECE**: Expected Calibration Error

- [ ] **TASK-007-3**: 临床决策指标
  - **Youden Index**: 敏感性+特异性最优点
  - **Diagnostic Odds Ratio**: 诊断优势比
  - **F1-Score**: 调和平均数
  - **Matthews Correlation**: 不平衡数据鲁棒性

#### 验收标准
- [ ] 所有医学指标计算正确
- [ ] 敏感性>90%达到筛查要求
- [ ] 校准指标符合临床标准

#### 技术要求
```python
# 伪代码示例
class MedicalMetrics:
    """医学专用评估指标"""

    @staticmethod
    def compute_core_metrics(y_true, y_pred, y_prob):
        """计算核心诊断指标"""
        metrics = {
            'auc_roc': roc_auc_score(y_true, y_prob),
            'auc_pr': average_precision_score(y_true, y_prob),
            'sensitivity': recall_score(y_true, y_pred),
            'specificity': specificity_score(y_true, y_pred),
            'npv': negative_predictive_value(y_true, y_pred),
            'ppv': precision_score(y_true, y_pred)
        }
        return metrics

    @staticmethod
    def compute_calibration_metrics(y_true, y_prob, n_bins=10):
        """计算校准指标"""
        brier_score = brier_score_loss(y_true, y_prob)
        ece = expected_calibration_error(y_true, y_prob, n_bins)
        calibration_slope = calibration_regression(y_true, y_prob)

        return {
            'brier_score': brier_score,
            'ece': ece,
            'calibration_slope': calibration_slope
        }
```

---

### TASK-008: 跨域适应效果评估
**优先级**: 🔥 High | **预计工时**: 4小时 | **截止**: Week 4

#### 子任务
- [ ] **TASK-008-1**: 性能保持率计算
  - **公式**: Target_AUC / Source_AUC
  - **目标**: >85%性能保持率
  - **分析**: 跨中心性能下降原因

- [ ] **TASK-008-2**: 域适应提升分析
  - **适应增益**: With_UDA_AUC - Without_UDA_AUC
  - **相对提升**: (After_UDA - Before_UDA) / Before_UDA
  - **统计显著性**: 配对t检验

- [ ] **TASK-008-3**: 分布对齐度评估
  - **MMD距离**: Maximum Mean Discrepancy
  - **Wasserstein距离**: Earth Mover's Distance
  - **CORAL损失**: 相关性对齐损失

#### 验收标准
- [ ] 跨域性能保持率>85%
- [ ] 域适应带来显著提升(p<0.05)
- [ ] 分布对齐指标有效度量

#### 技术要求
```python
# 伪代码示例
class DomainAdaptationMetrics:
    """跨域适应效果评估"""

    def __init__(self):
        self.baseline_metrics = None
        self.adapted_metrics = None

    def evaluate_adaptation(self, source_model, target_model, X_source, X_target, y_target):
        """评估域适应效果"""
        # 源域性能
        source_performance = self._evaluate_model(source_model, X_source, X_target, y_target)

        # 目标域无适应性能
        no_uda_performance = self._evaluate_model(source_model, X_target, y_target)

        # 目标域有适应性能
        uda_performance = self._evaluate_model(target_model, X_target, y_target)

        # 计算适应效果指标
        adaptation_metrics = {
            'performance_retention': uda_performance['auc'] / source_performance['auc'],
            'adaptation_gain': uda_performance['auc'] - no_uda_performance['auc'],
            'relative_improvement': (uda_performance['auc'] - no_uda_performance['auc']) / no_uda_performance['auc']
        }

        return adaptation_metrics

    def compute_alignment_metrics(self, X_source, X_target, X_source_adapted, X_target_adapted):
        """计算分布对齐指标"""
        # 原始分布距离
        original_distance = self._compute_mmd_distance(X_source, X_target)

        # 适应后分布距离
        adapted_distance = self._compute_mmd_distance(X_source_adapted, X_target_adapted)

        return {
            'mmd_reduction': (original_distance - adapted_distance) / original_distance,
            'wasserstein_distance': wasserstein_distance(X_source_adapted, X_target_adapted)
        }
```

---

### TASK-009: 对比模型评估
**优先级**: 🔥 High | **预计工时**: 8小时 | **截止**: Week 5

#### 子任务
- [ ] **TASK-009-1**: 传统机器学习基线模型
  - **逻辑回归**: LASSO/LR，医学常用解释性模型
  - **XGBoost**: 梯度提升树，表格数据SOTA
  - **随机森林**: 集成学习，鲁棒性强
  - **SVM**: 支持向量机，经典分类器
  - **KNN**: K近邻，简单基线

- [ ] **TASK-009-2**: 跨域适应对比实验
  - **无域适应**: 直接迁移测试
  - **传统域适应**: CORAL, TCA, SA
  - **深度域适应**: DANN, MMD
  - **特征对齐**: PCA, CCA

- [ ] **TASK-009-3**: 统计显著性分析
  - **配对t检验**: PANDA vs 基线模型
  - **Wilcoxon检验**: 非参数显著性
  - **效应量计算**: Cohen's d
  - **多重比较**: Bonferroni校正

#### 验收标准
- [ ] 5个基线模型完整实现
- [ ] 跨域适应方法全面对比
- [ ] 统计显著性分析正确

#### 技术要求
```python
# 伪代码示例
class BaselineModels:
    """传统机器学习基线模型"""

    def __init__(self):
        self.models = {
            'logistic_regression': {
                'model': LogisticRegression(penalty='l1', solver='saga', class_weight='balanced'),
                'name': 'LASSO_LR',
                'interpretable': True
            },
            'xgboost': {
                'model': XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    class_weight='balanced'
                ),
                'name': 'XGBoost',
                'interpretable': False
            },
            'random_forest': {
                'model': RandomForestClassifier(
                    n_estimators=200,
                    max_depth=None,
                    class_weight='balanced'
                ),
                'name': 'Random_Forest',
                'interpretable': True
            },
            'svm': {
                'model': SVC(kernel='rbf', C=1.0, probability=True, class_weight='balanced'),
                'name': 'SVM',
                'interpretable': False
            },
            'knn': {
                'model': KNeighborsClassifier(n_neighbors=5, weights='distance'),
                'name': 'KNN',
                'interpretable': False
            }
        }

    def compare_with_panda(self, X_train, y_train, X_test, y_test, panda_model):
        """对比基线模型与PANDA"""
        results = {}

        # 基线模型评估
        for model_name, model_config in self.models.items():
            model = model_config['model']

            # 训练
            model.fit(X_train, y_train)

            # 预测
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            # 评估
            metrics = self._compute_metrics(y_test, y_pred, y_prob)
            results[model_config['name']] = metrics

        # PANDA模型评估
        panda_pred = panda_model.predict(X_test)
        panda_prob = panda_model.predict_proba(X_test)
        panda_metrics = self._compute_metrics(y_test, panda_pred, panda_prob)
        results['PANDA_TabPFN'] = panda_metrics

        # 统计显著性分析
        significance_analysis = self._statistical_significance_test(results)

        return {
            'performance_comparison': results,
            'statistical_analysis': significance_analysis,
            'best_model': self._select_best_model(results)
        }
```

#### 跨域适应对比实现
```python
class DomainAdaptationComparison:
    """域适应方法对比"""

    def __init__(self):
        self.adaptation_methods = {
            'no_uda': None,
            'pca': PCA(n_components=0.95),
            'cca': CCA(n_components=0.95),
            'tca': TCAAdapter(kernel='linear', mu=0.1),
            'coral': CORALAdapter(),
            'sa': SAAdapter(),
            'dann': DANN(),
            'mmd': MMDMatcher()
        }

    def run_comprehensive_comparison(self, X_source, y_source, X_target, y_target):
        """运行全面的域适应对比"""
        results = {}

        for method_name, adapter in self.adaptation_methods.items():
            print(f"Testing {method_name}...")

            # 域适应训练和测试
            if adapter is None:
                # 无域适应基线
                X_source_train = X_source
                X_target_test = X_target
            else:
                # 域适应变换
                adapter.fit(X_source, X_target)
                X_source_train = adapter.transform(X_source)
                X_target_test = adapter.transform(X_target)

            # 训练和评估基线模型
            baseline_results = self._evaluate_all_baselines(
                X_source_train, y_source, X_target_test, y_target
            )

            # 训练和评估PANDA模型
            panda_results = self._evaluate_panda(
                X_source_train, y_source, X_target_test, y_target
            )

            results[method_name] = {
                'baseline_models': baseline_results,
                'panda_model': panda_results,
                'best_performance': max(baseline_results + [panda_results])
            }

        # 方法排序和分析
        method_ranking = self._rank_adaptation_methods(results)

        return {
            'method_comparison': results,
            'method_ranking': method_ranking,
            'best_baseline_method': method_ranking['baseline_best'],
            'best_panda_method': method_ranking['panda_best'],
            'improvement_analysis': self._analyze_improvements(results)
        }
```

---

## 🔧 实现细节

### 综合评估框架设计
```python
@dataclass
class EvaluationConfig:
    """评估配置"""
    medical_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'min_sensitivity': 0.90,  # 筛查要求
        'min_specificity': 0.80,
        'max_brier_score': 0.15,
        'max_ece': 0.05
    })

    domain_adaptation_targets: Dict[str, float] = field(default_factory=lambda: {
        'min_performance_retention': 0.85,
        'min_adaptation_gain': 0.03,
        'min_mmd_reduction': 0.20
    })

    baseline_models: Dict[str, Any] = field(default_factory=lambda: {
        'logistic_regression': LogisticRegression(penalty='l1', solver='saga', class_weight='balanced'),
        'xgboost': XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1),
        'random_forest': RandomForestClassifier(n_estimators=200, class_weight='balanced'),
        'svm': SVC(kernel='rbf', C=1.0, probability=True, class_weight='balanced'),
        'knn': KNeighborsClassifier(n_neighbors=5, weights='distance')
    })

class HeartDiseaseEvaluator:
    """心脏病综合评估主类"""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.medical_metrics = MedicalMetrics()
        self.domain_metrics = DomainAdaptationMetrics()
        self.baseline_models = BaselineModels()
        self.domain_comparison = DomainAdaptationComparison()

    def comprehensive_model_comparison(self, X_train, y_train, X_test, y_test):
        """综合模型对比评估"""
        comparison_results = {}

        # 1. 基线模型评估
        baseline_results = self.baseline_models.compare_with_panda(
            X_train, y_train, X_test, y_test, panda_model=None
        )
        comparison_results['baseline_comparison'] = baseline_results

        # 2. 医学指标评估
        for model_name, metrics in baseline_results['performance_comparison'].items():
            medical_eval = self.medical_metrics.compute_all_medical_metrics(y_test, metrics)
            comparison_results['medical_evaluation'] = medical_eval

        # 3. 统计显著性分析
        significance_results = self._comprehensive_statistical_analysis(baseline_results)
        comparison_results['statistical_analysis'] = significance_results

        return comparison_results

    def evaluate_domain_adaptation_methods(self, X_source, y_source, X_target, y_target):
        """域适应方法综合评估"""
        # 1. 域适应方法对比
        domain_results = self.domain_comparison.run_comprehensive_comparison(
            X_source, y_source, X_target, y_target
        )

        # 2. 性能提升分析
        improvement_analysis = self._analyze_performance_improvements(domain_results)

        # 3. 最优方法推荐
        best_methods = self._recommend_best_methods(domain_results)

        return {
            'domain_comparison': domain_results,
            'improvement_analysis': improvement_analysis,
            'recommendations': best_methods
        }
```

### 批量评估接口
```python
def evaluate_all_experiments(self, experiment_results):
    """批量评估所有实验"""
    evaluation_summary = []

    for exp in experiment_results:
        if exp['type'] == 'single_center':
            eval_result = self.evaluate_single_center(
                exp['model'], exp['X'], exp['y'], exp['center']
            )
        elif exp['type'] == 'domain_adaptation':
            eval_result = self.evaluate_domain_adaptation(
                exp['source_model'], exp['adapted_model'],
                exp['X_source'], exp['X_target'], exp['y_target'],
                exp['source_center'], exp['target_center']
            )

        evaluation_summary.append(eval_result)

    # 生成汇总报告
    summary_report = self._generate_summary_report(evaluation_summary)

    return {
        'detailed_results': evaluation_summary,
        'summary_report': summary_report
    }
```

---

## 🧪 测试计划

### 单元测试
- [ ] **医学指标计算**: 验证计算准确性
- [ ] **校准指标**: 验证校准评估
- [ ] **跨域指标**: 验证适应效果度量
- [ ] **解释性方法**: 验证特征重要性

### 集成测试
- [ ] **完整评估流程**: 从模型到报告
- [ ] **多中心对比**: 批量评估一致性
- [ ] **报告生成**: 结果格式正确性

### 临床验证
- [ ] **医学专家评审**: 评估指标合理性
- [ ] **阈值验证**: 临床可接受性
- [ ] **解释性验证**: 医学解释正确性

---

## 📊 预期输出

### 评估结果文件
- `evaluation/single_center_results.json` - 单中心评估结果
- `evaluation/domain_adaptation_results.json` - 跨域适应结果
- `evaluation/clinical_interpretability.json` - 可解释性分析
- `evaluation/evaluation_summary.json` - 汇总报告

### 可视化报告
- `evaluation/metrics_comparison.png` - 指标对比图
- `evaluation/calibration_plots.png` - 校准曲线图
- `evaluation/feature_importance.png` - 特征重要性图
- `evaluation/shap_explanations.png` - SHAP解释图

---

## 🚨 风险与缓解

### 风险识别
1. **医学指标计算错误** (临床意义)
2. **校准评估不准确** (概率可靠性)
3. **特征重要性偏差** (可解释性)

### 缓束策略
1. **参考标准库 + 医学验证**
2. **多种校准方法 + 交叉验证**
3. **多方法对比 + 专家验证**

---

## 📞 联系信息
**负责人**: [待分配]
**医学顾问**: [临床统计学家]
**技术支持**: [评估工程师]

*最后更新: 2025-11-18*