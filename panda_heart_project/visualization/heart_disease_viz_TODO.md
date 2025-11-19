# 可视化模块 TODO

## 模块概述
**文件**: `visualization/heart_disease_viz.py`
**功能**: 心脏病跨域诊断结果可视化
**负责人**: [待分配]
**预计工时**: 18小时

---

## 📋 详细任务清单

### TASK-016: 模型性能可视化
**优先级**: 🔥 High | **预计工时**: 8小时 | **截止**: Week 6

#### 子任务
- [ ] **TASK-016-1**: ROC和PR曲线
  - **ROC曲线**: 多模型ROC对比，AUC值标注
  - **PR曲线**: 精确率-召回率曲线，AP值计算
  - **置信区间**: 交叉验证ROC/PR曲线置信区间
  - **跨域对比**: 源域vs目标域ROC对比

- [ ] **TASK-016-2**: 混淆矩阵热图
  - **多中心混淆矩阵**: 每个中心的分类性能
  - **敏感性特异性**: 临床关键指标突出显示
  - **错误分析**: 误诊和漏诊案例分布
  - **阈值优化**: 不同决策阈值的性能变化

- [ ] **TASK-016-3**: 校准曲线
  - **概率校准**: 预测概率与实际概率对比
  - **Brier得分**: 概率预测准确度可视化
  - **ECE曲线**: 期望校准误差分析
  - **可靠性图**: 分桶概率校准评估

#### 验收标准
- [ ] ROC/PR曲线清晰准确，AUC值标注完整
- [ ] 混淆矩阵医学指标突出显示
- [ ] 校准曲线临床可解释性强

#### 技术要求
```python
# 伪代码示例
class HeartDiseasePerformanceViz:
    """心脏病性能可视化"""

    def __init__(self, figsize=(12, 8), style='seaborn', dpi=300):
        self.figsize = figsize
        self.style = style
        self.dpi = dpi
        self.clinical_colors = {
            'positive': '#FF6B6B',  # 红色 - 阳性
            'negative': '#4ECDC4',  # 青色 - 阴性
            'threshold': '#FFD93D'  # 黄色 - 阈值
        }

    def plot_roc_curves(self, y_true_dict, y_prob_dict, models=None, centers=None):
        """绘制ROC曲线对比"""
        plt.figure(figsize=(10, 8))

        for name, (y_true, y_prob) in y_prob_dict.items():
            # 计算ROC曲线
            fpr, tpr, auc = self._compute_roc_curve(y_true_dict[name], y_prob)

            # 绘制曲线
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)

        # 添加对角线和标签
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
        plt.title('ROC Curves: Heart Disease Detection Across Centers', fontsize=14, fontweight='bold')

        # 突出显示临床重要区域
        self._add_clinical_regions()

        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        return plt.gcf()

    def plot_confusion_matrices(self, y_true_dict, y_pred_dict, centers=None):
        """绘制多中心混淆矩阵"""
        n_centers = len(y_true_dict)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()

        for idx, (center, (y_true, y_pred)) in enumerate(zip(centers, y_pred_dict.items())):
            cm = confusion_matrix(y_true, y_pred[1])

            # 计算医学指标
            sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])
            specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
            ppv = cm[1, 1] / (cm[1, 1] + cm[0, 1])
            npv = cm[0, 0] / (cm[0, 0] + cm[1, 0])

            # 绘制热图
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['No Disease', 'Disease'],
                       yticklabels=['No Disease', 'Disease'])

            axes[idx].set_title(f'{center} Center\n'
                              f'Sensitivity: {sensitivity:.2f}, '
                              f'Specificity: {specificity:.2f}\n'
                              f'PPV: {ppv:.2f}, NPV: {npv:.2f}',
                              fontweight='bold')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')

        plt.suptitle('Cross-Center Heart Disease Classification Performance',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        return fig

    def plot_calibration_curves(self, y_true, y_prob, model_names, n_bins=10):
        """绘制概率校准曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()

        for idx, (name, prob) in enumerate(zip(model_names, y_prob)):
            # 计算校准曲线
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, prob, n_bins=n_bins
            )

            # 计算校准指标
            brier = brier_score_loss(y_true, prob)
            ece = self._compute_expected_calibration_error(y_true, prob, n_bins)

            # 绘制校准曲线
            axes[idx].plot(mean_predicted_value, fraction_of_positives, "s-",
                          label=f'{name}', linewidth=2, markersize=6)
            axes[idx].plot([0, 1], [0, 1], "k:", label="Perfect calibration")

            # 添加校准指标
            axes[idx].text(0.05, 0.95, f'Brier: {brier:.3f}\nECE: {ece:.3f}',
                          transform=axes[idx].transAxes, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            axes[idx].set_xlabel('Mean Predicted Probability')
            axes[idx].set_ylabel('Fraction of Positives')
            axes[idx].set_title(f'{name} Calibration')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        plt.suptitle('Probability Calibration for Heart Disease Prediction',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        return fig
```

---

### TASK-017: 跨域分析可视化
**优先级**: 🔥 High | **预计工时**: 6小时 | **截止**: Week 6

#### 子任务
- [ ] **TASK-017-1**: 分布对齐可视化
  - **特征分布**: 源域vs目标域特征分布对比
  - **TCA降维**: TCA变换前后的2D/3D可视化
  - **MMD距离**: 域间距离热图和趋势分析
  - **协方差对齐**: CORAL协方差矩阵对比图

- [ ] **TASK-017-2**: 性能保持分析
  - **性能下降图**: 源域→目标域性能变化
  - **适应增益**: 域适应带来的性能提升
  - **最优方法**: 不同场景下的最优域适应方法
  - **失败案例**: 域适应失败的案例分析

- [ ] **TASK-017-3**: 多中心对比
  - **LOCO-CV结果**: Leave-One-Center-Out结果对比
  - **中心特征**: 各医院数据特征雷达图
  - **迁移学习**: 跨医院知识迁移效果
  - **集成策略**: 多中心集成策略分析

#### 验收标准
- [ ] 域适应效果可视化清晰直观
- [ ] 跨中心对比分析全面
- [ ] 临床解释性可视化充分

#### 技术要求
```python
# 伪代码示例
class CrossDomainViz:
    """跨域分析可视化"""

    def plot_domain_adaptation_effects(self, X_source, X_target,
                                     X_source_ada, X_target_ada,
                                     method='TCA'):
        """绘制域适应效果"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. 原始分布
        self._plot_feature_distributions(axes[0, 0], X_source, X_target, 'Original')

        # 2. 域适应后分布
        self._plot_feature_distributions(axes[0, 1], X_source_ada, X_target_ada, f'After {method}')

        # 3. MMD距离对比
        original_mmd = self._compute_mmd_distance(X_source, X_target)
        adapted_mmd = self._compute_mmd_distance(X_source_ada, X_target_ada)
        self._plot_mmd_comparison(axes[0, 2], original_mmd, adapted_mmd, method)

        # 4. 2D降维可视化
        self._plot_2d_visualization(axes[1, 0], X_source, X_target, 'Original Space')
        self._plot_2d_visualization(axes[1, 1], X_source_ada, X_target_ada, f'{method} Space')

        # 5. 协方差矩阵对比
        self._plot_covariance_matrices(axes[1, 2], X_source, X_target,
                                      X_source_ada, X_target_ada, method)

        plt.suptitle(f'Domain Adaptation Effects: {method}', fontsize=16, fontweight='bold')
        plt.tight_layout()

        return fig

    def plot_loco_cv_results(self, results_dict, metrics=['auc_roc', 'sensitivity', 'specificity']):
        """绘制LOCO-CV结果"""
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 6))

        if n_metrics == 1:
            axes = [axes]

        for idx, metric in enumerate(metrics):
            # 提取数据
            models = []
            source_scores = []
            target_scores = []

            for model, model_results in results_dict.items():
                models.append(model)
                source_scores.append(np.mean([r['source'][metric] for r in model_results]))
                target_scores.append(np.mean([r['target'][metric] for r in model_results]))

            # 绘制柱状图
            x = np.arange(len(models))
            width = 0.35

            bars1 = axes[idx].bar(x - width/2, source_scores, width,
                                 label='Source Domain', alpha=0.8, color='skyblue')
            bars2 = axes[idx].bar(x + width/2, target_scores, width,
                                 label='Target Domain', alpha=0.8, color='lightcoral')

            # 添加数值标签
            self._add_bar_labels(axes[idx], bars1, target_scores)
            self._add_bar_labels(axes[idx], bars2, target_scores)

            axes[idx].set_xlabel('Models')
            axes[idx].set_ylabel(metric.upper())
            axes[idx].set_title(f'LOCO-CV: {metric.upper()} Comparison')
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(models, rotation=45)
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        plt.suptitle('Leave-One-Center-Out Cross Validation Results',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        return fig
```

---

### TASK-018: 医学解释可视化
**优先级**: 🔥 Medium | **预计工时**: 4小时 | **截止**: Week 7

#### 子任务
- [ ] **TASK-018-1**: 特征重要性可视化
  - **SHAP值**: SHAP特征重要性图和依赖图
  - **排列重要性**: 特征排列重要性对比
  - **医学特征**: 临床特征重要性解释
  - **跨中心稳定性**: 特征重要性跨中心稳定性

- [ ] **TASK-018-2**: 决策曲线分析
  - **临床净获益**: 不同阈值的净获益曲线
  - **模型对比**: 多模型决策曲线对比
  - **临床实用性**: 临床决策阈值分析
  - **成本效益**: 误诊和漏诊的成本分析

- [ ] **TASK-018-3**: 风险分层可视化
  - **风险分层**: 患者风险分层和分布
  - **预后分析**: 不同风险层预后对比
  - **临床路径**: 基于风险的临床路径建议

#### 验收标准
- [ ] 特征重要性医学解释清晰
- [ ] 决策曲线临床实用性明确
- [ ] 风险分层临床可操作性强

#### 技术要求
```python
# 伪代码示例
class MedicalInterpretabilityViz:
    """医学解释性可视化"""

    def plot_shap_analysis(self, X, y, model, feature_names, patient_idx=None):
        """绘制SHAP分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 计算SHAP值
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # 1. 特征重要性摘要图
        shap.summary_plot(shap_values[1], X, feature_names=feature_names,
                         plot_type="bar", ax=axes[0], show=False)
        axes[0].set_title('Feature Importance (SHAP Values)', fontweight='bold')

        # 2. SHAP摘要散点图
        shap.summary_plot(shap_values[1], X, feature_names=feature_names,
                         ax=axes[1], show=False)
        axes[1].set_title('SHAP Value Distribution', fontweight='bold')

        # 3. 单个患者解释
        if patient_idx is not None:
            shap.force_plot(explainer.expected_value[1], shap_values[1][patient_idx],
                          X.iloc[patient_idx], feature_names=feature_names,
                          matplotlib=True, ax=axes[2])
            axes[2].set_title(f'Patient {patient_idx} Prediction Explanation', fontweight='bold')

        # 4. 特征依赖图
        feature_idx = np.argsort(np.abs(shap_values[1]).mean(0))[-1]
        shap.dependence_plot(feature_idx, shap_values[1], X, feature_names=feature_names,
                            ax=axes[3], show=False)
        axes[3].set_title(f'{feature_names[feature_idx]} Dependence Plot', fontweight='bold')

        plt.suptitle('SHAP Analysis for Heart Disease Prediction',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        return fig

    def plot_decision_curve_analysis(self, y_true, y_probs, model_names,
                                   treat_all_thresholds=None):
        """绘制决策曲线分析"""
        plt.figure(figsize=(12, 8))

        # 定义阈值范围
        thresholds = np.arange(0, 1, 0.01)

        # 计算每个模型的净获益
        for name, y_prob in zip(model_names, y_probs):
            net_benefit = self._compute_net_benefit(y_true, y_prob, thresholds)
            plt.plot(thresholds, net_benefit, label=f'{name}', linewidth=2.5)

        # 添加基准线
        if treat_all_thresholds:
            net_benefit_treat_all = self._compute_treat_all_benefit(
                y_true, thresholds, treat_all_thresholds
            )
            plt.plot(thresholds, net_benefit_treat_all, 'k--',
                    label='Treat All', linewidth=2, alpha=0.7)

        # 添加无获益线
        plt.plot(thresholds, np.zeros_like(thresholds), 'k-',
                label='No Benefit', linewidth=1, alpha=0.5)

        plt.xlabel('Risk Threshold', fontsize=12)
        plt.ylabel('Net Benefit', fontsize=12)
        plt.title('Decision Curve Analysis: Heart Disease Screening',
                 fontsize=14, fontweight='bold')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)

        # 添加临床注释
        self._add_clinical_annotations()

        plt.tight_layout()
        return plt.gcf()

    def plot_risk_stratification(self, y_true, y_prob, risk_groups=['Low', 'Medium', 'High']):
        """绘制风险分层分析"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 计算风险分组
        risk_percentiles = [33, 67]
        risk_labels = np.digitize(y_prob,
                                 np.percentile(y_prob, risk_percentiles))

        # 1. 风险分布直方图
        axes[0, 0].hist(y_prob[y_true == 0], bins=30, alpha=0.7,
                       label='No Disease', color='blue', density=True)
        axes[0, 0].hist(y_prob[y_true == 1], bins=30, alpha=0.7,
                       label='Disease', color='red', density=True)
        axes[0, 0].axvline(np.percentile(y_prob, 33), color='orange',
                          linestyle='--', label='Low/Medium Threshold')
        axes[0, 0].axvline(np.percentile(y_prob, 67), color='purple',
                          linestyle='--', label='Medium/High Threshold')
        axes[0, 0].set_xlabel('Predicted Risk')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Risk Distribution')
        axes[0, 0].legend()

        # 2. 风险组混淆矩阵
        for i, (ax, risk_label) in enumerate(zip(axes[0, 1:], ['Low', 'Medium', 'High'])):
            mask = risk_labels == i
            if mask.sum() > 0:
                cm = confusion_matrix(y_true[mask], (y_prob[mask] > 0.5).astype(int))
                sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
                ax.set_title(f'{risk_label} Risk Group (n={mask.sum()})')

        # 3. 风险组性能指标
        metrics = ['sensitivity', 'specificity', 'ppv', 'npv']
        risk_metrics = self._compute_risk_group_metrics(y_true, y_prob, risk_labels)

        x = np.arange(len(metrics))
        width = 0.25

        for i, risk_label in enumerate(['Low', 'Medium', 'High']):
            values = [risk_metrics[risk_label][metric] for metric in metrics]
            axes[1, 1].bar(x + i*width, values, width, label=risk_label)

        axes[1, 1].set_xlabel('Metrics')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_title('Performance by Risk Group')
        axes[1, 1].set_xticks(x + width)
        axes[1, 1].set_xticklabels(metrics)
        axes[1, 1].legend()

        plt.suptitle('Risk Stratification Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()

        return fig
```

---

## 🔧 实现细节

### 可视化配置
```python
@dataclass
class VisualizationConfig:
    """可视化配置"""
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 300
    style: str = 'seaborn-v0_8'
    color_palette: str = 'Set2'
    font_size: int = 12
    save_format: str = 'png'
    transparent: bool = False

    # 医学专用配置
    clinical_colors: Dict[str, str] = field(default_factory=lambda: {
        'disease': '#FF6B6B',
        'no_disease': '#4ECDC4',
        'uncertain': '#FFD93D',
        'high_risk': '#FF4757',
        'medium_risk': '#FFA502',
        'low_risk': '#26DE81'
    })

    threshold_values: Dict[str, float] = field(default_factory=lambda: {
        'sensitivity_target': 0.90,
        'specificity_target': 0.80,
        'risk_low_medium': 0.33,
        'risk_medium_high': 0.67
    })

class HeartDiseaseVisualizer:
    """心脏病可视化主类"""

    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.performance_viz = HeartDiseasePerformanceViz()
        self.domain_viz = CrossDomainViz()
        self.interpret_viz = MedicalInterpretabilityViz()

        # 设置绘图样式
        plt.style.use(self.config.style)
        sns.set_palette(self.config.color_palette)

    def generate_comprehensive_report(self, results, save_dir='results/visualization'):
        """生成综合可视化报告"""
        os.makedirs(save_dir, exist_ok=True)

        # 1. 模型性能可视化
        roc_fig = self.performance_viz.plot_roc_curves(
            results['roc_data']['y_true_dict'],
            results['roc_data']['y_prob_dict'],
            results['model_names'],
            results['centers']
        )
        roc_fig.savefig(f'{save_dir}/roc_curves.{self.config.save_format}',
                       dpi=self.config.dpi, bbox_inches='tight')

        # 2. 跨域分析可视化
        domain_fig = self.domain_viz.plot_loco_cv_results(
            results['loco_cv_results']
        )
        domain_fig.savefig(f'{save_dir}/loco_cv_results.{self.config.save_format}',
                          dpi=self.config.dpi, bbox_inches='tight')

        # 3. 医学解释可视化
        interpret_fig = self.interpret_viz.plot_decision_curve_analysis(
            results['dca_data']['y_true'],
            results['dca_data']['y_probs'],
            results['model_names']
        )
        interpret_fig.savefig(f'{save_dir}/decision_curve_analysis.{self.config.save_format}',
                            dpi=self.config.dpi, bbox_inches='tight')

        print(f"Visualization report saved to {save_dir}")
```

---

## 🧪 测试计划

### 单元测试
- [ ] **绘图函数**: 验证各种绘图函数正确性
- [ ] **数据格式**: 验证输入数据格式兼容性
- [ ] **配置参数**: 验证配置参数有效性
- [ ] **保存功能**: 验证图片保存功能

### 集成测试
- [ ] **完整报告**: 验证综合报告生成功能
- [ ] **多数据源**: 验证不同数据源的可视化
- [ ] **批量处理**: 验证批量可视化生成

### 医学验证
- [ ] **医学专家评审**: 可视化医学解释性
- [ ] **临床可接受性**: 可视化临床实用性
- [ ] **标准符合**: 医学可视化标准符合性

---

## 📊 预期输出

### 可视化报告
- `visualization/roc_curves.png` - ROC曲线对比图
- `visualization/calibration_curves.png` - 校准曲线图
- `visualization/confusion_matrices.png` - 混淆矩阵热图
- `visualization/domain_adaptation_effects.png` - 域适应效果图
- `visualization/loco_cv_results.png` - LOCO-CV结果图
- `visualization/feature_importance.png` - 特征重要性图
- `visualization/decision_curve_analysis.png` - 决策曲线分析
- `visualization/risk_stratification.png` - 风险分层分析

### 交互式可视化
- `visualization/interactive_dashboard.html` - 交互式仪表板
- `visualization/shap_interactive.html` - 交互式SHAP分析

---

## 🚨 风险与缓解

### 风险识别
1. **可视化误导** (图表设计不当)
2. **医学误解** (临床指标解释错误)
3. **技术问题** (图片生成、保存失败)

### 缓解策略
1. **医学专家审查 + 最佳实践遵循**
2. **临床统计学家 + 医学专家双重验证**
3. **异常处理 + 备份方案**

---

## 📞 联系信息
**负责人**: [待分配]
**医学顾问**: [心脏病学专家]
**可视化专家**: [数据可视化工程师]

*最后更新: 2025-11-18*