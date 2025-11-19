# 配置文件 TODO

## 模块概述
**文件**: `config/heart_disease_config.py`
**功能**: PANDA-Heart项目全局配置和参数管理
**负责人**: [待分配]
**预计工时**: 12小时

---

## 📋 详细任务清单

### TASK-019: 数据配置管理
**优先级**: 🔥 High | **预计工时**: 4小时 | **截止**: Week 3

#### 子任务
- [ ] **TASK-019-1**: UCI数据配置
  - **数据源配置**: 4个中心的数据下载和处理配置
  - **特征映射**: 14个临床特征的标准化映射
  - **缺失值处理**: 各中心缺失值处理策略
  - **数据验证**: 数据质量和一致性检查配置

- [ ] **TASK-019-2**: 路径和文件配置
  - **数据路径**: 原始数据、处理数据的存储路径
  - **结果路径**: 实验结果、可视化结果的输出路径
  - **模型路径**: 训练好的模型存储和加载路径
  - **日志路径**: 日志文件的存储和轮转配置

- [ ] **TASK-019-3**: 环境配置
  - **计算资源**: CPU/GPU/MPS资源配置
  - **并行计算**: 多进程/多线程配置
  - **内存管理**: 大数据集的内存优化配置
  - **随机种子**: 实验可重现性配置

#### 验收标准
- [ ] 数据配置覆盖所有4个中心
- [ ] 路径配置跨平台兼容
- [ ] 环境配置适应不同硬件

#### 技术要求
```python
# 伪代码示例
@dataclass
class DataConfig:
    """UCI心脏病数据配置"""
    # 数据源配置
    cleveland_url: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    hungarian_url: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data"
    va_url: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data"
    switzerland_url: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data"

    # 本地存储路径
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    interim_data_dir: str = "data/interim"

    # 特征配置
    feature_names: List[str] = field(default_factory=lambda: [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'
    ])

    # 缺失值配置
    missing_values: List[str] = field(default_factory=lambda: ['?', '-9', '-9.0'])
    missing_threshold: float = 0.3  # 缺失率阈值

    # 数据质量配置
    age_range: Tuple[int, int] = (0, 120)
    bp_range: Tuple[int, int] = (50, 250)
    chol_range: Tuple[int, int] = (100, 600)

@dataclass
class PathConfig:
    """路径配置"""
    # 项目根目录
    project_root: str = "."

    # 数据路径
    data_dir: str = "data"
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"

    # 模型路径
    models_dir: str = "models"
    checkpoints_dir: str = "models/checkpoints"

    # 结果路径
    results_dir: str = "results"
    figures_dir: str = "results/figures"
    tables_dir: str = "results/tables"

    # 日志路径
    logs_dir: str = "logs"
    log_file: str = "logs/panda_heart.log"

@dataclass
class EnvironmentConfig:
    """环境配置"""
    # 计算设备
    device: str = "auto"  # auto, cpu, mps, cuda
    num_workers: int = 4
    batch_size: int = 32

    # 内存配置
    max_memory_usage: float = 0.8  # 最大内存使用率
    chunk_size: int = 1000  # 大数据集分块大小

    # 随机种子
    random_seed: int = 42
    numpy_seed: int = 42
    torch_seed: int = 42

    # 并行配置
    use_multiprocessing: bool = True
    n_jobs: int = -1  # -1表示使用所有可用CPU核心
```

---

### TASK-020: 模型配置管理
**优先级**: 🔥 High | **预计工时**: 4小时 | **截止**: Week 4

#### 子任务
- [ ] **TASK-020-1**: TabPFN配置
  - **模型参数**: TabPFN预训练模型配置
  - **集成配置**: 32成员集成的多样性配置
  - **推理配置**: TabPFN推理的超参数配置
  - **认证配置**: HuggingFace认证配置

- [ ] **TASK-020-2**: 域适应配置
  - **TCA配置**: 核函数、mu值、维度选择配置
  - **CORAL配置**: 正则化参数、白化变换配置
  - **SA配置**: PCA维度、子空间对齐配置
  - **参数网格**: 各方法的超参数搜索网格

- [ ] **TASK-020-3**: 基线模型配置
  - **传统ML模型**: LR、XGBoost、RF、SVM、KNN配置
  - **医学约束**: 分类阈值、类别权重配置
  - **交叉验证**: CV折数、分层采样配置
  - **评估指标**: 医学专用评估指标配置

#### 验收标准
- [ ] TabPFN配置完整且可用
- [ ] 域适应方法参数覆盖全面
- [ ] 基线模型配置医学合理

#### 技术要求
```python
# 伪代码示例
@dataclass
class TabPFNConfig:
    """TabPFN模型配置"""
    # 基础配置
    model_name: str = "priorlabs/tabpfn_2_5"
    ensemble_size: int = 32
    device: str = "auto"

    # 模型参数
    n_layers: int = 10
    d_model: int = 128
    n_heads: int = 4
    max_epochs: int = 1000

    # 推理配置
    batch_size: int = 32
    max_inference_samples: int = 1024

    # 集成多样性
    feature_subsets: bool = True
    input_transformations: bool = True
    seed_variation: bool = True
    model_variants: int = 8

    # HuggingFace配置
    use_auth_token: bool = True
    cache_dir: str = "models/huggingface_cache"

@dataclass
class DomainAdaptationConfig:
    """域适应配置"""
    # TCA配置
    tca_kernel_types: List[str] = field(default_factory=lambda: ['linear', 'rbf'])
    tca_mu_values: List[float] = field(default_factory=lambda: [0.01, 0.1, 1.0, 10.0])
    tca_n_components: List[float] = field(default_factory=lambda: [0.8, 0.9, 0.95])

    # CORAL配置
    coral_reg_params: List[float] = field(default_factory=lambda: [1e-4, 1e-3, 1e-2, 1e-1])
    coral_adaptive_weights: bool = True

    # SA配置
    sa_n_components: List[float] = field(default_factory=lambda: [0.8, 0.9, 0.95, 0.99])
    sa_alignment_methods: List[str] = field(default_factory=lambda: ['procrustes', 'least_squares'])

    # 通用配置
    cv_folds: int = 5
    scoring_metric: str = "roc_auc"

@dataclass
class BaselineModelsConfig:
    """基线模型配置"""
    # 逻辑回归
    lr_penalty: str = "l1"
    lr_c_values: List[float] = field(default_factory=lambda: [0.01, 0.1, 1.0, 10.0])
    lr_solver: str = "saga"
    lr_class_weight: str = "balanced"

    # XGBoost
    xgb_n_estimators: List[int] = field(default_factory=lambda: [50, 100, 200])
    xgb_max_depth: List[int] = field(default_factory=lambda: [3, 6, 9])
    xgb_learning_rate: List[float] = field(default_factory=lambda: [0.01, 0.1, 0.2])
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8

    # 随机森林
    rf_n_estimators: List[int] = field(default_factory=lambda: [100, 200, 500])
    rf_max_depth: List[Union[int, None]] = field(default_factory=lambda: [None, 10, 20])
    rf_class_weight: str = "balanced"

    # SVM
    svm_c_values: List[float] = field(default_factory=lambda: [0.1, 1.0, 10.0])
    svm_kernel: str = "rbf"
    svm_probability: bool = True
    svm_class_weight: str = "balanced"

    # KNN
    knn_n_neighbors: List[int] = field(default_factory=lambda: [3, 5, 7])
    knn_weights: str = "distance"
    knn_metric: str = "euclidean"
```

---

### TASK-021: 实验配置管理
**优先级**: 🔥 Medium | **预计工时**: 4小时 | **截止**: Week 4

#### 子任务
- [ ] **TASK-021-1**: 实验设计配置
  - **实验类型**: 单中心、两中心、多中心实验配置
  - **对比矩阵**: 7模型×6UDA方法的实验组合配置
  - **重复实验**: 实验重复次数和随机种子配置
  - **结果记录**: 实验结果记录和存储配置

- [ ] **TASK-021-2**: 评估配置
  - **医学指标**: AUC、敏感性、特异性、NPV阈值配置
  - **统计检验**: 显著性检验方法和阈值配置
  - **交叉验证**: LOCO-CV和分层CV配置
  - **性能基准**: 各方法性能基准和目标配置

- [ ] **TASK-021-3**: 可视化配置
  - **图表风格**: 颜色、字体、尺寸配置
  - **医学标注**: 临床阈值、重要性标注配置
  - **输出格式**: 图片格式、分辨率、透明度配置
  - **交互式**: 交互式可视化功能配置

#### 验收标准
- [ ] 实验配置覆盖126个组合
- [ ] 评估配置符合医学标准
- [ ] 可视化配置符合发表要求

#### 技术要求
```python
# 伪代码示例
@dataclass
class ExperimentConfig:
    """实验配置"""
    # 实验类型
    experiment_types: List[str] = field(default_factory=lambda: [
        'single_center', 'two_center', 'multi_center'
    ])

    # 模型配置
    models: List[str] = field(default_factory=lambda: [
        'PANDA_TabPFN', 'TabPFN_Only', 'LASSO_LR', 'XGBoost',
        'Random_Forest', 'SVM', 'KNN'
    ])

    # 域适应方法
    uda_methods: List[str] = field(default_factory=lambda: [
        'TCA', 'CORAL', 'SA', 'PCA', 'CCA', 'No_UDA'
    ])

    # 实验重复
    n_repetitions: int = 5
    random_seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 999])

    # 数据中心配置
    centers: List[str] = field(default_factory=lambda: [
        'Cleveland', 'Hungarian', 'VA', 'Switzerland'
    ])

@dataclass
class EvaluationConfig:
    """评估配置"""
    # 主要评估指标
    primary_metrics: List[str] = field(default_factory=lambda: [
        'auc_roc', 'auc_pr', 'sensitivity', 'specificity', 'npv', 'ppv'
    ])

    # 医学阈值
    medical_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'min_sensitivity': 0.90,  # 筛查要求
        'min_specificity': 0.80,
        'max_brier_score': 0.15,
        'max_ece': 0.05
    })

    # 域适应指标
    domain_metrics: Dict[str, float] = field(default_factory=lambda: {
        'min_performance_retention': 0.85,
        'min_adaptation_gain': 0.03,
        'min_mmd_reduction': 0.20
    })

    # 统计检验
    significance_level: float = 0.05
    multiple_correction: str = "bonferroni"
    effect_size_threshold: float = 0.5  # Cohen's d

@dataclass
class VisualizationConfig:
    """可视化配置"""
    # 基础配置
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 300
    style: str = "seaborn-v0_8"
    color_palette: str = "Set2"
    font_size: int = 12

    # 医学颜色
    clinical_colors: Dict[str, str] = field(default_factory=lambda: {
        'disease': '#FF6B6B',
        'no_disease': '#4ECDC4',
        'high_risk': '#FF4757',
        'medium_risk': '#FFA502',
        'low_risk': '#26DE81'
    })

    # 输出配置
    save_format: str = "png"
    transparent: bool = False
    bbox_inches: str = "tight"

    # 交互式配置
    interactive_plots: bool = True
    plotly_config: Dict[str, Any] = field(default_factory=lambda: {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d']
    })
```

---

## 🔧 实现细节

### 配置管理主类
```python
class HeartDiseaseConfigManager:
    """PANDA-Heart配置管理器"""

    def __init__(self, config_file: str = "config/panda_heart_config.yaml"):
        self.config_file = config_file
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            return self._create_default_config()

    def _create_default_config(self) -> Dict[str, Any]:
        """创建默认配置"""
        return {
            'data': asdict(DataConfig()),
            'paths': asdict(PathConfig()),
            'environment': asdict(EnvironmentConfig()),
            'tabpfn': asdict(TabPFNConfig()),
            'domain_adaptation': asdict(DomainAdaptationConfig()),
            'baseline_models': asdict(BaselineModelsConfig()),
            'experiment': asdict(ExperimentConfig()),
            'evaluation': asdict(EvaluationConfig()),
            'visualization': asdict(VisualizationConfig())
        }

    def save_config(self):
        """保存配置文件"""
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        with open(self.config_file, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)

    def get_config(self, section: str) -> Any:
        """获取配置部分"""
        return self.config.get(section, {})

    def update_config(self, section: str, updates: Dict[str, Any]):
        """更新配置部分"""
        if section not in self.config:
            self.config[section] = {}
        self.config[section].update(updates)
        self.save_config()

    def validate_config(self) -> bool:
        """验证配置有效性"""
        try:
            # 验证路径配置
            paths_config = self.get_config('paths')
            for path_key, path_value in paths_config.items():
                if path_key.endswith('_dir'):
                    os.makedirs(path_value, exist_ok=True)

            # 验证数据配置
            data_config = self.get_config('data')
            assert len(data_config['feature_names']) == 14, "应该有14个特征"

            # 验证模型配置
            models_config = self.get_config('tabpfn')
            assert models_config['ensemble_size'] > 0, "集成大小必须大于0"

            return True
        except Exception as e:
            print(f"配置验证失败: {e}")
            return False

# 全局配置实例
config_manager = HeartDiseaseConfigManager()
```

---

## 🧪 测试计划

### 单元测试
- [ ] **配置加载**: 验证YAML配置文件加载
- [ ] **配置验证**: 验证配置参数有效性
- [ ] **配置更新**: 验证动态配置更新
- [ ] **默认配置**: 验证默认配置生成

### 集成测试
- [ ] **跨平台兼容**: 不同操作系统的配置兼容性
- [ ] **环境适应**: 不同硬件环境的配置适应
- [ ] **配置继承**: 配置参数的继承和覆盖

### 实际使用测试
- [ ] **完整实验**: 使用默认配置运行完整实验
- [ ] **配置修改**: 修改配置后实验的稳定性
- [ ] **错误处理**: 配置错误时的错误处理

---

## 📊 预期输出

### 配置文件
- `config/panda_heart_config.yaml` - 主配置文件
- `config/experiment_templates/` - 实验模板配置
- `config/environments/` - 不同环境配置
- `config/hyperparameter_grids/` - 超参数网格配置

### 配置文档
- `config/README.md` - 配置使用说明
- `config/configuration_guide.md` - 详细配置指南
- `config/troubleshooting.md` - 配置问题排查

---

## 🚨 风险与缓解

### 风险识别
1. **配置错误** (参数值错误、类型不匹配)
2. **路径问题** (跨平台路径兼容性)
3. **环境差异** (不同环境的配置适应)

### 缓解策略
1. **参数验证 + 默认值设置**
2. **跨平台库 + 路径标准化**
3. **环境检测 + 自适应配置**

---

## 📞 联系信息
**负责人**: [待分配]
**技术支持**: [配置管理工程师]
**用户支持**: [系统管理员]

*最后更新: 2025-11-18*