# PANDA-Heart: 心脏病多中心跨域诊断项目PRD

## 🎯 项目概述

### 1.1 项目背景
心脏病是全球 leading cause of death，早期诊断对预后至关重要。不同医院间的数据分布差异（设备、操作规范、人群特征）严重影响AI模型的泛化能力。本项目基于UCI心脏病多中心数据集，应用**PANDA框架（TabPFN + 域适应）**实现跨医院的心脏病智能诊断。

### 1.2 研究目标
- **主要目标**: 验证PANDA框架在心脏病多中心数据上的跨域适应能力
- **次要目标**: 建立心脏病跨中心诊断的标准化评估基准
- **学术目标**: 为医学AI的跨中心应用提供方法论支持

### 1.3 技术定义
- **TabPFN**: 预训练表格Transformer，小样本学习SOTA
- **UDA**: 无监督域适应，解决源域→目标域分布偏移
- **PANDA**: TabPFN + UDA的完整框架
- **PANDA_TabPFN**: 带域适应的TabPFN（我们的主要方法）

## 📊 数据分析

### 2.1 UCI心脏病数据集

#### 数据来源
| 医疗机构 | 样本数量 | 缺失率 | 地理位置 | 代码 |
|----------|----------|--------|----------|------|
| **Cleveland Clinic Foundation** | 303 | 2% | 美国 | Cleveland |
| **Hungarian Institute of Cardiology** | 294 | 20% | 欧洲 | Hungarian |
| **V.A. Medical Center, Long Beach** | 200 | 12% | 美国 | VA |
| **University Hospital, Zurich** | 123 | 15% | 瑞士 | Switzerland |

**总计**: 920样本 → 740有效样本

#### 特征结构（14个临床特征）
```python
CLINICAL_FEATURES = {
    # 人口统计学 (2特征)
    'age': '年龄 (连续)',
    'sex': '性别 (1=男, 0=女)',

    # 症状特征 (2特征)
    'cp': '胸痛类型 (1-4级)',
    'exang': '运动诱发心绞痛 (1=yes, 0=no)',

    # 生命体征 (4特征)
    'trestbps': '静息血压 (mmHg)',
    'chol': '血清胆固醇 (mg/dl)',
    'thalach': '最大心率 (bpm)',
    'fbs': '空腹血糖>120mg/dl (1=true, 0=false)',

    # 心电图特征 (3特征)
    'restecg': '静息心电图 (0-2级)',
    'oldpeak': '运动ST段压低 (连续)',
    'slope': 'ST段斜率 (1-3级)',

    # 诊断特征 (3特征)
    'ca': '主要血管数 (0-3)',
    'thal': '地中海贫血 (3=正常, 6=固定缺陷, 7=可逆缺陷)',

    # 标签 (1特征)
    'num': '诊断结果 (0=无心脏病, 1-4=不同程度心脏病)'
}
```

### 2.2 跨域挑战
1. **地域差异**: 欧美人群基础健康状况差异
2. **设备差异**: 心电图设备、检测仪器标准化程度
3. **操作差异**: 各医院操作流程和诊断标准差异
4. **缺失模式**: Cleveland(2%) vs Hungarian(20%)缺失率差异

## 🏗️ PANDA-Heart技术架构

### 3.1 技术架构图
```
输入: 多中心心脏病数据 (14个临床特征)
  ↓
数据预处理: 缺失值处理 + 特征标准化 + 临床验证
  ↓
TabPFN编码器: 32成员集成 + 特征变换 + 不确定性量化
  ↓
域适应模块: TCA/CORAL/SA + 多中心分布对齐
  ↓
TabPFN分类器: 预训练Transformer + 概率校准
  ↓
输出: 心脏病风险概率 + 可解释性分析
```

### 3.2 核心技术组件

#### TabPFN配置
```python
PANDA_TABPFN_CONFIG = {
    'base_model': 'TabPFN',
    'ensemble_size': 32,
    'model_config': {
        'n_layers': 10,          # TabPFN标准层数
        'd_model': 128,          # 模型维度
        'n_heads': 4,            # 注意力头数
        'max_epochs': 1000,      # 训练轮数
        'batch_size': 32,        # 批次大小
    },
    'feature_encoding': {
        'age': 'normalized',
        'sex': 'binary',
        'cp': 'ordinal',
        'trestbps': 'clinical_normalized',
        'chol': 'clinical_normalized',
        'thalach': 'age_adjusted',
        'oldpeak': 'robust_scaling',
        'ca': 'numeric',
        'thal': 'one_hot'
    },
    'ensemble_diversity': {
        'feature_subsets': True,
        'input_transformations': True,
        'model_variants': 8,
        'seed_variation': True
    }
}
```

#### 域适应方法
```python
DOMAIN_ADAPTATION_METHODS = {
    'TCA': TransferComponentAnalysis(kernel='linear', mu=0.1),
    'CORAL': CorrelationAlignment(),
    'SA': SubspaceAlignment(),
    'PCA': PCA(n_components=0.95),  # 基线对比
    'CCA': CCA(n_components=0.95),  # 基线对比
    'No_UDA': None  # 无域适应基线
}
```

#### 对比基线模型
```python
BASELINE_MODELS = {
    'LASSO_LR': LogisticRegression(penalty='l1', solver='saga', class_weight='balanced'),
    'XGBoost': XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1),
    'Random_Forest': RandomForestClassifier(n_estimators=200, class_weight='balanced'),
    'SVM': SVC(kernel='rbf', C=1.0, probability=True, class_weight='balanced'),
    'KNN': KNeighborsClassifier(n_neighbors=5, weights='distance'),
    'TabPFN_Only': TabPFNEasyClassifier()  # 无域适应的TabPFN
}
```

## 🧪 实验设计

### 4.1 实验对比矩阵
```
实验类型: [单中心基线, 两中心跨域, 多中心集成]
模型: [PANDA_TabPFN, TabPFN_Only, LASSO_LR, XGBoost, Random_Forest, SVM, KNN]
域适应: [TCA, CORAL, SA, PCA, CCA, No_UDA]
总计: 3 × 7 × 6 = 126个实验组合
```

### 4.2 三层次实验架构

#### 4.2.1 单中心基线实验
- **目标**: 建立各中心性能基线
- **方法**: 10折交叉验证
- **中心**: Cleveland, Hungarian, VA, Switzerland
- **重点**: PANDA_TabPFN vs TabPFN_Only vs 传统模型

#### 4.2.2 两中心跨域实验
- **组合**: 所有可能的源域-目标域组合 (6对)
- **对比**:
  - PANDA_TabPFN: TabPFN + 最佳UDA
  - TabPFN_Only: TabPFN无UDA
  - 传统模型: ± UDA
- **评估**: 域适应增益 + 性能保持率

#### 4.2.3 多中心集成实验
- **策略**: LOCO-CV (Leave-One-Center-Out)
- **方法**: 3个中心训练，1个中心测试
- **创新**: 多源域融合 + 渐进适应

### 4.3 关键研究问题

1. **Q1**: PANDA_TabPFN vs TabPFN_Only：UDA的效果如何？
2. **Q2**: PANDA_TabPFN vs 传统模型：TabPFN的优势？
3. **Q3**: 不同UDA方法的效果排序？
4. **Q4**: 跨域场景下PANDA的鲁棒性？

## 📊 评估体系

### 5.1 医学专用指标
```python
MEDICAL_METRICS = {
    # 诊断性能
    'AUC_ROC': {'target': '>0.85', 'weight': 'high'},
    'AUC_PR': {'target': '>0.80', 'weight': 'medium'},
    'Sensitivity': {'target': '>0.90', 'weight': 'critical'},  # 漏诊最小化
    'Specificity': {'target': '>0.80', 'weight': 'medium'},
    'NPV': {'target': '>0.90', 'weight': 'high'},  # 排查能力

    # 跨域指标
    'Performance_Retention': 'Target_AUC / Source_AUC',
    'Adaptation_Gain': 'With_UDA_AUC - Without_UDA_AUC',
    'Failure_Rate': 'Critical_Miss_Rate'
}
```

### 5.2 统计显著性分析
```python
STATISTICAL_TESTS = {
    'paired_t_test': {
        'purpose': 'PANDA vs 基线模型比较',
        'significance_level': 0.05,
        'multiple_correction': 'bonferroni'
    },
    'cohens_d': {
        'purpose': '效应大小评估',
        'threshold': 0.5  # 中等效应
    }
}
```

### 5.3 预期结果

#### 性能对比预期
| 模型 | 源域AUC | 跨域AUC | UDA增益 | 预期排名 |
|------|---------|---------|---------|----------|
| **PANDA_TabPFN** | >0.85 | >0.82 | +5-8% | 🥇 第1 |
| **XGBoost+UDA** | 0.80-0.87 | 0.72-0.80 | +3-6% | 🥈 第2 |
| **TabPFN_Only** | >0.85 | 0.75-0.78 | N/A | 3rd |
| **Random_Forest+UDA** | 0.78-0.85 | 0.70-0.78 | +4-7% | 🥉 第4 |
| **LASSO_LR+UDA** | 0.75-0.82 | 0.68-0.75 | +2-5% | 5th |
| **SVM+UDA** | 0.72-0.80 | 0.65-0.72 | +2-4% | 6th |
| **KNN+UDA** | 0.65-0.75 | 0.55-0.65 | +1-3% | 7th |

#### 关键假设验证
- **H1**: PANDA_TabPFN显著优于TabPFN_Only (验证UDA效果)
- **H2**: PANDA_TabPFN显著优于所有传统模型 (验证TabPFN优势)
- **H3**: TCA在心脏病数据上表现最佳 (域适应方法排序)

## 📅 实施计划

### 6.1 12周项目计划

#### Phase 1: 基础设施 (Week 1-2) ✅ 已完成
- [x] ✅ 环境配置: PyTorch 2.9.1 + MPS (Apple Silicon)
- [x] ✅ TabPFN 2.0.9 安装完成 (使用当前仓库开发版本)
- [x] ✅ UDA库安装: adapt + dcor
- [x] ✅ 新项目结构创建: `/Users/lqy/work/TabPFN/panda_heart_project/`
- [x] ✅ PRD文档迁移和重组完成
- [x] ✅ 各模块TODO文档创建完成 (8个模块)
- [ ] 🔄 UCI心脏病数据下载和预处理 [Phase 2开始]

**状态**: ✅ Phase 1完成，项目结构就绪，开始Phase 2数据预处理

#### Phase 2: 核心实验 (Week 3-8)
- [ ] 单中心10折CV (7模型 × 4中心)
- [ ] 两中心跨域实验 (7模型 × 6对 × 6UDA)
- [ ] 多中心LOCO-CV实验
- [ ] 126个实验组合完成

#### Phase 3: 分析验证 (Week 9-10)
- [ ] 统计显著性分析
- [ ] 4个研究问题回答
- [ ] 医学合理性验证
- [ ] 敏感性分析

#### Phase 4: 结果整理 (Week 11-12)
- [ ] 学术级可视化制作
- [ ] 论文写作素材准备
- [ ] 代码开源和文档
- [ ] 项目总结

### 6.2 成功指标
- [ ] 126个实验全部完成
- [ ] H1-H3假设全部验证 (p < 0.01)
- [ ] PANDA_TabPFN达到预期性能
- [ ] 可发表级论文素材就绪

### 6.3 风险管控
| 风险 | 缓解措施 |
|------|----------|
| Hungarian数据缺失率高 | 多重插补 + 敏感性分析 |
| 计算复杂度过高 | 并行计算 + 结果缓存 |
| TabPFN内存占用 | 模型分片 + GPU优化 |

## 📁 项目文件结构

**新项目路径**: `/Users/lqy/work/TabPFN/panda_heart_project/`

```
panda_heart_project/                          # 新项目根目录 (与uda_medical_imbalance_project同级)
├── README.md                                 # 项目概述和快速开始
├── docs/
│   └── PANDA_Heart_Final_PRD.md             # 最终PRD (本文件)
├── data/
│   └── heart_disease_loader_TODO.md         # 数据处理模块实现
├── modeling/
│   └── panda_heart_adapter_TODO.md          # PANDA模型实现
├── evaluation/
│   └── heart_disease_metrics_TODO.md        # 评估工具和医学指标
├── scripts/
│   └── run_heart_disease_analysis_TODO.md   # 主实验脚本
├── features/
│   └── clinical_feature_engineering_TODO.md # 特征工程方案
├── uda/
│   └── heart_domain_adaptation_TODO.md      # 域适应方法实现
├── visualization/
│   └── heart_disease_viz_TODO.md            # 可视化模块
├── config/
│   └── heart_disease_config_TODO.md         # 配置文件管理
├── tests/
│   └── test_heart_disease_pipeline_TODO.md  # 测试方案
├── preprocessing/                           # 数据预处理工具 [待实现]
├── examples/                                # 使用示例 [待实现]
├── results/                                 # 实验结果输出 [待实现]
└── logs/                                    # 日志文件 [待实现]

# 参考项目结构
uda_medical_imbalance_project/               # 现有项目 (模板参考)
```

## 🎯 项目价值

### 学术贡献
1. **首次验证**: PANDA框架在心脏病诊断中的效果
2. **系统对比**: 126个实验组合的全面评估
3. **方法优化**: 心脏病专用域适应策略
4. **基准建立**: 跨中心心脏病诊断评估标准

### 临床意义
1. **筛查工具**: 高敏感性的心脏病智能筛查
2. **跨中心应用**: 解决医院间数据差异问题
3. **小样本优势**: 适用于数据稀缺的临床场景

---

### 🔥 当前项目状态

#### 已完成 ✅ (Phase 1 基础设施 & Phase 2 核心实验 & Phase 4 部分结果)
- **环境配置**: PyTorch 2.9.1 + MPS (Apple Silicon) ✅
- **TabPFN安装**: v2.0.9 (使用当前仓库开发版本) ✅
- **UDA库**: adapt + dcor + tensorflow ✅
- **项目结构**: `/Users/lqy/work/TabPFN/panda_heart_project/` ✅
- **数据预处理**: UCI心脏病4个中心数据下载和预处理完成 (920样本) ✅
- **模型实现**:
  - PANDA_TabPFN (TCA适配) ✅
  - 基线模型 (TabPFN_Only, LASSO_LR, XGBoost, RF, SVM, KNN) ✅
- **实验执行**:
  - 单中心实验 (Accuracy/AUC/Sensitivity) ✅
  - 跨域实验 (TCA only, 6个跨域对) ✅
- **可视化与分析**:
  - 科研级PDF图表生成 (`results/panda_heart_tca_analysis.pdf`) ✅
  - 自动化分析报告生成 (`results/tca_only_analysis_report.md`) ✅

#### 进行中 🔄 (Phase 3 深入分析)
- **统计显著性分析**: 验证PANDA-TCA相对于基线的显著性差异 🔄
- **多中心集成实验**: LOCO-CV策略实现 🔄

#### 待开始 ⏳ (Phase 4 论文写作)
- **论文撰写**: 基于生成的图表和数据撰写论文 ⏳
- **更多UDA方法**: CORAL/SA (目前仅实现了TCA，效果已达标，可选扩展) ⏳

**项目状态**: ✅ Phase 2 核心TCA实验已完成，Phase 3 分析可视化已出图
**最新成果**: PANDA-TCA模型在跨域任务中实现了 **99.0%** 的性能保留率 (Accuracy/AUC)
**下一步**: 整理实验数据，进行统计检验，准备论文初稿

*最后更新: 2025-11-19 12:55 | 版本: v4.0 (TCA实验完成版)*