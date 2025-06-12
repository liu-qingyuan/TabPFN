# Analytical MMD A2B Feature58 项目文档索引

## 项目概述

本项目是一个基于 Maximum Mean Discrepancy (MMD) 的医疗数据跨域适应分析包，专注于 AI4Health 到河南癌症医院数据集的域转移学习。项目包含多种 MMD 方法（线性变换、核PCA、均值标准差对齐等）和 AutoTabPFN 集成。

## 📚 核心文档

### 🚀 快速入门
- **[工作流程详解 (workflow.md)](workflow.md)** - 主函数运行流程完整说明
- **[配置管理 (config.md)](config.md)** - 项目配置系统详解
- **[设置文件 (config/settings.md)](config/settings.md)** - 详细的配置参数说明

### 🧮 算法实现
- **[MMD算法 (mmd_algorithms.md)](mmd_algorithms.md)** - MMD理论基础和算法实现
- **[预处理模块 (preprocessing/mmd.md)](preprocessing/mmd.md)** - MMD核心算法详细实现

### 🤖 模型系统
- **[模型架构 (models.md)](models.md)** - 支持的模型类型和架构
- **[模型选择器 (modeling/model_selector.md)](modeling/model_selector.md)** - 模型选择和管理系统

### 📊 数据处理
- **[数据加载器 (data/loader.md)](data/loader.md)** - Excel数据加载和验证

### 📈 可视化系统
- **[t-SNE可视化 (visualization/tsne_plots.md)](visualization/tsne_plots.md)** - 域适应效果可视化

### 🔍 API参考
- **[API参考 (api_reference.md)](api_reference.md)** - 完整的API文档

## 📁 完整文件文档索引

### 配置模块 (config/)
- **[config/settings.py](config/settings.md)** - 项目配置管理核心
  - 58个特征配置和管理
  - AutoTabPFN、TunedTabPFN等模型参数设置
  - 线性MMD、核PCA MMD、均值标准差对齐算法配置
  - 数据路径管理和实验配置

### 数据处理模块 (data/)
- **[data/loader.py](data/loader.md)** - 数据加载和验证系统
  - Excel文件读取和解析
  - 数据验证、清洗和预处理
  - 多数据集管理和兼容性检查
  - 缓存机制和性能优化

### 预处理模块 (preprocessing/)
- **[preprocessing/mmd.py](preprocessing/mmd.md)** - MMD算法核心实现
  - MMD距离计算和无偏估计
  - 线性MMD变换和分阶段训练
  - 核PCA MMD变换和降维
  - 统一变换接口和参数优化

- **[preprocessing/threshold_optimizer.py](preprocessing/threshold_optimizer.md)** - 阈值优化器
  - 分类阈值自动优化
  - ROC曲线分析和最优点选择
  - 多指标平衡优化
  - 交叉验证阈值选择

- **[preprocessing/class_conditional_mmd.py](preprocessing/class_conditional_mmd.md)** - 类条件MMD
  - 按类别计算MMD距离
  - 类别平衡的域适应
  - 条件分布对齐
  - 类别权重优化

- **[preprocessing/scaler.py](preprocessing/scaler.md)** - 数据标准化器
  - 特征标准化和归一化
  - 鲁棒性缩放方法
  - 类别特征处理
  - 逆变换支持

### 模型管理模块 (modeling/)
- **[modeling/model_selector.py](modeling/model_selector.md)** - 模型选择和管理
  - 统一模型接口和创建
  - AutoTabPFN、TunedTabPFN、原生TabPFN支持
  - 配置管理和参数验证
  - 错误处理和优雅降级

- **[modeling/cross_domain_runner.py](modeling/cross_domain_runner.md)** - 跨域实验运行器
  - 完整的跨域实验流程
  - 多种MMD方法集成
  - 批量实验管理
  - 结果收集和分析

- **[modeling/tabpfn_runner.py](modeling/tabpfn_runner.md)** - TabPFN运行器
  - TabPFN模型的专用运行器
  - 参数优化和调优
  - 性能监控和评估
  - 实验结果管理

### 可视化模块 (visualization/)
- **[visualization/tsne_plots.py](visualization/tsne_plots.md)** - t-SNE可视化核心
  - 域适应前后对比可视化
  - 交互式图表和网格布局
  - 批量可视化处理
  - 样式和主题管理

- **[visualization/histogram_plots.py](visualization/histogram_plots.md)** - 直方图可视化
  - 特征分布直方图
  - 域间分布对比
  - 统计分布分析
  - 多变量分布可视化

- **[visualization/comparison_plots.py](visualization/comparison_plots.md)** - 比较图表
  - 方法效果对比图
  - 性能指标比较
  - 多实验结果展示
  - 统计显著性可视化

- **[visualization/roc_plots.py](visualization/roc_plots.md)** - ROC曲线图
  - ROC曲线绘制和分析
  - AUC计算和比较
  - 多模型ROC对比
  - 阈值选择可视化

- **[visualization/metrics.py](visualization/metrics.md)** - 可视化指标
  - 可视化质量评估
  - 图表美观度指标
  - 信息传达效果评估
  - 可视化最佳实践

- **[visualization/utils.py](visualization/utils.md)** - 可视化工具
  - 通用绘图函数
  - 颜色和样式管理
  - 图表保存和导出
  - 格式转换工具

### 指标评估模块 (metrics/)
- **[metrics/classification.py](metrics/classification.md)** - 分类性能指标
  - 准确率、精确率、召回率计算
  - F1分数和AUC评估
  - 混淆矩阵分析
  - 多类别分类指标

- **[metrics/evaluation.py](metrics/evaluation.md)** - 模型评估框架
  - 交叉验证评估
  - 性能指标汇总
  - 统计显著性检验
  - 评估结果报告

- **[metrics/discrepancy.py](metrics/discrepancy.md)** - 域差距度量
  - MMD距离计算和分析
  - 其他域差距指标
  - 分布差异量化
  - 域适应效果评估

- **[metrics/cross_domain_metrics.py](metrics/cross_domain_metrics.md)** - 跨域评估指标
  - 跨域性能评估
  - 域转移效果量化
  - 适应质量指标
  - 综合评估报告

### 实用工具模块 (utils/)
- **[utils/logging_setup.py](utils/logging_setup.md)** - 日志配置系统
  - 统一日志格式和级别
  - 文件和控制台输出
  - 模块化日志管理
  - 调试和错误追踪

### 脚本模块 (scripts/)
- **[scripts/run_analytical_mmd.py](scripts/run_analytical_mmd.md)** - 主要实验脚本
  - 完整的MMD分析流程
  - 命令行参数解析
  - 批量实验执行
  - 结果保存和报告生成

### 示例模块 (examples/)
- **[examples/multi_model_demo.py](examples/multi_model_demo.md)** - 多模型演示
  - 完整的使用示例
  - 多种模型对比演示
  - 最佳实践展示
  - 快速入门指南

### 测试模块 (tests/)
- **[tests/test_mmd_basic.py](tests/test_mmd_basic.md)** - MMD基础测试
  - MMD算法正确性验证
  - 数值稳定性测试
  - 边界条件测试
  - 性能基准测试

- **[tests/test_model_selector.py](tests/test_model_selector.md)** - 模型选择器测试
  - 模型创建和配置测试
  - 错误处理测试
  - 兼容性验证
  - 性能测试

- **[tests/test_visualization.py](tests/test_visualization.md)** - 可视化测试
  - 图表生成测试
  - 样式和格式验证
  - 大数据集处理测试
  - 输出质量检查

- **[tests/test_multi_model_integration.py](tests/test_multi_model_integration.md)** - 多模型集成测试
  - 端到端集成测试
  - 多模型协同测试
  - 工作流程验证
  - 结果一致性检查

- **[tests/test_statistics_consistency.py](tests/test_statistics_consistency.md)** - 统计一致性测试
  - 统计指标计算验证
  - 数值精度测试
  - 算法一致性检查
  - 重现性验证

- **[tests/test_new_modules.py](tests/test_new_modules.md)** - 新模块测试
  - 新增功能测试
  - 向后兼容性验证
  - 接口一致性检查
  - 功能完整性测试

- **[tests/test_skip_cv.py](tests/test_skip_cv.md)** - 交叉验证跳过测试
  - 快速验证模式测试
  - 性能优化验证
  - 结果准确性检查
  - 时间效率测试

### 项目配置文件
- **[pytest.ini](project_config/pytest.md)** - 测试配置
  - 测试框架配置
  - 测试发现规则
  - 覆盖率设置
  - 测试报告配置

- **[requirements.txt](project_config/requirements.md)** - 依赖管理
  - 核心依赖包列表
  - 版本兼容性要求
  - 可选依赖说明
  - 安装指南

## 🚀 快速开始

### 新用户入门路径
1. 🚀 [工作流程详解](workflow.md) - 理解主函数完整执行流程
2. 📖 [项目README](../README.md) - 了解项目背景和特色
3. ⚙️ [配置管理](config.md) - 设置项目环境
4. 📊 [数据加载器](data/loader.md) - 学习数据加载
5. 🧮 [MMD算法](mmd_algorithms.md) - 理解核心算法
6. 🤖 [模型选择器](modeling/model_selector.md) - 选择合适的模型
7. 📈 [可视化系统](visualization/tsne_plots.md) - 生成结果图表

### 算法研究者路径
1. 🧮 [MMD算法详解](mmd_algorithms.md) - 深入理解MMD理论
2. 🔧 [预处理实现](preprocessing/mmd.md) - 查看具体实现细节
3. 📊 [类条件MMD](preprocessing/class_conditional_mmd.md) - 高级MMD方法
4. 📈 [域差距度量](metrics/discrepancy.md) - 评估指标详解
5. 🔍 [API参考](api_reference.md) - 查找具体函数接口

### 应用开发者路径
1. 🤖 [模型系统](models.md) - 了解可用模型
2. ⚙️ [配置系统](config/settings.md) - 学习参数配置
3. 🏃 [跨域运行器](modeling/cross_domain_runner.md) - 集成实验流程
4. 📈 [可视化工具](visualization/tsne_plots.md) - 集成可视化功能
5. 🔍 [API参考](api_reference.md) - 查找开发接口

### 测试和验证路径
1. 🧪 [基础测试](tests/test_mmd_basic.md) - 验证算法正确性
2. 🔧 [集成测试](tests/test_multi_model_integration.md) - 端到端验证
3. 📊 [统计测试](tests/test_statistics_consistency.md) - 数值精度验证
4. 📈 [可视化测试](tests/test_visualization.md) - 图表质量检查

## 🔍 按功能分类查找

### 数据处理
- [数据加载和验证](data/loader.md)
- [特征选择和配置](config/settings.md#特征配置)
- [数据预处理](preprocessing/mmd.md#数据预处理功能)
- [数据标准化](preprocessing/scaler.md)

### 算法实现
- [MMD距离计算](preprocessing/mmd.md#compute_mmd)
- [线性MMD变换](preprocessing/mmd.md#MMDLinearTransform类)
- [核PCA变换](preprocessing/mmd.md#核PCA-MMD变换)
- [类条件MMD](preprocessing/class_conditional_mmd.md)
- [阈值优化](preprocessing/threshold_optimizer.md)

### 模型使用
- [模型选择](modeling/model_selector.md#get_model)
- [AutoTabPFN配置](modeling/model_selector.md#AutoTabPFN)
- [跨域实验运行](modeling/cross_domain_runner.md)
- [TabPFN专用运行器](modeling/tabpfn_runner.md)

### 可视化
- [t-SNE对比图](visualization/tsne_plots.md#plot_tsne_comparison)
- [直方图分析](visualization/histogram_plots.md)
- [ROC曲线](visualization/roc_plots.md)
- [方法比较图](visualization/comparison_plots.md)
- [性能对比图](visualization/performance_plots.md)
- [可视化工具](visualization/utils.md)
- [可视化指标](visualization/metrics.md)

### 评估指标
- [分类性能](metrics/classification.md)
- [域差距度量](metrics/discrepancy.md)
- [跨域评估](metrics/cross_domain_metrics.md)
- [模型评估框架](metrics/evaluation.md)

### 配置管理
- [基础配置](config/settings.md#详细配置项)
- [模型配置](config/settings.md#模型配置)
- [MMD配置](config/settings.md#MMD方法配置)
- [实验配置](config/settings.md#实验配置)

## 🛠️ 按问题类型查找

### 安装和环境
- [依赖管理](project_config/requirements.md)
- [测试配置](project_config/pytest.md)
- [环境配置](config/settings.md#环境适应配置)
- [日志设置](utils/logging_setup.md)

### 数据问题
- [数据加载失败](data/loader.md#故障排除)
- [特征不匹配](data/loader.md#常见问题)
- [内存不足](data/loader.md#性能优化建议)
- [数据标准化问题](preprocessing/scaler.md#故障排除)

### 算法问题
- [MMD计算错误](preprocessing/mmd.md#故障排除)
- [收敛问题](preprocessing/mmd.md#常见问题)
- [数值不稳定](preprocessing/mmd.md#数值不稳定)
- [阈值优化失败](preprocessing/threshold_optimizer.md#故障排除)

### 模型问题
- [模型创建失败](modeling/model_selector.md#常见问题)
- [性能不佳](modeling/model_selector.md#模型配置优化)
- [内存限制](modeling/model_selector.md#内存不足)
- [跨域实验错误](modeling/cross_domain_runner.md#故障排除)

### 可视化问题
- [t-SNE收敛问题](visualization/tsne_plots.md#故障排除)
- [图表质量差](visualization/tsne_plots.md#可视化质量差)
- [大数据集处理](visualization/tsne_plots.md#大数据集处理)
- [ROC曲线异常](visualization/roc_plots.md#故障排除)

## 📖 使用场景指南

### 场景1: 快速验证域适应效果
```python
# 推荐文档路径
1. 数据加载器 → 模型选择器 → 可视化系统
2. 重点关注: 快速配置和自动化流程
3. 参考: examples/multi_model_demo.py
```

### 场景2: 深入研究MMD算法
```python
# 推荐文档路径  
1. MMD算法理论 → 预处理实现 → 类条件MMD → 域差距度量
2. 重点关注: 算法细节和参数调优
3. 参考: tests/test_mmd_basic.py
```

### 场景3: 集成到现有项目
```python
# 推荐文档路径
1. API参考 → 配置管理 → 跨域运行器 → 模型选择器
2. 重点关注: 接口设计和配置选项
3. 参考: modeling/cross_domain_runner.py
```

### 场景4: 性能优化和调试
```python
# 推荐文档路径
1. 故障排除 → 性能优化 → 日志配置 → 测试验证
2. 重点关注: 问题诊断和解决方案
3. 参考: utils/logging_setup.py
```

### 场景5: 自定义可视化
```python
# 推荐文档路径
1. 可视化模块 → 图表工具 → 样式配置 → 批量处理
2. 重点关注: 可视化扩展和定制
3. 参考: visualization/ 目录下所有文件
```

### 场景6: 算法验证和测试
```python
# 推荐文档路径
1. 测试框架 → 统计一致性 → 基础测试 → 集成测试
2. 重点关注: 测试设计和验证方法
3. 参考: tests/ 目录下所有文件
```

## 🔗 相关资源

### 理论背景
- [MMD理论介绍](mmd_algorithms.md#理论基础)
- [域适应概述](mmd_algorithms.md#域适应应用)
- [TabPFN原理](models.md#TabPFN模型系列)
- [类条件MMD理论](preprocessing/class_conditional_mmd.md#理论基础)

### 实践指南
- [最佳实践](config.md#最佳实践)
- [性能优化](preprocessing/mmd.md#性能优化)
- [实验设计](modeling/cross_domain_runner.md#实验设计)
- [可视化指南](visualization/tsne_plots.md#使用示例和最佳实践)

### 扩展开发
- [自定义模型](modeling/model_selector.md#高级功能)
- [新增算法](preprocessing/mmd.md#高级功能)
- [可视化扩展](visualization/tsne_plots.md#高级功能)
- [测试扩展](tests/test_new_modules.md#扩展测试)

## 📝 文档贡献

### 贡献指南
1. **文档结构**: 遵循现有的文档组织结构
2. **代码示例**: 提供完整可运行的示例代码
3. **中英文规范**: 自然语言使用中文，代码和技术术语使用英文
4. **更新索引**: 新增文档后及时更新本索引文件

### 文档标准
- **完整性**: 包含概述、参数说明、使用示例、故障排除
- **准确性**: 确保代码示例可以正常运行
- **可读性**: 使用清晰的结构和适当的格式
- **实用性**: 提供实际的使用场景和最佳实践

### 维护说明
- 定期检查文档与代码的一致性
- 及时更新过时的信息和示例
- 收集用户反馈并改进文档质量
- 保持文档索引的完整性和准确性

## 📊 文档完成状态

### ✅ 已完成文档
- **核心理论文档**: config.md, models.md, mmd_algorithms.md, api_reference.md
- **配置模块**: config/settings.py
- **数据处理**: data/loader.py
- **预处理模块**: preprocessing/mmd.py, preprocessing/scaler.py, preprocessing/threshold_optimizer.py, preprocessing/class_conditional_mmd.py
- **模型管理**: modeling/model_selector.py, modeling/cross_domain_runner.py, modeling/tabpfn_runner.py
- **可视化模块**: visualization/tsne_plots.py, visualization/histogram_plots.py, visualization/comparison_plots.py, visualization/roc_plots.py, visualization/metrics.py, visualization/utils.py
- **指标评估**: metrics/classification.py, metrics/evaluation.py, metrics/discrepancy.py, metrics/cross_domain_metrics.py
- **工具模块**: utils/logging_setup.py
- **脚本模块**: scripts/run_analytical_mmd.py
- **示例模块**: examples/multi_model_demo.py
- **测试模块**: tests/test_skip_cv.py
- **项目配置**: project_config/requirements.md, project_config/pytest.md

### 🚧 待完成文档 (需要根据实际代码创建)
- 其余测试文件文档 (test_mmd_basic.py, test_model_selector.py, test_visualization.py, test_multi_model_integration.py, test_statistics_consistency.py, test_new_modules.py)

---

**最后更新**: 2024年12月
**文档版本**: v1.3
**维护者**: Analytical MMD A2B Feature58 项目团队
**文档完成度**: 约85% (17/20+ 核心模块已完成) 