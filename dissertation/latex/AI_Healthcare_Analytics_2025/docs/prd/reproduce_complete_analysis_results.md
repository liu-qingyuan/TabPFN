# 完整分析结果复现指南 (PRD)

## 目标
复现 `complete_analysis_20250904_222331_8:8:8:8` 目录中的分析结果

## ⚠️ 重要发现：无法精确复现的原因

### 1. 脚本版本差异
- **目标版本**: commit `2cc7ab2d9e44533cdf23c505bc157b06467df660` (2025-09-04)
- **当前版本**: 最新版本，已大幅扩展和修改
- **关键参数变更**: `feature_set` → `feature_type`

### 2. 模型数量差异
- **旧版本**: 基础模型对比，数量较少
- **当前版本**: 包含大量新增模型：
  - 传统基线: PKUPH, Mayo, Paper_LR
  - 机器学习基线: SVM, DT, RF, GBDT, XGBoost
  - UDA方法: TCA, SA, CORAL, KMM

### 3. 8:8:8:8 配置解析
- **第一个8**: `feature_set='best8'` (8个最佳特征)
- **后面三个8**: 目录命名约定，不是参数配置
- **完整含义**: best8特征集的标准分析配置

## 关键约束
- **版本锁定**: 必须回到commit `2cc7ab2` 才能精确复现
- **结果差异**: 当前版本会产生更复杂的结果，包含更多模型对比
- **配置兼容**: `--feature_type` 与旧版 `--feature_set` 可能不完全等价

## 复现步骤清单

### 1. 环境准备
- [ ] 确认Python 3.9+环境
- [ ] 安装TabPFN和相关依赖
- [ ] 验证CUDA/GPU可用性（可选但推荐）
- [ ] 确认ADAPT库已安装

### 2. 数据准备
- [ ] 验证医疗数据文件完整性：
  - `data/AI4healthcare.xlsx` (源域数据)
  - `data/HenanCancerHospital_features63_58.xlsx` (目标域数据)
- [ ] 确认特征集配置文件存在且匹配
- [ ] 验证`best8`特征集定义

### 3. 版本回退 (关键步骤)
- [ ] 回退到目标commit版本：
  ```bash
  git checkout 2cc7ab2d9e44533cdf23c505bc157b06467df660
  ```

### 4. 参数配置 (旧版本参数)
- [ ] 使用以下精确参数运行：
  ```bash
  cd uda_medical_imbalance_project
  python scripts/run_complete_analysis.py \
    --feature_set best8 \
    --scaler_type none \
    --imbalance_method none \
    --cv_folds 10 \
    --random_state 42 \
    --verbose
  ```
- **注意**: 旧版本使用 `--feature_set` 参数，不是 `--feature_type`

### 5. 执行分析
- [ ] 在uda_medical_imbalance_project目录执行脚本
- [ ] 监控运行过程，确认无错误
- [ ] 验证生成结果包含所有预期文件

### 6. 结果验证
- [ ] 检查新生成的 `results/complete_analysis_YYYYMMDD_HHMMSS` 目录
- [ ] 对比关键指标：
  - TabPFN源域CV AUC应接近0.826
  - TCA域适应AUC应接近0.709
  - 基线模型性能应在合理范围内
- [ ] 验证生成的可视化图表完整性

### 7. 文件结构检查
确认生成以下文件：
- [ ] `complete_results.json` - 完整结果数据
- [ ] `source_domain_cv_results.json` - 源域交叉验证结果
- [ ] `uda_methods_results.json` - UDA方法结果
- [ ] `analysis_report.md` - 分析报告
- [ ] 可视化图表文件：
  - [ ] `combined_analysis_figure.png/pdf`
  - [ ] `calibration_curves.png/pdf`
  - [ ] `decision_curve_analysis.png/pdf`
  - [ ] `roc_curves/` 目录及内容
  - [ ] `uda_TCA/` 目录及内容

**注意**: 旧版本生成的文件结构可能比当前版本简单，因为新增的基线模型尚未加入。

## 技术注意事项

### 参数兼容性
- **版本差异**: 当前版本使用 `feature_type`，目标版本使用 `feature_set`
- **必须回退**: 只有在commit `2cc7ab2` 中才能使用正确的参数格式
- **脚本位置**: 目标版本脚本位于 `uda_medical_imbalance_project/scripts/`

### 随机性控制
- 设置固定随机种子 (42)
- TabPFN结果可能因硬件环境略有差异
- 重点关注统计指标而非精确数值匹配

### 资源需求
- 推荐使用GPU加速（8GB+ VRAM）
- CPU运行时间显著增长
- 确保充足内存（16GB+推荐）

## 预期结果范围

### 关键性能指标
- **TabPFN源域CV**: AUC 0.80-0.85
- **TabPFN无UDA**: AUC 0.65-0.75
- **TCA域适应**: AUC 0.68-0.75
- **传统基线**: AUC 0.60-0.70

### 可接受差异
- AUC差异: ±0.02
- 训练时间: 依赖硬件
- 可视化样式: 版本差异可能导致细微差别

## 故障排除

### 常见问题
1. **数据加载失败**: 检查数据文件路径和格式
2. **特征集不匹配**: 验证`best8`特征定义
3. **ADAPT库问题**: 重新安装域适应库
4. **内存不足**: 减少交叉验证折数或使用更少特征
5. **TabPFN错误**: 检查模型权重和网络连接

### 备用方案
#### 方案1: 使用当前脚本 (结果会有差异)
```bash
python scripts/run_complete_analysis.py --feature_type best8 --cv_folds 5
```
- 会生成更复杂的对比结果
- 包含新增的基线模型
- 文件结构更丰富

#### 方案2: 版本回退 (精确复现)
```bash
# 回退到目标版本
git checkout 2cc7ab2d9e44533cdf23c505bc157b06467df660
# 运行旧版本脚本
cd uda_medical_imbalance_project
python scripts/run_complete_analysis.py --feature_set best8
# 回到最新版本
git checkout main
```

#### 方案3: 手动目录重命名
如果希望生成相同的目录名格式，可以手动重命名：
```bash
mv results/complete_analysis_YYYYMMDD_HHMMSS results/complete_analysis_20250904_222331_8:8:8:8_reproduction
```

## 成功标准

### 精确复现 (使用commit 2cc7ab2)
- [ ] 所有分析步骤无错误完成
- [ ] 生成与原始文件结构一致的结果
- [ ] 关键性能指标在预期范围内 (TabPFN AUC ~0.826, TCA AUC ~0.709)
- [ ] 文件数量和类型与原始结果匹配

### 当前版本复现 (推荐)
- [ ] 分析步骤完成，包含更丰富的模型对比
- [ ] 生成包含新增基线模型的完整结果
- [ ] 核心TabPFN和TCA性能在合理范围内
- [ ] 可视化图表更加全面和详细

## 文档和报告
- [ ] 保存运行日志和git commit信息
- [ ] 记录实际使用的参数配置和版本
- [ ] 对比原始结果差异，分析模型扩展带来的影响
- [ ] 更新实验记录，注明使用的脚本版本

---

## 总结建议

**推荐方案**: 使用当前版本脚本进行复现，虽然结果不会完全一致，但会提供更全面的分析和更丰富的模型对比。如果需要精确匹配原始结果，必须回退到commit `2cc7ab2`。

**关键差异**: 当前版本新增了大量基线模型 (PKUPH, Mayo, SVM, RF, XGBoost等)，提供了更完整的算法对比分析。