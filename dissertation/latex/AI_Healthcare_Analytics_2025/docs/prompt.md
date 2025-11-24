# AI Assistant Execution Guidelines

**Core Rule**: 执行任何PRD任务后必须立即更新对应PRD文档标记完成状态

---

## ⚠️ 通用执行流程

### 1. 任务执行前
- 查看对应PRD文档了解任务详情
- 确认任务依赖和优先级
- 明确成功标准

### 2. 任务执行中
- 遵循PRD中的具体要求
- 保持专业标准（技术准确性、学术规范等）
- 遇到问题及时记录

### 3. 任务完成后 (关键步骤)
- **立即更新对应PRD文档**
- 将任务标记为 `[x]`
- 添加完成日期和备注
- 更新完成百分比
- 记录任何重要变更或建议

---

## 🔄 PRD更新标准格式

```markdown
- [x] **任务编号** 任务描述
  - **Status**: Completed
  - **Completion Date**: YYYY-MM-DD
  - **Completion Time**: HH:MM:SS
  - **Notes**: [简要完成说明，遇到的问题，改进建议等]
```

**时间获取**: 使用 `mcp__yokingma-time-mcp__current_time` 获取准确时间

同时更新文档底部的完成度统计。

---

## 📋 质量保证检查点

标记任务完成前确认：
- ✅ 技术准确性
- ✅ 符合目标受众
- ✅ 文档格式正确
- ✅ 无破坏性变更

---

## ⏰ 时间管理MCP工具

### 可用工具
- **`mcp__yokingma-time-mcp__current_time`**: 获取当前时间
- **`mcp__yokingma-time-mcp__relative_time`**: 计算相对时间
- **其他时间工具**: 格式转换、时区转换等

### 使用时机
- **任务开始时**: 记录开始时间
- **任务完成时**: 记录完成时间和总耗时
- **PRD更新时**: 准确的时间戳
- **进度跟踪时**: 计算任务执行效率

### 示例用法
```markdown
# 任务开始
开始时间: 2025-11-08 14:30:15

# 任务完成
- [x] **2.1** 标题优化
  - **Status**: Completed
  - **Completion Date**: 2025-11-08
  - **Completion Time**: 14:35:22
  - **Duration**: 5分7秒
  - **Notes**: 标题成功更新为AI技术导向
```

---

## 🔄 PRD修改协议

如果需要在执行过程中修改PRD：
1. 向用户说明修改原因和具体变更
2. 获得用户明确同意后再修改
3. 在PRD中记录修改历史（包含时间戳）
4. 更新完成度统计

---

## 📚 参考论文资源

执行任何写作或改写任务前，请优先查阅 `docs/reference/` 中的核心论文，以保证技术叙述和引用与最新研究保持一致：

- Accurate Predictions on Small Data with a Tabular Foundation Model（Nature, 2024）：阐述TabPFN等表格基础模型在小样本任务上的性能突破，为强调基础模型价值提供权威来源。
- Foundation Models（Business & Information Systems Engineering, 2024）：定义“基础模型”概念及跨领域影响，可用于说明方法论定位。
- Domain Generalization based on Transfer Component Analysis（Grubinger et al., 2015）：介绍TCA原理，是撰写域适应部分的技术依据。
- Addressing Cross-Population Domain Shift in Chest X-ray Classification through Supervised Adversarial Domain Adaptation（Scientific Reports, 2025）：展示医疗场景中跨域偏移的现实挑战，可引用其实验设计与结论。
- Attention Is All You Need（NeurIPS, 2017）：Transformer原始论文，支撑自注意力架构的描述与术语使用。
- A Survey on Deep Tabular Learning（arXiv, 2024）：总结深度表格学习模型演进，可帮助撰写相关工作与背景段落。
- TabNet: Attentive Interpretable Tabular Learning（AAAI, 2021）：可解释深度表格模型代表，用于比较与讨论注意力机制的可解释性。

若执行任务过程中需要额外论文，请在PRD中补充引用需求并等待资料更新后再继续，确保引用链条可追溯。

---

**重要提示**: 每个PRD都是独立的项目文档，执行前请确认正在操作正确的PRD文件！使用时间MCP工具确保准确的时间记录。
