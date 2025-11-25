# AI_Healthcare_Analytics_2025 论文扩写 PRD（目标 80 页+）

- 背景：当前正文约 32 页，导师要求扩充到 ~80 页，同时满足“引言/相关工作/方法/实验”四大部分的深度要求，强化跨医院/跨领域与肺结节分类的 AI 视角。
- 主要参考资料：`docs/AI Healthcare Analytics Introduction Generation.docx`、`docs/Introduction.txt`、`docs/Related_Work.txt`、`docs/Related Work Expansion for Medical AI.docx`、`docs/Related Work.txt`、`docs/1. 小样本、类不平衡和高维稀疏场景的特征选择方法.docx`，引文以 `refs.bib` 中已有条目为准（缺失则不强行引用）。
- 约束：不改动 `main.tex` 章节顺序；保持现有章节命名；新增内容以章节内新增小节/段落/表格/图为主；确保图片路径相对 `main.tex`（`img/...`）。

## 🎯 目标

- 将正文从 ~32 页扩展到 ≥80 页，确保各章均衡膨胀（引言/相关工作/方法/实验/分析/结论）。
- 引言：补充任务相关 AI 技术，解释跨院/跨域场景与肺结节分类挑战，以及现有方法的不足。
- 相关工作：系统覆盖跨医院/跨领域学习与肺结节分类演进。
- 方法：逐组件点明挑战与对应方案（TabPFN 表格基础模型、跨域 RFE、TCA/UDA 等）。
- 实验：增强讨论，解释为何 PANDA 优于基线，并覆盖新跨域数据集。

## 📋 TODO（按阶段推进）

### 阶段 0：基线盘点与版面预算

- [X] 0.1 编译当前 PDF，记录各章节页数/字数，生成“缺口表”指明每章需增加的页数。
- [X] 0.2 建立页数追踪表（章节-目标页-当前页-缺口），随迭代更新。
- [X] 0.3 当前编译版 `main.pdf` 实际页数：34（`pdfinfo`），需继续扩写至 ≥80；每次大改后复核实页数。
- [ ] 0.4 精确查看各章节页数：在章节首尾加入 `\label{sec:<name>-start}` 与 `\label{sec:<name>-end}`，编译后查 `main.aux` 中对应 `\newlabel` 的页码字段（第二个花括号数字），比按字数估算更可靠。

### 阶段 0b：章节页数缺口追踪（需持续推进，页数为整数、基于 label 定界）

- [X] Introduction：5 / 5 页（缺口 +0）
- [ ] Related Work：5 / 25 页（缺口 +20）
- [ ] Problem Formulation：2 / 4 页（缺口 +2）
- [ ] Solution：2 / 4 页（缺口 +2）
- [ ] Methods：4 / 10 页（缺口 +6）
- [ ] Analysis：5 / 15 页（缺口 +10）
- [ ] Evaluation：8 / 13 页（缺口 +5）
- [ ] Conclusion：1 / 3 页（缺口 +2）
- [ ] Acknowledgements：1 / 1 页（缺口 +0）

> 每次大幅扩写后重新统计 `wc -w Section/*.tex` 并刷新此列表，直到估算页数 ≥80。

### 阶段 1：引言扩写（AI 技术 & 场景挑战）

- [X] 1.1 以“传统树模型 → 深度表格 → 表格基础模型”演进框架重写背景，引用 TabNet/TabTransformer/SAINT/TabPFN 代表。
- [X] 1.2 补充跨医院/跨领域挑战：特征异构、分布偏移（covariate/label/concept shift）、小样本、类不平衡。
- [X] 1.3 结合肺结节分类案例，说明现有 AI 方法（树模型、深度表格、FM/LLM 序列化）为何不足。
- [X] 1.4 明确本工作定位：TabPFN 式表格 FM + 跨域 RFE + TCA 对齐的组合创新。
- [X] 1.5 引言增页计划：

  - [X] 1.5a 扩充“跨医院/跨领域挑战”实例，引用 docx 中的具体失败案例（设备差异、特征缺失、标签漂移），保持文字阐述，不添加图表。
  - [X] 1.5b 补充“AI 技术演进”段落，以文字对比树模型/深度表格/基础模型的优缺点（与 Related Work 的技术表呼应但更聚焦问题动机）。
- [X] 1.5c 加入“安全与监管背景”段：说明跨院部署失败的潜在风险（过度/不足诊断），以及为何必须在方法设计阶段考虑分布偏移。
- [X] 1.5d 增设“研究空白与本文定位”段：用文字串联小样本、高维稀疏、特征错配、无标签目标域四个痛点，并指出现有 AI 方法未解的核心问题（闭域假设、特征对齐缺失、阈值失配），字数目标 ≥500。
- [X] 1.6 引言补充待办（新增段落，先执行再勾选）：

  - [X] 1.6a 明示各类 AI 技术的不足：树模型（不可微/难迁移/小样本过拟合）、深度表格（数据饥饿/调参敏感/对批统计和编码漂移脆弱）、表格基础模型（闭域假设/特征错配/协变量漂移导致错误注意）及成像 DA 方法在缺失特征、标签漂移、无标签目标域下的失效。
  - [X] 1.6b 针对跨医院/肺结节情境，补充“为什么现有方法无法解决这些挑战”的总结段，纯文字，引用现有文献，不加图表。
- [X] 1.7 引言进一步扩写（基于指定资料落地，目标补齐 4.2 页缺口）：

  - [X] 1.7a 利用 `docs/AI Healthcare Analytics Introduction Generation.docx` 中的“AI 技术演进”素材，新增一段对比树模型、深度表格、表格 FM、TabLLM 在医疗表格的小样本与漂移场景下的优劣。
  - [X] 1.7b 结合 `docs/Introduction.txt` 与 `docs/1. 小样本、类不平衡和高维稀疏场景的特征选择方法.docx`，添加“小样本+高维稀疏”挑战描述，突出 RFE/稳定特征选择的必要性（纯文字）。
  - [X] 1.7c 引入 `docs/Related_Work.txt` 和 `docs/Related Work Expansion for Medical AI.docx` 中的跨院失败案例（设备差异、标签漂移、特征缺失），转述为引言中的“案例引子”段，强调动机。
  - [X] 1.7d 增设“监管与安全”补充段：引用现有监管/分布检测文献说明跨院部署的潜在过诊/漏诊风险，保持文字阐述，不放图。

### 阶段 2：相关工作大幅扩展（从 5 页 → 25 页）

#### 2.0 结构总览（先改 LaTeX 结构，再往里塞内容）

- [X] 在 `Section/Related_Work.tex` 中统一小节标题为以下 5 个（保持章节号/引用标签不变，仅改名称）：
  - 2.x Pulmonary nodule malignancy prediction: from clinical scores to multi-modal AI
  - 2.x Tabular learning for medical data: tree ensembles, deep tabular networks, and tabular foundation models
  - 2.x Domain shift and domain adaptation in medical AI
  - 2.x Feature selection and domain-aware stability for small medical cohorts
  - 2.x Benchmarks and open problems（含 TableShift/Wild-Time + gap analysis）

#### 2.1 肺结节恶性风险预测谱系（目标 ≥ 4 页）

- [X] 2.1.1 传统临床评分（Mayo / VA / Brock / PKUPH 等）
  - [X] 数据来源：`Related Work Expansion for Medical AI.docx`、`Related Work.docx`
  - [X] 汇总每个评分：开发队列规模、内部 AUC、外部/交叉人群 AUC 下跌、变量类型
  - [X] 写共性缺陷段：单中心小样本、logistic 假设、外部验证 AUC 下跌到 0.6–0.7
  - [X] ≥600 字；可选表格（4–5 个评分：名字/人群/样本量/内部 AUC/外部 AUC）
- [X] 2.1.2 Radiomics + 传统 ML
  - [X] 数据来源：`Related Work Expansion for Medical AI.docx`、`AI Healthcare Analytics Introduction Generation.docx`
  - [X] 描述 pipeline：分割→特征提取→特征选择（LASSO/稳定性）→分类器（SVM/RF/GBDT）
  - [X] 强调三点：高维小样本；对 scanner/thickness/kernel 敏感（ICC<0.5、AUC 下跌 0.1–0.2）；多为单中心/有限外验
  - [X] ≥500 字；结尾引到“tabular + clinical 变量适合跨院”
- [X] 2.1.3 Deep learning CAD（3D CNN / multi-view / multi-task）
  - [X] 数据来源：`Related Work Expansion for Medical AI.docx`、现有 Related Work
  - [X] 概述端到端 CT 学习（3D、patch、多尺度），给 NLST 等 AUC/sensitivity 示例
  - [X] 局限：对 scanner/protocol/vendor 极敏感；外院掉点；可解释性差；难融非影像特征
  - [X] ≥500 字；结尾过渡到表格/biomarker 模型优势
- [X] 2.1.4 Tabular / multi-modal nodule models
  - [X] 数据来源：`Related Work.docx`、`AI Healthcare Analytics Introduction Generation.docx`
  - [X] 总结表格变量 logistic/GBDT；影像+tabular late fusion/stacking
  - [X] 指出共性：单中心评估；少有显式 domain shift/feature mismatch 处理；缺乏真实跨院评测
  - [X] ≥400–500 字；结尾点明 gap：缺少表格基础模型 + 显式域对齐的风险预测

#### 2.2 Tabular learning for medical data（目标 ≥ 5 页）

- [X] 2.2.1 Tree ensembles for clinical tabular data
  - [X] 数据来源：`Related Work Expansion for Medical AI.docx`、`Related Work.docx`
  - [X] 概括 RF/XGBoost/LightGBM 优势：缺失值、非线性、交互、小中样本稳定
  - [X] 写跨院失败：新医院需重训/校准；无内置 feature alignment/DA
  - [X] ≥300–400 字；结尾：GBDT 擅长局部拟合但缺乏可迁移表征
- [X] 2.2.2 Deep tabular networks
  - [X] 数据来源：`Related Work Expansion for Medical AI.docx`、`AI Healthcare Analytics Introduction Generation.docx`
  - [X] 逐个 1–2 句概括 TabNet/TabTransformer/SAINT/FT-Transformer 设计
  - [X] 写公开基准表现与医疗小样本问题：过拟合、调参敏感、编码/批统计跨院崩溃
  - [X] ≥500 字；结论：few-hundred 病人级别未优于 GBDT，跨院鲁棒性弱
- [X] 2.2.3 Tabular foundation models
  - [X] 数据来源：`Related Work Expansion for Medical AI.docx`、`1. 小样本、类不平衡和高维稀疏场景的特征选择方法.docx`
  - [X] 介绍 TabPFN（合成任务元训练、单前向≈4h GBDT search、few-shot 友好）
  - [X] 补充 TabPFN-2.5、drift-resilient、TabLLM/LLM-over-tables 思路与局限
  - [X] 重点写闭域假设与特征错配：需同特征空间；无 feature alignment；covariate shift+缺失时 attention 错列
  - [X] ≥600–800 字；结尾过渡：把 tabular FM 作为强先验组件

#### 2.3 Domain shift & domain adaptation in medical AI（目标 ≥ 5 页）

- [X] 2.3.1 Domain shift taxonomy & clinical examples
  - [X] 数据来源：`Introduction.docx`、`Related Work Expansion for Medical AI.docx`
  - [X] 定义 covariate/label/concept shift；用肺结节场景举例（TB 高发、吸烟率、病理谱差异）
  - [X] 引 1–2 个 cross-hospital failure（如 Zech pneumonia）；≥400 字
- [X] 2.3.2 DA/DG 方法族在医疗 AI 中的尝试
  - [X] 数据来源：`Related Work Expansion for Medical AI.docx`
  - [X] 按方法族写统计对齐（MMD/CORAL/TCA）、对抗式（DANN/ADDA）、IRM/GroupDRO、federated
  - [X] 每类回答：公式/直觉、影像/表格代表性结果、小样本 tabular 失败模式
  - [X] 每类至少 1 段，整体 ≥700–800 字
- [X] 2.3.3 Tabular domain adaptation 的特殊挑战
  - [X] 数据来源：`Related Work Expansion for Medical AI.docx`、`1. 小样本、类不平衡和高维稀疏场景的特征选择方法.docx`
  - [X] 写特征异构、missingness shift、尺度/编码不一致；说明 imaging DA 直接套用困难
  - [X] 结尾引出 cross-domain RFE + TCA；≥500 字
- [X] 2.3.4 为什么选择 TCA 这一类统计对齐
  - [X] 数据来源：`Related Work Expansion for Medical AI.docx`、Methods/Problem Formulation 中已有数学描述
  - [X] 解释 TCA 在 RKHS 找共享子空间；列举 EHR/small tabular 应用
  - [X] 强调 closed-form/无需大量 target label；≥400 字；结尾连到“将 TCA 施加在 TabPFN latent space”

#### 2.4 Feature selection & domain-aware stability（目标 ≥ 4 页）

- [X] 2.4.1 Small-sample & high-dimensional feature selection
  - [X] 数据来源：`1. 小样本、类不平衡和高维稀疏场景的特征选择方法.docx`
  - [X] 总结 WPFS/GRACES/DeepFS 思想：训练中联合优化特征权重，适配高维稀疏/类不平衡
  - [X] 强调医疗动机：降采集成本、可解释、缓解过拟合；≥600 字
- [X] 2.4.2 基于 Transformer / foundation model 的特征选择
  - [X] 数据来源：同上 docx、`Related Work Expansion for Medical AI.docx`
  - [X] 描述 TabNet/Transformer 注意力或 mask 获取 importance；引出用强表格模型做 permutation/ranking
  - [X] 说明小样本下稳定性收益；≥400–500 字；写明“cross-cohort RFE 属于此范式”
- [X] 2.4.3 Domain-aware / cross-site feature selection
  - [X] 数据来源：同上 docx（FSDA/multi-site FS）
  - [X] 写多域共同优化特征子集或 DA objective 的特征惩罚；对应论文设计：shared features 上 TabPFN+RFE 同看两院性能，保留一致有效 7–8 特征
  - [X] ≥400 字；结尾：为 TCA 对齐提供低维稳定输入

#### 2.5 Benchmarks & open problems（目标 ≥ 3–4 页）

- [X] 2.5.1 TableShift / Wild-Time / 其他 tabular benchmarks
  - [X] 数据来源：`Related Work Expansion for Medical AI.docx`、Evaluation 中 BRFSS 段
  - [X] 介绍 TableShift（任务数、ID/OOD 构造、核心发现）；如需提 Wild-Time/temporal shifts
  - [X] 对比影像基准，强调 schema 变化/missingness/label shift 贴合本文问题；≥400–500 字
- [X] 2.5.2 Gap analysis：对号入座到 PANDA
  - [X] 数据来源：`Related Work Expansion for Medical AI.docx` “Gap Analysis” 段、Methods/Problem Formulation 的组件-挑战映射
  - [X] 总结前述局限（classical scores/radiomics/deep CAD/tree/deep tabular/DA/FS/benchmarks）
  - [X] 列 3–4 个 gap → 映射 PANDA 组件（tabular FM、cross-domain RFE、TCA on TabPFN latent、私有肺结节+公共基准实验设计）
  - [X] ≥600 字；收尾一句引到 PANDA 框架

#### 2.6 文献搬运与细化（对应原 2.7a–d，细化成步骤）

- [X] 2.6.1 结构对齐：在 `Section/Related_Work.tex` 写好 2.1–2.5.2 标题，留 `% TODO: fill from <doc>`
- [ ] 2.6.2 分段搬运+改写：逐个 docx（尤其 `Related Work Expansion for Medical AI.docx`）拷贝到临时文本并改写降重，按写作任务补缺（外部 AUC、子人群差异等）
- [ ] 2.6.3 引用检查：每次引用先查 `refs.bib`；找不到则换同类或删句，避免悬空引用
- [ ] 2.6.4 篇幅检查：每完成一大块运行 `wc -w Section/Related_Work.tex`，不足则回对应小节增段落（失败模式/子群差异等）

### 阶段 3A：问题定义扩写（完成）

- [X] 3A.1 记号与域定义：在 Problem\_Formulation 中明确 $\mathcal{D}_s=(X_s,Y_s)$、$\mathcal{D}_t=(X_t,Y_t)$、共享特征子集 $\mathcal{F}_{\cap}$、缺失特征集 $\mathcal{F}_{\setminus}$，并用表格/列表区分肺结节 vs.\ BRFSS 任务的变量集。
- [X] 3A.2 偏移类型与风险：解释 covariate/label/concept shift 在两类任务中的具体表现，附带风险界或误差分解公式，说明为什么需要显式域对齐。

### 阶段 3B：方法细化（完成）

- [X] 3B.1 组件-挑战映射：为 TabPFN、跨域 RFE、TCA/UDA、标签漂移处理分别撰写“挑战→机制→收益”段落，结合 best7/best8 稳定性结果和 BRFSS 设定。
- [X] 3B.2 训练/推理流程：插入伪代码或流程图，覆盖“特征对齐→RFE→TabPFN 上下文构建→TCA 投影→预测”的步骤，强调各环节的输入/输出、数据依赖、约束（共享特征、病种标签、无标签目标域）以及与肺结节/BRFSS 任务的对应关系。
- [X] 3B.3 失败模式与假设：罗列闭域假设、特征对齐假设、采样独立性假设等，说明违背时的后果，并设计表格总结潜在风险（特征缺失、新病种、极端标签漂移）及缓解策略（再训练、增量对齐、人工审查）。
- [X] 3B.4 方法-结果分离：通读 Methods 相关小节，删除或迁移任何带有实验结论/性能评价的语句，确保该章节只描述流程、假设、实现细节，所有结果留在 Analysis/Evaluation。

### 阶段 4：实验与讨论扩写（主要增页来源）

- [ ] 4.1 跨院肺结节实验：补充分层描述（按性别、吸烟、解剖位置），加上列联表/分布直方图；阐明阈值、校准、敏感性/特异性取舍。
- [ ] 4.2 TableShift 公共基准：在 Evaluation 中加入 BRFSS 任务小节（文字 + 表 + 图），说明 race shift 设置、采样方式、OOD/ID 统计差异。
- [ ] 4.3 指标矩阵与消融：新增表格对比无 TCA、无 RFE、无多分支预处理、树模型、TabPFN-only；列 AUC/ACC/F1/Calibration/Net Benefit。
- [ ] 4.4 误差分析：加入误判案例、混淆矩阵摘要、按子群体（性别、吸烟、年龄段、种族）分层的性能表，解释偏差来源。
- [ ] 4.5 精度偏低说明：在 Evaluation 文字解释 BRFSS precision/recall/F1 偏低的原因（正例占比、阈值、无类权重），避免读者误解。
- [ ] 4.6 图表扩充：保证跨院实验与 TableShift 各有至少 2–3 幅图（ROC/heatmap/降维投影），路径统一到 `img/hospital_cross_domain/` 与 `img/tableshift/`。

### 阶段 5：分析与结论补强

- [ ] 5.1 Analysis 结尾增加“PANDA 优势机理”段：预训练表征缩小域间差距、RFE 去除场景特异噪声、TCA 进一步对齐、树模型与无 TCA 的劣化原因。
- [ ] 5.2 新增“局限性与未来工作”段落：讨论闭域假设、特征缺失、极端标签漂移、推理成本；提出联邦/自适应再对齐等后续方向。
- [ ] 5.3 Conclusion 收尾补充“跨私有与公共基准的泛化”1–2 句，保持简洁。

### 阶段 6：引用与校对

- [ ] 6.1 交叉检查新增引用是否存在于 `refs.bib`，缺失则删除或换同义文献。
- [ ] 6.2 统一术语与符号，避免重复定义；检查跨章节衔接（Introduction ↔ Related Work ↔ Problem Formulation）。
- [ ] 6.3 编译检查（LaTeX）与页数核对，确保版面达标且无编译错误；更新 0b 页数缺口列表。
- [ ] 6.4 图表与路径复核：所有图片放在 `img/...` 下，caption 区分任务（跨院 vs TableShift），无遗漏文件。

## ✅ 验收标准

- 页数 ≥80，且各章均有实质扩写（非堆砌）。
- 引言/相关工作/方法/实验/分析均覆盖导师指定点；结果讨论能解释性能优势与指标偏低原因。
- 图表路径正确、字体可印刷；引用全部可解析。
