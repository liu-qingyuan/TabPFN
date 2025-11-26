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
- [ ] Related Work：13 / 25 页（缺口 +12）
- [ ] Problem Formulation：2 / 4 页（缺口 +2）
- [ ] Solution：2 / 4 页（缺口 +2）
- [ ] Methods：6 / 10 页（缺口 +4）
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

#### 2.7 Related Work 定点加深计划（在现有结构上增厚 3–5 页）

- [X] 2.7.1 强化 2.1 Tabular learning for medical data
  - [X] 在树模型小节中单独补一段“XGBoost/GBDT 在 EHR/医疗表格上的优势与小样本局限”（稀疏感知分裂、缺失值处理、对单调变换不敏感，但 N≈300 时易过拟合、不可微、难以做端到端 DA）
  - [X] 在 deep tabular 小节中补一段“TabNet/TabTransformer/FT-Transformer 在大规模公开基准 vs. 医疗小样本上的表现差异”，强调需要大样本和复杂调参，在 noisy、小样本医疗队列上不稳定
  - [X] 收束一句话对比传统 deep tabular 的 end-to-end 学习思路与 TabPFN 的 meta-learning/few-shot 路线，并指出二者都默认单域、未显式处理 cross-hospital shift
- [X] 2.7.2 新增 2.2.4 Domain adaptation and transfer learning for clinical tabular/EHR data
  - [X] 总结 1–2 个典型 EHR UDA 工作（如 AdaDiag 等），说明在 MIMIC vs UCLA 等场景中 baseline AUROC 掉 5–10 个点，DA 方法能回一部分但仍不完美
  - [X] 引出 multi-center EHR foundation model + transfer learning 的趋势，说明这类模型在跨院转移上更 sample-efficient，但任务多为 EHR 序列，与本论文的“结构化风险评分 + UDA”设定不同
  - [X] 用 Ben-David 误差上界（$\epsilon_T \le \epsilon_S + \text{divergence} + \lambda$）做半页左右的理论收束，强调“只在 source 上追求低误差不够，必须显式降低 divergence”
  - [X] 小结一句：Existing work has largely focused either on temporal shift in large EHR cohorts or on imaging-based DA, leaving a gap for small tabular cohorts with heterogeneous features…
- [X] 2.7.3 深化 2.3 Feature selection & domain-aware stability
  - [X] 在 2.3.1 中补一段“经典 RFE/LASSO vs 新一代 WPFS/GRACES/DeepFS”的优缺点对比：经典方法假设线性/单特征效应、小样本下稳定性差；新方法通过辅助网络、图结构或自编码器，应对超高维、小样本、类不平衡
  - [X] 在 2.3.3 末尾或新增 2.3.4 小结段，强调在跨域场景下特征选择还承担“domain alignment”的角色：选出各域都稳定的特征，抛弃 site-specific spurious 特征
  - [X] 以 1–2 句点明 PANDA 的 cross-domain RFE 实现了这一思想：利用预训练 TabPFN 在两家医院联合排名特征，只保留在两域都重要的 best7/best8
- [X] 2.7.4 深化 2.4 Pulmonary nodule malignancy prediction
  - [X] 在 2.4.1 中补一段“external validation & failure modes”，总结 Brock/Mayo/VA/Herder 等模型在中国队列、TB 高发地区和不同筛查人群中的外部 AUC 下跌和失配人群
  - [X] 在 2.4.2 中概括 radiomics + SVM/RF/XGBoost 在肺结节任务上的典型内部 AUC 范围（多在 0.75–0.90）以及跨扫描协议、特征可重复性不足导致的 external validation 薄弱
  - [X] 在 2.4.3 中补充 1–2 个代表性的 deep/multimodal 模型（3D-CNN、radiomics+DL+clinical），强调它们在本地或双中心数据上很强，但需要 site-specific fine-tuning，几乎没有“train at A, deploy at B without labels”
  - [X] 视需要补一个 2.4.4 Attempts to improve cross-hospital generalization 小节，用最新多中心中国小结节工作为例说明：即便针对新队列更新模型，外部 AUC 仍不稳定
  - [X] 收尾一段：Taken together, these studies show that neither handcrafted scores, radiomics pipelines, nor deep CNNs currently offer reliable malignancy prediction across hospitals without local retraining…
- [X] 2.7.5 补强 2.5 Benchmarks and open problems
  - [X] 在现有 TableShift/Wild-Tab/Wild-Time 段落后补一段，引用泛 tabular benchmark/survey（如 deep vs tree 在多数据集对比中的结论），总结在现实 tabular 场景下 GBDT 仍然是强 baseline，许多 deep/DG 方法没有系统性优势
  - [X] 强调 label shift 在误差中的主导作用，单纯对齐 feature moments 意义有限，为 PANDA 中显式考虑标签分布和校准做铺垫
  - [X] 在 Gap analysis 收尾处加一句话，点明本工作首次将 tabular foundation model、cross-domain RFE 和 TCA 组合到跨院肺结节 + 公共 TableShift 任务上

#### 2.6 文献搬运与细化（对应原 2.7a–d，细化成步骤）

- [X] 2.6.1 结构对齐：在 `Section/Related_Work.tex` 写好 2.1–2.5.2 标题，留 `% TODO: fill from <doc>`
- [ ] 2.6.2 分段搬运+改写：逐个 docx（尤其 `Related Work Expansion for Medical AI.docx`）拷贝到临时文本并改写降重，按写作任务补缺（外部 AUC、子人群差异等）
- [ ] 2.6.3 引用检查：每次引用先查 `refs.bib`；找不到则换同类或删句，避免悬空引用
- [ ] 2.6.4 篇幅检查：每完成一大块运行 `wc -w Section/Related_Work.tex`，不足则回对应小节增段落（失败模式/子群差异等）

### 阶段 3A：问题定义扩写（目标：尽可能详尽，≥6-8 页）



- [X] 3A.1 **符号定义与数学基础 (Mathematical Notation)**:

    - [X] 建立统一的符号体系表：特征空间 $\mathcal{X} \in \mathbb{R}^d$，标签空间 $\mathcal{Y}$，源域 $\mathcal{D}_S$，目标域 $\mathcal{D}_T$。

    - [X] 定义表格数据的特殊结构：异构特征（数值/类别）、缺失值掩码 $M$、特征子集 $\mathcal{F}$。

- [X] 3A.2 **表格数据生成过程 (Tabular Data Generation Process - PFN Perspective)**:

    - [X] 形式化描述 TabPFN 的先验假设：将数据集视为从先验分布 $P_{prior}(\mathcal{D})$ 中采样的结果。

    - [X] 定义 PFN 的元学习目标：在合成数据集上最大化后验概率 $P(y_{new} | x_{new}, \mathcal{D}_{train})$。

    - [X] 阐述为何这种生成式视角适合小样本学习（In-context Learning 能力）。

- [X] 3A.3 **域偏移问题的形式化 (Formalizing Domain Shift)**:

    - [X] 数学定义源域与目标域的分布差异：$P_S(X,Y) \neq P_T(X,Y)$。

    - [X] 细分偏移类型并给出数学描述（结合具体临床案例）：

        -   **Covariate Shift**: $P_S(X) \neq P_T(X)$ 但 $P_S(Y|X) = P_T(Y|X)$。*案例：CT 扫描仪差异（Sharp vs Smooth Kernels）导致的纹理特征分布平移。*

        -   **Label Shift**: $P_S(Y) \neq P_T(Y)$。*案例：三级医院（癌症中心，患病率 ~60%）与社区筛查（患病率 ~5%）的先验差异。*

        -   **Concept Shift**: $P_S(Y|X) \neq P_T(Y|X)$。*案例：地域性病理干扰，如 Ohio River Valley 的组织胞浆菌病（Histoplasmosis）或亚洲的结核病（TB）结节模仿恶性肿瘤特征，改变了 $P(Y|X)$。*

    - [X] 引入 Ben-David 的泛化误差上界定理，说明 $\epsilon_T(h) \leq \epsilon_S(h) + d_{\mathcal{H}\Delta\mathcal{H}}(S, T) + \lambda$，论证减小域间距离 $d(S, T)$ 的必要性。

- [ ] 3A.4 **现有模型的理论局限性分析 (Theoretical Constraints of Existing Models)**:

    - [ ] **GBDT (XGBoost/LightGBM)**: 分析其在小样本下的过拟合风险，以及**不可微性 (Non-Differentiability)** 导致无法直接应用梯度反向传播类 DA 方法；指出树模型在特征空间外推（Extrapolation）能力的缺失。

    - [ ] **Deep Tabular Models (TabNet/FT-Transformer)**: 分析其**数据饥饿 (Data Hunger)** 特性（在 $N \approx 300$ 时难以收敛）以及缺乏表格归纳偏置（如旋转不变性在表格数据中的不适用性）。

- [X] 3A.5 **迁移成分分析 (TCA) 的优化目标**:

    - [X] 定义再生核希尔伯特空间 (RKHS) 中的最大均值差异 (MMD)。

    - [X] 形式化 TCA 的优化问题：最小化 $\text{tr}(K L K) + \mu \text{tr}(K)$，同时保持数据方差 $\text{tr}(K H K)$。

    - [X] 说明如何将 TabPFN 的上下文嵌入 (Contextual Embeddings) 作为 TCA 的输入核矩阵 $K$。

- [X] 3A.6 **PANDA 框架的数学统一**:

    - [X] 定义 PANDA 为三阶段函数复合：$f(x) = (h \circ \psi \circ \phi)(x)$。

        -   $\phi$: TabPFN 编码器（Feature Extractor）。

        -   $\psi$: TCA 适配层（Domain Adapter）。

        -   $h$: 最终分类器（Classifier）。

    - [X] **形式化递归特征消除 (RFE) 算法**:

        -   参考 `predict_healthcare_RFE.py`，定义特征重要性评分函数 $\mathcal{I}(f; \mathcal{D}_S, \phi)$ 为基于 TabPFN 的 Permutation Importance。

        -   数学化描述 RFE 的迭代过程：$S_{k-1} = S_k \setminus \{ \text{argmin}_{j \in S_k} \mathcal{I}(f_j) \}$。

- [ ] 3A.7 **临床挑战与 PANDA 组件映射表**:

    - [ ] 插入 "Table: Mathematical Mapping of Clinical Problems to PANDA Components"，明确 Scanner Variance $\to$ Covariate Shift $\to$ TCA；Referral Patterns $\to$ Label Shift $\to$ Ensemble/Temperature Scaling；Biological Confounders $\to$ Concept Shift $\to$ RFE。

- [X] 3A.8 **医疗场景的约束条件**:

    - [X] **小样本约束**: $N_S, N_T \ll 1000$，导致大参数模型过拟合风险。

    - [X] **隐私与联邦约束**: 目标域标签 $Y_T$ 不可见（Unsupervised DA），且源域数据不能直接传输（虽然本论文主要处理集中式 DA，但可提及隐私隐含约束）。

    - [X] **不平衡约束**: 定义不平衡比率 $\rho = N_{neg}/N_{pos}$，引入加权损失函数形式。
- [ ] 3A.7 **对齐 Dissertation Problem Formulation Expansion.pdf问题设定的缺口**:
  - [ ] 在 Problem Formulation 增补“盲部署”场景（隐私/无标签目标域）与实际样本规模（$n_s \approx 295$, $n_t \approx 190$），明确 i.i.d. 假设失效。
  - [ ] 写出 schema mismatch 三分法表格（交集/源特有/目标特有）及映射函数 $\Phi_{schema}$，并给出临床含义示例。
  - [ ] 为 covariate/label/concept shift 各加至少 1 个肺结节临床例子（扫描协议、TB 高发、转诊率差异），强调对误差和重要性权重的影响。
  - [ ] 补充 GBDT 与深度表格模型在小样本/跨域下的失败分析，作为选择 TabPFN 的动机。
  - [ ] 明确 PANDA Stage 4 多视图+温度缩放 ensemble（Raw/Rotated/Quantile/Ordinal × 多随机种子）及其针对标签漂移/方差的作用。
  - [ ] 在 TCA 描述中写出线性核假设、TabPFN embedding 构造 $K$、分段 $L_{ij}$ 定义，并将其与 Ben-David 界中 $d_{\mathcal{H}\Delta\mathcal{H}}$ 收缩挂钩。
  - [ ] 增补 open-world/输入范围安全约束与隐私合规表述，删除或弱化复杂度推导，保持仅给实践可行性结论。
  - [ ] 添加“临床挑战 ↔ PANDA 组件 ↔ 理论依据”对照表的写作占位。

### 阶段 3C：解决方案详解扩写（目标：4-6 页）

- [ ] 3C.1 **PANDA 架构蓝图 (Architectural Overview)**:
    - [ ] 绘制并描述 PANDA 的整体数据流图（Data Flow Diagram）：从异构原始数据输入，经由 Feature Alignment & RFE，进入 TabPFN Encoder，再通过 TCA Projector，最终由 LogReg/ProtoNet 分类。
    - [ ] 重点描述数据在各模块间的流转逻辑，而非具体的张量维度。
- [ ] 3C.2 **核心组件 I：基于 TabPFN 的特征提取器 (TabPFN as a Feature Extractor)**:
    - [ ] 详细解释为何选择 TabPFN 作为 Backbone：不仅是分类器，更是强力的特征提取器。
    - [ ] **代码对应**：明确引用 `src/tabpfn/classifier.py` 中的 `get_embeddings` 方法，说明这即是“切除分类头、提取 Transformer 上下文嵌入”的工程实现。
    - [ ] 理论论证：预训练先验 $P_{prior}$ 如何帮助模型在医疗小样本上快速收敛并提取鲁棒特征。
- [ ] 3C.3 **核心组件 II：跨域 RFE 特征选择机制与 Cost-Effectiveness Index**:
    - [ ] 详述 RFE 的具体实施步骤：基于 TabPFN 的 Permutation Importance 进行迭代剔除。
    - [ ] **关键新增**：形式化定义 **Cost-Effectiveness Index** 优化目标:
        $ \mathcal{F}^* = \arg\max_{k} ( w_1 S_{perf}(k) + w_2 S_{eff}(k) + w_3 S_{stab}(k) + w_4 S_{simp}(k) ) $
    - [ ] 定义四个子项：$S_{perf}$ (AUC/Accuracy), $S_{eff}$ (Efficiency), $S_{stab}$ (Stability, 1-CV), $S_{simp}$ (Sparsity, $\exp(-\alpha k)$)。
    - [ ] 关联代码实现的细节（如 `predict_healthcare_RFE.py` 中的参数设置、交叉验证策略）。
- [ ] 3C.4 **核心组件 III：隐空间 TCA 适配 (Latent Space TCA Adaptation)**:
    - [ ] 深入解释为何在 TabPFN 的隐空间（Latent Space）而非原始特征空间做 TCA。
    - [ ] 阐述 TCA 如何通过核矩阵 $K$ 对齐源域和目标域的分布，并写清线性核 $K_{ij} = \langle \phi(x_i), \phi(x_j) \rangle$ 的选择理由（TabPFN 已处理非线性）。
    - [ ] 定义 MMD 矩阵 $L_{ij}$ 的分段形式。
- [ ] 3C.5 **核心组件 IV：多视图集成与校准 (Multi-Branch Ensemble & Calibration)**:
    - [ ] 描述 **4-Branch Preprocessing Strategy**（参考代码 `preprocessing/uda_processor.py`）：
        1.  **Raw**: 原始分布。
        2.  **Rotated**: 循环特征置换（打破 Transformer 位置偏置）。
        3.  **Quantile Transformed**: 映射至 $N(0,1)$（应对 Covariate Shift，对齐量纲）。
        4.  **Ordinal Encoded**: 处理类别特征漂移。
    - [ ] 定义 **Temperature Scaling** 与平均机制:
        $ \hat{p}(y=1|x) = \frac{1}{B \times S} \sum_{i=1}^{B \times S} \sigma\left(\frac{z_i(x)}{T}\right) $
        其中 $T=0.9$ (参考 `src/tabpfn/classifier.py`)，$B=4$ (分支数)，$S=8$ (种子数)。
- [ ] 3C.6 **理论论证与挑战应对 (Theoretical Justification & Challenge Addressing)**:
    - [ ] 对照 Problem Formulation 中的挑战（小样本、Covariate Shift、Label Shift），逐一说明 PANDA 的组件如何解决这些问题。
    - [ ] 论证 PANDA 框架的通用性：不仅适用于肺结节分类，也可扩展至其他医疗表格数据任务（如 TableShift）。
    - [ ] **实时推理可行性**: 给出实验中的实际推理耗时（ms级），证明满足临床实时要求。

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
