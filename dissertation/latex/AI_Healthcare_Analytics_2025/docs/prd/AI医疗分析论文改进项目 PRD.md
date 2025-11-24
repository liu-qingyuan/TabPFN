# AI医疗分析论文改进项目 PRD

- 项目：将现有医学导向的论文转化为以AI技术为核心的研究文章
- 目标刊物：AI领域顶级会议/期刊（如 NeurIPS、ICML、JMLR 等）
- 截止日期：待定
- 版本：1.0
- 创建时间：2025-11-07
- 状态：进行中

## 🎯 项目目标

- 表格医疗数据上的基础模型创新：强调在表格型医疗数据中引入基础模型（Foundation Model）的技术突破，例如 TabPFN 等表格基础模型在小样本数据集上取得了显著性能超越[1]。基础模型指在大规模通用数据上预训练、可适应多任务的模型[2]。通过将这一范式引入医疗表格数据分析领域，有望大幅提升小数据情境下模型的预测准确性和泛化能力[1]。
- 领域自适应技术贡献：突出本研究在领域自适应/迁移学习（Domain Adaptation）方面的技术创新。具体而言，解决不同医院数据分布差异导致的跨域泛化难题，例如利用迁移组件分析 (TCA) 等方法寻找源域与目标域的共享特征空间，从而最小化域间差异[3]。开发域适应技术使模型能够应对跨医院的数据分布偏移，对于医疗AI模型的稳健性尤为重要[4]。
- 跨机构AI部署策略：制定模型在多医院跨机构部署时的适应策略，确保模型在不同医院环境下保持性能。跨机构应用需要解决数据异质性和分布偏移问题[5]。本项目将探讨如联合学习（Federated Learning）或受监督域适应等方案，促进模型在各医疗机构间的可迁移性和一致性，从而降低单一机构偏差带来的性能下降风险[4]。
- TabPFN为开创性的表格基础模型：将TabPFN定位为本研究方法的核心，并强调其作为表格数据基础模型的先锋地位[6]。TabPFN是近期提出的一种基于生成式Transformer的表格基础模型，在2.8秒内即可完成小样本训练，并超越需要4小时调参的集成基线模型[7]。作为首个在表格小数据任务上全面超越传统方法的基础模型，TabPFN展示了跨领域可迁移学习算法的强大潜力[1][6]。

## 📋 任务追踪（TODO List）

### ✅ 阶段1：分析与规划

- [X] **1.1 当前论文内容分析** – 审查现有论文的医学导向内容，寻找需要加强AI技术侧重点的部分。状态：已完成 (2025-11-07)备注：列出了论文中偏重医学应用的段落，确定了可引入AI方法论阐释的切入点。
- [X] **1.2 AI内容增强方案制定** – 拟定具体的AI技术内容增强建议。状态：已完成 (2025-11-07)备注：确定了四大重点方向（基础模型、域适应、跨机构、TabPFN），形成针对性增强思路。
- [X] **1.3 文档结构与PRD撰写** – 创建文档结构和产品需求文档 (PRD)。状态：已完成 (2025-11-07)备注：建立了docs/文件夹结构，撰写了初始PRD，明确各阶段任务和目标。
- [X] **1.4 提示文档编写** – 编写任务执行指导的提示文档。
  状态：已完成 (2025-11-07)
  备注：已完成详细的提示文档，包括任务说明和PRD更新规范，为后续内容改写提供指南。

### 🔄 阶段2：内容改写与强化

- [X] **2.1 标题优化** – 将论文标题改为突出AI技术的方向。当前标题：“PANDA: Pretrained Adaptation Network with Domain Alignment for Feature-Efficient Cross-Hospital Pulmonary Nodule Classification”。备选方向：突出我们自研框架 **PANDA**（整合TabPFN式表格基础模型、TCA域对齐与RFE特征选择）的标题，例如“PANDA: … foundation … domain alignment … feature selection”等，需兼顾科研严谨性与可发表性。要求：新标题需显式体现PANDA框架名称，并暗含其包含的基础模型（如TabPFN）与域适应/特征选择能力，同时点明跨医院/跨域应用场景。状态：已完成；优先级：高。完成时间：2025-11-13 08:30:00 (Asia/Shanghai)。备注：标题更新为“PANDA: Pretrained Adaptation Network with Domain Alignment for Feature-Efficient Cross-Hospital Pulmonary Nodule Classification”，并在摘要中说明TabPFN式Transformer、RFE特征筛选与TCA域对齐三阶段特性。
- [X] **2.2 摘要强化** – 提升摘要对AI技术贡献的描述力度。要求：摘要应突出以下几点：• 表格数据基础模型创新：强调本研究引入了针对表格数据的基础模型方法，如Transformer架构在表格小数据上的突破[8]，可说明与传统梯度提升树相比，新方法在小数据集上取得显著性能提升[1]。• Transformer架构创新：点明我们使用了Transformer为核心的模型架构，这是AI领域的最新进展之一[9]，其自注意力机制可建模高阶特征交互，克服深度学习处理异构特征的瓶颈[10]。• TCA融合的新颖性：阐述将迁移组件分析 (TCA) 等域适应算法与基础模型相结合的创新之处，传统TCA通过在源域和目标域间学习共同特征子空间来缩小分布差异[3]，本研究首次尝试将TCA嵌入表格基础模型训练，实现跨医院数据的特征对齐和迁移学习融合。• 超越医学应用的AI贡献：强调方法对AI领域的贡献不仅局限于医疗应用，例如可推广到其他小样本表格任务，体现更广泛的AI研究价值[11]，基础模型结合迁移学习的范式将推动表格数据学习在各领域的创新[11]。状态：已完成；优先级：高。完成时间：2025-11-13 08:10:35 (Asia/Shanghai)。备注：摘要重写为强调TabPFN式Transformer基础模型在小样本表格数据上的突破、TCA嵌入式域对齐流程，以及方法对更广泛跨域AI任务的可迁移价值。
- [x] **2.3 引言增强** – 增加AI技术发展的背景和本研究定位。  
  要求：
  • 表格AI演进脉络：梳理从传统机器学习（如决策树、随机森林）到深度表格学习，再到表格基础模型的发展历程[12]，并说明深度学习在表格领域遇到的小样本和异构性挑战[10]，引出基础模型（如TabPFN）通过大量合成数据预训练带来的性能提升[1]。  
  • TabPFN的开创性：介绍TabPFN作为首个表格数据基础模型的意义[6]，指出其采用Transformer架构并在数百万合成数据集上进行元学习，使其成为表格领域“预训练-微调”范式代表[7]，在小样本数据集上明显优于传统方法[1]。  
  • 小样本学习挑战：阐明小数据集对AI模型的挑战，以及基础模型和元学习的重要性，传统深度学习在小型表格数据上因特征类型多样、缺失值、类别不均衡而表现不佳[10]，本研究通过预训练基础模型提高小样本学习效果。  
  • 跨域学习技术定位：指出跨医院数据存在领域偏移，需要专门的跨域学习技术[13][14]，本研究定位于表格基础模型+领域自适应的交叉创新，利用基础模型的通用表示能力并结合域适应方法（如TCA）应对医院间数据差异，填补跨领域表格小数据建模空白。  
  状态：已完成；优先级：高。  
  完成时间：2025-11-13 12:41:54 (Asia/Shanghai)。  
  备注：引言新增“传统→深度→基础模型”演进段落、TabPFN开创性、小样本与跨域挑战的对比，并引用TabNet/Transformer/TabPFN/TCA文献以明确PANDA定位。
- [x] **2.4 方法部分技术加深** – 提升方法论描述的AI理论深度。  
  要求：
  • Transformer在表格数据中的改进：详细说明如何针对表格数据特点改造Transformer架构，解释对数值和类别型特征的处理方式，以及与TabTransformer、FT-Transformer类似的嵌入和注意力机制来应对不同类型特征[15]，并引用Transformer在表格数据上的成功案例[8]。  
  • 与现有表格AI方法对比：增加与其它深度表格模型（TabNet、TabTransformer、FT-Transformer、SAINT 等）的比较，说明本研究方法在架构和理念上的异同，强调我们引入基础模型预训练优势和迁移学习模块，与上述方法形成差异并引用其成果[16][17]。  
  • 集成学习优势理论说明：强调可能融合的集成思路（如多模型集成或模型集成与预训练结合）的理论依据，说明集成方法通过结合多个基模型预测提升泛化能力和鲁棒性，分析其在表格小数据场景下的价值。  
  • 注意力机制的可解释性：讨论Transformer中自注意力机制如何帮助分析特征重要性和交互关系，可视化注意力权重以提供模型决策解释，并提及类似TabNet通过注意力选择特征实现可解释性的案例[19]，凸显对模型透明度的关注。  
  • RFE特征筛选与小样本稳定性：明确PANDA流水线中递归特征消除（RFE）的角色，说明其为何能在58维临床特征中稳健筛出8个核心变量，并对比近年来面向小样本/不平衡/高维稀疏场景的代表性特征选择研究（如AAAI 2023的WPFS、Bioinformatics 2023的GRACES、Neurocomputing 2022的DeepFS）[24][25][26]。需要总结这些工作的机制（辅助权重预测网络、图卷积筛选、深度特征筛选）以及PANDA相较于它们的优势：结合TabPFN式基础模型、RFE包裹式选择与TCA域对齐，在保持可解释性的同时提升跨域泛化。  
  状态：已完成；优先级：高。  
  完成时间：2025-11-13 12:41:54 (Asia/Shanghai)。  
  备注：Methods 已补充Transformer结构定制、与深度表格模型/SAINT的定位对比、注意力可解释性论述，并在RFE部分引用WPFS/GRACES/DeepFS/FSDA等文献强调跨院一致性。
  • 文献支撑要点梳理（如需新增多站点/多模态案例，请先确认具体引用）：
  – 小样本高维必要性：WPFS、GRACES、DeepFS 等工作[24][25][26]表明在 295 例训练且类不平衡背景下执行 RFE 能有效缓解过拟合，呼应“先压缩特征再训练”的策略。
  – 基础模型赋能 RFE：TabPFN 式 Transformer 的注意力与重要性评分[1][6][8][16][19]让我们可以用基础模型输出驱动 RFE，契合“强表征模型 + 特征选择”趋势[15]。
  – 可解释性与部署：可解释 AI 与临床部署研究[16][17][19][28]强调减少至关键特征便于跨院采集、沟通成本，我们的 8 项指标与之保持一致。
  – 跨域稳健性：域适应与多中心研究（TCA[3][5]、跨人群对抗适应[4][13][22]、Informative Feature Selection for Domain Adaptation[27]）支持“挑选跨域共享特征”以增强泛化。
  – 经典与现代结合：特征选择综述及多目标/稳定性工作[2][24][26]提示可结合经典 RFE 与现代深度模型，我们的实现也为未来加入成本或稳定性约束打下基础。
  状态：已完成；优先级：高。
  完成时间：2025-11-13 12:41:54 (Asia/Shanghai)。
  备注：Methods 已补充 Transformer 结构定制、与深度表格模型/SAINT 的定位对比、注意力可解释性论述，并在 RFE 部分引用 WPFS/GRACES/DeepFS/FSDA 等文献强调跨院一致性。
- [x] **2.5 相关工作扩展** – 增加AI方向文献综述的深度和广度。  
  要求：
  • Transformer基础论文：引用“Attention Is All You Need”[9]阐述自注意力网络提出背景，说明Transformer架构已成功应用于各类任务，为模型选择提供依据。  
  • 基础模型综述：加入对基础模型概念的介绍和讨论，如Bommasani等人2021年报告中的定义[2]，说明基础模型在NLP、CV等领域的发展趋势及其在医疗AI中的应用潜力[20]。  
  • 表格深度学习文献：梳理近期在ICML、NeurIPS、ICLR等顶会发表的表格数据深度学习模型，如TabNet (AAAI 2021)[16]、SAINT (arXiv 2021)[21]、TabTransformer (NeurIPS 2020)等，突显本研究与现有工作的关系。  
  • 领域自适应研究：补充医疗AI中领域自适应的相关研究，例如通过对抗训练实现跨人群域适应的工作[22]，说明域间分布差异的普遍存在及现有方法如何缓解，为采用TCA等方法提供文献支持。  
  状态：已完成；优先级：中。  
  完成时间：2025-11-13 13:05:00 (Asia/Shanghai)。  
  备注：Introduction 增设 Related Work 段，系统引用 Attention/TabNet/TabTransformer/SAINT/基础模型综述与域适应文献，明确PANDA与现有工作的差异与联系。
- [x] **2.6 实验结果与讨论强化** – 从AI角度深入讨论结果意义。  
  要求：在结果分析部分增加以下讨论，以凸显AI贡献：
  • 对表格基础模型领域的技术意义：解释实验结果对表格数据基础模型研究的启示，说明基础模型+迁移学习范式在表格领域的可行性[11]，并讨论如何填补现有文献空白及在其他任务上的应用潜力。  
  • 超越特定医疗任务的贡献：强调方法的通用性，说明虽然以肺结节良恶性预测为例，但方法论可推广到其他医疗预测乃至非医疗的小样本分类任务，对AI社区具有广泛价值[11]。  
  • 计算效率与可扩展性：从AI算法角度评估模型效率，比较训练时长和推理速度相对于传统方法的优势[7]，讨论预训练如何减少超参数调优需求，并分析在增大数据规模或特征维度时的可扩展性，引用相关基础模型在扩展性上的研究来支持。  
  • 模型可解释性（AI视角）：从AI研究角度讨论模型的可解释性成果，分析注意力权重或特征选择结果，说明模型关注的重要因素，并与现有可解释AI方法（如决策树、TabNet）对比[19]，强调在不牺牲精度的前提下实现模型透明度。  
  状态：已完成；优先级：中。  
  完成时间：2025-11-13 13:13:49 (Asia/Shanghai)。  
  备注：Results 新增 AI-Oriented Discussion 段，讨论 foundation model + UDA 范式的意义、跨任务通用性、效率与可扩展性、以及注意力/RFE 带来的可解释性，并引用相关文献支撑。
- [X] **2.7 基础模型段落引用与动机校准** – 统一“Foundation models … medical applications”段的叙述，补齐基础模型与医疗域异质性论证。  
  要求：
  • 定义基础模型概念并结合LLM/ViT案例，引用Bommasani等综述与BISE文章[2][20]；  
  • 将医疗数据稀缺、跨机构分布差异与跨人群域适应挑战对应到医学域对抗式迁移研究[4]；  
  • 顺畅引出TabPFN在小样本表格数据上的性能优势[1]，说明为何其最适合作为我们方案的基础模型。  
  状态：已完成；优先级：中。  
  完成时间：2025-11-13 15:05:00 (Asia/Shanghai)。  
  备注：Methods 内 foundation 段改写为“LLM/ViT→医疗域异质性→TabPFN”链路，补齐 \\cite{bommasani2021opportunities,schneider2024foundation,brown2020language,dosovitskiy2020image,guan2021domain,musa2025addressing,hollmann2025accurate} 引用。
- [X] **2.8 TabPFN架构与PANDA贡献澄清** – 修正“TabPFN treats each feature as a token”类表述，使其符合原论文以训练样本为token、在查询样本上一次性推理的设定，并突出PANDA新增模块。  
  要求：
  • 精确描述TabPFN token化流程、查询阶段和few-shot预训练策略，引用Nature论文及表格FM综述[1][6]；  
  • 交代我们在输入归一化、量纲感知嵌入、掩码/旋转增强、跨医院共享编码器与TCA后处理方面的附加贡献；  
  • 对比TabNet/TabTransformer/SAINT等架构如何处理特征序列，说明我们per-feature embedding block的新意与借鉴点[16][21]。  
  状态：已完成；优先级：高。  
  完成时间：2025-11-13 15:07:00 (Asia/Shanghai)。  
  备注：说明TabPFN以上下文样本为token并一次前向推理，列出PANDA的量纲感知嵌入/旋转增强/TCA模块贡献，并加入TabNet/TabTransformer/SAINT对比引用。
- [X] **2.9 可解释性与RFE文献支撑** – 为“Permutation-based RFE rankings”段补强引用并量化解释输出。  
  要求：
  • 引入Explainable feature selection与Permutation重要性研究作为支撑，如Deep Feature Screening、XAI-based FS与Weight Predictor Network论文[24][25][28]；  
  • 说明best7/best8等子集的选择流程、重复试验稳定性以及Permutation重要性如何映射到临床指标；  
  • 联系领域自适应特征选择研究（例如Informative Feature Selection for Domain Adaptation[27]），阐明在跨医院场景中暴露RFE权重的价值。  
  状态：已完成；优先级：中。  
  完成时间：2025-11-13 15:12:00 (Asia/Shanghai)。  
  备注：RFE段加入 \\cite{breiman2001random,guyon2002gene,liu2022deepfs,margeloiu2023weight,zacharias2022designing,luo2021fsda}，并描述best7/best8ΔAUC、跨院稳定性与临床可审计输出。
- [X] **2.10 传统/ML 基线引用与方法说明** – 明确实验对比中各传统与机器学习基线的来源与实现细节。  
  要求：
  • 指出PKUPH、Mayo等临床基线对应的公开模型或风险评分来源，并描述其特征输入与适用人群；  
  • LASSO LR需注明参考He等人提出的多中心肺结节恶性预测模型[he2021novel]，阐述其特征选择策略、阈值与我们实现的差异；  
  • 为SVM、DT/CART、Random Forest、GBDT、XGBoost等ML基线补充标准引用（Cortes & Vapnik 1995、Breiman 1984/2001、Friedman 2001、Chen & Guestrin 2016等），并在正文或附录用一句话说明超参配置；  
  • 在实验章节的表格/图例中确保首次出现时附上引用，避免“无来源的基线”影响审稿人信任。  
  状态：已完成；优先级：中。  
  完成时间：2025-11-13 15:15:00 (Asia/Shanghai)。  
  备注：Baselines 段加入PKUPH/Mayo/LASSO引用（\\cite{li2011development,swensen1997chest,swensen1997archives,he2021novel,tibshirani1996regression}）及SVM/DT/RF/GBDT/XGBoost实现说明和标准引用。

### 🔄 阶段3：质量保障

- [ ] **3.1 AI会议定位检查** – 验证论文内容是否契合AI顶会的期望。要求：逐条核对论文对AI审稿人关注点的满足情况：是否清晰阐明技术创新、引用足够AI文献、强调热点（基础模型、迁移学习等[20]），确保贡献一目了然并符合NeurIPS/ICML等会议投稿要求。状态：未完成；优先级：中。
- [ ] **3.2 技术细节审查** – 全面验证AI技术细节的正确性与严谨性。要求：由独立技术成员审阅方法和实验部分，确保理论推导无误，公式符号定义清晰；与引用算法（Transformer结构、TCA算法等）一致；实验设计合理，评价指标和统计检验符合AI论文规范，对比实现与文献公开基准，保证技术陈述严谨可靠。状态：未完成；优先级：中。
- [ ] **3.3 最终润色** – 面向AI读者的语言和格式优化。
  要求：整体通读论文，优化表述使其专业且流畅，符合英语科技论文风格，确保章节衔接自然、重点突出，检查参考文献格式与目标会议要求一致，插图表格排版清晰，术语一致（如“foundation model”统一译为“基础模型”），避免中英文混杂或不恰当翻译，满足AI与医疗读者的阅读习惯。
  状态：未完成；优先级：低。

## 🎯 关键成功指标

- 技术深度：论文需充分体现AI技术深度，例如对Transformer改进、基础模型预训练、域适应算法等都有深入讨论[8]，让评审感受到AI方法贡献明显高于应用型文章。
- 创新清晰：突出基础模型与域适应融合这一核心创新点，在标题、摘要和结论中反复强调其如何将大规模预训练优势引入表格医疗数据并结合迁移学习解决跨域问题。
- 受众契合：内容和行文风格要符合AI研究社群期待，引用最新AI文献，讨论结果时上升到方法论高度，确保话题切中AI热点（小样本学习、跨域泛化、基础模型等），体现前沿性和学术价值[20]。
- 理论严谨：保证理论推导和实验设计经得起推敲，说明自注意力机制为何适合表格数据、TCA为何能减少分布差异[3]，并采用公认实验方案，做到有新意且严谨。
- 影响阐释：清晰表述本研究在表格数据AI领域的定位和影响，说明填补的空白及推动的进展，例如验证基础模型在表格小数据上的可行性，为后续研究提供新方向[11]。

## 📝 进度追踪笔记

- 当前重点：阶段2全部结题，转入阶段3任务（3.1~3.3）的AI顶会定位校对、技术复核与英语润色。
- 下一步：聚焦阶段3检查项，依次完成AI会议定位自查、方法/实验技术审阅以及最终语言与排版润色。
- 阻塞事项：暂无，进展顺利，如在文献或实验复现上遇到困难将及时反馈协同解决。
- 依赖关系：无显著外部依赖，阶段2子任务需按逻辑顺序进行，章节改写可并行但需保持内容一致性。

## 🔄 更新历史

- 2025-11-07：创建初始PRD，完成阶段1分析规划任务的记录。
- 2025-11-07：整理任务结构为TODO清单格式，设定优先级并分配至各阶段。
- 2025-11-07：阶段1全部完成——已建立文档结构，撰写PRD和提示文档。
- 2025-11-08：修正PRD内容——更新表格AI演进路径描述，去除无关的NLP/视觉内容引用。
- 2025-11-13：PRD修订——融入最新领域研究参考文献，强化基础模型和表格深度学习相关内容，完善各任务细节说明。
- 2025-11-13：根据PANDA框架命名需求重新打开任务2.1，要求标题同时体现基础模型、域适应和RFE特征选择能力。
- 2025-11-13：完成PANDA命名版标题及摘要同步更新，明确TabPFN式基础模型+RFE+TCA域对齐三阶段亮点。
- 2025-11-13：完成引言（2.3）与方法论（2.4）增强，新增TabPFN/TabNet/Transformer发展脉络及RFE与跨域特征选择的文献支撑。
- 2025-11-13：完成引言（2.3）与方法论（2.4）增强，新增TabPFN/TabNet/Transformer发展脉络及RFE与跨域特征选择的文献支撑。
- 2025-11-13：Related Work 段落完成（2.5），补全Attention/TabTransformer/SAINT/基础模型综述与域适应文献引用。
- 2025-11-13：结果+讨论强化（2.6），总结基础模型+域适应范式的技术意义、跨任务通用性、效率与可解释性分析。
- 2025-11-13：完成阶段2新增任务（2.7~2.10），补齐foundation段引用、TabPFN/PANDA描述、RFE解释性与传统/ML基线实现说明。

## 📊 完成度统计

- 阶段1：4/4 已完成 (✅ 100%)
- 阶段2：10/10 已完成 (✅ 100%)
- 阶段3：0/3 已完成 (🚧 0%)
- 总进度：14/17 已完成 (82%)[1]

## 📚 参考文献

[1] [7] [10] [11] [12] [14] Accurate predictions on small data with a tabular foundation model | Nature
https://www.nature.com/articles/s41586-024-08328-6?error=cookies_not_supported&code=3e810af8-1cae-4399-bf14-1b6767f5ee57

[2] [20] Foundation Models | Business & Information Systems Engineering
https://link.springer.com/article/10.1007/s12599-024-00851-0

[3] [5] ancalime.de
https://www.ancalime.de/download/publications/Grubinger2015_DomainGeneralizationBasedOnTransferComponentAnalysis.pdf

[4] [13] [22] Addressing cross-population domain shift in chest X-ray classification through supervised adversarial domain adaptation | Scientific Reports
https://www.nature.com/articles/s41598-025-95390-3?error=cookies_not_supported&code=290481be-bf74-42d4-a2e1-82d68ac4b03d

[6] [8] [15] [17] [2410.12034] A Survey on Deep Tabular Learning
https://arxiv.org/abs/2410.12034

[9] [1706.03762] Attention Is All You Need
https://arxiv.org/abs/1706.03762

[16] [19] [1908.07442] TabNet: Attentive Interpretable Tabular Learning
https://arxiv.org/abs/1908.07442

[21] [2106.01342] SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training
https://arxiv.org/abs/2106.01342

[24] Liu X., "Deep Feature Screening: Feature Selection for Ultra High-Dimensional Data via Deep Neural Networks," arXiv:2204.01682, 2024. https://arxiv.org/html/2204.01682v3

[25] Margeloiu R. et al., "Weight Predictor Network with Feature Selection for Small Sample Tabular Biomedical Data," arXiv:2211.15616, 2022. https://arxiv.org/abs/2211.15616

[26] Chen X. et al., "Graph Convolutional Network-based Feature Selection for High-dimensional and Low-sample Size Data," arXiv:2211.14144, 2022. https://arxiv.org/abs/2211.14144

[27] Luo T. et al., "Informative Feature Selection for Domain Adaptation," Hong Kong University of Science and Technology Technical Report, 2021. https://researchportal.hkust.edu.hk/en/publications/informative-feature-selection-for-domain-adaptation/

[28] Domingos J. et al., "Designing a Feature Selection Method Based on Explainable Artificial Intelligence," Electronic Markets, 2022. https://link.springer.com/article/10.1007/s12525-022-00608-1
