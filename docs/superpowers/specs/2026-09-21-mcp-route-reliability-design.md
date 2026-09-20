# MCP Route Reliability Lab 设计文档

- 状态：已确认设计，等待用户审阅书面规格
- 日期：2026-09-21
- 仓库名：`mcp-route-reliability`
- Python 包名：`mcp_route_reliability`
- 目标周期：4 周，每天投入 2–3 小时
- 目标用户：需要在大量 MCP Tools 中可靠筛选候选工具的 Agent/MCP 开发者
- 项目目的：作为面向 AI 应用开发 / Agent 工程岗位的开源求职项目

## 1. 背景与问题

当 Agent 同时接入几十至上百个 MCP Tools 时，将全部 Tool Schema 塞入模型上下文会增加 Token 成本，也可能降低工具选择准确率。现有项目已经实现 BM25、Embedding、Hybrid、Cross-encoder、LLM Selector、Top-K 路由和基础指标，因此本项目不再重复实现一个通用 Tool Router。

本项目研究更具体、也更缺少可靠答案的问题：

> Agent 什么时候可以相信 MCP Tool Router，什么时候必须拒绝路由或请求用户补充信息？

项目围绕 Tool Router 的可靠性、置信度校准、分布扰动和选择性预测建立公开双语 Benchmark。它提供少量内置路由基线，但核心价值是评测、校准、拒绝决策和失败分析。

## 2. 目标与成功标准

### 2.1 产品目标

1. 提供可复现的中英双语 MCP Tool 路由可靠性数据集。
2. 使用统一 Adapter 评测内置或外部 Router 的排名结果。
3. 测量 Router 在跨语言、改写、噪声、歧义及 Tool Schema 退化下的稳定性。
4. 将 Router 分数转换为可解释的正确概率，并在给定最大错误率时自动选择接受/拒绝阈值。
5. 生成可供工程决策使用的 Reliability Card、校准曲线、风险—覆盖率曲线和失败报告。
6. 通过 SDK、CLI 和只读 Dashboard 提供可运行、可测试、可集成的开源工程。

### 2.2 量化目标

以下是项目目标，不是必须美化或伪造的结果：

- 校准后的 Expected Calibration Error 相比原始 Router 分数下降至少 30%。
- 在已接受查询错误率不高于 5% 时，自动处理覆盖率达到 60% 以上。
- No-match 与应澄清样本的识别 F1 达到 0.80。
- 多特征 Logistic 校准器优于固定最大分数阈值和单一 Margin 阈值。
- 每种扰动均报告性能变化、置信区间和失败类型，不只展示最佳数字。

若目标未达到，最终报告必须保留真实结果并分析原因。

## 3. 非目标

第一版明确不做以下内容：

- 不执行真实 MCP Tool。
- 不生成 Tool 调用参数。
- 不做多 Agent 编排。
- 不做运行时安全网关、权限系统或在线 Router SaaS。
- 不做 MCP Marketplace。
- 不宣称提出新的通用路由算法。
- 不真正支持 Agent Skills，只保留可扩展的数据适配边界。
- 不为复杂 Dashboard 或账号系统牺牲数据、评测和测试质量。

## 4. 竞争定位

现有 `semantic-tool-router` 已覆盖 BM25、Embedding、Hybrid、Cross-encoder、MCP 导入、Top-K 路由、基础 Benchmark、消融实验和 Tool 执行；MCPToolBench++ 等项目也已提供更大规模的 MCP 工具使用评测。

本项目与它们的区别是：

- 关注“错误能否被提前识别”，而不只关注总体准确率。
- 将校准、选择性预测、拒绝路由和风险—覆盖率作为一等能力。
- 建立配对的中英双语、语义改写、噪声和 Schema 退化测试。
- 把模糊查询与 No-match 明确建模为应拒绝场景。
- 支持导入外部 Router 结果，不要求用户改用本项目的路由器。

## 5. 用户流程

典型用户流程如下：

1. 用户导入 MCP Tool Schemas，或使用仓库自带的公开 Tool Corpus。
2. 用户导入带标签的查询，或使用内置 Benchmark。
3. 用户选择内置 Router，或通过 JSON Adapter 导入外部 Router 的 Top-K 结果。
4. 系统验证数据、运行排名、提取置信特征并保存逐条预测。
5. 系统仅使用校准集拟合校准器和选择拒绝阈值。
6. 系统在冻结测试集上计算总体、分组、扰动和选择性预测指标。
7. 系统生成机器可读产物、Reliability Card 和 Dashboard。
8. 用户依据目标最大风险选择推荐阈值，将其用于自己的 Agent 系统。

## 6. 系统架构

```text
MCP Schemas ──> Loader / Validator ──> Tool Corpus
                                             |
Benchmark Queries ──> Dataset Validator ─────+
                                             v
                    Built-in or External Router Adapter
                                             |
                              [(tool_id, score), ...]
                                             v
                               Confidence Feature Extractor
                                             v
                         Threshold / Isotonic / Logistic Calibrator
                                             v
                                  Accept Top-1 / Abstain
                                             v
                    Metrics + Artifacts + Reliability Card + Dashboard
```

### 6.1 模块边界

建议目录：

```text
src/mcp_route_reliability/
├── models/          # Pydantic 数据模型
├── loaders/         # MCP Schema 与数据集加载
├── routers/         # 内置基线和外部结果 Adapter
├── confidence/      # 特征提取、校准器与拒绝策略
├── evaluation/      # 指标、分组分析、Bootstrap 与统计比较
├── perturbations/   # Query 与 Tool Schema 扰动
├── reporting/       # Reliability Card、表格和图表
└── cli.py
```

各模块只通过显式数据模型通信。Dashboard 读取标准实验产物，不复制评测逻辑。

### 6.2 核心数据模型

- `ToolRecord`：稳定 Tool ID、名称、描述、输入 Schema、来源、Commit、许可证及可选领域标签。
- `QueryCase`：查询文本、语言、难度、查询类型、相关 Tool ID 集合、`should_abstain`、基础查询组 ID 和切分。
- `RankedCandidate`：Tool ID、原始分数、归一化分数和排名。
- `RankingResult`：Router、Query、Top-K 候选、延迟、Token、费用和错误状态。
- `ConfidenceFeatures`：最大分数、Margin、Top-K 熵、跨 Router 一致性、词汇覆盖和数据质量特征。
- `CalibratedDecision`：正确概率、阈值、`accept/abstain`、理由和候选 Tool。
- `RunManifest`：数据版本、模型版本、配置、随机种子、环境和产物哈希。

数据模型包含显式 Schema Version，未来升级时通过迁移函数处理，禁止静默解释旧格式。

## 7. Router 与 Adapter

### 7.1 内置基线

- BM25。
- 多语言 Sentence Transformer 余弦检索。
- BM25 与 Embedding 的 Hybrid 融合。
- 可选云端 LLM 对照组。

内置 Router 只是 Benchmark 基线，不追求替代成熟路由项目。

### 7.2 外部 Adapter

统一接口接收查询和 Tool Corpus，并返回有序的 `(tool_id, score)` 列表。第一版提供：

- Python Protocol，供程序内集成。
- JSON/JSONL Adapter，供任意外部 Router 导入预测。
- 一个针对现有语义 Router 输出格式的示例转换器，但不把外部项目作为核心运行依赖。

若外部 Router 不提供可比较的分数，系统仍可评测排名指标，但不会伪造概率校准结果；相关校准实验会明确标记为不可用。

## 8. 数据集设计

### 8.1 Tool Corpus

- 收集 50–80 个真实公开 MCP Tools。
- 仅选取来源和许可证清晰的项目。
- 保存上游仓库、版本或 Commit、许可证和抓取日期。
- 规范化字段，但不把第三方 Tool Schema 错误地重新授权。

### 8.2 基础查询

共 300 条人工标注查询，语言比例约为 60% 英文、40% 中文：

| 类型 | 数量 | 预期行为 |
|---|---:|---|
| 清晰的单工具查询 | 120 | 路由到唯一或主要 Tool |
| 相似工具困难样本 | 60 | 区分语义接近的候选 |
| 多个工具均可接受 | 30 | 任一相关 Tool 均可判对 |
| 信息不足、需要澄清 | 30 | 拒绝路由 |
| 无匹配工具 | 60 | 拒绝路由 |

每条查询包含 `relevant_tool_ids` 和 `should_abstain`。对于多标签样本，Top-1 命中任一有效 Tool 即可判为正确，但排名指标使用完整相关集合。

### 8.3 配对扰动集

从基础查询中分层抽取 100 条，每条构造三个保持标签语义的版本：

- 语义改写。
- 中英文跨语言改写。
- 拼写、口语或轻微表面噪声。

形成 300 条扰动查询，总可执行样本约 600 条。所有扰动必须人工确认标签仍然有效。

### 8.4 Tool Schema 扰动

程序化生成以下条件：

- 隐藏工具名称。
- 截短或移除描述。
- 添加无关冗余信息。
- 注入名称相似但参数不兼容的困难干扰项。

扰动器必须保存变换类型、参数和随机种子。对可能改变真实标签语义的变换，必须人工抽检并排除有歧义的样本。

### 8.5 数据切分与泄漏控制

- 开发集：120 条，用于开发基线和特征。
- 校准集：90 条，只用于拟合校准器和选择拒绝阈值。
- 测试集：90 条，只用于最终报告。
- 同一基础查询及其全部扰动版本必须处于同一切分。
- 按语言、难度、查询类型和 Tool 领域进行分层。
- 测试集结果不得反向用于调参；如发现数据错误，修订后必须提升数据集版本并重新运行全部最终实验。

### 8.6 标注质量

- 项目作者完成全部 300 条基础查询和 300 条扰动查询的标注/复核。
- 第二位人工标注者独立复核约 100 条基础查询，重点覆盖困难、模糊和 No-match 样本。
- 保存两位标注者的原始结果，不覆盖分歧记录。
- 报告 Tool 标签集合的一致程度，并对“路由/拒绝”计算 Cohen's Kappa。
- 所有分歧经讨论形成裁决标签；Dataset Card 如实说明标注人数、抽样方法和局限性。

## 9. 置信度校准与拒绝策略

### 9.1 特征

第一版使用以下可解释特征：

- Top-1 原始及归一化分数。
- Top-1 与 Top-2 的差值和比值。
- Top-K 分数熵与离散程度。
- BM25 与 Embedding 是否选中相同 Top-1。
- 多 Router 排名一致性。
- 查询与 Tool 名称、描述及参数名的词汇覆盖。
- 查询长度、语言、Tool 描述长度及缺失状态。

### 9.2 方法

按复杂度依次比较：

1. 固定最大相似度阈值。
2. 固定 Top-1/Top-2 Margin 阈值。
3. 基于单一分数的 Isotonic Regression。
4. 基于多特征的 Logistic Regression 错误预测器。

所有模型仅在校准集上拟合。第一版不使用复杂深度模型，避免样本量不足时过拟合。

### 9.3 决策语义

校准器输出 Top-1 路由正确的估计概率。系统根据用户配置的 `max_risk` 在校准集上选择阈值：满足错误率约束时最大化覆盖率。

低于阈值时返回结构化拒绝结果：

- `no_match`：语料库中没有足够可信的 Tool。
- `ambiguous`：多个候选过于接近。
- `insufficient_information`：查询缺少完成选择所需的信息。

拒绝原因来自显式规则和特征，不要求 LLM 生成解释。

## 10. 指标与统计分析

### 10.1 路由指标

- Top-1 Accuracy。
- Recall@K。
- MRR。
- NDCG。
- No-match / Clarification Precision、Recall、F1。

### 10.2 可靠性指标

- Expected Calibration Error。
- Brier Score。
- Error-detection AUROC。
- Selective Accuracy / Selective Risk。
- Coverage。
- Risk-Coverage AUC。
- 指定最大风险下的覆盖率。

### 10.3 稳健性与分组分析

- 原始查询与配对扰动的性能差值。
- 中英文与跨语言差异。
- 简单、困难、多标签、模糊和 No-match 差异。
- 不同 Tool Schema 退化条件下的下降幅度。
- 不同 Router 及校准方法的对比。

主要指标使用按基础查询组重采样的 Bootstrap 置信区间，避免把同一查询的多个改写错误地视为独立样本。方法比较使用成对样本，并同时报告效果量和不确定性。

## 11. CLI、SDK 与 Dashboard

### 11.1 CLI

第一版提供：

- `validate`：验证 Tool Corpus、查询、标签、切分和来源元数据。
- `benchmark`：运行 Router、特征提取和基础指标。
- `calibrate`：拟合校准器并选择拒绝阈值。
- `report`：生成 Reliability Card、图表和汇总文件。
- `route`：对单条查询输出 Top-K、校准概率和接受/拒绝决策。

命令采用非零退出码表达失败，并支持结构化 JSON 输出，方便 CI 使用。

### 11.2 SDK

SDK 暴露稳定的 Loader、Router Adapter、Evaluator、Calibrator 和 Reporter 接口。CLI 仅负责参数解析并调用 SDK，不包含独立业务逻辑。

### 11.3 Dashboard

Streamlit Dashboard 只读取标准实验产物，展示：

- Router 与校准器排行榜。
- Reliability Diagram。
- 风险—覆盖率曲线。
- No-match 混淆矩阵。
- 查询与 Schema 扰动对比。
- 可筛选的逐条错误案例。

Dashboard 不负责训练、在线路由、用户账户或数据持久化。

## 12. 实验产物与可复现性

每次运行保存：

- `run-manifest.json`：配置、代码版本、数据版本、模型版本、随机种子和环境。
- `predictions.parquet`：逐条排名、特征、校准概率、决策和延迟。
- `metrics.json`：总体、分组、扰动及置信区间指标。
- `reliability-card.md`：适用范围、推荐阈值、失败模式和限制。
- 图表文件及可选 HTML 报告。

产物包含输入和配置哈希。相同哈希的云端请求可读取缓存；缓存记录模型和请求版本，禁止不同模型之间误用。

## 13. 错误处理与成本控制

- 缺失 Tool 字段、无效 JSON Schema、重复 ID、错误标签和切分重叠均在运行前失败，并指出文件与字段。
- 本地模型加载失败时，不静默替换模型。
- 一个 Router 失败不影响其他 Router 的有效结果；失败状态写入 Manifest。
- API 超时或限流只进行有限重试，仍失败时保存已完成进度。
- 支持从检查点恢复，避免重复 API 费用。
- 云端实验运行前估算 Token 和费用；超过显式预算即停止。
- API Key 只从环境变量读取，不进入日志、缓存、配置或产物。
- 云端模型采用低温度并记录完整模型标识，但报告仍承认其非确定性。

## 14. 测试与 CI

### 14.1 测试层次

- 单元测试：数据模型、指标、Top-K、特征、缓存、费用和阈值选择。
- 性质测试：排名长度、分数顺序、重复 Tool、空语料库、极端分数和损坏输入。
- 小型集成测试：约 10 个 Tools、20 条查询，完整跑通 Benchmark、Calibration 和 Report。
- Golden Tests：固定输入的指标和报告结构不能意外漂移。
- CLI 冒烟测试：验证命令退出码、标准输出和产物。
- Dashboard 加载测试：有效、缺失和损坏产物都得到明确反馈。

### 14.2 CI 原则

- 公共 CI 只运行 CPU 可复现的小型实验。
- CI 不下载大型模型，也不需要秘密 API Key。
- 真实 API 测试只能手动触发。
- Ruff、mypy 和 pytest 必须通过。
- 完整 Benchmark 在本地运行，并通过 Manifest 和哈希验证结果完整性。

## 15. 技术栈与运行环境

- Python 3.11+。
- Pydantic、Typer。
- scikit-learn、pandas、NumPy、PyArrow。
- sentence-transformers。
- Plotly、Streamlit。
- pytest、Ruff、mypy。
- `uv` 管理依赖并提交锁文件。
- GitHub Actions 执行 CI。

开发机器为 Intel i5-13500H、16GB 内存和 RTX 4050 Laptop GPU。默认配置必须可在 CPU 上运行；GPU 仅用于加速，不作为复现要求。

## 16. 开源、隐私与许可证

- 项目原创代码采用 Apache-2.0 许可证，并在仓库根目录和 README 中明确。
- 自行创作的查询、标注指南和 Dataset Card 采用 CC BY 4.0 数据许可证。
- 第三方 Tool Schema 保留原始来源和许可证，不统一改写其授权。
- 数据集不包含 API Key、私人文件、用户聊天或个人身份信息。
- 发布前运行秘密扫描并检查样本与实验产物。
- 项目不是安全认证工具；Reliability Card 不构成对 MCP Server 安全性的背书。

## 17. 四周实施排期

### 第 1 周：基础工程与数据规范

- 初始化 Python 包、CI 和代码质量工具。
- 定义数据模型、Loader、Validator 和产物 Schema。
- 收集并规范化 50–80 个 Tools。
- 编写标注规范并完成约 100 条基础查询。
- 跑通 BM25 最小基线。

### 第 2 周：数据集与路由基线

- 完成 300 条基础查询和固定分组切分。
- 第二标注者开始独立复核。
- 完成 300 条配对扰动查询。
- 实现 Embedding、Hybrid 和 JSON Adapter。
- 实现路由、No-match 和基础分组指标。

### 第 3 周：可靠性与校准

- 完成标注分歧裁决。
- 实现置信特征、阈值、Isotonic 和 Logistic 方法。
- 实现 ECE、Brier、AUROC、风险—覆盖率和 Bootstrap。
- 实现 Tool Schema 扰动。
- 冻结配置并运行主要实验与失败归因。

### 第 4 周：工程化与求职包装

- 完成 CLI、缓存、中断恢复和结构化日志。
- 完成 Streamlit Dashboard 与 Reliability Card。
- 完成单元、集成、Golden 和 CLI 测试。
- 完成 Docker、README、Dataset Card 和复现说明。
- 运行冻结测试集并发布完整结果。
- 准备两分钟演示视频脚本、简历描述和面试讲解材料。

## 18. 优先级与降级方案

交付优先级从高到低为：

1. 数据集和标注质量。
2. 可复现实验及冻结测试纪律。
3. 校准、拒绝策略和可靠性指标。
4. 测试、Dataset Card 和 Reliability Card。
5. Dashboard。
6. 云端 LLM 对照组和界面美化。

若进度落后，首先移除云端 LLM 对照组和 Dashboard 美化；不缩减核心标注复核、不使用测试集调参，也不牺牲真实结果。

## 19. 主要风险与缓解措施

| 风险 | 影响 | 缓解措施 |
|---|---|---|
| 与现有 Router 项目再次重合 | 差异化不足 | 坚持以校准、拒绝、稳健性和外部 Adapter 为核心 |
| 300 条基础查询对复杂模型过少 | 过拟合 | 只使用简单校准器，报告置信区间并限制结论范围 |
| 单一作者产生标注偏差 | 评测不可信 | 第二人工标注者复核 100 条，保存分歧并报告一致性 |
| 生成式扰动改变原意 | 标签错误 | 人工确认所有查询扰动，Schema 扰动抽检并排除歧义 |
| 云端模型费用或不可复现 | 成本与稳定性问题 | 设预算上限、缓存请求，云端基线作为低优先级可选项 |
| Dashboard 消耗过多时间 | 核心实验延误 | Dashboard 只读产物，进度不足时保留静态报告 |
| 第三方 Schema 许可证不清 | 无法安全发布数据 | 只采集来源明确项目，并保存逐项来源与许可证 |

## 20. Definition of Done

项目只有同时满足以下条件才算完成第一版：

- 数据集、标注指南、Dataset Card 和来源清单已公开。
- 第二标注结果、一致性指标和分歧裁决可审计。
- 至少三个 Router 基线及三种校准/拒绝策略可复现运行。
- 冻结测试集报告包含总体、分组、扰动、校准和选择性预测指标。
- CLI 与 Python SDK 可从干净环境安装并运行。
- CPU 小型集成测试、静态检查和 CI 通过。
- Reliability Card 明确适用范围、失败模式和推荐阈值。
- README 包含问题定义、架构、复现命令、主要结果和诚实限制。
- Streamlit Dashboard 能从标准实验产物加载并展示核心结果。
- 仓库中不存在真实密钥或私人数据。

## 21. 后续扩展

第一版稳定后，才考虑以下扩展：

- Agent Skills Adapter。
- Conformal Prediction 或 Set-valued Tool Prediction。
- 更多语言和更大 Tool Corpus。
- 在线漂移监测与持续校准。
- 真实 Agent 任务中的端到端影响评估。
- 向现有语义 Router 项目贡献多语言、No-match 或校准 Adapter。

这些扩展不进入当前四周承诺。
