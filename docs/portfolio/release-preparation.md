# 本地发布准备记录

> 2026-10-09 发布更新：项目已公开上传至
> [svj-w/mcp-route-reliability-lab](https://github.com/svj-w/mcp-route-reliability-lab)。
> 下方“尚未发布”和“未配置远程”描述的是首次推送前的验收快照。
> 当前发布与 CI 修复记录见本文末节。

本轮目标已完成：修复真实配对稳健性评估、重建两份 V1 AI experimental
报告，并将项目隔离为可单独发布的 Git 仓库。尚未向 GitHub 发布。

## 数据与结果边界

这是 AI 标注实验，不是独立人工复核后的稳定 V1.0，也不是生产安全认证。
原始标注、最终数据和划分均未修改；112 个基础标签仍为 generator-provisional。
两种路由器都没有获得可用的 5% 风险策略，不能声称达到该安全目标。

| 检查项 | 本轮结果 |
|---|---|
| Test | 100 个基础查询 + 90 个变体，共 190 条 |
| 配对 | 90 对，paraphrase / cross_language / noise 各 30 对 |
| BM25 整体 Top-1 flip | 33/90 = 0.3667 |
| MiniLM 整体 Top-1 flip | 36/90 = 0.4000 |
| Decision flip | 不可用：JSON 为 null，报告为 not_available，不伪造为 0 |
| Report / metrics / figure | 使用同一组配对结果；逐对独立核算已通过 |
| 产物校验 | 两份 manifest 各 12 个 SHA-256 校验和均通过 |

90 个变体复用 30 个基础意图；190 条 Test 记录不是 190 个独立意图。
准确率和置信度变化均为“变体减基础”。配对计算没有重新拟合 Calibration
模型，没有根据 Test 调参；每个完成的运行均记录一次冻结 Test 评估。

数据集规范化哈希：
`sha256:3e5f030f545dad8386ae98e688d81d16640fe03c7b5838291f9baa5741b7b478`。
Tool Corpus 哈希：
`sha256:8983296fa7cc6343c94a11fe15307e87f3bf5c6f03bcd036e971b9e9e18de069`。

## 源代码与 Git 隔离

项目根目录现在是 `agent/`，分支为 `main`，未配置远程仓库。
导入了 30 个与本项目相关的历史提交，保留作者、日期和消息；因为移除了
外层 `agent/` 路径前缀，提交 ID 相应重写。不相关的桌面项目没有被导入。

隔离前的父仓库 HEAD：`b9021cbd090f6f1ad97219e1ce2819bbd0b41e84`。
其 `HEAD:agent` 子树：`b7800c257ba024b7bbdd07f7a8c80ce08d3a85ad`。
导入后的初始项目 HEAD：`dbd24727e8a5b98d2183c6796c46a7cc709114e5`。
导入前后 179 个受版本控制文件的路径和 Git 对象 ID 完全一致，原父仓库
没有被删除或重写；本轮只在独立项目仓库中产生后续提交。
验收时父仓库已因不相关的 `dsh/` 项目提交前进到
`7d3ea55bfdd971f18887031f025610028656e879`；该提交不属于本轮操作，
没有回退或覆盖它。上面的迁移校验使用隔离开始时的快照。

两份报告实际生成于干净提交
`6319120606b19f34802d0c9cad4dbf65866921cd`，manifest 和三个策略文件均绑定
该完整 SHA。随后修复的 PowerShell 特殊字符引号只影响命令展示，不改变
这两次计算；当前普通路径的复现命令也不受影响。报告不会把后来提交冒充
为当时运行的源代码。历史报告仍可从 Git 历史恢复。

## 复现

在仓库根目录安装 README 所列依赖；embedding 额外依赖和模型首次下载
需要联网，已缓存模型后可离线运行。不要覆盖已有输出目录。

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:OPENBLAS_NUM_THREADS='2'
$env:TOKENIZERS_PARALLELISM='false'
# 仅在所需模型已经完整缓存时设置：
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'

python -m mcp_route_reliability.cli benchmark --config configs/v1-ai-bm25.yaml `
  --tools data/v1-ai-experimental/tools.json `
  --queries data/v1-ai-experimental/queries.jsonl `
  --run-dir runs/reproduce-ai-experimental-bm25 --json
python -m mcp_route_reliability.cli benchmark --config configs/v1-ai-embedding.yaml `
  --tools data/v1-ai-experimental/tools.json `
  --queries data/v1-ai-experimental/queries.jsonl `
  --run-dir runs/reproduce-ai-experimental-embedding --json
```

模型：`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`，固定 revision
`e8f8c211226b894fcb81acc59f3b34ba3efd5f42`；本次 sentence-transformers 5.7.0、CPU。
运行其他依赖版本记录在各自 manifest 中。

相比修复前报告，390 条查询的候选工具及排序全部不变，排名指标完全一致。
BM25 预测文件字节相同；MiniLM 的 1,950 个候选分数中 1,824 个存在极小差异，
最大绝对差为 `1.4758793615010646e-7`。本轮改变了 CPU 线程设置；差异与浮点
计算顺序变化相符，但未单独做因果实验，不能保证跨环境逐字节一致。
重新生成后的实际分数、Calibration 模型和指标作为本轮完整产物保存。

## 验收与发布边界

本地完整 pytest：187 项通过；Ruff 通过；mypy 检查 32 个源模块通过。
两份产物共 24 个校验和及配对独立核算通过，README 和求职文档中的
42 个本地链接目标均存在。独立本地克隆再次通过 187 项测试、Ruff、mypy
以及全部校验和与链接检查；确认导入的是克隆中的源代码，复用了当前机器
已安装的依赖。这验证了不依赖外层桌面仓库，但不是在全新操作系统上安装
依赖的测试。
PowerShell 回归测试真实
调用原生 Python，验证含空格、单引号、美元符号和反引号的路径不会被拆分
或展开；也覆盖以 `@` 开头的配置文件名。有限的常见凭证格式扫描未发现
命中，但它不是完整的安全审计。

可审阅 [BM25 报告](../../reports/v1-ai-experimental/bm25/reliability-card.md)、
[MiniLM 报告](../../reports/v1-ai-experimental/embedding/reliability-card.md)、
[离线演示](demo.md) 和 [面试说明](interview-zh.md)。

下一步是选择目标 GitHub 仓库及公开/私有权限。没有这些信息，不配置 remote、
不推送、不创建线上 release。发布时保持“AI experimental”标签，不改称稳定
V1.0；独立人工复核仍是另一个尚未满足的发布门槛。

## GitHub 发布与 CI 兼容修复

首次公开推送保留了 36 个项目提交，线上 main 与本地
`f80972f05abffc4dc33887ae3623cb11c1fef837` 一致。origin 使用同一仓库的
HTTPS 地址；Git 登录通过官方 Git Credential Manager 完成，密码和令牌
不写入项目文件。

首次 CI 暴露了两处跨环境问题：Python 3.12 的依赖类型存根需要 3.12
解析，但 mypy 项目默认目标仍是 3.11；候选 CSV 默认 CRLF 行尾则与 Git
保存的 LF 不同。CI 的 424 行 CSV 差异去掉回车后全部相同。

用户确认的最小修复仅让 CI mypy 跟随 Python 版本矩阵，并让四个候选
CSV writer 固定生成 LF，保留 UTF-8/BOM、字段顺序和空白复核栏。
新增测试在独立临时根目录重建候选快照，检查全部文件内容和 CSV 行尾。
原始标注、已提交数据、冻结报告、依赖范围和报告来源 SHA 均未变更。

修复后本地 Python 3.11 验收：189 项测试、Ruff、32 个源模块的 mypy、
production-profile 候选校验、离线 fixture benchmark 均通过；独立只读
复核没有阻塞意见。本机未运行 Python 3.12，其验收必须由线上矩阵完成。
线上结果查看 [GitHub Actions](https://github.com/svj-w/mcp-route-reliability-lab/actions)。
