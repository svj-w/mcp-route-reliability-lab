# Resume bullets

Use only claims supported by the checked-in repository. Update the numbers after a
future V1.0 release; do not mix projected V1.0 scale with measured V0.1 results.

## English

- Built an open-source Python evaluation toolkit for MCP tool routing with strict
  Pydantic contracts, label-free router inputs, provenance-gated external prediction
  imports, calibration/Test separation, and fail-closed policy binding across router,
  model, corpus, feature, and score versions.
- Curated and documented an exploratory bilingual benchmark of 20 licensed tools and
  100 Chinese/English requests; recorded per-tool source/license transformations and
  published deterministic BM25 and pinned multilingual-embedding artifacts.
- Implemented ranking, calibration, selective-risk, grouped-bootstrap, robustness,
  and failure-slice reporting; V0.1 produced 12 checksum-verified artifacts per
  baseline and deliberately marked small-sample policies unavailable instead of
  claiming a production threshold.
- Added a typed Typer CLI and 157-test quality suite with strict mypy and Ruff gates;
  measured warm CPU runs at 8.659 s for BM25 and 55.312 s for the embedding baseline
  on an Intel Core i5-13500H laptop.

## 中文

- 独立开发 MCP 工具路由可靠性评测框架：使用 Pydantic 严格数据契约隔离金标，支持外部路由结果的
  哈希溯源校验，并将策略绑定到路由器、模型、工具语料、特征版本及分数语义，绑定不一致时拒绝执行。
- 构建并开源 V0.1 中英双语基准，覆盖 20 个有明确许可证的工具与 100 条查询；记录逐工具来源、
  许可证和转换过程，发布 BM25 与固定模型版本的多语种 embedding 可复现实验产物。
- 实现 Top-K/MRR、Brier/ECE、错误 AUROC、Risk-Coverage、分组 Bootstrap、鲁棒性和错误切片报告；
  两个基线各输出 12 个 checksum 可验证产物，并明确将小样本策略标记为不可用，避免夸大生产安全性。
- 建立 Typer CLI、157 项自动化测试、mypy strict 与 Ruff 质量门；在 i5-13500H CPU 笔记本上实测
  暖缓存完整运行耗时 BM25 8.659 秒、embedding 55.312 秒。
