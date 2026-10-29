# EchoMind 文档中心

本目录汇总产品设计、系统架构、工程实现与评测证据。建议根据阅读目标选择以下路径。

## 推荐阅读路径

### HR / AI 产品面试官

1. [AI 产品案例](AI_PRODUCT_CASE_STUDY.md)：用户、场景、产品目标、方案、优先级、指标与迭代；
2. [静态产品演示](PRODUCT_DEMO.md)：无需启动环境即可了解四类核心场景；
3. [当前完整链路报告](../backend/evaluation/reports/e2e_current_top3_20261026.md)：真实请求、路由、证据和回答契约结果。

### 技术 / 架构面试官

1. [系统架构](架构图.md)：服务、Agent、工具、记忆、存储与部署关系；
2. [工程案例](PORTFOLIO_CASE_STUDY.md)：关键工程问题、实现方案和验证方式；
3. [完整使用指南](完整使用指南.md)：配置、启动、接口、数据和排障说明；
4. [业务流程说明](业务流程说明.md)：意图、路由、工具、Skills、记忆与回复生成流程。

### RAG / 评测方向

1. [Embedding 对照实验](EMBEDDING_BENCHMARK.md)：MiniLM 与 BGE 的同集对比；
2. [Embedding 迁移记录](EMBEDDING_MIGRATION.md)：模型、维度、Collection 与回滚策略；
3. [RAG 评测协议](RAG评测协议.md)：数据口径、指标定义和复核流程；
4. [意图评测数据规范](意图评测数据规范.md)：标签、冻结集和审核规则；
5. [评测报告目录](../backend/evaluation/reports/)：意图、检索、证据门控与完整链路报告。

## 文档清单

| 文档 | 内容 |
|---|---|
| [AI_PRODUCT_CASE_STUDY.md](AI_PRODUCT_CASE_STUDY.md) | 面向 AI 产品岗位的完整案例复盘 |
| [PRODUCT_DEMO.md](PRODUCT_DEMO.md) | 产品界面和四类核心场景 |
| [PORTFOLIO_CASE_STUDY.md](PORTFOLIO_CASE_STUDY.md) | 工程改进、验证结果和下一阶段计划 |
| [EchoMind定位与技术亮点.md](EchoMind定位与技术亮点.md) | 项目定位与整体技术能力 |
| [架构图.md](架构图.md) | 六张系统架构图入口 |
| [业务流程说明.md](业务流程说明.md) | 端到端业务和 Agent 处理流程 |
| [完整使用指南.md](完整使用指南.md) | 本地运行、接口和运维说明 |
| [CHANGES.md](CHANGES.md) | 文件级工程变更摘要 |
| [EMBEDDING_BENCHMARK.md](EMBEDDING_BENCHMARK.md) | 中文 Embedding 小型离线对比 |
| [EMBEDDING_MIGRATION.md](EMBEDDING_MIGRATION.md) | Embedding 迁移与回滚说明 |
| [RAG评测协议.md](RAG评测协议.md) | RAG 指标与冻结评测流程 |
| [意图评测数据规范.md](意图评测数据规范.md) | 意图数据集和标注规范 |

## 证据层级

功能与效果结论按以下顺序复核：

1. 当前代码与配置；
2. 自动化测试；
3. 固定开发集与冻结集报告；
4. 真实请求的完整链路记录；
5. 产品与工程文档中的设计说明。

