# EchoMind 文档中心

按“了解产品 → 查看方案 → 核对效果 → 运行代码”的顺序阅读。

## 快速入口

| 阅读目标 | 文档 | 重点 |
|---|---|---|
| 看产品如何使用 | [演示与场景](PRODUCT_DEMO.md) | 视频、界面、输入和输出 |
| 理解产品决策 | [产品案例](AI_PRODUCT_CASE_STUDY.md) | 场景、需求、取舍与迭代 |
| 查看功能定义 | [需求与验收](PRODUCT_REQUIREMENTS.md) | 用户故事、流程、异常行为与指标 |
| 核对实验结果 | [评测总览](EVALUATION_COVERAGE.md) | 数据、版本与各层验证 |
| 理解工程实现 | [工程案例](PORTFOLIO_CASE_STUDY.md) | 路由、证据、记忆、工具与性能 |
| 本地运行 | [完整使用指南](完整使用指南.md) | 配置、启动、接口和排障 |

## 产品方案

- [产品案例](AI_PRODUCT_CASE_STUDY.md)：从场景问题到方案、评测与产品决策。
- [需求与验收说明](PRODUCT_REQUIREMENTS.md)：功能、用户流程、异常状态与验收条件。
- [演示说明](PRODUCT_DEMO.md)：88 秒视频及典型操作。
- [产品验证方案](PRODUCT_VALIDATION_PLAN.md)：五个固定任务、参与者观察与指标。

## 工程实现

- [实现与贡献范围](IMPLEMENTATION_SCOPE.md)：原有模块、本仓库改进与源码对应。
- [工程案例](PORTFOLIO_CASE_STUDY.md)：关键机制与验证。
- [系统架构](架构图.md)：服务、Agent、工具、记忆、存储与部署关系。
- [业务流程](业务流程说明.md)：请求、路由和回复生成。
- [技术能力](EchoMind定位与技术亮点.md)：各模块用途与入口。
- [完整使用指南](完整使用指南.md)：本地环境、模型配置与排障。
- [变更记录](CHANGES.md)：按时间记录主要交付。

## 评测与优化

| 主题 | 入口 | 数据或范围 |
|---|---|---|
| 意图识别 | [v2 首次结果](../backend/evaluation/reports/intent_test_v2_first_run_20261026.md) | 95 条、19 类冻结合成样本 |
| 中文检索 | [MiniLM/BGE 对照](../backend/evaluation/reports/retrieval_test_v1_first_run_20261026.md) | 60 条查询，36 条有答案 |
| 证据与回答 | [Grounded Chat v2](../backend/evaluation/reports/grounded_chat_test_v2_first_run_20261026.md) | 24 条支持与缺口样本 |
| 完整回答覆盖 | [60 条汇总](../backend/evaluation/reports/answer_coverage_60_v1_summary_20261106.md) | 五类场景首次目标回答 |
| 复合诉求 | [分项取证](COMPOSITE_EVIDENCE_ROUTING_20261106.md) | 四条已知问题定向回归 |
| 性能与 Token | [组件分析](PERFORMANCE_TOKEN_PROFILE_20261106.md) · [正式配对报告](../backend/evaluation/reports/performance_ab_4cases_20261106.md) | 四题、两组隔离对照 |
| 评分校准 | [方案](EVALUATION_CALIBRATION.md) · [结果分析](../backend/evaluation/reports/judge_calibration_v1_findings_20261104.md) | 双人评分，16/8 切分 |
| 历史综合链路 | [2026-10-26 结果](../backend/evaluation/reports/e2e_current_top3_20261026.md) | 7 场景、8 次请求 |
| 数据规则 | [意图规范](意图评测数据规范.md) · [RAG 协议](RAG评测协议.md) | 标注、冻结与复核 |
| 向量迁移 | [迁移说明](EMBEDDING_MIGRATION.md) · [早期开发实验](EMBEDDING_BENCHMARK.md) | 模型、collection 与回退 |
| 运行检查 | [2026-11-04](REPOSITORY_VERIFICATION_20261104.md) · [任务分流](TASK_ROUTING_VERIFICATION_20261105.md) | 历史配置与分项回归 |

完整历史报告保留在[评测报告目录](../backend/evaluation/reports/)。每份报告注明对应的任务、代码和样本，便于对照版本变化。

[返回项目首页](../README.md)

