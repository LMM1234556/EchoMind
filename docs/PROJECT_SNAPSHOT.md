# EchoMind 项目现状

核对日期：2026-11-15。

EchoMind 是面向企业软件客服场景的多 Agent 原型，围绕产品咨询、技术排障、账单售后和人工交接，提供诉求识别、知识查证、回复生成与质量评测。当前交付包括可运行代码、调试工作台、演示视频、产品方案及分层评测报告。

## 产品能力与实现

| 用户需要 | 当前能力 | 实现入口 |
|---|---|---|
| 找到合适的处理职责 | 19 类意图识别，General、Technical、Billing、Escalation 主辅路由 | [实现与贡献](IMPLEMENTATION_SCOPE.md) |
| 一条消息中的问题都得到回应 | 按明确子问题分别查证，合并已支持项与待核实项 | [迭代案例](CASE_COMPOSITE_REQUEST.md) |
| 确认政策回答的依据 | 中文 BGE / ChromaDB 检索、Top-3 Evidence Selector、候选来源校验 | [工程案例](PORTFOLIO_CASE_STUDY.md) |
| 连续交流并保留上下文 | Redis 工作记忆、ChromaDB 历史摘要与画像，按用户和会话读取 | [架构](架构图.md) |
| 扩展工具和业务规范 | 标准 MCP 客户端与知识服务端、角色白名单、动态 Skills | [MCP 接入](MCP_INTEGRATION.md) |
| 看清回答过程与改进方向 | Trace、阶段耗时、评测工作台、四维评分与逐题证据 | [评测工作台](EVALUATION_WORKBENCH_20261115.md) |

维护者在原有 EchoMind 架构上负责场景拆解、方案与验收设计、开发任务拆分、代码审阅、样本复核和结果分析，使用 Codex 辅助实现。基础能力与本仓库改进分别对应到[实现范围](IMPLEMENTATION_SCOPE.md)。

## 已取得的验证结果

| 关注点 | 已有结果 | 结果对应范围 |
|---|---|---|
| 意图分类 | Accuracy 98.95%，Macro-F1 98.94% | [95 条、19 类冻结合成问题](../backend/evaluation/reports/intent_test_v2_first_run_20261026.md) |
| 中文知识检索 | MiniLM → BGE，Hit@1 55.56% → 97.22% | [同一 60 条查询中的 36 条有答案问题](../backend/evaluation/reports/retrieval_test_v1_first_run_20261026.md) |
| 证据与事实 | 严格选证 21/24，事实忠实度人工复核 20/24 | [24 条 Grounded Chat v2](../backend/evaluation/reports/grounded_chat_test_v2_first_run_20261026.md) |
| 完整回答 | 首次覆盖检查 27 条通过、19 条部分完成、13 条未通过、1 条预算停止 | [60 条首次结果，用于选择改进项](../backend/evaluation/reports/answer_coverage_60_v1_summary_20261106.md) |
| 复合问题改进 | 四条已知问题在两组中均经人工确认完成目标 | [四题定向回归及配对报告](../backend/evaluation/reports/performance_ab_4cases_20261106.md) |
| 调用消耗 | 总 Token 36,817 → 22,099，下降 40.0% | 同一四题两组隔离配对；包含后台画像调用，控制组先运行 |
| 评分校准 | 完成 24 条双人评分、证据对齐复核和 16/8 校准分析 | [人工参考用于验收，Judge 辅助定位](../backend/evaluation/reports/judge_calibration_v1_findings_20261104.md) |
| 工程稳定性 | 2026-11-15 完整自动化回归 265 项通过 | [标点、分项取证及界面验证](COMPOSITE_PUNCTUATION_REGRESSION_20261115.md) |

各项结果按对应任务、样本和运行版本阅读。四题配对中平均响应由 82.22 秒降至 34.77 秒；顺序和服务商负载共同影响延迟，后续对照可采用交错顺序与重复运行。

## 最新交付与当前验收状态

- **完整迭代案例已整理**：从复合请求漏答出发，记录分句定位、分项取证、知识缺口表达约束，以及原始失败和两次真实复测。[查看案例](CASE_COMPOSITE_REQUEST.md)。
- **演示结果已留存**：退款审核与到账原题完成一次真实请求；密码与发票标点变体保存修复前失败及两次复测。最新回复均已完成助手核对，项目负责人复核状态保留为待确认。[查看场景状态](KEY_SCENARIO_ACCEPTANCE.md)。
- **问题解决流程设计已完成**：定义诊断、权限确认、处理、结果复测及交接的状态和验收条件。当前实现交付排障指引与交接摘要，任务操作、恢复回查和工单受理列入后续集成。[查看流程](TASK_RESOLUTION_DESIGN.md)。

八项场景清单中原题已执行 1 项，密码与发票变体另有记录；其余原题保留为待执行。当前展示以已留存的回答、Trace 和视频为依据；后续验证由具体改动、失败案例或新增业务需求触发。

## 后续产品推进

1. 在可控沙箱接入一项获授权操作，通过执行回执和功能复测衡量问题是否解决。
2. 按已有五个任务开展真实参与者观察，采集完成情况、等待时间和可信度；当前完成场次为 0。
3. 根据新增问题补充回归，在保持质量验收的基础上比较效率、成本和风险。

[项目首页](../README.md) · [88 秒演示](PRODUCT_DEMO.md) · [评测总览](EVALUATION_COVERAGE.md) · [详细进展](PROJECT_PROGRESS_20261113.md)
