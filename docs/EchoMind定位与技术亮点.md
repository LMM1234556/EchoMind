# EchoMind 定位与技术能力

EchoMind 是面向产品咨询、技术故障、账单售后与人工交接的多 Agent 客服系统。产品以证据回答、诉求覆盖和运行可观察为核心，使用分层评测驱动迭代。

## 能力结构

| 模块 | 具体实现 | 作用 |
|---|---|---|
| 意图识别 | LLM、字符 n-gram 相似度、关键词 Pattern 融合；19 类意图 | 理解问题并为路由提供结构化信息 |
| Agent 编排 | General、Technical、Billing、Escalation；主辅职责和并行处理 | 按场景组织工作流 |
| 中文 RAG | BGE、ChromaDB、快速关键词路径与增强召回 | 找到相关候选文档 |
| Evidence Selector | 每个子查询最多三篇候选、标题归属与充分性检查 | 确认回答依据 |
| 复合取证 | 明确子句拆分、并行选证、分项支持状态 | 覆盖同一消息中的多项诉求 |
| 内部工具治理 | MCPToolManager 参数、缓存、超时、熔断、降级 | 统一工具执行与错误处理 |
| Skills | Markdown/JSON 规则、按职责注入、热加载 | 分离处理规范与业务事实 |
| 记忆 | Redis 近期对话与摘要；ChromaDB 历史和用户画像 | 支持多轮理解与上下文管理 |
| 可观测 | Trace、Metrics、工具来源和阶段耗时 | 定位质量与运行问题 |
| Token 策略 | 结构化任务关闭思考，最终 Agent 保留默认配置 | 按任务分配模型计算量 |
| 评测 | 冻结集、人工量表、Judge 校准与自动化回归 | 对照效果并确定改进优先级 |

## 验证结果入口

- 中文检索：同一冻结集上的[MiniLM/BGE 对照](../backend/evaluation/reports/retrieval_test_v1_first_run_20261026.md)。
- 回答覆盖：五类场景的[60 条首次目标记录](../backend/evaluation/reports/answer_coverage_60_v1_summary_20261106.md)。
- 复合诉求与效率：[四题性能配对](../backend/evaluation/reports/performance_ab_4cases_20261106.md)。
- 评分工具：[24 条双人参考与 Judge 校准](../backend/evaluation/reports/judge_calibration_v1_findings_20261104.md)。
- 运行机制：[205 项自动化检查](../backend/tests/)。
- 使用体验：[产品验证方案](PRODUCT_VALIDATION_PLAN.md)。

## 阅读路径

[产品案例](AI_PRODUCT_CASE_STUDY.md) → [需求与验收](PRODUCT_REQUIREMENTS.md) → [工程实现](PORTFOLIO_CASE_STUDY.md) → [评测总览](EVALUATION_COVERAGE.md)。

原有模块、本仓库改进和协议适配规划见[实现范围](IMPLEMENTATION_SCOPE.md)。
