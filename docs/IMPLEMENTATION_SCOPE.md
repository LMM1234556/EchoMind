# EchoMind 实现与贡献范围

更新日期：2026-11-07。

当前版本在原 EchoMind Python 后端和 Vue 前端上完成中文知识检索、证据回答、运行追踪与评测迭代。本文将基础模块与本仓库改进对应到源码和报告。

## 1. 模块与改进

| 模块 | 原有基础 | 本仓库改进与验证 |
|---|---|---|
| Agent 与意图 | 四类 Agent；LLM、相似度、Pattern 融合；主辅路由 | 完善结构化调用、路由与 Trace；95 条、19 类冻结样本 |
| RAG | ChromaDB、查询改写、并行召回与 LLM 重排 | 本地中文 BGE、快速检索、证据门控；60 条冻结检索对照 |
| 证据回答 | 知识片段注入上下文 | Top-3 Selector、来源校验、知识缺口处理与复合诉求分项取证 |
| 工具与 Skills | 参数检查、缓存、超时、熔断、Skills 加载 | 调用来源、错误分类、降级语义及回归 |
| 分层记忆 | Redis 工作记忆；ChromaDB 历史与画像 | 组合过滤、会话归属、压缩顺序和存储异常处理 |
| 任务分流 | 消息写入与多轮历史 | 显式订单信息记录，混合问题保留取证路径 |
| 评测 | 基础分类和回答评分框架 | 冻结报告、双人评分与 Judge 校准、60 条覆盖验收 |
| 性能 | 多阶段模型调用 | 三类结构化任务关闭思考、组件用量归因、隔离配对对照 |
| 产品交付 | 对话工作台 | 需求与验收、产品/工程案例、演示视频、文档导航 |

维护者负责场景与需求梳理、方案取舍、开发任务拆分、代码审阅、样本复核、验证和结果分析，使用 Codex 辅助开发与迭代。[来源说明](../NOTICE.md)保留上游署名与展示授权信息。

## 2. 当前关键实现

### 意图相似度与知识库向量

[意图识别器](../backend/core/intent_recognizer.py)的本地相似度路径使用字符 n-gram；[知识库](../backend/mcp/knowledge_base.py)使用 BAAI/bge-small-zh-v1.5。两者分别服务于请求分类与中文文档召回。

### 工具管理

[工具管理器](../backend/mcp/tool_manager.py)沿用 MCPToolManager 名称，负责项目内 handler 注册、参数检查、缓存、超时、熔断和 fallback。当前 Agent 调用通过这一内部接口执行。

标准 MCP 客户端/服务端列为协议适配扩展。2026-11-05 核对的上游分发版本包含相应 SDK 及服务端能力，本仓库按现有 Grounded RAG 链路单独规划接入。

### 证据选择

[Evidence Selector](../backend/core/evidence_selector.py)对每次子查询读取最多三篇候选，选择足以支持该子问题的文档，并校验返回标题属于当前候选。

[复合取证工具](../backend/agents/tools.py)分别处理可识别的子问题，再合并来源与支持状态，最多保留三篇最终证据。生成阶段同时看到支持项和知识缺口，按项回复。

### 思考模式

[统一调用入口](../backend/core/llm_utils.py)按组件应用 Qwen 思考策略。意图识别、证据选择、画像提取默认关闭思考，最终 Agent 使用提供商默认配置。配置可覆盖，效果见[性能配对报告](../backend/evaluation/reports/performance_ab_4cases_20261106.md)。

## 3. 验证入口

| 目标 | 证据 |
|---|---|
| 意图识别 | [95 条冻结结果](../backend/evaluation/reports/intent_test_v2_first_run_20261026.md) |
| 中文检索 | [60 条 MiniLM/BGE 对照](../backend/evaluation/reports/retrieval_test_v1_first_run_20261026.md) |
| 证据与回答 | [24 条 Grounded Chat v2](../backend/evaluation/reports/grounded_chat_test_v2_first_run_20261026.md) |
| 回答覆盖 | [60 条首次请求验收](../backend/evaluation/reports/answer_coverage_60_v1_summary_20261106.md) |
| 复合取证与性能 | [四题隔离配对](../backend/evaluation/reports/performance_ab_4cases_20261106.md) |
| Judge 校准 | [双人参考与 16/8 分析](../backend/evaluation/reports/judge_calibration_v1_findings_20261104.md) |
| 工程回归 | [当前 205 项测试](../backend/tests/) |
| 历史配置与运行 | [2026-11-04 验证记录](REPOSITORY_VERIFICATION_20261104.md) |

## 4. 扩展规划

- 标准 MCP 适配与真实订单、工单接口。
- 更大知识库和政策版本管理。
- 更多类型的性能配对与并发验证。
- 按五个固定任务开展参与者可用性测试。

[产品方案](AI_PRODUCT_CASE_STUDY.md) · [需求与验收](PRODUCT_REQUIREMENTS.md) · [工程案例](PORTFOLIO_CASE_STUDY.md)
