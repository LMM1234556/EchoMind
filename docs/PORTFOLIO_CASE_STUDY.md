# EchoMind 工程实现与验证

## 请求链路

1. FastAPI 接收消息，读取用户与会话上下文。
2. 意图识别器融合 LLM、字符 n-gram 相似度和 Pattern，输出意图、置信度、实体与紧急度。
3. 编排器分配主辅职责，区分信息补充、事实查询和人工升级。
4. 检索层按问题召回候选；明确的复合诉求分别检索并选择证据。
5. Agent 结合证据、Skills 和工具结果生成回复。
6. 保存本轮消息，返回 Trace 与阶段耗时，后台提取画像。

## 核心实现

| 能力 | 实现 | 源码 |
|---|---|---|
| 意图融合 | LLM、相似度和 Pattern 加权；结构化实体与置信度 | [intent_recognizer.py](../backend/core/intent_recognizer.py) |
| 主辅编排 | 职责评分、并行执行和结果整合 | [agent_orchestrator.py](../backend/agents/agent_orchestrator.py) |
| 中文检索 | FastEmbed 中文 BGE、ChromaDB、候选合并与重排 | [knowledge_base.py](../backend/tooling/knowledge_base.py) |
| 证据确认 | Top-3 候选、来源检查和充分性判定 | [evidence_selector.py](../backend/core/evidence_selector.py) |
| 复合取证 | 子句主题拆分、并行证据门控、分项状态 | [knowledge_routing.py](../backend/core/knowledge_routing.py) · [tools.py](../backend/agents/tools.py) |
| 工具治理 | 参数校验、TTL 缓存、超时、熔断与 fallback | [tool_manager.py](../backend/tooling/tool_manager.py) |
| 记忆 | Redis 工作记忆、摘要压缩、ChromaDB 历史和画像 | [conversation_memory.py](../backend/memory/conversation_memory.py) |
| 计算策略 | 根据组件决定思考模式，保留显式覆盖 | [llm_utils.py](../backend/core/llm_utils.py) |
| 运行追踪 | 结构化工具、证据、调用来源和耗时 | [main.py](../backend/api/main.py) |

工具层通过 MCPToolManager 统一治理内部 handler 和标准 MCP 远程工具。已整合上游官方 SDK 客户端／服务端，支持工具发现、角色白名单、结构化结果和调用来源追踪。[接入与验证](MCP_INTEGRATION.md)说明配置与运行方式，[实现范围](IMPLEMENTATION_SCOPE.md)对应来源和分支改进。

## 改进一：中文知识检索

采用 BAAI/bge-small-zh-v1.5 本地 Embedding，并使用独立 ChromaDB collection 管理向量。相同冻结集上，36 条有答案查询的 Hit@1 从 20/36 提升到 35/36；60 条整体还包括难负例和域外负例。

[模型比较、阈值与错误分析](../backend/evaluation/reports/retrieval_test_v1_first_run_20261026.md)保留完整方法。

## 改进二：复合诉求分别取证

60 条覆盖评测暴露单篇证据对复合问题的覆盖不足。实现中对有明确分隔符、可识别知识主题的子问题分别执行检索与证据选择，通过 asyncio.gather 并行处理。

每个子问题保存支持状态、来源和错误信息。最终回复区分已确认事实与需要核实的事项。四条受影响原题在后续定向与隔离配对中均经人工确认完成目标。

[复合取证说明](COMPOSITE_EVIDENCE_ROUTING_20261106.md) · [配对回答](../backend/evaluation/reports/performance_ab_4cases_20261106.json)

## 改进三：结构化任务的 Token 控制

三类结构化任务关闭 Qwen 思考，最终 Agent 保留提供商默认配置。两组固定代码、模型、文档和问题，各采用全新独立会话。

| 指标 | 控制组 | 优化组 |
|---|---:|---:|
| 调用 | 19 | 19 |
| 总 Token | 36,817 | 22,099 |
| 平均端到端耗时 | 82.22 秒 | 34.77 秒 |
| 人工任务验收 | 4/4 | 4/4 |

主要节省来自结构化输出减少。样本为四条固定复合诉求，运行顺序是控制组后优化组；延迟解释同时考虑服务商时段波动。

[配置与组件归因](PERFORMANCE_TOKEN_PROFILE_20261106.md) · [逐题性能报告](../backend/evaluation/reports/performance_ab_4cases_20261106.md)

## 记忆与运行可靠性

Redis 保存近期对话与摘要，默认 TTL 为 24 小时；ChromaDB 保存历史摘要与用户级画像。查询使用对应标识进行过滤，同步向量操作放入线程池。

工具错误、记忆存储异常与证据缺口有各自的降级路径。Trace 记录系统预取和模型主动调用，便于追踪请求经过了哪些实际步骤。

## 评测与质量验收

- 冻结意图、检索、证据与回答数据，保存原始报告。
- 60 条首次目标回答按事实、覆盖、操作真实性和安全验收，形成改动前基线。
- 24 条回答完成双人四维评分与 Judge 校准，人工复核承担当前验收。
- 217 项自动化检查通过，最近验证于 2026-11-08，包含标准 MCP 协议与故障场景。
- 接口提供健康状态、运行摘要、工具 Trace、Metrics 与 Skills 重载。

| 接口 | 用途 |
|---|---|
| GET /health | 服务和 Agent 状态 |
| GET /monitor | 运行监控摘要 |
| GET /metrics | 指标文本 |
| GET /trace/tool/{request_id} | 单次工具轨迹 |
| GET /trace/tools | 近期工具轨迹 |
| POST /skills/reload | 业务规则重载 |

## 后续验证

在更大知识库、更多复合表达和多轮请求上扩展覆盖；性能采用重复与交错运行；真实订单/工单连接、并发和公网访问按独立能力验证。

[评测总览](EVALUATION_COVERAGE.md) · [完整运行指南](完整使用指南.md)
