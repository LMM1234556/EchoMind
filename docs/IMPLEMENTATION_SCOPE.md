# EchoMind 实现范围与版本对照

更新日期：2026-11-05。

本仓库基于 EchoMind Python 后端和 Vue 前端开展工程改进，围绕中文客服的证据回答、运行追踪和分层评测形成当前版本。本文说明基础能力、个人分支改进及已核对的上游扩展，便于对应源码与验证结果。

## 1. 基础能力与个人改进

| 方面 | 继承的基础能力 | 本分支改进与验证 |
|---|---|---|
| Agent 与意图 | General、Technical、Billing、Escalation；LLM、Embedding、Pattern 三路意图融合；主辅 Agent 路由 | 完善结构化调用与 Trace，使用 95 条、19 类人工复核的合成冻结样本评估意图识别 |
| RAG | ChromaDB 知识库、查询改写、并行召回与 LLM 重排 | 接入本地中文 BGE，增加快速检索、政策预检索和证据门控，完成 60 条冻结检索对照 |
| 证据回答 | 将知识片段注入 Agent 上下文 | 增加 Top-3 Evidence Selector、选中文档校验及证据不足处理，保存 24 条完整回答与逐条事实复核 |
| 工具与 Skills | `MCPToolManager` 的参数校验、缓存、超时、熔断和降级；Skills 加载与重载 | 完善实际调用轨迹、失败分类、降级语义及对应回归 |
| 分层记忆 | Redis 工作记忆；ChromaDB 情景记忆和用户画像 | 修正组合过滤、会话隔离、压缩顺序及存储异常处理 |
| 评测与展示 | 基础意图评测、对话评测、Judge 和在线监控 | 建立冻结数据与报告、24 条双人评分和 Judge 校准、产品／工程案例、演示视频与可观察工作台 |

业务实现使用 Codex 辅助开发与迭代；维护者负责场景拆解、方案取舍、代码审阅、样本复核、功能验证和结果分析。文件级改进见[工程变更摘要](CHANGES.md)，项目来源见[NOTICE](../NOTICE.md)。

## 2. 当前分支的关键实现

### 工具管理与标准 MCP

当前内部工具由 [`MCPToolManager`](../backend/mcp/tool_manager.py) 管理，Agent 通过注册的 handler 执行调用。标准 MCP 客户端／服务端接入列入上游能力移植范围。

2026-11-05 核对的上游分发版本，其随包 Git HEAD 为 `4ee28fa`，已包含官方 MCP SDK、客户端和服务端，支持初始化、工具发现及调用。本分支保留当前内部工具链和 Grounded RAG，后续可通过独立分支接入协议适配层。

### 意图向量与知识库向量

- [意图识别器](../backend/core/intent_recognizer.py)当前使用的本地路径为字符 n-gram，服务于请求类别判断。
- [知识库](../backend/mcp/knowledge_base.py)使用 `BAAI/bge-small-zh-v1.5`，服务于中文文档检索。
- 上游新版的在线 `BAAI/bge-m3` 属于意图分支扩展；知识库检索对照报告衡量的是 MiniLM 与本地中文 BGE 的文档召回效果。

### Top-3 Evidence Selector

当前[选择器](../backend/core/evidence_selector.py)读取最多 3 篇候选，选择其中一篇能够完整支持当前问题的文档，并校验返回的 `selected_title` 是否属于本次候选标题。证据不足或选择器失败时进入核验引导流程。

本分支以单篇充分证据为当前回答策略，多文档联合取证和部分问题分别回答可作为后续能力扩展。

## 3. 上游扩展与本分支状态

以下内容已在上游新版源码中定位，本仓库按自身请求链路分项评估移植：

| 上游扩展 | 上游实现 | 本分支状态 |
|---|---|---|
| 标准 MCP | 官方 SDK 客户端、Streamable HTTP／stdio、知识库服务端及协议测试 | 待接入；当前使用内部工具管理 |
| 在线意图向量 | 独立 BGE-M3 HTTP 客户端、模板缓存、校验与降级 | 待接入；当前保留本地意图向量路径 |
| 会话可靠性 | 会话锁、请求 ID、回执与事务提交 | 独立评估；保留本分支已有隔离与异常处理 |
| 异步评测 | HTTP 202、任务 ID 与状态轮询 | 待接入；需同步适配评测接口与前端 |
| 性能与权限 | 阶段 Token 记录、调用节流和可选身份映射 | 按场景分项引入 |

上游协议测试和运行报告用于核查相应上游实现；本仓库的版本验证见[2026-11-04 验证记录](REPOSITORY_VERIFICATION_20261104.md)。

## 4. 当前评测资料

| 评测目标 | 数据范围 | 对应材料 |
|---|---|---|
| 意图识别 | 95 条、19 类合成冻结样本 | [意图 v2 首次结果](../backend/evaluation/reports/intent_test_v2_first_run_20261026.md) |
| 知识库检索 | 60 条冻结查询，其中 36 条有答案 | [MiniLM／BGE 正式对照](../backend/evaluation/reports/retrieval_test_v1_first_run_20261026.md) |
| 证据与回答 | 24 条 Grounded Chat v2 | [首次结果](../backend/evaluation/reports/grounded_chat_test_v2_first_run_20261026.md) |
| Judge 校准 | 复用同一批 24 条回答，16 条校准、8 条同批留出验证 | [校准结果与采用决策](../backend/evaluation/reports/judge_calibration_v1_findings_20261104.md) |
| 综合链路 | 7 个场景、8 次真实请求 | [Top-3 综合回归](../backend/evaluation/reports/e2e_current_top3_20261026.md) |
| 工程回归 | 2026-11-04 隔离检出，163 项通过 | [版本验证记录](REPOSITORY_VERIFICATION_20261104.md) |

样本扩充围绕类别覆盖、难例和真实业务场景开展。每轮固定数据、模型配置和评价规则，按对应任务报告指标；当前规范见[意图数据规范](意图评测数据规范.md)和[RAG 评测协议](RAG评测协议.md)。
