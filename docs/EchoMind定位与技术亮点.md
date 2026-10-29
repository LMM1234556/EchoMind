# EchoMind 定位与技术亮点

这份文档面向第一次接触 EchoMind 的读者，用来快速理解项目定位、多 Agent 架构、关键演进和完整客服运行链路。

## 一句话定位

EchoMind 是一个面向复杂客服任务的多 Agent 客服编排运行时。

系统将意图识别、知识检索、记忆、路由、工具、监控和评测连接为完整协同链路。它先理解用户问题，再决定主处理 Agent、辅助 Agent、知识检索与人工升级策略，最后把结果写回记忆和观测系统。

## 它解决什么问题

真实客服场景包含多类型、多轮次和跨领域诉求。

用户常常会同时提到：

- 订单、物流、会员、积分等通用咨询
- 登录失败、401/500、页面崩溃等技术问题
- 退款、发票、重复扣款、支付失败等账务问题
- 转人工、投诉、紧急处理等升级诉求

EchoMind 重点解决三类产品挑战：

1. **分流精准度**
   让技术问题进入技术处理链路，让账单问题进入账单处理链路。

2. **上下文连续性**
   多轮对话里用户补充订单号、金额、错误码后，需要稳定延续上下文。

3. **持续优化**
   通过评测、监控和规则热更新建立持续优化闭环。

EchoMind 将这些能力组织为一条完整、可观测、可评测的客服链路。

## 当前是什么架构

当前 EchoMind 采用 **路由驱动的多 Agent 编排架构**，核心形态可以概括成：

```text
意图识别 -> 路由决策 -> 单 Agent / 并行多 Agent 执行 -> 响应合并 -> 记忆回写 -> 监控/评测反馈
```

更具体一点：

- `IntentRecognizer` 先做三路融合意图识别
- `AgentOrchestrator` 基于意图、实体、关键词、运行状态做路由
- `GeneralAgent / TechnicalAgent / BillingAgent / EscalationAgent` 是具备独立角色契约、工具白名单和输入输出边界的运行时角色
- 当请求同时覆盖多个业务域时，编排器会并行派发多个 Agent，再由 `ResponseComposer` 合并结果
- 运行过程会把 Agent 成功率、延迟、工具质量回写到路由评分里，形成闭环

EchoMind 的多 Agent 设计重点是**路由、协作、降级和治理**。

## 和之前版本的不同

早期版本的核心链路为：

```text
用户消息 -> 一个编排器 -> 几个 prompt 不同的 Agent
```

当前版本完成了四项关键演进：

### 1. 从“prompt 区分”变成“角色契约区分”

现在每个 Agent 都有自己的 `AgentProfile`，其中包括：

- `role`
- `mission`
- `workflow`
- `input_contract`
- `output_contract`
- `handoff_conditions`
- `tool_scope`

Agent 的差异由以下运行契约定义：

- 接收什么输入
- 产出什么结构
- 能用哪些工具
- 什么情况下必须升级

这比单纯换 prompt 稳定得多。

### 2. 从“单点路由”变成“结构化路由 + 主辅协作”

现在编排器会生成结构化 `RoutingDecision`：

- `primary_agent`
- `supporting_agents`
- `routing_reason`
- `routing_confidence`

这允许系统处理复合问题。  
比如“登录报错 + 重复扣款”，可以由技术 Agent 主处理、账单 Agent 辅助处理，再统一合并两个领域的结果。

### 3. 从“一个 Agent 一套能力”变成“共享工具 + 角色白名单”

现在工具已经集中到 `agents/tools.py`，并且：

- 有共享 RAG 工具
- 有通用工具
- 有技术工具
- 有账单工具
- 有升级工具

Agent 通过显式工具白名单获得与职责匹配的调用权限。

### 4. 从“能回答”变成“可治理”

现在系统里有：

- 监控：看成功率、延迟、熔断、工具质量
- 降权：运行差的 Agent 会被路由权重压低
- 评测：意图识别 Accuracy / Macro-F1，回复质量 LLM-as-Judge

这使系统形成可根据运行表现持续调整的客服运行时。

## 核心处理链路

```text
用户请求
  -> /chat
  -> 读取 Redis 工作记忆、ChromaDB 历史摘要和用户画像
  -> 识别细粒度意图、意图组、置信度和结构化实体
  -> 按意图判断是否触发 RAG 知识库检索
  -> 生成结构化路由决策
     - primary_agent
     - supporting_agents
     - routing_reason
     - routing_confidence
  -> 单 Agent 执行或并行多 Agent 执行
  -> 注入记忆、知识库、结构化实体和动态 Skills
  -> LLM 生成回复
  -> 写入工作记忆
  -> 异步更新用户画像
  -> Monitor 和 Evaluator 形成观测与评测闭环
```

链路中的每一步都对应一项独立职责：

- 记忆解决上下文
- 意图解决分流
- 路由解决主辅协作
- RAG 解决事实正确性
- Skills 解决业务规范
- 监控解决在线健康度
- 评测解决迭代质量

## 核心技术亮点

### 1. 细粒度意图识别

EchoMind 同时支持“咨询、投诉、技术、账单”等意图组和更贴近业务的细粒度分类。

例如：

| 细粒度意图 | 归一化意图组 | 示例 |
|---|---|---|
| `logistics` | `query` | 快递什么时候到 |
| `refund` | `billing` | 退款多久到账 |
| `invoice` | `billing` | 帮我开发票 |
| `payment_issue` | `billing` | 为什么重复扣款 |
| `technical_login` | `technical` | 登录一直报 401 |
| `technical_crash` | `technical` | 应用一直崩溃 |
| `human_handoff` | `escalation` | 我要找人工客服 |

意图识别使用三路融合：

- **LLM**：负责语义理解和上下文判断
- **Embedding / 本地哈希向量**：负责模板相似度匹配
- **Pattern**：负责关键词兜底

最终输出包括：

- `intent_group`
- `intent_confidence`
- `intent_source_scores`
- `urgency`
- `entities`

意图识别由此成为后续路由、澄清和升级的结构化输入层。

### 2. 结构化多 Agent 路由

EchoMind 的多 Agent 路由综合意图、关键词、实体和运行状态进行决策。

当前实现是一个路由驱动的多 Agent 编排架构：

1. 先通过意图识别得到业务方向
2. 再按意图、关键词和实体打分
3. 选出主 Agent
4. 对足够强的其他领域选择辅助 Agent
5. 必要时并行执行并合并结果

系统内部的打分逻辑综合考虑：

- 意图类别
- 关键词命中
- 结构化实体
- 当前 Agent 是否可用
- 在线表现和监控降权

#### 现在的路由结构

```text
general
technical
billing
escalation
```

主处理 Agent 由 `primary_agent` 表示，辅助 Agent 由 `supporting_agents` 表示。  
对于复合问题，系统让多个 Agent 同时工作并合并结果。

#### 和之前版本的区别

之前更像是“某个 Agent 负责一个 prompt 版本”。  
现在是“多个具备契约和白名单的运行时角色 + 一个可解释的路由层 + 一个结果合并层”。

这一演进将角色执行、路由决策和结果合并拆分为可独立治理的层次。

### 3. RAG 知识库增强

EchoMind 使用 ChromaDB 构建知识库，用于存放退款政策、配送说明、技术排障、会员规则等文档。

检索链路包括：

```text
原始问题
  -> 查询改写
  -> 多子查询并行召回
  -> 合并去重
  -> LLM 重排
  -> Top-K 注入 Agent 上下文
```

RAG 按需触发。系统先识别意图，业务类问题进入知识库检索；问候、反馈、转人工和未知意图由对应流程直接处理，从而控制检索成本与上下文噪声。

### 4. Redis + ChromaDB 记忆体系

EchoMind 把记忆拆成三层：

| 记忆类型 | 存储 | 作用 |
|---|---|---|
| 工作记忆 | Redis | 当前会话最近消息 |
| 情景记忆 | ChromaDB `episodic` | 历史对话摘要，支持语义检索 |
| 用户画像 | ChromaDB `user_profile` | 用户偏好和关键实体 |

Redis 读写使用异步客户端，ChromaDB 的同步操作放入线程池，减少主请求链路阻塞。

当前会话消息达到阈值时，系统会压缩旧消息、生成摘要并保留最近几轮对话，使上下文规模稳定可控。

### 5. 动态 Skills 注入

知识库解决的是“业务事实是什么”，Skills 解决的是“客服应该怎么处理”。

例如：

- 技术支持需要先收集错误码、版本、操作步骤
- 账单退款按平台规则说明审核与到账周期
- 通用客服需要先澄清用户诉求
- 涉及敏感信息时主动提示用户妥善保护密码和验证码

EchoMind 支持从 `skills/` 目录加载 Markdown / JSON / TXT 规则文件，并根据 Agent 类型和关键词动态注入到 system prompt。

修改规则后可以通过接口热加载，不需要重启服务。

### 6. MCP 工具可靠性治理

EchoMind 把知识库检索封装成工具，并加入完整的可靠性机制：

- 参数校验
- TTL 缓存
- 超时控制
- 熔断器
- fallback 降级
- 查询改写
- LLM 重排
- 工具成功率和延迟统计

工具调用由此具备可治理、可观测和可降级能力。

### 7. Monitor 在线观测和路由降权

Monitor 会定期采集：

- Agent 成功率
- Agent 平均延迟
- 工具成功率
- 工具平均延迟
- 连续失败次数
- 熔断状态

如果某个 Agent 表现变差，Monitor 会写回 `monitor_penalty`，影响后续 `routing_score`。

监控结果既用于展示运行指标，也参与后续路由选择。

### 8. LLM-as-Judge 端到端评测

EchoMind 内置 `/eval/run` 评测入口。

评测内容包括：

- 意图识别 Accuracy
- Macro-F1
- 端到端 Agent 回复质量
- LLM-as-Judge 四维评分
- 回归检测
- 优化建议

LLM-as-Judge 会从四个维度评价回复：

- 相关性
- 准确性
- 完整性
- 有用性

这让项目能够持续评估并改进回答质量。

## 完整客服运行链路

基础对话链路可以概括为：

```text
用户输入 -> LLM 回复
```

EchoMind 在此基础上扩展为：

```text
用户输入
  -> 意图识别
  -> 实体提取
  -> 记忆读取
  -> 按意图 RAG
  -> 结构化多 Agent 路由
  -> Skills 注入
  -> Agent 回复
  -> 记忆写入
  -> 画像更新
  -> 监控反馈
  -> 评测回归
```

整体实现形成了一个小型 Multi-Agent Runtime，覆盖从请求理解到评测回归的完整闭环。
