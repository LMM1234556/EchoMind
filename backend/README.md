# EchoMind

EchoMind 是一个面向客服/运营场景的多 Agent 智能系统，将以下能力连接成完整闭环：

- 细粒度意图识别
- 路由驱动的多 Agent 编排
- 意图驱动 RAG 检索
- Redis + ChromaDB 分层记忆
- 动态 Skills 注入
- 在线监控与路由降权
- LLM-as-Judge 端到端评测

## 推荐阅读

- [工程案例](../docs/PORTFOLIO_CASE_STUDY.md)
- [工程变更摘要](../docs/CHANGES.md)
- [业务流程说明](../docs/业务流程说明.md)
- [完整使用指南](../docs/完整使用指南.md)
- [LLM-as-Judge 人工校准方案](../docs/EVALUATION_CALIBRATION.md)
- [产品验证方案](../docs/PRODUCT_VALIDATION_PLAN.md)

## 快速开始

### 1. 准备环境

- Docker
- Docker Compose
- `ANTHROPIC_API_KEY`

如果使用兼容 Anthropic 协议的第三方模型服务，也可以配置：

```env
ANTHROPIC_BASE_URL=https://dashscope.aliyuncs.com/apps/anthropic
ANTHROPIC_MODEL=qwen3.7-plus
ANTHROPIC_API_KEY=your_key
```

### 2. 配置环境变量

复制示例配置：

```bash
cp .env.example .env
```

最少确认这些变量可用：

```env
ANTHROPIC_API_KEY=your_api_key
REDIS_PASSWORD=echomind123
```

RAG 语义证据选择器默认启用。它从最多 3 条候选中选择一条证据充分的文档；知识缺口、超时或格式异常会进入核验引导流程。该步骤会为需要 RAG 的请求增加一次模型调用：

```env
EVIDENCE_SELECTOR_ENABLED=true
EVIDENCE_SELECTOR_TOP_K=3
EVIDENCE_SELECTOR_MAX_TOKENS=512
EVIDENCE_SELECTOR_TIMEOUT_SECONDS=45
```

### 3. 启动服务

推荐直接启动全栈：

```bash
docker compose up -d --build
```

查看状态：

```bash
docker compose ps
```

看日志：

```bash
docker compose logs -f echomind
```

### 4. 访问入口

- API: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`
- Nginx: `http://localhost`
- Health: `http://localhost:8000/health`

## 核心功能

### 对话主链路

`POST /chat`

流程是：

```text
读取记忆 -> 意图识别 -> 知识检索 -> Agent 路由 -> 回复生成 -> 写回记忆
```

### 知识库

- `POST /search`
- `POST /knowledge/add`
- `POST /knowledge/upload`
- `GET /knowledge/stats`

### Skills

- `GET /skills`
- `POST /skills/reload`

### 监控与评测

- `GET /monitor`
- `POST /eval/run`
- `POST /chat` 返回 `llm_call_count` 与 `llm_calls_by_component`，只统计响应返回前的前台应用层 `messages.create` 调用；SDK 内部自动重试不计入。

固定真实链路试运行：

```powershell
python -m evaluation.e2e_benchmark --base-url http://127.0.0.1:8000
```

用 `--case <case_id>` 可只运行指定场景，避免不必要的模型费用。运行器会逐条保存结果，超时也会作为失败写入报告。

## 项目结构

```text
api/main.py                  FastAPI 入口
agents/agent_orchestrator.py 多 Agent 编排
core/intent_recognizer.py    三路融合意图识别
core/skill_loader.py         动态 Skills 加载
memory/conversation_memory.py  Redis + ChromaDB 记忆
mcp/tool_manager.py          工具层、缓存、熔断、重排
mcp/knowledge_base.py        ChromaDB 知识库
monitor/performance_monitor.py 在线监控
evaluation/evaluator.py      端到端评测
evaluation/judge_calibration.py  人工评分与 Judge 一致性分析
evaluation/product_validation.py 产品可用性会话指标汇总
evaluation/e2e_cases.json    固定人工标注试运行集
evaluation/e2e_benchmark.py  真实 API 试运行与指标汇总
evaluation/intent_dataset.py 可审计意图数据集校验与已复核样本加载
evaluation/data/intent_eval_candidates.json  首批待人工复核候选
evaluation/reports/intent_dev_20261025.md  7 条开发集管线试运行与结论边界
evaluation/reports/intent_dev_batch2_20261025.md  15 条开发集诊断、修复与回归链
evaluation/reports/intent_dev_batch3_20261025.md  第三批 8 条开发样本评测
evaluation/reports/intent_dev_batch4_20261025.md  第四批 8 条开发样本评测
evaluation/reports/intent_dev_batch5_20261025.md  第五批 6 条开发样本评测
evaluation/reports/intent_dev_seed_summary_20261025.md  37 条开发样本建设总结
skills/                     动态业务规则
data/                       持久化数据
```

## 运行时架构

```text
用户请求
  -> /chat
  -> MemoryManager 读取工作记忆、情景记忆、用户画像
  -> IntentRecognizer 输出 intent / intent_group / urgency / entities
  -> 按意图决定是否检索知识库
  -> AgentOrchestrator 路由到 General / Technical / Billing / Escalation
  -> Skills 注入、工具调用、回复生成
  -> 写回 Redis 和 ChromaDB
  -> Monitor 采集在线指标
  -> Evaluator 做意图识别和回复质量评测
```

## 主要端口

| 服务 | 端口 |
|---|---:|
| EchoMind API | 8000 |
| ChromaDB | 8001 |
| Redis | 6379 |
| Prometheus | 9090 |
| Nginx | 80 |

## 开发和调试

常用顺序：

```text
1. /health
2. /chat
3. /skills
4. /monitor
5. /eval/run
```

如果希望快速了解项目，建议阅读：

- [项目定位与技术亮点](../docs/EchoMind定位与技术亮点.md)
- [工程案例](../docs/PORTFOLIO_CASE_STUDY.md)
- [Embedding 评测](../docs/EMBEDDING_BENCHMARK.md)

## 一句话概括

EchoMind 是一个可观测、可评测、可降级的多 Agent 客服运行时。
