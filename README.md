# EchoMind

> 面向企业软件客服场景的 Multi-Agent AI 产品原型：通过意图路由、RAG 证据门控、分层记忆和评测闭环，减少跨领域漏处理与无依据业务回答。

**作品集导航：** [视频演示](#视频演示) · [AI 产品案例](docs/AI_PRODUCT_CASE_STUDY.md) · [静态产品演示](docs/PRODUCT_DEMO.md) · [工程案例](docs/PORTFOLIO_CASE_STUDY.md) · [当前评测报告](backend/evaluation/reports/e2e_current_top3_20261026.md) · [系统架构](docs/架构图.md)

![EchoMind 调试工作台](docs/assets/portfolio/echomind-dashboard.png)

## 视频演示

[![播放 EchoMind 产品演示](docs/assets/portfolio/echomind-demo-cover.jpg)](docs/assets/portfolio/echomind-demo.mp4)

点击封面观看约 88 秒的产品演示，内容包括意图路由、RAG 证据确认、技术故障处理、知识缺口识别、人工交接、产品架构和分层评测。

## 30 秒了解项目

| 项目要素 | 内容 |
|---|---|
| 目标用户 | 企业软件客服团队及需要处理产品、技术、账单问题的一线客服 |
| 核心问题 | 单 Agent 容易漏掉复合诉求；模型可能编造退款时效、费用等业务事实；问题发生后难以定位具体链路 |
| 产品方案 | 用细粒度意图识别进行主/辅 Agent 路由；针对业务问题检索知识库；通过 Top-3 Evidence Selector 决定基于证据回答或引导进一步核验 |
| 我的工作 | 基于已有 EchoMind 项目完成环境复现、Qwen 接入、中文 Embedding 迁移、证据门控、路由与记忆修复、故障降级和分层评测 |
| 当前状态 | 已完成本地可运行的作品集原型；下一阶段聚焦公网安全加固与真实业务试点 |

## 已验证的关键结果

| 结果 | 价值 | 口径 |
|---|---|---|
| 中文检索 Hit@1：55.56% → 97.22% | 改善中文口语问题的首位文档召回 | 同一 60 条冻结检索集上的 MiniLM/BGE 对比 |
| 知识缺口正确识别率：91.67% | 识别证据不足的业务问题，并引导用户进一步核验 | 24 条 Grounded Chat 冻结集 |
| 自动化回归：151/151 通过 | 覆盖路由、工具、记忆、异常处理与降级机制 | Mock 与本地依赖测试 |

这些数据用于衡量固定开发/冻结样本中的分层能力，样本范围和完整评测发现见[分层评测与迭代发现](#分层评测与迭代发现)。

## 产品处理流程

## 核心链路

```mermaid
flowchart LR
    U[用户请求] --> API[FastAPI /chat]
    API --> I[意图识别\nLLM + Embedding + Pattern]
    I --> O[Orchestrator\n结构化路由]
    O --> A1[General Agent]
    O --> A2[Technical Agent]
    O --> A3[Billing Agent]

    O --> G{是否需要业务知识}
    G -->|是| R[RAG 检索与证据校验]
    R --> C[(ChromaDB)]
    G -->|否| X[跳过检索]

    O --> S[Skills 与工具治理]
    O --> M[会话记忆]
    M --> RD[(Redis)]
    M --> C

    A1 --> Z[合并并生成回复]
    A2 --> Z
    A3 --> Z
    R --> Z
    S --> Z
    M --> Z
    Z --> T[Trace / Monitor / Evaluation]
```

## 典型用户场景

| 用户请求 | 产品行为 | 风险控制 |
|---|---|---|
| “退款审核需要多久？” | 路由至 Billing Agent，检索并选择退款政策证据 | 依据已确认的证据提供具体时效 |
| “登录一直提示 401” | 路由至 Technical Agent，结合错误码工具给出排障步骤 | 区分平台事实与通用技术建议 |
| “电子发票多久开出来？” | 检索后识别当前知识库的时效证据缺口 | 明确说明依据范围，并引导进一步核验 |
| “请转人工客服” | 进入 Escalation Agent 生成交接摘要 | 只使用结构化意图触发升级，不解析自由文本关键词 |

查看[静态产品演示](docs/PRODUCT_DEMO.md)，可了解界面字段和四类场景的观察重点。

## 关键产品决策

1. **基于业务职责与风险设计 Agent。** General、Technical、Billing、Escalation 分别承担接待、排障、账单和人工交接；复合请求可形成主 Agent 与辅助 Agent。
2. **建立“召回—证据确认”两阶段链路。** 向量检索负责找到候选，Evidence Selector 进一步确认平台业务事实的依据完整性。
3. **采用分级响应策略。** Billing/General 依据业务证据回答，Technical 在证据范围之外提供明确标注的通用排障建议。
4. **以评测反馈驱动产品迭代。** Trace 记录意图、路由、证据、工具和阶段耗时，评测分别覆盖意图、检索、证据选择和完整链路。

## 分层评测与迭代发现

不同评测集分别衡量意图、检索、证据选择和完整链路，各项结果按对应样本范围独立解读。

| 评测 | 样本 | 当前结果 |
|---|---:|---|
| 意图冻结集 v2 | 95 条、19 类 | 准确率 98.95%，Macro-F1 98.94% |
| 检索冻结集 | 60 条 | BGE Hit@1 97.22%、Hit@3 100%；含负例的检索决策准确率 78.33% |
| Grounded Chat v2 | 24 条 | 严格证据选择 87.50%；知识缺口正确识别 91.67%；项目负责人确认的回答事实忠实度 83.33%（20/24） |
| 当前 Top-3 综合回归 | 7 场景、8 次真实请求 | 意图 75.00%；完整路由 75.00%；回答契约 75.00%；严格端到端 62.50%（5/8） |
| 自动化回归 | 151 项 | 151/151 通过 |

当前综合回归采用单次完整运行口径，定位出三个明确的迭代方向：提升问候标签稳定性、完善复合领域路由，以及优化记忆指令与证据门控的任务分流。原始结果见[当前 Top-3 综合回归](backend/evaluation/reports/e2e_current_top3_20261026.md)。以上结果适用于当前固定开发/冻结样本，并为后续真实业务试点提供基线。

从产品角度，下一轮将优先提升复合意图召回，并完善“保存信息”与“询问政策”的任务分流。完整产品复盘见 [AI_PRODUCT_CASE_STUDY.md](docs/AI_PRODUCT_CASE_STUDY.md)。

## 技术实现

- Backend：Python、FastAPI、asyncio
- Model：Qwen API、Anthropic-compatible SDK
- Retrieval：ChromaDB、BGE Chinese Embedding
- Memory：Redis、ChromaDB
- Frontend：Vue、Vite
- Runtime：Docker Compose
- Observability：Prometheus-style metrics、structured trace

## 仓库结构

```text
backend/   Python + FastAPI 多 Agent 后端
frontend/  Vue + Vite 调试前端
docs/      工程改动、评测、架构与作品集案例资料
```

Java 版本未参与本轮工程验证，因此未纳入仓库。

## 快速启动

### 1. 准备环境

- Docker Desktop
- Node.js 20 或更高版本
- 可用的阿里云百炼 API Key，并已授权所选模型

### 2. 配置后端

```powershell
Set-Location backend
Copy-Item .env.example .env.local
```

只在本机编辑 `backend/.env.local`：

```env
ANTHROPIC_API_KEY=your_api_key
```

真实密钥保存在本地 `.env.local`，该文件已由 Git 忽略。

语义证据选择器默认启用，会为需要 RAG 的请求增加一次模型调用。它接收最多 3 条候选的标题和正文，仅将通过验证的文档交给回复模型；超时或格式异常时进入证据核验引导流程。可在本地 `.env.local` 调整：

```env
EVIDENCE_SELECTOR_ENABLED=true
EVIDENCE_SELECTOR_TOP_K=3
EVIDENCE_SELECTOR_MAX_TOKENS=512
EVIDENCE_SELECTOR_TIMEOUT_SECONDS=45
```

需要回滚到旧证据门控时，可设置 `EVIDENCE_SELECTOR_ENABLED=false`。

### 3. 启动后端

```powershell
Set-Location backend
docker compose up -d --build redis chromadb echomind
Invoke-RestMethod http://127.0.0.1:8000/health
```

### 4. 启动前端

```powershell
Set-Location frontend
npm ci
npm run dev -- --port 5174
```

访问 <http://127.0.0.1:5174/>，选择 Python 后端。

## 测试

```powershell
Set-Location backend
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m pytest -q
```

真实模型调用与使用 mock 或本地依赖的自动化测试分别运行，便于独立统计 API 用量、响应耗时与模型效果。

## 适用范围与下一阶段

- 当前综合回归 P95 为 68.00 秒，下一阶段将围绕模型调用次数、阶段耗时和 token 成本开展性能优化。
- LLM-as-Judge 评测框架已完成，下一阶段将通过人工评分样本进行一致性校准。
- 当前 8 次综合回归已形成可复核基线，并定位问候识别、复合领域路由和记忆指令分流三个优化方向。
- `MCPToolManager` 当前承担项目内工具治理，标准 MCP Server/Client 协议可作为后续扩展方向。
- 当前版本面向本地作品展示；公网试点将进一步补充认证、授权、限流、日志脱敏和知识库审核机制。

## 项目来源与个人工作

本仓库基于原 EchoMind Python 后端和 Vue 前端开展工程改进，重点呈现环境复现、Qwen 接入、中文 Embedding 迁移、证据门控、路由与记忆修复、故障降级和分层评测。根据仓库维护者确认，项目提供方已授权将该项目用于实习求职和公开作品展示；当前仓库采用保留权利方式公开展示，具体说明见 [NOTICE.md](NOTICE.md)。
