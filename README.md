# EchoMind

EchoMind 是一个面向客服与运营场景的多 Agent 系统。本仓库在原始 EchoMind Python 后端和 Vue 前端的基础上，完成了本地部署、模型接入、RAG 检索、会话记忆、路由、监控与测试方面的工程改进和验证。

## 来源与定位

- 原 Python 后端：<https://github.com/Biscuit-AI531/EchoMind>
- 原 Vue 前端：<https://github.com/Biscuit-AI531/EchoMindFrontend>
- 当前仓库性质：基于原项目的工程改进版本，不宣称原始项目的原创权。
- 原仓库未提供项目级开源许可证。当前整理版应保持为本地或私有仓库；公开发布前需取得原作者许可。详见 [NOTICE.md](NOTICE.md)。

## 仓库结构

```text
backend/   Python + FastAPI 多 Agent 后端
frontend/  Vue + Vite 调试前端
docs/      原项目包中的架构、业务流程、使用和重点代码资料
```

Java 版本没有参与本次工程验证，因此未纳入本仓库。

## 已完成的工程改进

- 接入阿里云百炼 Qwen API（Anthropic 兼容接口），统一读取模型配置。
- 增加按意图触发的知识库检索门控、快速关键词召回和 RAG 回退。
- 修正工具调用后回复生成、复合领域路由以及 Billing/Technical 并行执行。
- 修正 Redis 工作记忆窗口、TTL、压缩顺序和跨会话隔离。
- 修正 ChromaDB 用户与会话过滤，增加检索评测与去重逻辑。
- 增加监控告警去重、恢复处理、可配置延迟阈值和阶段耗时追踪。
- 前端展示 Agent 路由、工具调用、知识库使用情况和阶段耗时。
- 补充路由、知识库、记忆、工具、监控和检索评测测试。

更完整的文件级变化见 [docs/CHANGES.md](docs/CHANGES.md)。

## 快速启动

### 1. 准备环境

- Docker Desktop
- Node.js 20 或更高版本
- 可用的阿里云百炼 API Key，并已授权所选模型

### 2. 配置后端

在 PowerShell 中执行：

```powershell
Set-Location backend
Copy-Item .env.example .env
```

然后只在本机编辑 `backend/.env`，填写：

```env
ANTHROPIC_API_KEY=你的密钥
```

`.env` 已被 Git 忽略，不要提交密钥。

### 3. 启动后端依赖和 API

```powershell
Set-Location backend
docker compose up -d --build redis chromadb echomind
```

检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

### 4. 启动前端

```powershell
Set-Location frontend
npm ci
npm run dev -- --port 5174
```

浏览器访问 <http://127.0.0.1:5174/>，选择 Python 后端。

## 本地测试

```powershell
Set-Location backend
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m pytest -q
```

测试会使用 mock 或本地服务验证大部分逻辑；真实模型调用会消耗 API 额度，应单独控制次数。

### 固定端到端试运行

后端健康且已配置模型服务后，可运行 7 个固定场景（8 次真实请求）：

```powershell
Set-Location backend
.\venv\Scripts\python.exe -m evaluation.e2e_benchmark `
  --base-url http://127.0.0.1:8000 `
  --output evaluation\reports\e2e_latest.json
```

评测分别统计人工标注意图/路由准确率、回答契约通过率、P50/P95 延迟和前台应用层 LLM SDK 调用次数。SDK 内部自动重试不计入该次数。回答契约检查不等同于完整语义质量或生产准确率。2026-10-18 的实测结果与边界见 [端到端试运行报告](backend/evaluation/reports/e2e_final_20261018.md)。

## 文档说明

`docs/` 来自项目包中的正式技术资料。资料中的设计描述不自动等于当前环境已完成验证；以代码、自动化测试和实际运行结果为准。

