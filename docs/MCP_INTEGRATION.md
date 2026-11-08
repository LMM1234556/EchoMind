# 标准 MCP 接入与验证

更新日期：2026-11-08。

EchoMind 已整合上游分发版本的官方 MCP SDK 客户端和知识库服务端，并适配当前证据回答链路。使用 `mcp==1.26.0`，内部工具模块统一放在 `backend/tooling/`。

## 两条协议链路

| 方向 | 执行过程 | 实现 |
|---|---|---|
| EchoMind 调用其他服务 | Agent → 工具管理器 → MCP 客户端 → stdio／Streamable HTTP 服务 | [客户端](../backend/tooling/mcp_client.py) |
| 其他客户端检索 EchoMind | MCP 客户端 → `/mcp/` → 本地知识库检索或统计资源 | [服务端](../backend/tooling/mcp_server.py) |

客户端通过官方 SDK 完成初始化、工具发现和调用。配置中的 `allowed_tools` 决定可注册工具，`agents` 决定哪些 Agent 可以使用。只读模式进一步筛选工具注解，调用继续经过参数校验、缓存、超时和熔断。连接中断时按退避策略重连，服务目录变更时刷新工具和缓存版本。

服务端提供：

- `search_local_knowledge(query, top_k)`：返回知识库候选文档与检索分数，`top_k` 范围为 1–20。
- `echomind://knowledge/stats`：读取知识库片段数量。

MCP 检索结果属于候选资料。聊天回答仍按现有取证流程检查来源与充分性；复合诉求分项取证，Evidence Selector 继续执行。

## 本地启动与配置

后端启动后，协议地址为 `http://127.0.0.1:8000/mcp/`，连接状态为 `GET /integrations/mcp`。原 `/chat`、`/search` 和工作台入口继续使用。

```dotenv
MCP_SERVER_ENABLED=true
MCP_EXTERNAL_ENABLED=false
```

默认提供本地知识服务；外部配置文件 `backend/config/mcp_servers.json` 的 `servers` 数组为空。添加服务时参考 [示例配置](../backend/config/mcp_servers.example.json)，明确服务地址、工具和角色，再设置 `MCP_EXTERNAL_ENABLED=true`。配置文件位置可由 `MCP_SERVERS_CONFIG` 覆盖。

stdio 服务配置示例：

```json
{
  "servers": [
    {
      "name": "docs",
      "transport": "stdio",
      "command": "python",
      "args": ["/path/to/read_only_mcp_server.py"],
      "allowed_tools": ["read_document"],
      "agents": ["general", "technical"],
      "read_only": true,
      "timeout_s": 25,
      "cache_ttl": 300,
      "max_result_chars": 12000
    }
  ]
}
```

将命令和脚本路径改为运行环境中的实际值。模型 Key 不默认传入外部子进程；服务所需变量通过 `env_vars` 显式传递，HTTP 认证头通过 `headers_env` 引用环境变量。配置文件保存变量名。

## 用官方客户端验证

在已安装后端依赖的 Python 环境运行：

```powershell
# 在 backend 目录，一次检查工具发现、检索与资源读取
python -m tooling.check_mcp
```

检查器输出协议版本、工具、候选标题和片段数。也可直接使用 SDK：

```python
import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

async def main():
    async with streamable_http_client("http://127.0.0.1:8000/mcp/") as (read, write, _):
        async with ClientSession(read, write) as session:
            initialized = await session.initialize()
            print("protocol:", initialized.protocolVersion)
            print("tools:", [tool.name for tool in (await session.list_tools()).tools])
            result = await session.call_tool("search_local_knowledge", {"query": "退款审核", "top_k": 3})
            print(result.structuredContent)
            print(await session.read_resource("echomind://knowledge/stats"))

asyncio.run(main())
```

协议检索使用本地知识库，以上步骤不请求 Qwen。

## 回归与运行验证

当前全量自动化回归为 **217/217**，日期为 2026-11-08，其中 [MCP 协议回归](../backend/tests/test_standard_mcp.py)覆盖 12 项：

- 真实 stdio 子进程中的初始化、白名单、调用和目录变更通知。
- 外部工具进入 Agent 调用循环，检查角色过滤与来源 Trace；生成模型使用测试替身。
- Streamable HTTP 客户端通过本地 TCP 发现并调用服务。
- 服务端工具、结构化结果和资源读取，以及 FastAPI `/mcp/` 挂载。
- 参数错误、执行错误、超时、连接失败、输出裁剪和关闭清理。
- 外部调用失败时，内部工具仍可执行；降级检索结果按 MCP 工具错误返回。

```powershell
# 在 backend 目录
python -m pytest tests -q
python -m pytest tests/test_standard_mcp.py -q
```

运行记录见 [MCP 验证记录](MCP_VERIFICATION_20261108.md)。已有意图、检索、回答与性能报告继续对应各自原运行版本。

## 运行与扩展

`MCP_SERVER_ENABLED=false` 可关闭协议服务端；`MCP_EXTERNAL_ENABLED=false` 可关闭外部客户端连接，两者均在重启后生效，原内部工具与 RAG 路径继续工作。

当前交付面向本地部署与作品展示。接入具体业务时按服务配置认证、Host／Origin 允许范围和网关权限；真实订单、退款执行和工单接口按独立业务集成推进。

SDK 用法参考 [官方 Python SDK v1.26.0](https://github.com/modelcontextprotocol/python-sdk/tree/v1.26.0)。
