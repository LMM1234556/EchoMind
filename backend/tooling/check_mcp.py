"""用官方客户端检查 EchoMind MCP 服务；不调用生成模型。"""
import argparse
import asyncio
import json
from pathlib import Path
import time

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


async def check(url: str, query: str, top_k: int) -> dict:
    started = time.monotonic()
    async with httpx.AsyncClient(timeout=30, follow_redirects=False, trust_env=False) as http:
        async with streamable_http_client(url, http_client=http) as (read, write, _):
            async with ClientSession(read, write) as session:
                initialized = await session.initialize()
                tools = await session.list_tools()
                resources = await session.list_resources()
                result = await session.call_tool('search_local_knowledge', {'query': query, 'top_k': top_k})
                if result.isError:
                    raise RuntimeError('MCP knowledge search returned a tool error')
                structured = result.structuredContent
                if not isinstance(structured, dict) or not isinstance(structured.get('results'), list):
                    raise RuntimeError('MCP knowledge search returned an invalid structure')
                stats = await session.read_resource('echomind://knowledge/stats')
                counts = json.loads(stats.contents[0].text)
                return {
                    'protocol_version': initialized.protocolVersion,
                    'server': initialized.serverInfo.name,
                    'tools': [tool.name for tool in tools.tools],
                    'resources': [str(resource.uri) for resource in resources.resources],
                    'query': query,
                    'candidate_titles': [row.get('title') for row in structured['results'] if isinstance(row, dict)],
                    'total_chunks': counts['total_chunks'],
                    'tool_error': result.isError,
                    'latency_ms': round((time.monotonic()-started)*1000, 1),
                }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8000/mcp/')
    parser.add_argument('--query', default='退款审核')
    parser.add_argument('--top-k', type=int, default=3, choices=range(1, 21))
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = asyncio.run(check(args.url, args.query, args.top_k))
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text+'\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
