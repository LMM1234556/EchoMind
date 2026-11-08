import asyncio
import json
import pathlib
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from agents.agent_orchestrator import GeneralAgent, BillingAgent, Request
from tooling.mcp_client import MCPServerConfig, StandardMCPClient, tool_alias
from tooling.tool_manager import MCPToolManager
from tooling import mcp_server


def test_stdio_discovery_whitelist_and_agent_tool_roundtrip():
    async def run():
        manager = MCPToolManager(api_key='test')
        config = MCPServerConfig(name='fixture', transport='stdio',
            allowed_tools=['read_document','dangerous_write'], agents=['general','technical'],
            command=sys.executable, args=[str(pathlib.Path(__file__).with_name('mcp_fixture_server.py'))])
        client = StandardMCPClient(manager,[config])
        try:
            await client.start()
            assert client.summary()['servers'][0]['state'] == 'connected'
            alias = 'mcp_fixture_read_document'
            assert client.owns_tool(alias) and not client.owns_tool('mcp_fixture_dangerous_write')
            remote = await manager.call(alias, {'query':'hello'})
            assert remote.success and remote.data['structured']['text'] == 'HELLO'
            cached = await manager.call(alias, {'query':'hello'})
            assert cached.success and cached.cached
            version = manager._cache_versions[alias]
            await client.connections['fixture'].session.call_tool('refresh_catalog', {})
            for _ in range(100):
                if manager._cache_versions[alias] > version:
                    break
                await asyncio.sleep(0.01)
            assert manager._cache_versions[alias] > version
            assert client.summary()['servers'][0]['state'] == 'connected'
            bad = await manager.call(alias, {'query': True})
            assert not bad.success
            llm = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(side_effect=[
                SimpleNamespace(stop_reason='tool_use',content=[SimpleNamespace(type='tool_use',id='t1',name=alias,input={'query':'proof'})]),
                SimpleNamespace(stop_reason='end_turn',content=[SimpleNamespace(type='text',text='读取完成')]),
            ])))
            general = GeneralAgent(llm,'test-model')
            general.set_shared_tools(client.agent_tools())
            billing = BillingAgent(llm,'test-model')
            billing.set_shared_tools(client.agent_tools())
            assert alias in general.get_tools() and alias not in billing.get_tools()
            result = await general._call_llm(Request(message='读取资料',user_id='u',conv_id='c'))
            assert result == '读取完成'
            assert general._last_tool_traces[0]['source'] == 'external_mcp'
            assert general._last_tool_traces[0]['mcp_server'] == 'fixture'
            assert general._last_tool_traces[0]['result_success'] is True
        finally:
            await client.close()
            await manager._client.close()
        assert not client.agent_tools() and client.summary()['servers'][0]['state'] == 'closed'
    asyncio.run(run())


def test_local_standard_http_server_tools_and_resources():
    async def run():
        kb = SimpleNamespace(search_async=AsyncMock(return_value=[{'title':'政策','content':'退款规则'}]),
                             doc_count_async=AsyncMock(return_value=8))
        mcp_server.configure(kb)
        server = mcp_server.create_server()
        app = server.streamable_http_app()
        try:
            async with server.session_manager.run():
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as http:
                    async with streamable_http_client('http://localhost/',http_client=http) as (read,write,_):
                        async with ClientSession(read,write) as session:
                            await session.initialize()
                            catalog = await session.list_tools()
                            assert [t.name for t in catalog.tools] == ['search_local_knowledge']
                            result = await session.call_tool('search_local_knowledge',{'query':'退款','top_k':2})
                            assert not result.isError and result.structuredContent['results'][0]['title'] == '政策'
                            stats = await session.read_resource('echomind://knowledge/stats')
                            assert json.loads(stats.contents[0].text)['total_chunks'] == 8
                            invalid = await session.call_tool('search_local_knowledge',{'query':'退款','top_k':0})
                            assert invalid.isError
                            bad_type = await session.call_tool('search_local_knowledge',{'query':'退款','top_k':True})
                            assert bad_type.isError
        finally:
            mcp_server.configure(None)
    asyncio.run(run())


@pytest.mark.parametrize('changes', [dict(allowed_tools=[]),dict(agents=['unknown']),dict(url='file:///tmp/data'),dict(timeout_s=0)])
def test_invalid_mcp_configuration_is_rejected(changes):
    values=dict(name='test',transport='streamable_http',url='https://example.com/mcp',allowed_tools=['search'],agents=['general'])
    values.update(changes)
    with pytest.raises(ValueError):
        MCPServerConfig(**values)


def test_alias_is_stable_and_valid_for_model_tool_names():
    import re
    a=tool_alias('test','very.long.name'*12)
    assert len(a)<=64 and re.fullmatch('[A-Za-z0-9_-]+',a)
    assert a==tool_alias('test','very.long.name'*12)
    assert tool_alias('test','some.name') != tool_alias('test','some_name')


def test_large_external_result_remains_structured_and_keeps_source_url():
    from tooling.mcp_client import bounded_result
    result = bounded_result({'structured':{'results':[{'content':'大段资料' * 1000,
        'title':'身份认证文档','url':'https://learn.microsoft.com/example'}]},'content':[]}, 'mslearn', 'search', 500)
    assert result['truncated']
    row = result['structured']['results'][0]
    assert row['url'] == 'https://learn.microsoft.com/example'
    assert row['content'].endswith('…')
    assert len(json.dumps(result,ensure_ascii=False,separators=(',', ':'))) <= 500
    assert json.loads(json.dumps(result)) == result


def test_remote_error_timeout_and_close_keep_local_tools_available():
    from tooling.tool_manager import Tool

    async def run():
        manager = MCPToolManager(api_key='test')
        async def local(params, context):
            return {'local': True}
        manager.register(Tool(name='local', description='local', handler=local, schema={'type':'object'}))
        config = MCPServerConfig(name='failure', transport='stdio',
            allowed_tools=['fail_read','slow_read'], agents=['technical'],
            command=sys.executable, args=[str(pathlib.Path(__file__).with_name('mcp_fixture_server.py'))],
            timeout_s=1, cache_ttl=0)
        client = StandardMCPClient(manager, [config])
        try:
            await client.start()
            assert client.summary()['servers'][0]['state'] == 'connected'
            error = await manager.call('mcp_failure_fail_read', {})
            assert not error.success and '返回执行错误' in error.error
            timeout = await manager.call('mcp_failure_slow_read', {})
            assert not timeout.success
            assert (await manager.call('local', {})).data == {'local':True}
        finally:
            await client.close()
            await manager._client.close()
        assert client.summary()['servers'][0]['state'] == 'closed'
        assert not client.agent_tools() and 'mcp_failure_fail_read' not in manager._tools
    asyncio.run(run())


def test_unavailable_server_is_reported_without_registering_tools():
    async def run():
        manager = MCPToolManager(api_key='test')
        config = MCPServerConfig(name='unavailable', transport='stdio', allowed_tools=['read'],
            agents=['general'], command=str(pathlib.Path(sys.executable).with_name('missing-mcp-executable.exe')),
            timeout_s=1)
        client = StandardMCPClient(manager, [config])
        try:
            await client.start()
            summary = client.summary()['servers'][0]
            assert summary['state'] == 'disconnected' and summary['error']
            assert not client.agent_tools()
        finally:
            await client.close()
            await manager._client.close()
    asyncio.run(run())


def test_mounted_fastapi_endpoint_and_degraded_result(monkeypatch):
    from api import main
    from tooling.tool_manager import ToolResult

    async def run():
        kb = SimpleNamespace(doc_count_async=AsyncMock(return_value=6))
        manager = SimpleNamespace(call=AsyncMock(return_value=ToolResult(
            success=True, data=[{'fallback':True}], tool_name='knowledge_search', error='unavailable')))
        mcp_server.configure(kb, manager)
        server = mcp_server.create_server()
        http_app = server.streamable_http_app()
        mount = next(route for route in main.app.routes if getattr(route, 'path', None) == '/mcp')
        monkeypatch.setattr(mount, 'app', http_app)
        try:
            async with server.session_manager.run():
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app)) as http:
                    async with streamable_http_client('http://localhost/mcp/',http_client=http) as (read,write,_):
                        async with ClientSession(read,write) as session:
                            await session.initialize()
                            result = await session.call_tool('search_local_knowledge',{'query':'退款'})
                            assert result.isError
                            stats = await session.read_resource('echomind://knowledge/stats')
                            assert json.loads(stats.contents[0].text)['total_chunks'] == 6
        finally:
            mcp_server.configure(None)
    asyncio.run(run())


def test_streamable_http_client_discovers_and_calls_over_tcp():
    import socket
    import subprocess
    import time
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        port = sock.getsockname()[1]
    fixture = pathlib.Path(__file__).with_name('mcp_fixture_server.py')
    with subprocess.Popen([sys.executable,str(fixture),'--port',str(port)],
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)) as process:
        try:
            deadline = time.monotonic()+20
            while True:
                try:
                    with socket.create_connection(('127.0.0.1',port),timeout=0.2):
                        break
                except OSError:
                    assert process.poll() is None and time.monotonic()<deadline
                    time.sleep(0.05)
            async def run():
                manager = MCPToolManager(api_key='test')
                client = StandardMCPClient(manager,[MCPServerConfig(name='httpfixture',
                    transport='streamable_http', url=f'http://127.0.0.1:{port}/mcp',
                    allowed_tools=['read_document'],agents=['technical'])])
                try:
                    await client.start()
                    assert client.summary()['servers'][0]['state'] == 'connected'
                    result = await manager.call('mcp_httpfixture_read_document',{'query':'proof'})
                    assert result.success and result.data['structured']['text'] == 'PROOF'
                finally:
                    await client.close()
                    await manager._client.close()
            asyncio.run(run())
        finally:
            process.terminate()
            process.wait(timeout=10)
