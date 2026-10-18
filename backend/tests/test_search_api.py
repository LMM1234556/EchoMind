import asyncio

from mcp.tool_manager import ToolResult


def test_search_defaults_to_local_mode_without_llm(monkeypatch):
    from api import main

    class Manager:
        def __init__(self):
            self.local_calls = []

        async def call(self, tool_name, params, use_cache=True):
            self.local_calls.append((tool_name, dict(params), use_cache))
            return ToolResult(
                success=True,
                data=[{"title": "退款政策", "content": "支持退款"}],
                tool_name=tool_name,
                cached=True,
            )

        async def search_with_rewrite(self, *args, **kwargs):
            raise AssertionError("local 模式不应调用查询改写或 LLM 重排")

    manager = Manager()
    monkeypatch.setattr(main, "_tool_manager", manager)

    response = asyncio.run(main.search("买错东西后想退掉", top_k=20))

    assert manager.local_calls == [
        ("knowledge_search", {"query": "买错东西后想退掉", "top_k": 5}, True)
    ]
    assert response["success"] is True
    assert response["mode"] == "local"
    assert response["top_k"] == 5
    assert response["reranked"] is False
    assert response["cached"] is True
    assert response["llm_assisted"] is False
    assert response["evidence_status"] == "unverified_candidates"


def test_search_enhanced_mode_is_explicit_and_marked_llm_assisted(monkeypatch):
    from api import main

    class Manager:
        async def call(self, *args, **kwargs):
            raise AssertionError("enhanced 模式应走完整 RAG")

        async def search_with_rewrite(self, tool_name, query, top_k=5):
            assert (tool_name, query, top_k) == ("knowledge_search", "退款时效", 3)
            return ToolResult(
                success=True,
                data=[{"title": "退款政策", "content": "1-3 日审核"}],
                tool_name=tool_name,
                reranked=True,
            )

    monkeypatch.setattr(main, "_tool_manager", Manager())

    response = asyncio.run(main.search("退款时效", top_k=3, mode="enhanced"))

    assert response["mode"] == "enhanced"
    assert response["reranked"] is True
    assert response["llm_assisted"] is True
    assert response["evidence_status"] == "unverified_candidates"
