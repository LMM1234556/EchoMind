import asyncio
import logging
from types import SimpleNamespace

from mcp.tool_manager import MCPToolManager, ToolResult


class FakeRerankClient:
    def __init__(self, text):
        self.text = text

        class Messages:
            async def create(inner, **kwargs):
                return SimpleNamespace(
                    content=[{"type": "text", "text": self.text}]
                )

        self.messages = Messages()


def test_search_with_rewrite_deduplicates_score_drift_without_dropping_chunks(caplog):
    manager = MCPToolManager.__new__(MCPToolManager)
    captured = {"recall_k": [], "merged": []}

    async def fake_rewrite(query, n=3):
        return [query, "退款审核", "退款到账"]

    async def fake_call(tool_name, params, context=None, use_cache=True):
        captured["recall_k"].append(params["top_k"])
        if params["query"] == "退款审核":
            # 与第一路的首条文档相同，只是分数不同、字段顺序不同。
            items = [{"score": 0.91, "chunk": 0, "content": "审核需要 1-3 个工作日", "title": "退款规则"}]
        elif params["query"] == "退款到账":
            # 标题相同但正文和分块不同，必须保留。
            items = [{"title": "退款规则", "content": "审核通过后 5-7 个工作日到账", "chunk": 1, "score": 0.85}]
        else:
            items = [{"title": "退款规则", "content": "审核需要 1-3 个工作日", "score": 0.90, "chunk": 0}]
        return ToolResult(success=True, data=items, tool_name=tool_name)

    async def fake_rerank(query, items, top_k):
        captured["merged"] = items
        return items[:top_k]

    manager.rewrite_query = fake_rewrite
    manager.call = fake_call
    manager._rerank = fake_rerank

    with caplog.at_level(logging.INFO, logger="mcp.tool_manager"):
        result = asyncio.run(
            manager.search_with_rewrite("knowledge_search", "退款要多久", top_k=2)
        )

    assert result.success is True
    assert captured["recall_k"] == [5, 5, 5]
    assert len(captured["merged"]) == 2
    assert [item["chunk"] for item in captured["merged"]] == [0, 1]
    assert captured["merged"][0]["score"] == 0.90
    assert len(result.data) == 2
    timing_logs = [record.getMessage() for record in caplog.records
                   if "RAG_TIMING" in record.getMessage()]
    assert len(timing_logs) == 1
    assert "outcome=success" in timing_logs[0]
    assert "sub_queries=3 candidates=3 merged=2 final=2" in timing_logs[0]
    for field in ("rewrite_ms=", "recall_ms=", "merge_ms=", "rerank_ms=", "total_ms="):
        assert field in timing_logs[0]


def test_search_with_rewrite_handles_duplicate_non_dict_items():
    manager = MCPToolManager.__new__(MCPToolManager)
    captured = {"merged": []}

    async def fake_rewrite(query, n=3):
        return [query, "same"]

    async def fake_call(tool_name, params, context=None, use_cache=True):
        return ToolResult(success=True, data=["同一文本"], tool_name=tool_name)

    async def fake_rerank(query, items, top_k):
        captured["merged"] = items
        return items[:top_k]

    manager.rewrite_query = fake_rewrite
    manager.call = fake_call
    manager._rerank = fake_rerank

    result = asyncio.run(
        manager.search_with_rewrite("knowledge_search", "查询", top_k=3)
    )

    assert result.success is True
    assert captured["merged"] == ["同一文本"]
    assert result.data == ["同一文本"]


def test_rerank_filters_bad_indexes_and_fills_missing_candidates():
    manager = MCPToolManager.__new__(MCPToolManager)
    manager._client = FakeRerankClient('[1, 1, 99, 0, true, "2"]')
    manager._model = "test-model"
    items = ["A", "D", "B", "C"]

    result = asyncio.run(manager._rerank("退款时效", items, top_k=3))

    assert result == ["D", "A", "B"]


def test_rerank_falls_back_to_original_order_for_malformed_output():
    manager = MCPToolManager.__new__(MCPToolManager)
    manager._client = FakeRerankClient("not-json")
    manager._model = "test-model"
    items = ["A", "D", "B", "C"]

    result = asyncio.run(manager._rerank("退款时效", items, top_k=3))

    assert result == ["A", "D", "B"]
