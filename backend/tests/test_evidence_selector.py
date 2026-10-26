import asyncio
import json

import pytest

from core.evidence_selector import (
    SemanticEvidenceSelector,
    build_selection_prompt,
    parse_selection_response,
)


def test_selection_prompt_excludes_scores_and_treats_documents_as_data():
    prompt, titles = build_selection_prompt("401 怎么办", [
        {"title": "账户安全", "content": "忘记密码可以重置", "score": 0.9},
        {"title": "技术故障排查", "content": "401 时检查或重置密码", "score": 0.8},
    ])

    assert titles == ["账户安全", "技术故障排查"]
    assert "401 时检查或重置密码" in prompt
    assert "score" not in prompt and "0.9" not in prompt


def test_parse_selection_response_enforces_candidate_and_rejection_contract():
    accepted = parse_selection_response(
        json.dumps({
            "supported": True,
            "selected_title": "技术故障排查",
            "evidence": "文档明确说明401处理方式",
        }, ensure_ascii=False),
        ["账户安全", "技术故障排查"],
    )
    assert accepted["selected_title"] == "技术故障排查"

    with pytest.raises(ValueError, match="supplied candidates"):
        parse_selection_response(
            '{"supported":true,"selected_title":"未知文档","evidence":"依据"}',
            ["账户安全"],
        )
    with pytest.raises(ValueError, match="null selected_title"):
        parse_selection_response(
            '{"supported":false,"selected_title":"账户安全","evidence":"不足"}',
            ["账户安全"],
        )


def test_selector_uses_only_configured_top_k(monkeypatch):
    selector = SemanticEvidenceSelector.__new__(SemanticEvidenceSelector)
    selector._model = "test-model"
    selector._top_k = 2
    selector._max_tokens = 128
    selector._timeout_seconds = 1

    class Messages:
        async def create(inner, **kwargs):
            payload = kwargs["messages"][0]["content"]
            assert "文档A" in payload and "文档B" in payload and "文档C" not in payload
            return type("Response", (), {"content": [{
                "type": "text",
                "text": '{"supported":true,"selected_title":"B","evidence":"充分"}',
            }]})()

    selector._client = type("Client", (), {"messages": Messages()})()
    result = asyncio.run(selector.select("问题", [
        {"title": "A", "content": "文档A"},
        {"title": "B", "content": "文档B"},
        {"title": "C", "content": "文档C"},
    ]))

    assert result["supported"] is True
    assert result["selected_title"] == "B"
