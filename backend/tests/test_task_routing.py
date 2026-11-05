"""Development regression for context updates, mixed requests and memory flow."""
import asyncio
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from agents.agent_orchestrator import (
    AgentOrchestrator, AgentType, BillingAgent, EscalationAgent, GeneralAgent,
    Request, TechnicalAgent,
)
from agents.tools import AgentToolSpec
from core.intent_recognizer import IntentCategory, UrgencyLevel
from core.task_routing import parse_order_context_update


@pytest.mark.parametrize("message,order_id", [
    ("请记住我的订单号是 EM-2026-1018，我正在申请退款。", "EM-2026-1018"),
    ("请记住订单号：#12345。", "12345"),
    ("帮我记下订单编号 ABCD_2026", "ABCD_2026"),
    ("请帮我记录一下我的订单号为 EM-2026-1018！", "EM-2026-1018"),
    ("保存订单号 EM-2026-1018，我目前正在申请退货", "EM-2026-1018"),
    ("我的订单号是 EM-2026-1018，请帮我记住。", "EM-2026-1018"),
])
def test_explicit_context_update_extracts_only_the_supplied_order_id(message, order_id):
    assert parse_order_context_update(message) == order_id


@pytest.mark.parametrize("message", [
    "请记住我的订单号是 EM-2026-1018，退款多久到账？",
    "请记住我的订单号是 EM-2026-1018，退款多久到账",
    "请记住订单号 EM-2026-1018，并帮我申请退款。",
    "请记住订单号 EM-2026-1018，直接退给我。",
    "请记住订单号 EM-2026-1018，转人工。",
    "请记住订单号 EM-2026-1018，登录一直提示401。",
    "请记住订单号 EM-2026-1018，我的银行卡已注销。",
    "请记住订单号 EM-2026-1018，我正在申请退款，最晚什么时候到账？",
    "不要记住订单号 EM-2026-1018。",
    "请不要保存订单号 EM-2026-1018。",
    "请忘记订单号 EM-2026-1018。",
    "请记住我的密码是 secret123。",
    "请记住我的银行卡号是 1234567890123456。",
    "请记住订单号 EM-2026-1018 和 EM-2026-1019。",
    "请记住我的订单号。",
    "请记住订单号 ABC。",
    "请记住订单号 " + "A" * 33,
    "订单号是 EM-2026-1018",  # Could be answering a previous clarification.
    "退款审核需要多久？",
    "",
])
def test_mixed_ambiguous_and_sensitive_messages_keep_the_normal_path(message):
    assert parse_order_context_update(message) is None


def make_orchestrator():
    client = SimpleNamespace(messages=SimpleNamespace(
        create=AsyncMock(side_effect=AssertionError("unexpected model call"))))
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._pool = {
        AgentType.GENERAL: [GeneralAgent(client, "test-model")],
        AgentType.TECHNICAL: [TechnicalAgent(client, "test-model")],
        AgentType.BILLING: [BillingAgent(client, "test-model")],
        AgentType.ESCALATION: [EscalationAgent(client, "test-model")],
    }
    orchestrator._recent_tool_traces = deque(maxlen=200)
    return orchestrator, client


def request(message, **overrides):
    values = dict(message=message, user_id="user-1", conv_id="conversation-1",
                  intent=IntentCategory.REFUND, intent_group="billing",
                  urgency=UrgencyLevel.LOW, intent_confidence=.95)
    values.update(overrides)
    return Request(**values)


def test_context_update_acks_before_rag_or_generation_and_records_the_task():
    async def run():
        orchestrator, client = make_orchestrator()
        orchestrator._execute = AsyncMock(side_effect=AssertionError("unexpected Agent execution"))
        result = await orchestrator.run(request(
            "请记住我的订单号是 EM-2026-1018，我正在申请退款。",
            entities={"order_id": ["MODEL_INVENTED_ID"]},
        ))
        assert "EM-2026-1018" in result.response
        assert "MODEL_INVENTED_ID" not in result.response
        assert "未找到足够" not in result.response
        assert "永久" not in result.response and "已保存" not in result.response
        assert result.primary_agent == AgentType.BILLING
        assert result.tools_used == result.tool_traces == []
        assert result.failure_kind == "" and not result.escalated
        assert "task=context_update" in result.routing_reason
        assert orchestrator.get_tool_trace(result.request_id)["routing_reason"] == result.routing_reason
        orchestrator._execute.assert_not_awaited()
        client.messages.create.assert_not_awaited()
    asyncio.run(run())


def test_explicit_context_update_does_not_need_generic_intent_clarification():
    async def run():
        orchestrator, client = make_orchestrator()
        result = await orchestrator.run(request(
            "请记住订单号 EM-2026-1018", intent=IntentCategory.OTHER,
            intent_group="other", intent_confidence=.1,
        ))
        assert "EM-2026-1018" in result.response
        assert "task=context_update" in result.routing_reason
        client.messages.create.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize("overrides", [
    {"urgency": UrgencyLevel.CRITICAL},
    {"intent": IntentCategory.HUMAN_HANDOFF},
    {"intent": IntentCategory.ESCALATION},
])
def test_context_update_preserves_escalation_priority(overrides):
    async def run():
        orchestrator, client = make_orchestrator()
        result = await orchestrator.run(request("请记住订单号 EM-2026-1018", **overrides))
        assert result.primary_agent == AgentType.ESCALATION
        assert result.escalated
        assert "task=context_update" not in result.routing_reason
        client.messages.create.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize("message", [
    "请记住我的订单号是 EM-2026-1018，退款多久到账？",
    "请记住我的订单号是 EM-2026-1018，我正在申请退款，审核需要多久",
    "这个订单审核通过后多久能到账？",
])
def test_policy_questions_still_retrieve_and_stop_without_evidence(message):
    async def run():
        orchestrator, client = make_orchestrator()
        search = AsyncMock(return_value={"success": True, "results": [], "evidence_accepted": False})
        tool = AgentToolSpec(
            name="search_knowledge_base", description="test search",
            input_schema={"type": "object", "properties": {
                "query": {"type": "string"}, "top_k": {"type": "integer"}},
                "required": ["query", "top_k"]}, handler=search,
        )
        orchestrator.set_shared_tools({tool.name: tool})
        result = await orchestrator.run(request(message))
        search.assert_awaited_once()
        assert "未找到足够的可核验依据" in result.response
        assert result.tool_traces[0]["evidence_found"] is False
        assert "task=context_update" not in result.routing_reason
        client.messages.create.assert_not_awaited()
    asyncio.run(run())


def test_api_ack_uses_existing_memory_write_and_followup_context(monkeypatch):
    # Import without loading the developer's personal .env files.
    with patch("dotenv.load_dotenv", return_value=False):
        import api.main as api
    from memory.conversation_memory import MemoryContext, Message, MsgRole

    async def run():
        orchestrator, client = make_orchestrator()
        memory = SimpleNamespace(messages={})

        async def get_context(user_id, conv_id, query=""):
            return MemoryContext(list(memory.messages.get((user_id, conv_id), [])), [], {}, "")

        async def add_message(user_id, conv_id, role, content):
            memory.messages.setdefault((user_id, conv_id), []).append(Message(role, content))

        memory.get_context = AsyncMock(side_effect=get_context)
        memory.add_message = AsyncMock(side_effect=add_message)
        memory.update_profile = AsyncMock()
        recognized = SimpleNamespace(
            intent=IntentCategory.REFUND, intent_group="billing", urgency=UrgencyLevel.LOW,
            confidence=.95, source_scores={}, entities={"order_id": ["EM-2026-1018"]},
        )
        orchestrator._intent_recognizer = SimpleNamespace(recognize=AsyncMock(return_value=recognized))
        search = AsyncMock(return_value={"success": True, "results": [], "evidence_accepted": False})
        orchestrator.set_shared_tools({"search_knowledge_base": AgentToolSpec(
            name="search_knowledge_base", description="test search",
            input_schema={"type": "object", "properties": {
                "query": {"type": "string"}, "top_k": {"type": "integer"}},
                "required": ["query", "top_k"]}, handler=search,
        )})
        monkeypatch.setattr(api, "_orchestrator", orchestrator)
        monkeypatch.setattr(api, "_memory", memory)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as http:
            first = await http.post("/chat", json={
                "message": "请记住我的订单号是 EM-2026-1018，我正在申请退款。",
                "user_id": "user-1", "conv_id": "conversation-1",
            })
            assert first.status_code == 200
            assert not first.json()["knowledge_used"]
            assert "EM-2026-1018" in first.json()["response"]
            assert "task=context_update" in first.json()["routing_reason"]
            assert [m.role for m in memory.messages[("user-1", "conversation-1")]] == [MsgRole.USER, MsgRole.ASSISTANT]
            search.assert_not_awaited()

            followup = await http.post("/chat", json={
                "message": "这个订单审核通过后多久能到账？", "user_id": "user-1", "conv_id": "conversation-1"})
            assert followup.status_code == 200
            history = orchestrator._intent_recognizer.recognize.await_args.kwargs["history"]
            assert "EM-2026-1018" in history[0]["content"]
            assert "未找到足够" in followup.json()["response"]
            search.assert_awaited_once()
        client.messages.create.assert_not_awaited()
        await asyncio.sleep(0)  # Let the mocked background profile jobs finish.
    asyncio.run(run())
