import asyncio
import time
from collections import deque

import httpx
from anthropic import APIConnectionError

from agents.agent_orchestrator import (
    AgentProfile,
    AgentResponse,
    AgentType,
    AgentOrchestrator,
    BaseAgent,
    BillingAgent,
    EscalationAgent,
    GeneralAgent,
    Request,
    ResponseComposer,
    RoutingDecision,
    TechnicalAgent,
    build_shared_rag_tools,
)
from core.intent_recognizer import IntentCategory, UrgencyLevel
from core.knowledge_routing import should_use_knowledge


class FakeClient:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

        class Messages:
            async def create(inner, **kwargs):
                self.calls.append(kwargs)
                if self.error:
                    raise self.error
                return self.response

        self.messages = Messages()


def make_request(**kwargs):
    values = {
        "message": "登录时报 401，同时这笔订单被重复扣款",
        "user_id": "u1",
        "conv_id": "c1",
        "intent": IntentCategory.TECHNICAL_LOGIN,
        "intent_group": "technical",
        "urgency": UrgencyLevel.HIGH,
        "intent_confidence": 0.92,
        "entities": {"error_code": ["401"], "amount": ["99 元"]},
    }
    values.update(kwargs)
    return Request(**values)


def test_agent_profiles_have_distinct_contracts_and_generation_config():
    assert isinstance(GeneralAgent.profile, AgentProfile)
    assert GeneralAgent.profile.role != TechnicalAgent.profile.role
    assert TechnicalAgent.profile.workflow != BillingAgent.profile.workflow
    assert TechnicalAgent.profile.temperature < GeneralAgent.profile.temperature
    assert "search_knowledge_base" in GeneralAgent.profile.tool_scope
    assert "lookup_error_code" in TechnicalAgent.profile.tool_scope
    assert "check_billing_fields" in BillingAgent.profile.tool_scope


def test_domain_agents_build_different_role_packets():
    req = make_request()
    general_packet = GeneralAgent(FakeClient(), "test-model")._build_role_packet(req)
    technical_packet = TechnicalAgent(FakeClient(), "test-model")._build_role_packet(req)
    billing_packet = BillingAgent(FakeClient(), "test-model")._build_role_packet(req)

    assert "triage_targets" in general_packet
    assert "diagnostic_fields" in technical_packet
    assert "verification_fields" in billing_packet
    assert general_packet != technical_packet != billing_packet


def test_escalation_agent_is_a_real_non_llm_handoff_node():
    client = FakeClient()
    agent = EscalationAgent(client, "test-model")

    result = asyncio.run(agent.handle(make_request(
        intent=IntentCategory.HUMAN_HANDOFF,
        urgency=UrgencyLevel.CRITICAL,
    )))

    assert result.success is True
    assert result.escalate is True
    assert "人工升级" in result.content
    assert client.calls == []


def test_composer_fallback_preserves_primary_and_supporting_results():
    composer = ResponseComposer(FakeClient(error=RuntimeError("provider down")), "test-model")
    req = make_request()
    responses = [
        AgentResponse(AgentType.TECHNICAL, "先排查 Token 是否过期。", True),
        AgentResponse(AgentType.BILLING, "请提供两笔扣款的时间和金额。", True),
    ]

    content = asyncio.run(composer.compose(req, responses))

    assert content.startswith("先排查 Token 是否过期。")
    assert "补充说明" in content
    assert "两笔扣款" in content


def test_routing_decision_can_target_escalation_pool():
    # Keep this assertion close to the public data contract used by the API.
    decision = RoutingDecision(
        primary_agent=AgentType.ESCALATION,
        reason="critical request",
        confidence=1.0,
    )
    assert decision.agent_types == [AgentType.ESCALATION]
    assert not decision.multi_agent


def test_composite_request_routes_explicit_billing_signal_as_supporting_agent():
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._pool = {
        AgentType.GENERAL: [object()],
        AgentType.TECHNICAL: [object()],
        AgentType.BILLING: [object()],
    }

    decision = orchestrator._route_decision(make_request())

    assert decision.primary_agent is AgentType.TECHNICAL
    assert decision.supporting_agents == [AgentType.BILLING]
    assert decision.multi_agent is True


def test_broad_composite_intent_uses_only_available_domain_agents():
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._pool = {
        AgentType.GENERAL: [object()],
        AgentType.TECHNICAL: [object()],
        AgentType.BILLING: [object()],
    }
    for intent in (
        IntentCategory.REQUEST,
        IntentCategory.QUERY,
        IntentCategory.COMPLAINT,
        IntentCategory.OTHER,
    ):
        req = make_request(
            message="我登录时提示401，而且订单被重复扣款99元。请分别简要说明技术排查和账单核验步骤。",
            intent=intent,
            intent_group="request",
            intent_confidence=0.574,
            entities={"amount": ["99元"], "error_code": []},
        )
        decision = orchestrator._route_decision(req)

        assert decision.primary_agent is AgentType.BILLING
        assert decision.supporting_agents == [AgentType.TECHNICAL]
        assert AgentType.GENERAL not in decision.agent_types
        assert "复合领域优先" in decision.reason
        assert req.intent is intent  # Routing must preserve the recognizer's output.


def test_composite_domain_preference_preserves_escalation_and_missing_pool_behavior():
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._pool = {
        AgentType.GENERAL: [object()],
        AgentType.TECHNICAL: [object()],
        AgentType.BILLING: [object()],
    }
    for req in (
        make_request(intent=IntentCategory.REQUEST, urgency=UrgencyLevel.CRITICAL),
        make_request(intent=IntentCategory.HUMAN_HANDOFF, urgency=UrgencyLevel.MEDIUM),
    ):
        decision = orchestrator._route_decision(req)
        assert decision.agent_types == [AgentType.ESCALATION]

    # With no available specialists, retain the existing General fallback.
    orchestrator._pool = {AgentType.GENERAL: [object()]}
    decision = orchestrator._route_decision(make_request(intent=IntentCategory.REQUEST))
    assert decision.agent_types == [AgentType.GENERAL]

    orchestrator._pool = {
        AgentType.GENERAL: [object()],
        AgentType.TECHNICAL: [object()],
        AgentType.BILLING: [],
    }
    decision = orchestrator._route_decision(make_request(intent=IntentCategory.REQUEST))
    assert AgentType.BILLING not in decision.agent_types


def test_broad_composite_request_runs_two_specialists_and_preserves_knowledge_access():
    class ComposerStub:
        async def compose(self, req, responses):
            return " | ".join(response.content for response in responses)

    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._pool = {
        AgentType.GENERAL: [object()],
        AgentType.TECHNICAL: [object()],
        AgentType.BILLING: [object()],
    }
    orchestrator._composer = ComposerStub()
    orchestrator._recent_tool_traces = deque(maxlen=10)
    calls = []

    async def execute(req, agent_type):
        calls.append(agent_type)
        return AgentResponse(agent_type, f"{agent_type.value} result", True)

    orchestrator._execute = execute
    req = make_request(
        message="我登录时提示401，而且订单被重复扣款99元。请分别简要说明技术排查和账单核验步骤。",
        intent=IntentCategory.REQUEST,
        intent_group="request",
        intent_confidence=0.574,
        entities={"amount": ["99元"]},
    )

    result = asyncio.run(orchestrator.run(req))

    assert calls == [AgentType.BILLING, AgentType.TECHNICAL]
    assert result.intent is IntentCategory.REQUEST
    assert result.primary_agent is AgentType.BILLING
    assert result.supporting_agents == [AgentType.TECHNICAL]
    assert result.response == "billing result | technical result"
    assert "agent_general_ms" not in result.stage_timings
    assert orchestrator.get_tool_trace(req.request_id)["primary_agent"] == "billing"

    # Correcting routing preserves knowledge-tool access; it does not force a
    # search or overwrite the classifier's original intent/confidence.
    shared = build_shared_rag_tools(object())
    for agent_class in (BillingAgent, TechnicalAgent):
        agent = agent_class(FakeClient(), "test-model")
        agent.set_shared_tools(shared)
        assert "search_knowledge_base" in agent._tools_for_request(req)


def test_parallel_agents_overlap_in_time_and_preserve_roles():
    class ComposerStub:
        async def compose(self, req, responses):
            return " | ".join(response.content for response in responses)

    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._composer = ComposerStub()
    orchestrator._recent_tool_traces = deque(maxlen=10)
    starts = {}
    finishes = {}

    async def delayed_execute(req, agent_type):
        starts[agent_type] = time.monotonic()
        await asyncio.sleep(0.08)
        finishes[agent_type] = time.monotonic()
        return AgentResponse(
            agent_type=agent_type,
            content=f"{agent_type.value} result",
            success=True,
        )

    orchestrator._execute = delayed_execute
    decision = RoutingDecision(
        primary_agent=AgentType.TECHNICAL,
        supporting_agents=[AgentType.BILLING],
        reason="composite technical and billing request",
        confidence=0.92,
    )

    started = time.monotonic()
    result = asyncio.run(orchestrator.run_parallel(make_request(), decision))
    elapsed = time.monotonic() - started

    # Both executions must have started before either one finished. This proves
    # overlap directly and is more robust than relying only on a wall-clock limit.
    assert max(starts.values()) < min(finishes.values())
    assert elapsed < 0.14  # Sequential execution would take at least 0.16 s.
    assert result.primary_agent is AgentType.TECHNICAL
    assert result.supporting_agents == [AgentType.BILLING]
    assert result.agent_types == [AgentType.TECHNICAL, AgentType.BILLING]
    assert result.response == "technical result | billing result"
    # Event-loop timers may wake a few milliseconds early on Windows, so use a
    # tolerance while still checking the expected order of magnitude.
    assert 60 <= result.stage_timings["agent_technical_ms"] < 140
    assert 60 <= result.stage_timings["agent_billing_ms"] < 140
    assert 60 <= result.stage_timings["parallel_agents_ms"] < 140
    assert result.stage_timings["compose_ms"] >= 0

    orchestrator.update_tool_trace_stage_timings(
        result.request_id,
        {**result.stage_timings, "api_total_ms": 123.4},
    )
    trace = orchestrator.get_tool_trace(result.request_id)
    assert trace["stage_timings"]["api_total_ms"] == 123.4


def test_parallel_request_keeps_successful_result_when_one_agent_raises():
    client = FakeClient()
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._composer = ResponseComposer(client, "test-model")
    orchestrator._recent_tool_traces = deque(maxlen=10)

    async def partially_failing_execute(req, agent_type):
        if agent_type is AgentType.BILLING:
            raise RuntimeError("billing unavailable")
        return AgentResponse(
            agent_type=agent_type,
            content="technical result remains available",
            success=True,
        )

    orchestrator._execute = partially_failing_execute
    decision = RoutingDecision(
        primary_agent=AgentType.TECHNICAL,
        supporting_agents=[AgentType.BILLING],
        reason="composite technical and billing request",
        confidence=0.92,
    )

    result = asyncio.run(orchestrator.run_parallel(make_request(), decision))

    assert result.response == "technical result remains available"
    assert result.agent_types == [AgentType.TECHNICAL]
    assert result.primary_agent is AgentType.TECHNICAL
    assert result.supporting_agents == [AgentType.BILLING]
    # With only one successful response, the composer returns it directly and
    # avoids an unnecessary third model call.
    assert client.calls == []


def test_agent_tool_scopes_are_real_and_isolated():
    general_tools = set(GeneralAgent(FakeClient(), "test-model").get_tools())
    technical_tools = set(TechnicalAgent(FakeClient(), "test-model").get_tools())
    billing_tools = set(BillingAgent(FakeClient(), "test-model").get_tools())
    escalation_tools = set(EscalationAgent(FakeClient(), "test-model").get_tools())

    assert general_tools == {"inspect_request_context", "suggest_required_fields"}
    assert technical_tools == {"lookup_error_code", "build_diagnostic_plan"}
    assert billing_tools == {"check_billing_fields", "compare_amounts"}
    assert escalation_tools == {"create_handoff_summary"}
    assert not general_tools & technical_tools
    assert not technical_tools & billing_tools


def test_shared_rag_tool_is_available_to_all_agents():
    class RagManager:
        async def search_with_rewrite(self, tool_name, query, top_k=5):
            return type(
                "Result",
                (),
                {"success": True, "data": [{"title": "退款政策", "content": "7 天内可退款"}], "reranked": True},
            )()

    shared = build_shared_rag_tools(RagManager())

    general = GeneralAgent(FakeClient(), "test-model")
    technical = TechnicalAgent(FakeClient(), "test-model")
    billing = BillingAgent(FakeClient(), "test-model")
    escalation = EscalationAgent(FakeClient(), "test-model")

    for agent in (general, technical, billing, escalation):
        agent.set_shared_tools(shared)
        tools = agent.get_tools()
        assert "search_knowledge_base" in tools


def test_clear_business_intent_uses_fast_keyword_rag_without_llm_rag_stages():
    class RagManager:
        def __init__(self):
            self.fast_calls = []
            self.full_calls = []

        async def call(self, tool_name, params, use_cache=True):
            self.fast_calls.append((tool_name, params, use_cache))
            return type(
                "Result",
                (),
                {"success": True, "data": [{"title": "退款政策", "content": "1-3 日审核"}], "reranked": False},
            )()

        async def search_with_rewrite(self, tool_name, query, top_k=5):
            self.full_calls.append((tool_name, query, top_k))
            raise AssertionError("快速路径命中后不应调用完整 RAG")

    manager = RagManager()
    spec = build_shared_rag_tools(manager)["search_knowledge_base"]
    req = make_request(message="退款审核和到账需要多久", intent=IntentCategory.REFUND)

    result = asyncio.run(spec.handler(req, {"query": "退款审核时效 退款到账时间", "top_k": 3}))

    assert result["success"] is True
    assert result["retrieval_mode"] == "fast_keyword"
    assert result["reranked"] is False
    assert manager.fast_calls[0][1]["mode"] == "keyword"
    assert manager.full_calls == []


def test_fast_rag_falls_back_to_full_rag_when_keyword_search_is_empty():
    class RagManager:
        async def call(self, tool_name, params, use_cache=True):
            return type("Result", (), {"success": True, "data": [], "reranked": False})()

        async def search_with_rewrite(self, tool_name, query, top_k=5):
            return type(
                "Result",
                (),
                {"success": True, "data": [{"title": "退款政策", "content": "可以申请退款"}], "reranked": True},
            )()

    spec = build_shared_rag_tools(RagManager())["search_knowledge_base"]
    req = make_request(message="买错东西后想退掉", intent=IntentCategory.REFUND)

    result = asyncio.run(spec.handler(req, {"query": "买错东西后想退掉", "top_k": 3}))

    assert result["success"] is True
    assert result["retrieval_mode"] == "full_rag"
    assert result["reranked"] is True
    assert result["evidence_accepted"] is True
    assert result["evidence_gate"] == "intent_title_match"


def test_high_confidence_intent_uses_grounded_fast_vector_before_llm_rag():
    class RagManager:
        def __init__(self):
            self.calls = []
            self.full_calls = []

        async def call(self, tool_name, params, use_cache=True):
            self.calls.append(dict(params))
            if params.get("mode") == "keyword":
                data = []
            else:
                data = [
                    {"title": "退款政策", "content": "审核 1-3 日，到账 5-7 日", "score": 0.76},
                    {"title": "配送说明", "content": "标准配送 3-5 日", "score": 0.51},
                ]
            return type("Result", (), {"success": True, "data": data, "reranked": False})()

        async def search_with_rewrite(self, tool_name, query, top_k=5):
            self.full_calls.append((tool_name, query, top_k))
            raise AssertionError("本地向量证据通过门控后不应调用 LLM RAG")

    manager = RagManager()
    spec = build_shared_rag_tools(manager)["search_knowledge_base"]
    req = make_request(
        message="买错东西后想退掉",
        intent=IntentCategory.REFUND,
        intent_confidence=0.92,
    )

    result = asyncio.run(spec.handler(req, {"query": req.message, "top_k": 12}))

    assert result["success"] is True
    assert result["top_k"] == 5
    assert result["retrieval_mode"] == "fast_vector"
    assert result["reranked"] is False
    assert result["candidate_count"] == 2
    assert [item["title"] for item in result["results"]] == ["退款政策"]
    assert manager.calls == [
        {"query": req.message, "top_k": 5, "mode": "keyword"},
        {"query": req.message, "top_k": 5},
    ]
    assert manager.full_calls == []


def test_fast_vector_wrong_domain_falls_back_to_full_rag():
    class RagManager:
        def __init__(self):
            self.call_count = 0
            self.full_calls = []

        async def call(self, tool_name, params, use_cache=True):
            self.call_count += 1
            data = [] if params.get("mode") == "keyword" else [
                {"title": "配送说明", "content": "标准配送 3-5 日"},
            ]
            return type("Result", (), {"success": True, "data": data, "reranked": False})()

        async def search_with_rewrite(self, tool_name, query, top_k=5):
            self.full_calls.append((tool_name, query, top_k))
            return type(
                "Result",
                (),
                {"success": True, "data": [{"title": "退款政策", "content": "支持退款"}], "reranked": True},
            )()

    manager = RagManager()
    spec = build_shared_rag_tools(manager)["search_knowledge_base"]
    req = make_request(message="想退掉买错的东西", intent=IntentCategory.REFUND)

    result = asyncio.run(spec.handler(req, {"query": req.message, "top_k": 3}))

    assert manager.call_count == 2
    assert manager.full_calls == [("knowledge_search", req.message, 3)]
    assert result["retrieval_mode"] == "full_rag"
    assert result["reranked"] is True


def test_low_confidence_intent_does_not_trust_single_vector_recall():
    class RagManager:
        def __init__(self):
            self.call_params = []

        async def call(self, tool_name, params, use_cache=True):
            self.call_params.append(dict(params))
            return type("Result", (), {"success": True, "data": [], "reranked": False})()

        async def search_with_rewrite(self, tool_name, query, top_k=5):
            return type(
                "Result",
                (),
                {"success": True, "data": [{"title": "退款政策", "content": "支持退款"}], "reranked": True},
            )()

    manager = RagManager()
    spec = build_shared_rag_tools(manager)["search_knowledge_base"]
    req = make_request(
        message="好像想退一个东西",
        intent=IntentCategory.REFUND,
        intent_confidence=0.55,
    )

    result = asyncio.run(spec.handler(req, {"query": req.message, "top_k": 3}))

    assert manager.call_params == [{"query": req.message, "top_k": 3, "mode": "keyword"}]
    assert result["retrieval_mode"] == "full_rag"
    assert result["reranked"] is True


def test_full_rag_rejects_nearest_document_from_wrong_business_domain():
    class RagManager:
        async def call(self, tool_name, params, use_cache=True):
            return type("Result", (), {"success": True, "data": [], "reranked": False})()

        async def search_with_rewrite(self, tool_name, query, top_k=5):
            return type(
                "Result",
                (),
                {
                    "success": True,
                    "data": [{"title": "配送说明", "content": "标准配送 3-5 个工作日"}],
                    "reranked": True,
                },
            )()

    spec = build_shared_rag_tools(RagManager())["search_knowledge_base"]
    req = make_request(message="退款什么时候到账", intent=IntentCategory.REFUND)

    result = asyncio.run(spec.handler(req, {"query": req.message, "top_k": 3}))

    assert result["success"] is True
    assert result["candidate_count"] == 1
    assert result["results"] == []
    assert result["evidence_accepted"] is False
    assert result["evidence_gate"] == "intent_title_mismatch"


def test_full_rag_marks_missing_invoice_document_as_knowledge_gap():
    class RagManager:
        async def call(self, tool_name, params, use_cache=True):
            return type("Result", (), {"success": True, "data": [], "reranked": False})()

        async def search_with_rewrite(self, tool_name, query, top_k=5):
            return type(
                "Result",
                (),
                {
                    "success": True,
                    "data": [{"title": "退款政策", "content": "1-3 个工作日审核"}],
                    "reranked": True,
                },
            )()

    spec = build_shared_rag_tools(RagManager())["search_knowledge_base"]
    req = make_request(message="电子发票多久开出", intent=IntentCategory.INVOICE)

    result = asyncio.run(spec.handler(req, {"query": req.message, "top_k": 3}))

    assert result["candidate_count"] == 0
    assert result["results"] == []
    assert result["evidence_gate"] == "knowledge_gap"
    assert result["retrieval_mode"] == "knowledge_gap"


def test_semantic_selector_can_choose_second_grounded_candidate():
    class RagManager:
        async def call(self, tool_name, params, use_cache=True):
            return type("Result", (), {"success": True, "data": [], "reranked": False})()

        async def search_with_rewrite(self, tool_name, query, top_k=5):
            return type("Result", (), {
                "success": True,
                "data": [
                    {"title": "订单查询", "content": "查询订单状态"},
                    {"title": "配送说明", "content": "偏远地区增加2-3天"},
                ],
                "reranked": True,
            })()

    class Selector:
        def __init__(self):
            self.calls = []

        async def select(self, query, candidates):
            self.calls.append((query, candidates))
            return {
                "success": True,
                "supported": True,
                "selected_title": "配送说明",
                "evidence": "文档明确说明偏远地区时效",
            }

    selector = Selector()
    spec = build_shared_rag_tools(RagManager(), evidence_selector=selector)["search_knowledge_base"]
    req = make_request(
        message="偏远地区配送要多几天",
        intent=IntentCategory.LOGISTICS,
        intent_confidence=0.6,
    )

    result = asyncio.run(spec.handler(req, {"query": req.message, "top_k": 3}))

    assert result["evidence_accepted"] is True
    assert result["evidence_gate"] == "semantic_topk_selected"
    assert result["evidence_selected_title"] == "配送说明"
    assert [item["title"] for item in result["results"]] == ["配送说明"]
    assert len(selector.calls) == 1


def test_semantic_selector_rejection_and_failure_are_fail_closed():
    class RagManager:
        async def call(self, tool_name, params, use_cache=True):
            return type("Result", (), {
                "success": True,
                "data": [{"title": "退款政策", "content": "7天无理由退款"}],
                "reranked": False,
            })()

        async def search_with_rewrite(self, tool_name, query, top_k=5):
            raise AssertionError("关键词结果不应回退")

    class Rejector:
        async def select(self, query, candidates):
            return {
                "success": True,
                "supported": False,
                "selected_title": None,
                "evidence": "文档没有优惠券返还规则",
            }

    class BrokenSelector:
        async def select(self, query, candidates):
            raise TimeoutError("provider timeout")

    req = make_request(message="退款后优惠券退吗", intent=IntentCategory.REFUND)
    rejected = asyncio.run(
        build_shared_rag_tools(RagManager(), Rejector())["search_knowledge_base"].handler(
            req, {"query": req.message, "top_k": 3}
        )
    )
    failed = asyncio.run(
        build_shared_rag_tools(RagManager(), BrokenSelector())["search_knowledge_base"].handler(
            req, {"query": req.message, "top_k": 3}
        )
    )

    assert rejected["results"] == []
    assert rejected["evidence_accepted"] is False
    assert rejected["evidence_gate"] == "semantic_unsupported"
    assert failed["results"] == []
    assert failed["evidence_accepted"] is False
    assert failed["evidence_gate"] == "semantic_selector_failed"
    assert failed["evidence_selector_success"] is False


def test_semantic_selector_can_handle_general_knowledge_query_without_title_mapping():
    class RagManager:
        async def search_with_rewrite(self, tool_name, query, top_k=5):
            return type("Result", (), {
                "success": True,
                "data": [
                    {"title": "会员与积分", "content": "100积分等于1元"},
                    {"title": "订单查询", "content": "用订单号查询"},
                ],
                "reranked": True,
            })()

    class Selector:
        async def select(self, query, candidates):
            assert [item["title"] for item in candidates] == ["会员与积分", "订单查询"]
            return {
                "success": True,
                "supported": True,
                "selected_title": "会员与积分",
                "evidence": "积分换算规则明确",
            }

    req = make_request(
        message="两百积分能抵几元",
        intent=IntentCategory.QUERY,
        intent_confidence=0.7,
    )
    spec = build_shared_rag_tools(RagManager(), Selector())["search_knowledge_base"]

    result = asyncio.run(spec.handler(req, {"query": req.message, "top_k": 3}))

    assert result["evidence_accepted"] is True
    assert result["evidence_gate"] == "semantic_topk_selected"
    assert result["evidence_selected_title"] == "会员与积分"


def test_duplicate_charge_does_not_accept_order_document_as_payment_evidence():
    class RagManager:
        async def call(self, tool_name, params, use_cache=True):
            return type(
                "Result",
                (),
                {
                    "success": True,
                    "data": [{"title": "订单查询", "content": "订单状态说明"}],
                    "reranked": False,
                },
            )()

        async def search_with_rewrite(self, tool_name, query, top_k=5):
            raise AssertionError("词面候选应由标题门控拒绝，无需远程重排")

    spec = build_shared_rag_tools(RagManager())["search_knowledge_base"]
    req = make_request(
        message="同一笔订单被重复扣款",
        intent=IntentCategory.PAYMENT_ISSUE,
        intent_confidence=0.55,
    )

    result = asyncio.run(spec.handler(req, {"query": req.message, "top_k": 3}))

    assert result["results"] == []
    assert result["evidence_accepted"] is False
    assert result["evidence_gate"] == "intent_title_mismatch"
    assert result["retrieval_mode"] == "fast_keyword"


def test_high_confidence_specific_intent_prefetches_knowledge_before_first_llm_call():
    class RagManager:
        async def call(self, tool_name, params, use_cache=True):
            return type(
                "Result",
                (),
                {
                    "success": True,
                    "data": [{"title": "退款政策", "content": "1-3 日审核，5-7 日到账"}],
                    "reranked": False,
                },
            )()

        async def search_with_rewrite(self, tool_name, query, top_k=5):
            raise AssertionError("关键词预取命中后不应调用完整 RAG")

    class TextBlock:
        type = "text"
        text = "退款审核需要 1-3 个工作日。"

    client = FakeClient(response=type("Response", (), {"content": [TextBlock()]})())
    agent = BillingAgent(client, "test-model")
    agent.set_shared_tools(build_shared_rag_tools(RagManager()))
    req = make_request(
        message="退款审核需要多久，审核通过后多久能到账？",
        intent=IntentCategory.REFUND,
        intent_group="billing",
        intent_confidence=0.93,
    )

    response = asyncio.run(agent.handle(req))

    assert response.success is True
    assert response.tools_used == ["search_knowledge_base"]
    assert response.tool_traces[0]["prefetched"] is True
    assert response.tool_traces[0]["invocation_source"] == "deterministic_prefetch"
    assert response.tool_traces[0]["retrieval_mode"] == "fast_keyword"
    assert len(client.calls) == 1
    first_call = client.calls[0]
    assert "预取知识库结果" in str(first_call["messages"])
    assert "search_knowledge_base" not in {tool["name"] for tool in first_call["tools"]}


def test_prefetch_scopes_composite_query_for_each_domain_agent():
    mixed_query = "我登录时提示 401，而且订单被重复扣款 99 元"

    class RagManager:
        def __init__(self):
            self.queries = []

        async def call(self, tool_name, params, use_cache=True):
            self.queries.append(params["query"])
            return type(
                "Result",
                (),
                {"success": True, "data": [{"content": "命中知识"}], "reranked": False},
            )()

        async def search_with_rewrite(self, tool_name, query, top_k=5):
            raise AssertionError("关键词检索命中后不应调用完整 RAG")

    class TextBlock:
        type = "text"
        text = "已根据知识库处理。"

    manager = RagManager()
    response = type("Response", (), {"content": [TextBlock()]})()
    request = make_request(
        message=mixed_query,
        intent=IntentCategory.PAYMENT_ISSUE,
        intent_group="billing",
        intent_confidence=0.93,
    )

    traces = []
    for agent_class in (TechnicalAgent, BillingAgent):
        agent = agent_class(FakeClient(response=response), "test-model")
        agent.set_shared_tools(build_shared_rag_tools(manager))
        result = asyncio.run(agent.handle(request))
        assert result.success is True
        traces.append(result.tool_traces[0]["input"]["query"])

    assert manager.queries == ["我登录时提示 401", "订单被重复扣款 99 元"]
    assert traces == manager.queries


def test_empty_policy_prefetch_returns_grounded_fallback_without_calling_model():
    class EmptyRagManager:
        async def call(self, tool_name, params, use_cache=True):
            return type("Result", (), {"success": True, "data": [], "reranked": False})()

        async def search_with_rewrite(self, tool_name, query, top_k=5):
            return type("Result", (), {"success": True, "data": [], "reranked": False})()

    class NoModelClient:
        def __init__(self):
            self.calls = 0

            class Messages:
                async def create(inner, **kwargs):
                    self.calls += 1
                    raise AssertionError("空知识证据不应继续调用生成模型")

            self.messages = Messages()

    client = NoModelClient()
    agent = BillingAgent(client, "test-model")
    agent.set_shared_tools(build_shared_rag_tools(EmptyRagManager()))
    result = asyncio.run(agent.handle(make_request(
        message="电子发票最晚几天可以开出来？",
        intent=IntentCategory.INVOICE,
        intent_group="billing",
        intent_confidence=0.93,
        entities={},
    )))

    assert result.success is True
    assert "未找到足够的可核验依据" in result.content
    assert client.calls == 0
    assert result.tool_traces[0]["evidence_found"] is False


def test_empty_technical_prefetch_still_allows_low_risk_diagnostics():
    class EmptyRagManager:
        async def call(self, tool_name, params, use_cache=True):
            return type("Result", (), {"success": True, "data": [], "reranked": False})()

        async def search_with_rewrite(self, tool_name, query, top_k=5):
            return type("Result", (), {"success": True, "data": [], "reranked": False})()

    class TextBlock:
        type = "text"
        text = "401 通常表示认证失败，请先确认登录状态和凭证是否过期。"

    client = FakeClient(response=type("Response", (), {"content": [TextBlock()]})())
    agent = TechnicalAgent(client, "test-model")
    agent.set_shared_tools(build_shared_rag_tools(EmptyRagManager()))
    result = asyncio.run(agent.handle(make_request(
        message="登录时提示 401，怎么排查？",
        intent=IntentCategory.TECHNICAL_LOGIN,
        intent_group="technical",
        intent_confidence=0.93,
        entities={"error_code": ["401"]},
    )))

    assert result.success is True
    assert "401" in result.content
    assert len(client.calls) == 1
    assert result.tool_traces[0]["evidence_found"] is False


def test_low_confidence_specific_technical_intent_still_prefetches_knowledge():
    class RagManager:
        async def call(self, tool_name, params, use_cache=True):
            return type("Result", (), {
                "success": True,
                "data": [{"title": "技术故障排查", "content": "401 通常表示认证失败"}],
                "reranked": False,
            })()

        async def search_with_rewrite(self, tool_name, query, top_k=5):
            raise AssertionError("关键词结果已命中，不应进入完整 RAG")

    class Selector:
        async def select(self, query, candidates):
            return {
                "success": True,
                "supported": True,
                "selected_title": "技术故障排查",
                "evidence": "文档明确解释 401",
            }

    class TextBlock:
        type = "text"
        text = "401 通常表示认证失败。"

    client = FakeClient(response=type("Response", (), {"content": [TextBlock()]})())
    agent = TechnicalAgent(client, "test-model")
    agent.set_shared_tools(build_shared_rag_tools(RagManager(), Selector()))

    result = asyncio.run(agent.handle(make_request(
        message="登录报 401 是什么问题？",
        intent=IntentCategory.TECHNICAL_LOGIN,
        intent_group="technical",
        intent_confidence=0.55,
        entities={"error_code": ["401"]},
    )))

    assert result.tools_used == ["search_knowledge_base"]
    assert result.tool_traces[0]["invocation_source"] == "deterministic_prefetch"
    assert result.tool_traces[0]["evidence_selector_success"] is True
    assert result.tool_traces[0]["evidence_selected_title"] == "技术故障排查"
    assert result.tool_traces[0]["evidence_found"] is True


def test_low_confidence_policy_intent_prefetches_and_fails_closed_without_evidence():
    class TextBlock:
        type = "text"
        text = "请补充您的问题。"

    class EmptyRagManager:
        async def call(self, tool_name, params, use_cache=True):
            return type("Result", (), {"success": True, "data": [], "reranked": False})()

        async def search_with_rewrite(self, tool_name, query, top_k=5):
            return type("Result", (), {"success": True, "data": [], "reranked": False})()

    client = FakeClient(response=type("Response", (), {"content": [TextBlock()]})())
    agent = BillingAgent(client, "test-model")
    agent.set_shared_tools(build_shared_rag_tools(EmptyRagManager()))
    req = make_request(
        message="我想问一下",
        intent=IntentCategory.REFUND,
        intent_group="billing",
        intent_confidence=0.55,
    )

    response = asyncio.run(agent.handle(req))

    assert response.tools_used == ["search_knowledge_base"]
    assert "未找到足够的可核验依据" in response.content
    assert client.calls == []
    assert response.tool_traces[0]["invocation_source"] == "deterministic_prefetch"
    assert response.tool_traces[0]["evidence_found"] is False


def test_generated_handoff_words_do_not_control_escalation():
    class TextBlock:
        type = "text"
        text = "实际退款需要人工审核，但当前不需要转人工客服。"

    client = FakeClient(response=type("Response", (), {"content": [TextBlock()]})())
    agent = BillingAgent(client, "test-model")
    req = make_request(
        message="退款审核需要多久？",
        intent=IntentCategory.REFUND,
        intent_group="billing",
        urgency=UrgencyLevel.MEDIUM,
    )

    response = asyncio.run(agent.handle(req))

    assert response.success is True
    assert response.escalate is False


def test_structured_critical_signal_still_triggers_escalation():
    class TextBlock:
        type = "text"
        text = "我正在处理您的问题。"

    client = FakeClient(response=type("Response", (), {"content": [TextBlock()]})())
    agent = TechnicalAgent(client, "test-model")
    req = make_request(urgency=UrgencyLevel.CRITICAL)

    response = asyncio.run(agent.handle(req))

    assert response.success is True
    assert response.escalate is True


def test_core_customer_service_scenario_matrix():
    """四类核心请求的路由、RAG 门控和升级信号必须保持一致。"""
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._pool = {
        AgentType.GENERAL: [object()],
        AgentType.TECHNICAL: [object()],
        AgentType.BILLING: [object()],
        AgentType.ESCALATION: [object()],
    }
    cases = [
        {
            "name": "greeting",
            "request": make_request(
                message="你好",
                intent=IntentCategory.GREETING,
                intent_group="general",
                urgency=UrgencyLevel.LOW,
                entities={},
            ),
            "agent": AgentType.GENERAL,
            "knowledge": False,
            "escalate": False,
        },
        {
            "name": "refund",
            "request": make_request(
                message="退款审核需要多久？",
                intent=IntentCategory.REFUND,
                intent_group="billing",
                urgency=UrgencyLevel.MEDIUM,
                entities={},
            ),
            "agent": AgentType.BILLING,
            "knowledge": True,
            "escalate": False,
        },
        {
            "name": "login_failure",
            "request": make_request(
                message="登录失败并提示 401",
                intent=IntentCategory.TECHNICAL_LOGIN,
                intent_group="technical",
                urgency=UrgencyLevel.HIGH,
                entities={"error_code": ["401"]},
            ),
            "agent": AgentType.TECHNICAL,
            "knowledge": True,
            "escalate": False,
        },
        {
            "name": "human_handoff",
            "request": make_request(
                message="请帮我转人工客服",
                intent=IntentCategory.HUMAN_HANDOFF,
                intent_group="escalation",
                urgency=UrgencyLevel.MEDIUM,
                entities={},
            ),
            "agent": AgentType.ESCALATION,
            "knowledge": False,
            "escalate": True,
        },
    ]

    for case in cases:
        req = case["request"]
        decision = orchestrator._route_decision(req)
        assert decision.primary_agent is case["agent"], case["name"]
        assert decision.multi_agent is False, case["name"]
        assert should_use_knowledge(req.message, req.intent) is case["knowledge"], case["name"]
        assert BaseAgent._needs_escalation(req) is case["escalate"], case["name"]


def test_provider_connection_failure_preserves_prefetched_tool_trace():
    class RagManager:
        async def call(self, tool_name, params, use_cache=True):
            return type(
                "Result",
                (),
                {
                    "success": True,
                    "data": [{"title": "退款政策", "content": "1-3 日审核"}],
                    "reranked": False,
                },
            )()

        async def search_with_rewrite(self, tool_name, query, top_k=5):
            raise AssertionError("快速检索成功后不应进入完整 RAG")

    error = APIConnectionError(request=httpx.Request("POST", "https://provider.invalid/messages"))
    agent = BillingAgent(FakeClient(error=error), "test-model")
    agent.set_shared_tools(build_shared_rag_tools(RagManager()))
    req = make_request(
        message="退款审核需要多久？",
        intent=IntentCategory.REFUND,
        intent_group="billing",
        urgency=UrgencyLevel.MEDIUM,
        intent_confidence=0.93,
        entities={},
    )

    response = asyncio.run(agent.handle(req))

    assert response.success is False
    assert response.failure_kind == "provider_connection"
    assert response.tools_used == ["search_knowledge_base"]
    assert response.tool_traces[0]["prefetched"] is True
    assert response.tool_traces[0]["result_success"] is True


def test_provider_failure_skips_redundant_general_fallback():
    class Stats:
        @staticmethod
        def routing_score():
            return 1.0

    class AgentStub:
        def __init__(self, response):
            self.response = response
            self.stats = Stats()
            self.calls = 0

        async def handle(self, req):
            self.calls += 1
            return self.response

    technical = AgentStub(AgentResponse(
        agent_type=AgentType.TECHNICAL,
        content="暂时无法连接模型服务，请稍后重试。",
        success=False,
        failure_kind="provider_connection",
    ))
    general = AgentStub(AgentResponse(
        agent_type=AgentType.GENERAL,
        content="不应调用",
        success=True,
    ))
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._pool = {
        AgentType.TECHNICAL: [technical],
        AgentType.GENERAL: [general],
    }

    response = asyncio.run(orchestrator._execute(make_request(), AgentType.TECHNICAL))

    assert response.failure_kind == "provider_connection"
    assert technical.calls == 1
    assert general.calls == 0

    # Agent 自身逻辑错误仍保留原有 GeneralAgent 降级能力。
    technical.response = AgentResponse(
        agent_type=AgentType.TECHNICAL,
        content="内部处理失败",
        success=False,
        failure_kind="agent_error",
    )
    fallback_response = asyncio.run(orchestrator._execute(make_request(), AgentType.TECHNICAL))
    assert fallback_response.success is True
    assert general.calls == 1


def test_tool_input_validation_rejects_unknown_fields():
    agent = TechnicalAgent(FakeClient(), "test-model")
    spec = agent.get_tools()["lookup_error_code"]

    try:
        agent._validate_tool_input(spec, {"error_code": "401", "secret": "nope"})
    except ValueError as exc:
        assert "不允许的工具参数" in str(exc)
    else:
        raise AssertionError("unknown tool fields should be rejected")


def test_tool_use_round_trip_executes_only_whitelisted_tool():
    class ToolUseBlock:
        type = "tool_use"
        id = "toolu_1"
        name = "lookup_error_code"
        input = {"error_code": "401"}

    class TextBlock:
        type = "text"
        text = "已根据 401 错误码给出排查建议。"

    class ToolClient:
        def __init__(self):
            self.calls = []
            self.responses = [
                type("Response", (), {"content": [ToolUseBlock()]})(),
                type("Response", (), {"content": [TextBlock()]})(),
            ]

        class Messages:
            def __init__(self, owner):
                self.owner = owner

            async def create(self, **kwargs):
                self.owner.calls.append(kwargs)
                return self.owner.responses.pop(0)

        @property
        def messages(self):
            return self.Messages(self)

    client = ToolClient()
    agent = TechnicalAgent(client, "test-model")
    response = asyncio.run(agent.handle(make_request()))

    assert response.success is True
    assert response.tools_used == ["lookup_error_code"]
    assert len(response.tool_traces) == 1
    trace = response.tool_traces[0]
    assert trace["tool_name"] == "lookup_error_code"
    assert trace["tool_use_id"] == "toolu_1"
    assert trace["input"] == {"error_code": "401"}
    assert trace["success"] is True
    assert trace["prefetched"] is False
    assert trace["invocation_source"] == "model_tool_call"
    assert trace["latency_ms"] >= 0
    assert len(client.calls) == 2
    assert {tool["name"] for tool in client.calls[0]["tools"]} == {
        "lookup_error_code",
        "build_diagnostic_plan",
    }
    assert "tool_result" in str(client.calls[1]["messages"])


def test_model_requested_rag_query_is_scoped_before_tool_execution():
    mixed_query = "我登录时提示 401，而且订单被重复扣款 99 元"

    class ToolUseBlock:
        type = "tool_use"
        id = "toolu_rag"
        name = "search_knowledge_base"
        input = {"query": mixed_query, "top_k": 3}

    class TextBlock:
        type = "text"
        text = "请先检查登录凭证是否过期。"

    class ToolClient:
        def __init__(self):
            self.responses = [
                type("Response", (), {"content": [ToolUseBlock()]})(),
                type("Response", (), {"content": [TextBlock()]})(),
            ]

        class Messages:
            def __init__(self, owner):
                self.owner = owner

            async def create(self, **kwargs):
                return self.owner.responses.pop(0)

        @property
        def messages(self):
            return self.Messages(self)

    class RagManager:
        def __init__(self):
            self.queries = []

        async def call(self, tool_name, params, use_cache=True):
            self.queries.append(params["query"])
            return type(
                "Result",
                (),
                {"success": True, "data": [{"content": "401 表示认证失败"}], "reranked": False},
            )()

        async def search_with_rewrite(self, tool_name, query, top_k=5):
            self.queries.append(query)
            return type(
                "Result",
                (),
                {"success": True, "data": [{"content": "401 表示认证失败"}], "reranked": True},
            )()

    manager = RagManager()
    agent = TechnicalAgent(ToolClient(), "test-model")
    agent.set_shared_tools(build_shared_rag_tools(manager))
    response = asyncio.run(agent.handle(make_request(
        message=mixed_query,
        intent=IntentCategory.TECHNICAL,
        intent_group="technical",
        intent_confidence=0.55,
    )))

    assert response.success is True
    assert manager.queries == ["我登录时提示 401"]
    assert response.tool_traces[0]["input"]["query"] == "我登录时提示 401"
    assert response.tool_traces[0]["invocation_source"] == "model_tool_call"


def test_empty_model_requested_policy_search_stops_before_hallucinated_second_turn():
    class ToolUseBlock:
        type = "tool_use"
        id = "toolu_empty_rag"
        name = "search_knowledge_base"
        input = {"query": "电子发票开具时效", "top_k": 3}

    class HallucinatedTextBlock:
        type = "text"
        text = "电子发票一定会在 3 个工作日内开具。"

    class ToolClient:
        def __init__(self):
            self.calls = []
            self.responses = [
                type("Response", (), {"content": [ToolUseBlock()]})(),
                type("Response", (), {"content": [HallucinatedTextBlock()]})(),
            ]

        class Messages:
            def __init__(self, owner):
                self.owner = owner

            async def create(self, **kwargs):
                self.owner.calls.append(kwargs)
                return self.owner.responses.pop(0)

        @property
        def messages(self):
            return self.Messages(self)

    class EmptyRagManager:
        async def call(self, tool_name, params, use_cache=True):
            return type("Result", (), {"success": True, "data": [], "reranked": False})()

        async def search_with_rewrite(self, tool_name, query, top_k=5):
            return type("Result", (), {"success": True, "data": [], "reranked": False})()

    client = ToolClient()
    agent = BillingAgent(client, "test-model")
    agent.set_shared_tools(build_shared_rag_tools(EmptyRagManager()))
    result = asyncio.run(agent.handle(make_request(
        message="电子发票最晚几天可以开出来？",
        intent=IntentCategory.INVOICE,
        intent_group="billing",
        intent_confidence=0.55,
        entities={},
    )))

    assert result.success is True
    assert "未找到足够的可核验依据" in result.content
    assert "3 个工作日" not in result.content
    assert len(client.calls) == 0
    assert result.tool_traces[0]["evidence_found"] is False
    assert result.tool_traces[0]["retrieval_mode"] == "knowledge_gap"
