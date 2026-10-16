from agents.agent_orchestrator import GeneralAgent, Request
from agents.tools import build_shared_rag_tools
from core.intent_recognizer import IntentCategory
from core.knowledge_routing import should_use_knowledge


class FakeClient:
    class Messages:
        async def create(self, **kwargs):  # pragma: no cover - tests never call the model
            raise AssertionError("不应调用模型")

    messages = Messages()


def make_request(message: str, intent: IntentCategory) -> Request:
    return Request(
        message=message,
        user_id="u1",
        conv_id="c1",
        intent=intent,
    )


def test_specific_business_intent_can_use_knowledge_without_literal_keyword():
    assert should_use_knowledge("买错东西后想退掉", IntentCategory.REFUND) is True


def test_broad_or_unknown_intent_requires_business_keyword():
    assert should_use_knowledge("退款什么时候到账", IntentCategory.QUERY) is True
    assert should_use_knowledge("订单还没发货", IntentCategory.OTHER) is True
    assert should_use_knowledge("今天天气怎么样", IntentCategory.QUERY) is False
    assert should_use_knowledge("帮我写一首关于秋天的诗", IntentCategory.REQUEST) is False
    assert should_use_knowledge("比特币今天涨了多少", IntentCategory.OTHER) is False


def test_conversation_only_intents_never_use_knowledge():
    assert should_use_knowledge("你好，会员客服", IntentCategory.GREETING) is False
    assert should_use_knowledge("转人工查订单", IntentCategory.HUMAN_HANDOFF) is False
    assert should_use_knowledge("", IntentCategory.REFUND) is False


def test_agent_hides_rag_tool_for_out_of_domain_request():
    agent = GeneralAgent(FakeClient(), "test-model")
    agent.set_shared_tools(build_shared_rag_tools(object()))

    tools = agent._tools_for_request(
        make_request("帮我写一首关于秋天的诗", IntentCategory.REQUEST)
    )

    assert "search_knowledge_base" not in tools
    assert "inspect_request_context" in tools


def test_agent_exposes_rag_tool_for_business_request():
    agent = GeneralAgent(FakeClient(), "test-model")
    agent.set_shared_tools(build_shared_rag_tools(object()))

    tools = agent._tools_for_request(
        make_request("退款什么时候到账", IntentCategory.QUERY)
    )

    assert "search_knowledge_base" in tools
