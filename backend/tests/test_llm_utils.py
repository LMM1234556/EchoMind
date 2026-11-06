import asyncio

from core.llm_utils import (
    current_llm_call_stats,
    llm_call_scope,
    tracked_messages_create,
)


class _Messages:
    async def create(self, **kwargs):
        if kwargs.get("fail"):
            raise RuntimeError("provider failure")
        return kwargs


class _Client:
    messages = _Messages()


def test_llm_call_scope_counts_components_and_resets():
    async def run():
        client = _Client()
        with llm_call_scope() as meter:
            await tracked_messages_create(client, "intent", model="demo")
            await tracked_messages_create(client, "agent.billing", model="demo")
            await tracked_messages_create(client, "agent.billing", model="demo")
            assert meter == {
                "total": 3,
                "by_component": {"intent": 1, "agent.billing": 2},
            }
            assert current_llm_call_stats() == meter

        assert current_llm_call_stats() == {"total": 0, "by_component": {}}

    asyncio.run(run())


def test_llm_call_scope_counts_failed_attempts():
    async def run():
        client = _Client()
        with llm_call_scope() as meter:
            try:
                await tracked_messages_create(client, "intent", fail=True)
            except RuntimeError:
                pass
            assert meter["total"] == 1
            assert meter["by_component"] == {"intent": 1}

    asyncio.run(run())


def test_structured_qwen_calls_disable_thinking(monkeypatch):
    monkeypatch.delenv("ECHOMIND_DISABLE_THINKING_COMPONENTS", raising=False)

    async def run():
        client = _Client()
        result = await tracked_messages_create(
            client,
            "intent_recognizer",
            model="qwen3.6-plus",
            max_tokens=256,
        )
        assert result["extra_body"] == {"thinking": {"type": "disabled"}}

    asyncio.run(run())


def test_generation_policy_preserves_explicit_override(monkeypatch):
    monkeypatch.delenv("ECHOMIND_DISABLE_THINKING_COMPONENTS", raising=False)

    async def run():
        client = _Client()
        result = await tracked_messages_create(
            client,
            "evidence_selector",
            model="qwen3.6-plus",
            extra_body={"thinking": {"type": "enabled", "budget_tokens": 256}},
        )
        assert result["extra_body"]["thinking"]["type"] == "enabled"

    asyncio.run(run())


def test_final_agent_and_non_qwen_keep_provider_default(monkeypatch):
    monkeypatch.delenv("ECHOMIND_DISABLE_THINKING_COMPONENTS", raising=False)

    async def run():
        client = _Client()
        final_agent = await tracked_messages_create(
            client, "agent.billing", model="qwen3.6-plus"
        )
        other_model = await tracked_messages_create(
            client, "intent_recognizer", model="claude-demo"
        )
        assert "extra_body" not in final_agent
        assert "extra_body" not in other_model

    asyncio.run(run())
