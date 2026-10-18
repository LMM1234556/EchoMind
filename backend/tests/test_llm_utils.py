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
