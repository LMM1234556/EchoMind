import asyncio
import json
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

from chromadb.api.types import validate_where

from memory.conversation_memory import MemoryManager, MsgRole


def _serialized_message(index: int) -> str:
    return json.dumps({
        "role": MsgRole.USER.value,
        "content": f"message-{index}",
        "ts": (datetime(2026, 1, 1) + timedelta(seconds=index)).isoformat(),
        "metadata": {"index": index},
    })


class _FakeRedis:
    """Memory-only Redis subset used to verify the compression data flow."""

    def __init__(self):
        self.values = {}
        self.ttls = {}

    async def lpush(self, key, value):
        self.values.setdefault(key, []).insert(0, value)
        return len(self.values[key])

    async def expire(self, key, seconds):
        self.ttls[key] = seconds
        return True

    async def llen(self, key):
        return len(self.values.get(key, []))

    async def lrange(self, key, start, end):
        values = self.values.get(key, [])
        return values[start:] if end == -1 else values[start:end + 1]

    async def get(self, key):
        value = self.values.get(key)
        return value if isinstance(value, str) else None

    async def setex(self, key, seconds, value):
        self.values[key] = value
        self.ttls[key] = seconds
        return True

    async def delete(self, key):
        existed = key in self.values
        self.values.pop(key, None)
        self.ttls.pop(key, None)
        return int(existed)


def test_episodic_search_combines_user_and_conversation_with_and():
    memory = MemoryManager.__new__(MemoryManager)
    memory._query_episodic = AsyncMock(side_effect=[
        {"documents": [["current conversation"]]},
        {"documents": [["current conversation", "earlier conversation"]]},
    ])

    result = asyncio.run(memory._search_episodic("user-1", "conversation-1", "refund"))

    calls = memory._query_episodic.call_args_list
    assert len(calls) == 2
    assert calls[0].kwargs["where"] == {
        "$and": [{"user_id": "user-1"}, {"conv_id": "conversation-1"}]
    }
    assert calls[1].kwargs["where"] == {"user_id": "user-1"}
    for call in calls:
        validate_where(call.kwargs["where"])
    assert result == ["current conversation", "earlier conversation"]


def test_blank_episodic_query_does_not_query_chroma():
    memory = MemoryManager.__new__(MemoryManager)
    memory._query_episodic = AsyncMock()

    assert asyncio.run(memory._search_episodic("user-1", "conversation-1", "  ")) == []
    memory._query_episodic.assert_not_awaited()


def test_add_message_uses_isolated_key_and_refreshes_24_hour_ttl():
    memory = MemoryManager.__new__(MemoryManager)
    memory._redis = AsyncMock()
    memory._redis.llen.return_value = 1
    memory._compress = AsyncMock()

    asyncio.run(memory.add_message(
        "user-1",
        "conversation-1",
        MsgRole.USER,
        "hello",
    ))

    key = "wm:user-1:conversation-1"
    assert memory._redis.lpush.await_args.args[0] == key
    memory._redis.expire.assert_awaited_once_with(key, MemoryManager.WORKING_TTL_SECONDS)
    memory._compress.assert_not_awaited()
    assert MemoryManager._wm_key("user-1", "conversation-2") != key
    assert MemoryManager._wm_key("user-2", "conversation-1") != key


def test_redis_outage_degrades_to_empty_working_memory_without_failing_chat_context():
    memory = MemoryManager.__new__(MemoryManager)
    memory._redis = AsyncMock()
    memory._redis.lrange.side_effect = ConnectionError("redis unavailable")
    memory._redis.get.side_effect = ConnectionError("redis unavailable")
    memory._search_episodic = AsyncMock(return_value=["历史证据"])
    memory._get_profile = AsyncMock(return_value={"tier": "vip"})

    context = asyncio.run(memory.get_context("user-1", "conversation-1", "退款多久到账"))

    assert context.recent_messages == []
    assert context.summary == ""
    assert context.relevant_history == ["历史证据"]
    assert context.user_profile == {"tier": "vip"}


def test_redis_outage_does_not_fail_message_generation_after_response():
    memory = MemoryManager.__new__(MemoryManager)
    memory._redis = AsyncMock()
    memory._redis.lpush.side_effect = ConnectionError("redis unavailable")

    asyncio.run(memory.add_message(
        "user-1",
        "conversation-1",
        MsgRole.ASSISTANT,
        "仍然返回给用户的回答",
    ))

    memory._redis.lpush.assert_awaited_once()


def test_working_memory_reads_only_latest_window_in_chronological_order():
    memory = MemoryManager.__new__(MemoryManager)
    memory._redis = AsyncMock()
    # Redis LPUSH stores newest first. Simulate the newest 20 records from 25 total.
    memory._redis.lrange.return_value = [
        _serialized_message(index) for index in range(24, 4, -1)
    ]

    messages = asyncio.run(memory._get_working_memory("user-1", "conversation-1"))

    memory._redis.lrange.assert_awaited_once_with(
        "wm:user-1:conversation-1",
        0,
        MemoryManager.WORKING_MAX - 1,
    )
    assert len(messages) == MemoryManager.WORKING_MAX
    assert [message.content for message in messages] == [
        f"message-{index}" for index in range(5, 25)
    ]


def test_fifteenth_message_compresses_old_ten_and_keeps_latest_five():
    memory = MemoryManager.__new__(MemoryManager)
    memory._redis = _FakeRedis()
    memory._model = "test-model"
    generated_summary = "关键事实：订单 A123 的退款仍在审核，用户希望周五前收到结果。"
    create = AsyncMock(return_value=SimpleNamespace(
        content=[{"type": "text", "text": generated_summary}],
    ))
    memory._client = SimpleNamespace(messages=SimpleNamespace(create=create))
    memory._store_episodic = AsyncMock()

    async def exercise_compression():
        for index in range(MemoryManager.COMPRESS_AT):
            content = (
                "订单 A123 的退款仍在审核，用户希望周五前收到结果。"
                if index == 0
                else f"message-{index}"
            )
            await memory.add_message(
                "user-1",
                "conversation-1",
                MsgRole.USER,
                content,
            )
        return await memory._get_working_memory("user-1", "conversation-1")

    recent = asyncio.run(exercise_compression())

    assert [message.content for message in recent] == [
        f"message-{index}" for index in range(10, 15)
    ]
    working_key = "wm:user-1:conversation-1"
    summary_key = "summary:user-1:conversation-1"
    assert memory._redis.values[summary_key] == generated_summary
    assert memory._redis.ttls[working_key] == MemoryManager.WORKING_TTL_SECONDS
    assert memory._redis.ttls[summary_key] == MemoryManager.WORKING_TTL_SECONDS
    create.assert_awaited_once()
    prompt = create.await_args.kwargs["messages"][0]["content"]
    assert "订单 A123" in prompt
    assert "message-9" in prompt
    assert "message-10" not in prompt
    memory._store_episodic.assert_awaited_once()
    episodic_args = memory._store_episodic.await_args.args
    assert episodic_args[0:2] == ("user-1", "conversation-1")
    assert "订单 A123" in episodic_args[2]
    assert episodic_args[3] == generated_summary
