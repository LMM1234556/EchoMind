"""LLM helpers shared by Anthropic-compatible providers.

The request-scoped call meter counts application-level ``messages.create``
invocations, including invocations that eventually fail. Provider SDK retries
inside one invocation are not observable here and are therefore not counted.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterable, Iterator, List, Optional


_active_call_meter: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "echomind_llm_call_meter",
    default=None,
)


@contextmanager
def llm_call_scope() -> Iterator[Dict[str, Any]]:
    """Collect LLM call attempts made by the current async request context."""
    meter: Dict[str, Any] = {"total": 0, "by_component": {}}
    token = _active_call_meter.set(meter)
    try:
        yield meter
    finally:
        _active_call_meter.reset(token)


async def tracked_messages_create(client: Any, component: str, **kwargs: Any) -> Any:
    """Call ``messages.create`` and attribute the SDK invocation to a component."""
    meter = _active_call_meter.get()
    if meter is not None:
        meter["total"] += 1
        by_component = meter["by_component"]
        by_component[component] = by_component.get(component, 0) + 1
    return await client.messages.create(**kwargs)


def current_llm_call_stats() -> Dict[str, Any]:
    """Return an immutable-style snapshot for an API response or trace."""
    meter = _active_call_meter.get()
    if meter is None:
        return {"total": 0, "by_component": {}}
    return {
        "total": int(meter["total"]),
        "by_component": dict(meter["by_component"]),
    }


def extract_text_content(content: Iterable[Any]) -> str:
    """Return text blocks from Anthropic-style response content."""
    texts: List[str] = []
    for block in content or []:
        if isinstance(block, str):
            texts.append(block)
            continue

        block_type = getattr(block, "type", None)
        text = getattr(block, "text", None)
        if isinstance(block, dict):
            block_type = block.get("type", block_type)
            text = block.get("text", text)

        if isinstance(text, str) and (block_type in (None, "text")):
            texts.append(text)

    return "\n".join(t for t in texts if t)
