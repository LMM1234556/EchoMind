"""Recognize explicit order-context updates without bypassing business questions."""
import re
from typing import Optional


_ORDER_ID = r"(?P<order_id>[A-Za-z0-9][A-Za-z0-9_-]{3,31})"
_REQUEST = r"(?:(?:请帮我|麻烦你|请你|帮我|麻烦|请)\s*)?"
_REMEMBER = r"(?:记住|记下|记录(?:一下)?|保存(?:一下)?)"
_ORDER_LABEL = r"(?:我的\s*)?(?:订单号|订单编号)\s*(?:是|为)?\s*[:：]?\s*#?\s*"
# Full-message matching keeps policy questions, actions and unrelated extra facts
# on the normal Agent path. Only this explicit user-supplied status is accepted.
_REFUND_CONTEXT = r"(?:[，,；;]\s*我(?:目前|现在)?(?:正在|正)?申请退(?:款|货))?"
_ENDING = r"[。.!！\s]*"
_CONTEXT_PATTERNS = (
    re.compile(_REQUEST + _REMEMBER + r"\s*" + _ORDER_LABEL + _ORDER_ID + _REFUND_CONTEXT + _ENDING),
    re.compile(_ORDER_LABEL + _ORDER_ID + r"[，,；;]\s*" + _REQUEST + _REMEMBER + _ENDING),
)


def parse_order_context_update(message: str) -> Optional[str]:
    """Return one order ID only for a complete, explicit context-update request.

    This identifies a dialogue task. The existing API memory-write path remains
    responsible for persistence; acknowledgement does not imply a database update,
    a refund operation or permanent memory.
    """
    text = (message or "").strip()
    for pattern in _CONTEXT_PATTERNS:
        match = pattern.fullmatch(text)
        if match:
            return match.group("order_id")
    return None
