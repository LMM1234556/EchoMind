"""知识库检索门控。

这里只回答“是否应向 Agent 暴露 RAG 工具”，不执行检索。
规则保持确定性，便于用正反例测试，也避免把无关问题强行映射到
知识库中“最不坏”的文档。
"""

from typing import Any


_DIRECT_KNOWLEDGE_INTENTS = {
    "billing",
    "account",
    "technical",
    "order_status",
    "logistics",
    "refund",
    "invoice",
    "payment_issue",
    "account_security",
    "technical_login",
    "technical_crash",
}

_NEVER_KNOWLEDGE_INTENTS = {
    "greeting",
    "feedback",
    "escalation",
    "human_handoff",
}

_BUSINESS_KEYWORDS = (
    "退款",
    "退货",
    "订单",
    "物流",
    "快递",
    "配送",
    "运费",
    "发票",
    "扣款",
    "扣费",
    "支付",
    "账单",
    "订阅",
    "登录",
    "密码",
    "验证码",
    "账户",
    "账号",
    "会员",
    "积分",
    "崩溃",
    "闪退",
    "报错",
    "401",
    "500",
    "refund",
    "return",
    "order",
    "invoice",
    "payment",
    "billing",
    "login",
    "password",
    "delivery",
    "shipping",
    "crash",
)


def should_use_knowledge(message: str, intent: Any = None) -> bool:
    """仅当请求明确属于当前客服知识域时，允许 Agent 调用 RAG。"""
    msg = (message or "").strip().lower()
    if not msg:
        return False

    intent_value = getattr(intent, "value", intent)
    if intent_value in _NEVER_KNOWLEDGE_INTENTS:
        return False
    if intent_value in _DIRECT_KNOWLEDGE_INTENTS:
        return True

    # QUERY / REQUEST / COMPLAINT / OTHER 语义过宽，不能单凭类别
    # 就检索；它们需要再命中客服业务词。
    return any(keyword in msg for keyword in _BUSINESS_KEYWORDS)
