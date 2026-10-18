"""知识库检索门控与多 Agent 检索词作用域控制。

这里只回答“是否应向 Agent 暴露 RAG 工具”，不执行检索。
规则保持确定性，便于用正反例测试，也避免把无关问题强行映射到
知识库中“最不坏”的文档。
"""

import re
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

_TECHNICAL_KEYWORDS = (
    "登录", "登陆", "401", "403", "404", "500", "错误码", "报错", "故障",
    "崩溃", "闪退", "接口", "网络", "超时", "token", "api key", "认证",
    "权限", "配置", "版本", "同步", "login", "password", "error", "crash",
)

_BILLING_KEYWORDS = (
    "退款", "退货", "订单", "物流", "快递", "配送", "运费", "发票", "扣款",
    "扣费", "支付", "账单", "订阅", "金额", "到账", "收费", "refund", "return",
    "order", "invoice", "payment", "billing", "charge", "shipping", "delivery",
)

# 只在用户已经用标点或连接词划开子问题时做确定性拆分。对于“登录401订单
# 重复扣款99元”这类无法可靠划界的输入，宁可保留原文，也不猜测切分位置。
_CLAUSE_SEPARATOR = re.compile(
    r"(?:[，,；;。！？!?]+|而且|并且|同时|另外|此外|以及|\b(?:and|also|while)\b)",
    re.IGNORECASE,
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


def scope_knowledge_query(message: str, agent_type: Any) -> str:
    """将可分割的技术+账单复合问题缩小到当前 Agent 的知识域。

    只有同时满足以下条件才改写：
    1. 原问题可由显式标点或连接词分成多个子句；
    2. 不同子句分别明确命中技术域和账单域；
    3. 单个子句没有同时命中两个域。

    其余情况返回原文，避免因激进切词丢失金额、错误码或上下文。
    """
    original = (message or "").strip()
    if not original:
        return original

    agent_value = getattr(agent_type, "value", agent_type)
    agent_value = str(agent_value or "").strip().lower()
    if agent_value not in {"technical", "billing"}:
        return original

    clauses = [part.strip() for part in _CLAUSE_SEPARATOR.split(original) if part.strip()]
    if len(clauses) < 2:
        return original

    labels = []
    for clause in clauses:
        lowered = clause.lower()
        is_technical = any(keyword in lowered for keyword in _TECHNICAL_KEYWORDS)
        is_billing = any(keyword in lowered for keyword in _BILLING_KEYWORDS)
        if is_technical and is_billing:
            return original
        labels.append((clause, is_technical, is_billing))

    if not any(item[1] for item in labels) or not any(item[2] for item in labels):
        return original

    match_index = 1 if agent_value == "technical" else 2
    scoped = "；".join(item[0] for item in labels if item[match_index])
    return scoped or original
