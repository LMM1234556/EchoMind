"""知识库检索门控与多 Agent 检索词作用域控制。

这里只回答“是否应向 Agent 暴露 RAG 工具”，不执行检索。
规则保持确定性，便于用正反例测试，也避免把无关问题强行映射到
知识库中“最不坏”的文档。
"""

import re
from typing import Any, Optional


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

# 标点、换行和显式连接词用于划开子问题；没有可靠边界时保留原文。
_CLAUSE_SEPARATOR = re.compile(
    r"(?:[，,；;。！？!?\r\n]+|而且|并且|同时|另外|此外|以及|\b(?:and|also|while)\b)",
    re.IGNORECASE,
)

# “还有”也可能只是描述资料内容。只有连接词两侧都含询问或故障信号，
# 才按追加诉求处理，避免把“发票上还有退款信息”拆成两个问题。
_FOLLOWUP_SEPARATOR = re.compile(
    r"(?:还有(?:一个问题(?:是)?|想问(?:一下)?)?|还想问(?:一下)?|也想问(?:一下)?|顺便(?:再)?问(?:一下)?)"
)
_REQUEST_SIGNAL = re.compile(
    r"怎么|如何|多久|何时|什么时候|什么|哪里|哪些|几天|多少|能否|是否|怎么办|吗|忘了|忘记|失败|报错|崩溃|闪退|"
    r"重复扣款|扣了两次|扣费异常|(?<!\d)(?:401|403|404|500)(?!\d)"
)


def _explicit_clauses(message: str) -> list[str]:
    """拆出明确边界；口语连接还须满足两侧都是请求或问题。"""
    clauses = []
    for clause in _CLAUSE_SEPARATOR.split(message):
        clause = clause.strip()
        if not clause:
            continue
        followups = [part.strip() for part in _FOLLOWUP_SEPARATOR.split(clause) if part.strip()]
        if len(followups) > 1 and all(_REQUEST_SIGNAL.search(part) for part in followups):
            clauses.extend(followups)
        else:
            clauses.append(clause)
    return clauses


# 复合问题只有在每个子句都能确定映射到一个知识主题时才拆分。
# 主题映射同时提供可审计的文档标题边界；空集合表示已知知识缺口，
# 例如当前演示知识库没有电子发票办理时限。
_KNOWLEDGE_TOPIC_TITLES = {
    "technical": {"技术故障排查"},
    "account_security": {"账户安全"},
    "membership": {"会员与积分"},
    "delivery": {"配送说明"},
    "order": {"订单查询"},
    "refund": {"退款政策"},
    "invoice": set(),
    "payment_gap": set(),
}


def infer_knowledge_scope(message: str) -> tuple[Optional[str], Optional[set[str]]]:
    """从一个明确子问题推断知识主题和允许引用的文档标题。

    返回 ``(None, None)`` 表示无法安全确定主题，应继续使用既有意图门控。
    规则按更具体的业务词优先，避免“订单被重复扣款”误用订单状态文档。
    """
    text = (message or "").strip().lower()
    if not text:
        return None, None

    if any(keyword in text for keyword in ("发票", "invoice")):
        topic = "invoice"
    elif any(keyword in text for keyword in ("积分", "银卡", "金卡", "会员等级", "折扣")):
        topic = "membership"
    elif any(keyword in text for keyword in ("加急配送", "标准配送", "同城配送", "运费", "收货地址", "偏远地区", "配送时间")):
        topic = "delivery"
    elif any(keyword in text for keyword in ("退款", "退货", "退款审核", "退款到账", "原支付账户")):
        topic = "refund"
    elif any(keyword in text for keyword in ("重复扣款", "扣了两次", "多扣", "扣费异常")):
        topic = "payment_gap"
    elif any(keyword in text for keyword in ("401", "403", "404", "500", "崩溃", "闪退", "错误码", "报错", "接口", "网络超时")):
        topic = "technical"
    elif any(keyword in text for keyword in ("忘记密码", "密码忘了", "重置密码", "两步验证", "异常登录", "账户安全", "账号安全")):
        topic = "account_security"
    elif any(keyword in text for keyword in ("订单异常", "订单状态", "物流位置", "物流信息", "已发货", "未收到", "查件")):
        topic = "order"
    else:
        return None, None
    return topic, set(_KNOWLEDGE_TOPIC_TITLES[topic])


def split_knowledge_queries(message: str) -> list[str]:
    """保守拆分需要不同知识文档的显式复合问题。

    单一主题、多义子句或没有明确分隔依据时保持原文。这样既支持逐项取证，
    又不会把“登录401订单重复扣款99元”等不可可靠划界的文本强行拆开。
    """
    original = (message or "").strip()
    if not original:
        return []
    clauses = _explicit_clauses(original)
    if not 2 <= len(clauses) <= 3:
        return [original]

    topics = []
    for clause in clauses:
        topic, _ = infer_knowledge_scope(clause)
        if topic is None:
            return [original]
        topics.append(topic)
    if len(set(topics)) < 2:
        return [original]
    return clauses


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

    clauses = _explicit_clauses(original)
    if len(clauses) < 2:
        return original

    labels = []
    for clause in clauses:
        lowered = clause.lower()
        is_technical = any(keyword in lowered for keyword in _TECHNICAL_KEYWORDS)
        is_billing = any(keyword in lowered for keyword in _BILLING_KEYWORDS)
        if is_technical and is_billing:
            return original
        if not is_technical and not is_billing:
            return original
        labels.append((clause, is_technical, is_billing))

    if not any(item[1] for item in labels) or not any(item[2] for item in labels):
        return original

    match_index = 1 if agent_value == "technical" else 2
    scoped = "；".join(item[0] for item in labels if item[match_index])
    return scoped or original
