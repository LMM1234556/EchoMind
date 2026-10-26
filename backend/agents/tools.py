"""Agent 工具定义与实现。

所有 Agent 工具集中在这里，编排器只负责：
  1. 根据 Agent 类型暴露工具白名单
  2. 执行 LLM 返回的 tool_use
  3. 将工具结果回传给 LLM

工具本身保持确定性、可测试，并明确区分：
  - 当前请求分析
  - 技术排障建议
  - 账单字段核验
  - 人工升级摘要
  - 共享知识库 RAG

订单查询、退款执行、账单修改等需要真实业务系统授权的动作不在这里伪造。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from agents.agent_orchestrator import Request


AgentToolHandler = Callable[["Request", Dict[str, Any]], Union[Any, Awaitable[Any]]]

logger = logging.getLogger(__name__)


_FAST_RAG_INTENTS = {
    "order_status",
    "logistics",
    "refund",
    "invoice",
    "payment_issue",
    "account_security",
    "technical_login",
    "technical_crash",
}


# 完整向量检索总会返回“最相近”的文档，因此结果非空不等于有证据。
# 这里只为当前内置知识库建立可审计的意图-文档边界；没有对应文档的
# 意图（例如 invoice）应明确返回知识缺口，而不是借用别的文档作答。
_FULL_RAG_TITLES_BY_INTENT = {
    "order_status": {"订单查询"},
    "logistics": {"订单查询", "配送说明"},
    "refund": {"退款政策"},
    "invoice": set(),
    # “支付失败”可由关键词路径命中技术文档；语义召回不足以证明
    # 重复扣款、扣费规则等更宽泛的资金结论。
    "payment_issue": set(),
    "account_security": {"账户安全"},
    "technical_login": {"技术故障排查"},
    "technical_crash": {"技术故障排查"},
}

# 词面命中也必须落在意图允许的文档标题内。否则“重复扣款”中的“订单”
# 可能把《订单查询》误当成资金规则证据。支付失败是唯一可以由技术故障
# 文档支撑的 payment_issue 子场景；重复扣款等资金结论仍属于知识缺口。
_KEYWORD_TITLES_BY_INTENT = {
    **_FULL_RAG_TITLES_BY_INTENT,
    "payment_issue": {"技术故障排查"},
}


def filter_grounded_results(
    intent_value: str,
    retrieval_mode: str,
    results: Any,
) -> tuple[List[Dict[str, Any]], str]:
    """把“召回候选”转换成可以交给生成模型的证据。

    关键词快速路径已经要求实际词面交集。完整向量路径则必须同时满足
    具体意图与文档标题的领域映射；不能用未经验证的原始相似度阈值。
    """
    if not isinstance(results, list) or not results:
        return [], "no_candidates"

    dict_results = [item for item in results if isinstance(item, dict)]
    if retrieval_mode == "fast_keyword":
        allowed_titles = _KEYWORD_TITLES_BY_INTENT.get(str(intent_value or ""))
        if allowed_titles is None:
            return [], "intent_not_specific_enough"
        if not allowed_titles:
            return [], "knowledge_gap"
        accepted = [
            item for item in dict_results
            if str(item.get("title", "")) in allowed_titles
        ]
        if not accepted:
            return [], "intent_title_mismatch"
        return accepted, "lexical_intent_title_match"

    allowed_titles = _FULL_RAG_TITLES_BY_INTENT.get(str(intent_value or ""))
    if allowed_titles is None:
        return [], "intent_not_specific_enough"
    if not allowed_titles:
        return [], "knowledge_gap"

    accepted = [
        item for item in dict_results
        if str(item.get("title", "")) in allowed_titles
    ]
    if not accepted:
        return [], "intent_title_mismatch"
    return accepted, "intent_title_match"


def should_prefetch_knowledge(req: "Request") -> bool:
    """只对高置信度的具体业务意图执行确定性知识预取。"""
    intent_value = getattr(req.intent, "value", req.intent)
    return intent_value in _FAST_RAG_INTENTS and req.intent_confidence >= 0.8


@dataclass(frozen=True)
class AgentToolSpec:
    """Agent 可见工具的定义和执行函数。"""

    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: AgentToolHandler


def make_tool(
    name: str,
    description: str,
    properties: Dict[str, Any],
    handler: AgentToolHandler,
    required: Optional[List[str]] = None,
) -> AgentToolSpec:
    """创建带 JSON Schema 的 Agent 工具。"""
    return AgentToolSpec(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "properties": properties,
            "required": required or [],
            "additionalProperties": False,
        },
        handler=handler,
    )


def inspect_request_context(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """通用客服工具：返回脱敏后的当前请求快照。"""
    return {
        "intent": req.intent.value if req.intent else None,
        "intent_group": req.intent_group,
        "urgency": req.urgency.name if req.urgency else None,
        "intent_confidence": round(req.intent_confidence, 4),
        "entities": req.entities or {},
        "context_available": bool(req.context),
        "requested_focus": str(args.get("focus", "general"))[:40],
    }


def suggest_required_fields(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """通用客服工具：按业务类型计算下一轮只需询问的字段。"""
    intent = req.intent.value if req.intent else "other"
    fields: List[str] = []
    if intent in {"order_status", "logistics"}:
        fields = ["订单号或下单时间"]
    elif intent in {"account", "account_security"}:
        fields = ["登录方式或账号标识", "问题发生时间"]
    elif intent in {"complaint", "request"}:
        fields = ["事件时间", "期望处理方式"]
    elif intent == "other":
        fields = ["希望解决的具体问题"]
    return {
        "intent": intent,
        "required_fields": fields,
        "known_entities": req.entities or {},
    }


def lookup_error_code(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """技术工具：解释常见错误码的排查方向，不声称读取了服务端日志。"""
    code = str(args.get("error_code", "")).upper().strip()
    mapping = {
        "401": ("认证失败", ["确认 Token/API Key 是否过期", "确认请求时间戳和签名", "确认账号登录状态"]),
        "403": ("权限不足", ["确认账号或套餐权限", "确认资源权限和 IP 白名单"]),
        "404": ("资源或路径不存在", ["确认接口路径和环境", "确认资源标识是否正确"]),
        "500": ("服务端处理异常", ["记录 request_id 和发生时间", "检查依赖服务、参数格式和服务端日志"]),
    }
    meaning, steps = mapping.get(
        code,
        ("暂未识别的错误码", ["补充完整错误信息、发生时间和运行环境"]),
    )
    return {
        "error_code": code,
        "meaning": meaning,
        "next_steps": steps,
        "server_log_checked": False,
    }


def build_diagnostic_plan(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """技术工具：生成低风险排障顺序。"""
    environment = str(args.get("environment", "unknown"))[:80]
    reproduced = bool(args.get("reproduced", False))
    steps = [
        "复现并记录完整错误信息",
        "确认网络、DNS、代理和证书",
        "确认版本、配置和权限",
    ]
    if reproduced:
        steps.append("用最小请求复现并记录 request_id")
    return {
        "environment": environment,
        "reproduced": reproduced,
        "diagnostic_steps": steps,
    }


def check_billing_fields(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """账单工具：检查必要核验字段是否齐全。"""
    fields = {
        "order_id": bool(req.entities.get("order_id")),
        "amount": bool(req.entities.get("amount")),
        "date": bool(req.entities.get("date")),
        "payment_channel": bool(args.get("payment_channel")),
    }
    return {
        "fields": fields,
        "missing_fields": [name for name, present in fields.items() if not present],
        "can_confirm_refund": False,
        "reason": "当前工具只做字段检查，不连接订单或支付系统",
    }


def compare_amounts(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """账单工具：只做用户明确提供金额之间的算术。"""
    try:
        first = float(args["amount_a"])
        second = float(args["amount_b"])
    except (KeyError, TypeError, ValueError):
        return {"success": False, "error": "amount_a 和 amount_b 必须是数字"}
    return {
        "success": True,
        "amount_a": first,
        "amount_b": second,
        "difference": round(first - second, 2),
        "interpretation": "仅表示金额差值，不代表重复扣款或退款结论",
    }


def create_handoff_summary(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """升级工具：生成可交给人工客服的结构化摘要。"""
    return {
        "request_id": req.request_id,
        "reason": str(args.get("reason", "需要人工客服继续核验"))[:120],
        "intent": req.intent.value if req.intent else "unknown",
        "urgency": req.urgency.name if req.urgency else "UNKNOWN",
        "entities": req.entities or {},
        "sensitive_data_required": False,
    }


def build_shared_rag_tools(
    tool_manager: Any,
    evidence_selector: Optional[Any] = None,
) -> Dict[str, AgentToolSpec]:
    """构建所有 Agent 可共享的 RAG 工具。"""

    async def search_knowledge_base(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
        query = str(args.get("query") or req.message or "").strip()
        # 每路最多取 5 条候选，避免把过多弱相关文档交给后续生成阶段。
        top_k = max(1, min(int(args.get("top_k", 5) or 5), 5))
        if not query:
            return {"success": False, "error": "query 不能为空", "results": []}
        if tool_manager is None:
            return {"success": False, "error": "RAG 工具未初始化", "results": []}

        intent_value = getattr(req.intent, "value", req.intent)
        retrieval_mode = "full_rag"
        result = None

        # 明确的业务意图先尝试零 LLM 调用的词面检索。只有真正命中
        # 文档时才接受快速结果；空结果自动回退到完整 RAG。
        if intent_value in _FAST_RAG_INTENTS:
            fast_result = await tool_manager.call(
                "knowledge_search",
                {"query": query, "top_k": top_k, "mode": "keyword"},
                use_cache=True,
            )
            if (
                getattr(fast_result, "success", False)
                and isinstance(getattr(fast_result, "data", None), list)
                and fast_result.data
            ):
                result = fast_result
                retrieval_mode = "fast_keyword"

        # 关键词没有命中时，高置信度具体意图先走一次本地向量检索。
        # 只有结果通过既有的“意图-文档标题”证据门控才采用；否则仍回退
        # 到查询改写 + 多路召回 + LLM 重排，不能因提速而放宽证据标准。
        allowed_titles = _FULL_RAG_TITLES_BY_INTENT.get(str(intent_value or ""))
        # 如果关键词路径没有命中，而且当前意图在内置知识库中明确没有
        # 对应文档，就直接报告知识缺口。继续做查询改写和重排只会消耗
        # 两次远程模型调用，也不可能产生可接受证据。
        if result is None and allowed_titles is not None and not allowed_titles:
            return {
                "success": True,
                "query": query,
                "top_k": top_k,
                "results": [],
                "candidate_count": 0,
                "evidence_accepted": False,
                "evidence_gate": "knowledge_gap",
                "reranked": False,
                "retrieval_mode": "knowledge_gap",
            }
        if (
            result is None
            and req.intent_confidence >= 0.8
            and bool(allowed_titles)
        ):
            vector_result = await tool_manager.call(
                "knowledge_search",
                {"query": query, "top_k": 5},
                use_cache=True,
            )
            vector_candidates = (
                vector_result.data
                if getattr(vector_result, "success", False)
                and isinstance(getattr(vector_result, "data", None), list)
                else []
            )
            grounded_vector, _ = filter_grounded_results(
                str(intent_value or ""),
                "fast_vector",
                vector_candidates,
            )
            if grounded_vector:
                result = vector_result
                retrieval_mode = "fast_vector"

        if result is None:
            result = await tool_manager.search_with_rewrite(
                "knowledge_search",
                query,
                top_k=top_k,
            )
        if not getattr(result, "success", False):
            return {
                "success": False,
                "query": query,
                "error": getattr(result, "error", "知识库检索失败"),
                "results": [],
                "candidate_count": 0,
                "evidence_accepted": False,
                "evidence_gate": "retrieval_failed",
                "reranked": False,
            }

        candidate_results = result.data if isinstance(result.data, list) else []
        accepted_results, evidence_gate = filter_grounded_results(
            str(intent_value or ""),
            retrieval_mode,
            candidate_results,
        )
        # 旧规则对没有专属意图标签的知识查询默认拒绝。启用语义选择器后，
        # 这类查询可以把召回候选交给 Judge；具体业务意图仍保留标题白名单，
        # 明确为空的 invoice/payment 等知识缺口也不会被放宽。
        if evidence_selector is not None and candidate_results:
            title_map = (
                _KEYWORD_TITLES_BY_INTENT
                if retrieval_mode == "fast_keyword"
                else _FULL_RAG_TITLES_BY_INTENT
            )
            semantic_allowed_titles = title_map.get(str(intent_value or ""))
            if semantic_allowed_titles is None:
                accepted_results = [
                    item for item in candidate_results
                    if isinstance(item, dict)
                ][:3]
                evidence_gate = "semantic_candidate_pool"

        selector_used = False
        selector_success: Optional[bool] = None
        selected_title: Optional[str] = None
        selection_evidence = ""
        selector_error = ""
        if accepted_results and evidence_selector is not None:
            selector_used = True
            try:
                selection = evidence_selector.select(query, accepted_results[:3])
                if isinstance(selection, Awaitable):
                    selection = await selection
                if not isinstance(selection, dict) or not selection.get("success"):
                    raise RuntimeError("evidence selector returned an invalid result")
                selector_success = True
                selection_evidence = str(selection.get("evidence", "") or "")
                if selection.get("supported") is True:
                    selected_title = str(selection.get("selected_title", "") or "")
                    selected = [
                        item for item in accepted_results
                        if str(item.get("title", "")) == selected_title
                    ]
                    if not selected:
                        raise RuntimeError("evidence selector chose an unavailable document")
                    accepted_results = selected[:1]
                    evidence_gate = "semantic_topk_selected"
                else:
                    accepted_results = []
                    evidence_gate = "semantic_unsupported"
            except Exception as ex:
                selector_success = False
                selector_error = f"{type(ex).__name__}: {ex}"
                accepted_results = []
                evidence_gate = "semantic_selector_failed"
                logger.warning("语义证据选择失败，按无证据处理: %s", ex)

        return {
            "success": True,
            "query": query,
            "top_k": top_k,
            "results": accepted_results,
            "candidate_count": len(candidate_results),
            "evidence_accepted": bool(accepted_results),
            "evidence_gate": evidence_gate,
            "evidence_selector_used": selector_used,
            "evidence_selector_success": selector_success,
            "evidence_selected_title": selected_title,
            "evidence_selection_reason": selection_evidence,
            "evidence_selector_error": selector_error,
            "reranked": bool(getattr(result, "reranked", False)),
            "retrieval_mode": retrieval_mode,
        }

    return {
        "search_knowledge_base": make_tool(
            "search_knowledge_base",
            "检索知识库并返回最相关的文档片段；可用于通用、技术、账单和升级场景。",
            {
                "query": {"type": "string", "description": "用户问题或检索关键词"},
                "top_k": {"type": "integer", "description": "返回结果条数"},
            },
            search_knowledge_base,
            required=["query"],
        )
    }


def general_tools() -> Dict[str, AgentToolSpec]:
    return {
        "inspect_request_context": make_tool(
            "inspect_request_context",
            "查看当前请求的意图、紧急度、实体和上下文可用性；不查询外部业务系统。",
            {"focus": {"type": "string", "description": "希望关注的业务方向"}},
            inspect_request_context,
        ),
        "suggest_required_fields": make_tool(
            "suggest_required_fields",
            "根据当前意图建议下一轮只需向用户补充的字段。",
            {},
            suggest_required_fields,
        ),
    }


def technical_tools() -> Dict[str, AgentToolSpec]:
    return {
        "lookup_error_code": make_tool(
            "lookup_error_code",
            "解释常见 HTTP 错误码的可能含义和低风险排查方向；不会读取服务端日志。",
            {"error_code": {"type": "string", "description": "例如 401、403、500"}},
            lookup_error_code,
            required=["error_code"],
        ),
        "build_diagnostic_plan": make_tool(
            "build_diagnostic_plan",
            "根据运行环境和是否可复现生成排障顺序，不执行修改配置等操作。",
            {
                "environment": {"type": "string", "description": "App、浏览器、服务端或 Docker 等"},
                "reproduced": {"type": "boolean", "description": "问题是否可以稳定复现"},
            },
            build_diagnostic_plan,
            required=["environment", "reproduced"],
        ),
    }


def billing_tools() -> Dict[str, AgentToolSpec]:
    return {
        "check_billing_fields": make_tool(
            "check_billing_fields",
            "检查账单核验字段是否齐全；不连接订单、支付或退款系统。",
            {"payment_channel": {"type": "string", "description": "支付渠道，例如微信、支付宝、银行卡"}},
            check_billing_fields,
        ),
        "compare_amounts": make_tool(
            "compare_amounts",
            "计算用户明确提供的两笔金额差值；不判断是否重复扣款，也不执行退款。",
            {
                "amount_a": {"type": "number", "description": "第一笔金额"},
                "amount_b": {"type": "number", "description": "第二笔金额"},
            },
            compare_amounts,
            required=["amount_a", "amount_b"],
        ),
    }


def escalation_tools() -> Dict[str, AgentToolSpec]:
    return {
        "create_handoff_summary": make_tool(
            "create_handoff_summary",
            "生成交给人工客服的结构化交接摘要，不会创建真实工单。",
            {"reason": {"type": "string", "description": "需要升级的原因"}},
            create_handoff_summary,
        ),
    }
