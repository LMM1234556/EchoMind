"""Prompting, parsing, and metrics for an evidence-only semantic judge."""
from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Sequence


SYSTEM_PROMPT = """你是严格的证据充分性判定器。你只能依据提供的单篇文档判断它是否足以回答问题。

判定规则：
1. 只有文档明确包含回答问题所需的全部关键信息时，supported 才能为 true。
2. 仅仅主题相关、出现相同业务词、可以依靠常识猜测，或只回答了问题的一部分，都必须判为 false。
3. 允许文档明确给出规则后进行一步确定性的算术换算；不得补充文档之外的业务规则。
4. 不得使用你自己的知识，不得因为问题看起来合理就判为 true。
5. evidence 只能简短转述文档中的依据；若不充分，指出缺少的具体信息。

仅输出一个 JSON 对象，格式为：
{"decisions":[{"id":"原始ID","supported":true,"evidence":"简短依据"}]}
不要输出 Markdown、代码围栏或额外说明。"""


TOPK_SYSTEM_PROMPT = """你是严格的多文档证据选择器。对于每个问题，你只能检查提供的候选文档，并选择一篇足以回答问题的文档；如果没有任何一篇足够，就必须拒绝。

判定规则：
1. 只有某篇文档明确包含回答问题所需的全部关键信息时，supported 才能为 true，并将 selected_title 设置为该文档的原始标题。
2. 如果候选文档都只是主题相关、缺少具体规则、需要常识猜测，或只能回答问题的一部分，supported 必须为 false，selected_title 必须为 null。
3. 允许依据文档明确规则做一步确定性算术换算；禁止使用外部知识或补充业务规则。
4. 多篇文档都相关时，只选择证据最直接、最完整的一篇。不得拼接互不完整的内容制造答案。
5. evidence 只能简短转述所选文档的依据；拒绝时说明缺少的具体信息。

仅输出一个 JSON 对象，格式为：
{"decisions":[{"id":"原始ID","supported":true,"selected_title":"原始文档标题","evidence":"简短依据"},{"id":"另一个ID","supported":false,"selected_title":null,"evidence":"缺少的信息"}]}
不要输出 Markdown、代码围栏或额外说明。"""


def build_batch_items(
    rankings: Sequence[Mapping[str, Any]],
    documents_by_title: Mapping[str, str],
) -> list[Dict[str, str]]:
    items = []
    for row in rankings:
        top_title = str(row.get("top_title") or "").strip()
        if top_title not in documents_by_title:
            raise ValueError(f"top document is missing: {top_title!r}")
        items.append({
            "id": str(row["id"]),
            "question": str(row["query"]),
            "document_title": top_title,
            "document": documents_by_title[top_title],
        })
    return items


def build_user_prompt(items: Sequence[Mapping[str, str]]) -> str:
    return "请逐项判断以下问题：\n" + json.dumps(
        list(items), ensure_ascii=False, separators=(",", ":")
    )


def build_topk_batch_items(
    rankings: Sequence[Mapping[str, Any]],
    documents_by_title: Mapping[str, str],
    top_k: int,
) -> list[Dict[str, Any]]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    items: list[Dict[str, Any]] = []
    for row in rankings:
        titles = [str(title) for title in list(row.get("top_titles") or [])[:top_k]]
        if not titles:
            raise ValueError(f"ranking has no candidate documents: {row.get('id')}")
        if len(titles) != len(set(titles)):
            raise ValueError("candidate document titles must be unique")
        missing = [title for title in titles if title not in documents_by_title]
        if missing:
            raise ValueError(f"candidate documents are missing: {missing}")
        items.append({
            "id": str(row["id"]),
            "question": str(row["query"]),
            "candidates": [
                {"title": title, "document": documents_by_title[title]}
                for title in titles
            ],
        })
    return items


def build_topk_user_prompt(items: Sequence[Mapping[str, Any]]) -> str:
    return "请逐项选择证据充分的文档或拒绝：\n" + json.dumps(
        list(items), ensure_ascii=False, separators=(",", ":")
    )


def parse_judge_response(text: str, expected_ids: Sequence[str]) -> list[Dict[str, Any]]:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    payload = json.loads(cleaned)
    raw_decisions = payload.get("decisions") if isinstance(payload, dict) else None
    if not isinstance(raw_decisions, list):
        raise ValueError("judge response must contain a decisions list")

    decisions = []
    seen = set()
    for item in raw_decisions:
        if not isinstance(item, dict):
            raise ValueError("each judge decision must be an object")
        case_id = str(item.get("id", "")).strip()
        supported = item.get("supported")
        evidence = str(item.get("evidence", "")).strip()
        if not case_id or not isinstance(supported, bool) or not evidence:
            raise ValueError("judge decision has invalid required fields")
        if case_id in seen:
            raise ValueError("judge response contains duplicate ids")
        seen.add(case_id)
        decisions.append({"id": case_id, "supported": supported, "evidence": evidence})

    if set(seen) != set(expected_ids) or len(decisions) != len(expected_ids):
        raise ValueError("judge response ids do not match the requested batch")
    order = {case_id: index for index, case_id in enumerate(expected_ids)}
    return sorted(decisions, key=lambda item: order[item["id"]])


def parse_topk_judge_response(
    text: str,
    allowed_titles_by_id: Mapping[str, Sequence[str]],
) -> list[Dict[str, Any]]:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    payload = json.loads(cleaned)
    raw_decisions = payload.get("decisions") if isinstance(payload, dict) else None
    if not isinstance(raw_decisions, list):
        raise ValueError("judge response must contain a decisions list")

    decisions = []
    seen = set()
    for item in raw_decisions:
        if not isinstance(item, dict):
            raise ValueError("each judge decision must be an object")
        case_id = str(item.get("id", "")).strip()
        supported = item.get("supported")
        selected_title = item.get("selected_title")
        evidence = str(item.get("evidence", "")).strip()
        if not case_id or not isinstance(supported, bool) or not evidence:
            raise ValueError("judge decision has invalid required fields")
        if case_id in seen:
            raise ValueError("judge response contains duplicate ids")
        if case_id not in allowed_titles_by_id:
            raise ValueError("judge response contains an unexpected id")
        if supported:
            if selected_title not in allowed_titles_by_id[case_id]:
                raise ValueError("supported decision must select an allowed title")
        elif selected_title is not None:
            raise ValueError("unsupported decision must use a null selected_title")
        seen.add(case_id)
        decisions.append({
            "id": case_id,
            "supported": supported,
            "selected_title": selected_title,
            "evidence": evidence,
        })

    if seen != set(allowed_titles_by_id) or len(decisions) != len(allowed_titles_by_id):
        raise ValueError("judge response ids do not match the requested batch")
    order = {case_id: index for index, case_id in enumerate(allowed_titles_by_id)}
    return sorted(decisions, key=lambda item: order[item["id"]])


def semantic_judge_metrics(
    rankings: Sequence[Mapping[str, Any]],
    decisions: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    tp = fp = fn = tn = strict_correct = 0
    for row in rankings:
        expected = bool(row.get("expected_supported"))
        decision = decisions.get(str(row["id"]))
        if decision is None:
            raise ValueError(f"missing semantic decision for {row['id']}")
        predicted = bool(decision["supported"])
        if expected and predicted:
            tp += 1
        elif not expected and predicted:
            fp += 1
        elif expected:
            fn += 1
        else:
            tn += 1
        if expected:
            strict_correct += int(
                predicted and row.get("top_title") == row.get("candidate_title")
            )
        else:
            strict_correct += int(not predicted)

    def safe_div(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator else 0.0

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    positive_f1 = safe_div(2 * precision * recall, precision + recall)
    negative_precision = safe_div(tn, tn + fn)
    rejection = safe_div(tn, tn + fp)
    negative_f1 = safe_div(
        2 * negative_precision * rejection,
        negative_precision + rejection,
    )
    total = len(rankings)
    return {
        "total": total,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": round(safe_div(tp + tn, total), 4),
        "strict_decision_accuracy": round(safe_div(strict_correct, total), 4),
        "supported_precision": round(precision, 4),
        "supported_recall": round(recall, 4),
        "supported_f1": round(positive_f1, 4),
        "correct_rejection_rate": round(rejection, 4),
        "macro_f1": round((positive_f1 + negative_f1) / 2, 4),
    }


def semantic_selector_metrics(
    rankings: Sequence[Mapping[str, Any]],
    decisions: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    tp = fp = fn = tn = selected_correct = 0
    for row in rankings:
        expected = bool(row.get("expected_supported"))
        decision = decisions.get(str(row["id"]))
        if decision is None:
            raise ValueError(f"missing semantic decision for {row['id']}")
        predicted = bool(decision["supported"])
        selected_title = decision.get("selected_title")
        if expected and predicted:
            tp += 1
        elif not expected and predicted:
            fp += 1
        elif expected:
            fn += 1
        else:
            tn += 1
        if expected:
            selected_correct += int(predicted and selected_title == row.get("candidate_title"))
        else:
            selected_correct += int(not predicted and selected_title is None)

    def safe_div(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator else 0.0

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    positive_f1 = safe_div(2 * precision * recall, precision + recall)
    negative_precision = safe_div(tn, tn + fn)
    rejection = safe_div(tn, tn + fp)
    negative_f1 = safe_div(
        2 * negative_precision * rejection,
        negative_precision + rejection,
    )
    total = len(rankings)
    return {
        "total": total,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "classification_accuracy": round(safe_div(tp + tn, total), 4),
        "strict_selection_accuracy": round(safe_div(selected_correct, total), 4),
        "supported_precision": round(precision, 4),
        "supported_recall": round(recall, 4),
        "supported_f1": round(positive_f1, 4),
        "correct_rejection_rate": round(rejection, 4),
        "macro_f1": round((positive_f1 + negative_f1) / 2, 4),
    }
