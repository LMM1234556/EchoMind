import json
import sys

import pytest

from evaluation.semantic_evidence_judge import (
    build_topk_batch_items,
    build_topk_user_prompt,
    build_batch_items,
    build_user_prompt,
    parse_judge_response,
    parse_topk_judge_response,
    semantic_judge_metrics,
    semantic_selector_metrics,
)
from evaluation.run_semantic_evidence_topk_dev_eval import main as topk_main


def test_prompt_payload_does_not_include_evaluation_labels():
    rows = [{
        "id": "case-1",
        "query": "是否支持某项服务",
        "top_title": "说明",
        "expected_supported": False,
        "candidate_title": "hidden-label",
        "label_basis": "hidden-basis",
    }]
    items = build_batch_items(rows, {"说明": "这里只提供文档正文。"})
    prompt = build_user_prompt(items)

    assert "expected_supported" not in prompt
    assert "candidate_title" not in prompt
    assert "label_basis" not in prompt
    assert "hidden-label" not in prompt
    assert "hidden-basis" not in prompt
    assert "这里只提供文档正文" in prompt


def test_parse_judge_response_requires_exact_ids_and_booleans():
    text = json.dumps({"decisions": [
        {"id": "a", "supported": True, "evidence": "有明确规则"},
        {"id": "b", "supported": False, "evidence": "缺少时效"},
    ]}, ensure_ascii=False)

    decisions = parse_judge_response(text, ["a", "b"])

    assert [item["id"] for item in decisions] == ["a", "b"]
    with pytest.raises(ValueError, match="ids do not match"):
        parse_judge_response(text, ["a", "c"])


def test_parse_judge_response_accepts_code_fence_but_no_extra_text():
    payload = '{"decisions":[{"id":"a","supported":true,"evidence":"依据"}]}'

    assert parse_judge_response(f"```json\n{payload}\n```", ["a"])[0]["supported"] is True
    with pytest.raises(json.JSONDecodeError):
        parse_judge_response(f"说明\n{payload}", ["a"])


def test_semantic_judge_metrics_separates_classification_and_retrieval_error():
    rows = [
        {
            "id": "positive", "expected_supported": True,
            "top_title": "wrong", "candidate_title": "right",
        },
        {
            "id": "negative", "expected_supported": False,
            "top_title": "A", "candidate_title": "A",
        },
    ]
    decisions = {
        "positive": {"supported": True},
        "negative": {"supported": False},
    }

    metrics = semantic_judge_metrics(rows, decisions)

    assert metrics["accuracy"] == 1.0
    assert metrics["strict_decision_accuracy"] == 0.5
    assert metrics["correct_rejection_rate"] == 1.0


def test_topk_prompt_contains_candidates_but_not_labels():
    rows = [{
        "id": "case-1", "query": "问题", "top_titles": ["A", "B", "C"],
        "candidate_title": "hidden", "expected_supported": True,
    }]
    items = build_topk_batch_items(rows, {"A": "正文A", "B": "正文B", "C": "正文C"}, 2)
    prompt = build_topk_user_prompt(items)

    assert [item["title"] for item in items[0]["candidates"]] == ["A", "B"]
    assert "正文A" in prompt and "正文B" in prompt and "正文C" not in prompt
    assert "hidden" not in prompt and "expected_supported" not in prompt


def test_parse_topk_response_enforces_selected_title_contract():
    payload = json.dumps({"decisions": [{
        "id": "a", "supported": True, "selected_title": "B", "evidence": "明确依据",
    }]}, ensure_ascii=False)

    parsed = parse_topk_judge_response(payload, {"a": ["A", "B"]})

    assert parsed[0]["selected_title"] == "B"
    invalid = json.dumps({"decisions": [{
        "id": "a", "supported": False, "selected_title": "A", "evidence": "不充分",
    }]}, ensure_ascii=False)
    with pytest.raises(ValueError, match="null selected_title"):
        parse_topk_judge_response(invalid, {"a": ["A", "B"]})


def test_semantic_selector_metrics_requires_correct_document_selection():
    rows = [
        {"id": "p", "expected_supported": True, "candidate_title": "B"},
        {"id": "n", "expected_supported": False, "candidate_title": "A"},
    ]
    decisions = {
        "p": {"supported": True, "selected_title": "A"},
        "n": {"supported": False, "selected_title": None},
    }

    metrics = semantic_selector_metrics(rows, decisions)

    assert metrics["classification_accuracy"] == 1.0
    assert metrics["strict_selection_accuracy"] == 0.5
    assert metrics["correct_rejection_rate"] == 1.0


def test_topk_runner_cli_can_build_parser(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["topk-runner", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        topk_main()

    assert exc_info.value.code == 0
