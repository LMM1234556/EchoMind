import json

from evaluation.run_frozen_grounded_chat_v2 import (
    DEFAULT_CORPUS,
    DEFAULT_DATASET,
    DEFAULT_FREEZE,
    dry_run_plan,
    evaluate_case,
    load_frozen_dataset,
    summarize,
)


def _response(intent="refund", calls=3):
    return {
        "response": "根据退款政策，需要先完成退货流程。",
        "intent": intent,
        "failure_kind": "",
        "llm_call_count": calls,
        "llm_calls_by_component": {
            "intent_recognizer": 1,
            "evidence_selector": 1,
            "agent.billing": max(0, calls - 2),
        },
        "stage_timings": {"api_total_ms": 1000.0},
    }


def _trace(*, found, title="", gate="semantic_topk_selected"):
    return {
        "tool_calls": [
            {
                "tool_name": "search_knowledge_base",
                "evidence_selector_used": True,
                "evidence_selector_success": True,
                "evidence_found": found,
                "evidence_selected_title": title,
                "evidence_gate": gate,
            }
        ]
    }


def test_frozen_grounded_chat_dataset_integrity():
    dataset, freeze = load_frozen_dataset(DEFAULT_DATASET, DEFAULT_FREEZE, DEFAULT_CORPUS)

    assert len(dataset["cases"]) == 24
    assert freeze["dataset_sha256"]
    assert all(case["review"]["status"] == "verified" for case in dataset["cases"])


def test_dry_run_estimates_calls_without_execution():
    dataset = json.loads(DEFAULT_DATASET.read_text(encoding="utf-8"))

    plan = dry_run_plan(dataset, max_llm_calls=80)

    assert plan["mode"] == "dry_run_no_api_calls"
    assert plan["cases"] == 24
    assert plan["estimated_foreground_llm_calls"] == 60
    assert plan["max_observed_foreground_llm_calls"] == 80


def test_supported_case_requires_selector_and_expected_title():
    case = {
        "id": "supported",
        "query": "已发货能直接退款吗",
        "intended_intent": "refund",
        "expected_supported": True,
        "expected_title": "退款政策",
        "evidence_basis": "需要先退货",
        "expected_answer_facts": ["先完成退货流程"],
        "forbidden_claims": ["立即退款"],
    }

    result = evaluate_case(
        case,
        _response(),
        _trace(found=True, title="退款政策"),
        1200.0,
    )

    assert result["checks"]["strict_selection_correct"] is True
    assert result["automatic_pipeline_pass"] is True
    assert result["answer_review"]["status"] == "pending_human_review"


def test_gap_case_requires_successful_semantic_rejection():
    case = {
        "id": "gap",
        "query": "能改退款银行卡吗",
        "intended_intent": "refund",
        "expected_supported": False,
        "expected_title": None,
        "evidence_basis": "没有变更账户规则",
        "expected_answer_facts": ["没有规则"],
        "forbidden_claims": ["可以更换"],
    }

    result = evaluate_case(
        case,
        _response(calls=2),
        _trace(found=False, gate="semantic_unsupported"),
        800.0,
    )

    assert result["checks"]["evidence_classification_correct"] is True
    assert result["checks"]["strict_selection_correct"] is True
    assert result["automatic_pipeline_pass"] is True


def test_summary_separates_automatic_metrics_from_human_faithfulness():
    supported_case = {
        "id": "supported",
        "query": "已发货能直接退款吗",
        "intended_intent": "refund",
        "expected_supported": True,
        "expected_title": "退款政策",
        "evidence_basis": "需要先退货",
        "expected_answer_facts": [],
        "forbidden_claims": [],
    }
    gap_case = {
        "id": "gap",
        "query": "能改退款银行卡吗",
        "intended_intent": "refund",
        "expected_supported": False,
        "expected_title": None,
        "evidence_basis": "没有规则",
        "expected_answer_facts": [],
        "forbidden_claims": [],
    }
    results = [
        evaluate_case(
            supported_case,
            _response(calls=3),
            _trace(found=True, title="退款政策"),
            1200.0,
        ),
        evaluate_case(
            gap_case,
            _response(calls=2),
            _trace(found=False, gate="semantic_unsupported"),
            800.0,
        ),
    ]

    summary = summarize(results)

    assert summary["automatic_metrics"]["strict_evidence_selection_accuracy"] == 1.0
    assert summary["automatic_metrics"]["correct_rejection_rate"] == 1.0
    assert summary["llm_calls"]["foreground_total"] == 5
    assert summary["answer_faithfulness"]["status"] == "pending_human_review"
    assert summary["answer_faithfulness"]["pass_rate"] is None
