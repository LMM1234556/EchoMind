"""评测证据、失败计数与运行隔离检查，使用离线替身。"""

import asyncio
import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from core.intent_recognizer import IntentCategory
from evaluation import evaluator as module
from evaluation.evaluator import EndToEndEvaluator, IntentTestCase, LLMJudge, QualityScores


class StubOrchestrator:
    def __init__(self):
        self.requests = []

    async def run(self, request):
        self.requests.append(request)
        return SimpleNamespace(response="审核通常需 1–3 个工作日。", agent_type=SimpleNamespace(value="billing"), intent=IntentCategory.REFUND)


class StubJudge:
    def __init__(self):
        self.calls = []

    async def judge(self, question, response, context=None, **kwargs):
        self.calls.append(dict(question=question, context=context, **kwargs))
        if question == "评分失败":
            return QualityScores(.5, .5, .5, .5, judge_failed=True, error="timeout")
        return QualityScores(.9, .9, .9, .9)


def make_evaluator():
    evaluator = object.__new__(EndToEndEvaluator)
    evaluator._orchestrator = StubOrchestrator()
    evaluator._judge = StubJudge()
    evaluator._history = []
    evaluator._baseline = None
    evaluator._baseline_path = None

    async def recognize(cases):
        return dict(total=len(cases), correct=1, completed=len(cases), failed=0, accuracy=.5, macro_f1=.5)

    evaluator._intent_evaluator = SimpleNamespace(evaluate=recognize)
    return evaluator


def evidence_case(question="退款审核多久？"):
    return dict(case_id="refund-reference", question=question, reference_evidence=["退款申请提交后，系统会在 1-3 个工作日内审核。"], acceptance_criteria=["保留工作日单位与审核阶段"])


def test_reference_and_criteria_reach_judge_and_remain_reviewable():
    evaluator = make_evaluator()
    case = evidence_case()
    report = asyncio.run(evaluator.run(dialog_cases=[case]))
    call = evaluator._judge.calls[0]
    assert call["reference_evidence"] == case["reference_evidence"]
    assert call["acceptance_criteria"] == case["acceptance_criteria"]
    result = report.results[0]
    assert result.test_id == case["case_id"]
    assert result.metadata["grading_scope"] == "evidence_aligned"
    assert result.metadata["human_review_status"] == "pending"
    assert report.summary["dialog"]["human_review_required"] == 1


def test_conversation_only_scoring_has_explicit_scope():
    report = asyncio.run(make_evaluator().run(dialog_cases=[dict(question="你好")]))
    assert report.results[0].metadata["grading_scope"] == "conversation_only"
    assert report.summary["dialog"]["evidence_aligned"] == 0
    assert report.summary["dialog"]["human_review_required"] == 1


def test_failed_judge_excluded_from_average_and_retained_in_denominator():
    evaluator = make_evaluator()
    report = asyncio.run(evaluator.run(dialog_cases=[evidence_case(), evidence_case("评分失败")]))
    assert report.avg_scores["accuracy"] == .9
    assert report.results[1].scores == {}
    assert report.results[1].passed is False
    assert report.summary["dialog"]["total"] == 2
    assert report.summary["dialog"]["judge_failed"] == 1
    assert report.summary["dialog"]["auxiliary_pass_rate"] == .5


def test_intent_counts_are_separate_from_dialog_reference_scores():
    evaluator = make_evaluator()
    report = asyncio.run(evaluator.run(intent_cases=[IntentTestCase("a", "refund"), IntentTestCase("b", "refund")], dialog_cases=[evidence_case()]))
    assert report.summary["intent"]["correct"] == 1
    assert report.summary["intent"]["total"] == 2
    assert report.summary["dialog"]["auxiliary_passed"] == 1
    assert report.summary["dialog"]["total"] == 1


def test_each_run_isolates_default_users_and_preserves_multiturn_history():
    evaluator = make_evaluator()
    case = dict(turns=["我想退款", "订单号是 A12345"])
    asyncio.run(evaluator.run(dialog_cases=[case]))
    asyncio.run(evaluator.run(dialog_cases=[case]))
    first, second, third, fourth = evaluator._orchestrator.requests
    assert first.user_id == second.user_id
    assert first.conv_id == second.conv_id
    assert third.user_id == fourth.user_id
    assert first.user_id != third.user_id
    assert first.conv_id != third.conv_id
    assert "我想退款" in evaluator._judge.calls[1]["context"]


def test_regression_comparison_requires_same_case_evidence_and_protocol():
    evaluator = make_evaluator()
    report = asyncio.run(evaluator.run(dialog_cases=[evidence_case()]))
    assert evaluator._detect_regressions({"accuracy": .1}, report.case_set_id)
    changed = evidence_case()
    changed["reference_evidence"] = ["新政策"]
    changed_id = evaluator._case_set_id([], [changed])
    assert changed_id != report.case_set_id
    assert evaluator._detect_regressions({"accuracy": .1}, changed_id) == []
    legacy = evaluator._report_from_dict(dict(avg_scores={"accuracy": 1.0}))
    evaluator._history = []
    evaluator._baseline = legacy
    assert evaluator._detect_regressions({"accuracy": .1}, report.case_set_id) == []


def test_builtin_examples_have_explicit_reference_and_acceptance_conditions():
    assert all(case["reference_evidence"] and case["acceptance_criteria"] for case in module.DEFAULT_DIALOG_CASES)
    assert "1-3 个工作日" in str(module.DEFAULT_DIALOG_CASES[-1]["reference_evidence"])


@pytest.mark.parametrize("invalid", [None, "0.9", True, -0.1, 1.1, float("nan"), float("inf")])
def test_malformed_judge_scores_are_recorded_as_failures(monkeypatch, invalid):
    data = dict(relevance=.9, accuracy=invalid, completeness=.9, helpfulness=.9)

    async def reply(*args, **kwargs):
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(data))])

    monkeypatch.setattr(module, "tracked_messages_create", reply)
    result = asyncio.run(LLMJudge(None, "offline").judge("问题", "回答"))
    assert result.judge_failed is True


def test_judge_prompt_contains_case_reference_and_acceptance(monkeypatch):
    prompts = []

    async def reply(*args, **kwargs):
        prompts.append(kwargs["messages"][0]["content"])
        return SimpleNamespace(content=[SimpleNamespace(type="text", text='{"relevance": 1, "accuracy": 0.5, "completeness": 1, "helpfulness": 1}')])

    monkeypatch.setattr(module, "tracked_messages_create", reply)
    result = asyncio.run(LLMJudge(None, "offline").judge("审核多久", "三天内保证到账", reference_evidence=["审核需 1–3 个工作日"], acceptance_criteria=["区分审核与到账"]))
    assert result.judge_failed is False
    assert "审核需 1–3 个工作日" in prompts[0]
    assert "区分审核与到账" in prompts[0]
    assert "证据外具体承诺" in prompts[0]


def test_eval_api_preserves_reference_fields():
    # API 模块会读取本机 dotenv；将这些环境变更限制在本测试内。
    with patch.dict(os.environ):
        from api.main import EvalDialogInput

        case = evidence_case()
        assert EvalDialogInput(**case).model_dump(exclude_none=True) == case
