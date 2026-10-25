import asyncio
from types import SimpleNamespace

from core.intent_recognizer import IntentCategory
from evaluation.evaluator import IntentEvaluator, IntentTestCase


class FakeRecognizer:
    async def recognize(self, message):
        if message == "slow":
            await asyncio.sleep(0.1)
        intent = IntentCategory.QUERY if message != "complaint" else IntentCategory.COMPLAINT
        return SimpleNamespace(
            intent=intent,
            confidence=0.9,
            reasoning="fake result",
        )


def test_intent_evaluator_records_timeout_without_creating_fake_class():
    evaluator = IntentEvaluator(FakeRecognizer(), case_timeout_s=0.02, max_concurrency=2)
    cases = [
        IntentTestCase("normal", "query"),
        IntentTestCase("complaint", "complaint"),
        IntentTestCase("slow", "query"),
    ]

    report = asyncio.run(evaluator.evaluate(cases))

    assert report["total"] == 3
    assert report["completed"] == 2
    assert report["failed"] == 1
    assert report["completion_rate"] == 0.6667
    assert report["accuracy"] == 0.6667
    assert "__timeout__" not in report["per_class"]
    assert report["cases"][2]["predicted"] == "__timeout__"
    assert report["cases"][2]["failed"] is True


def test_intent_evaluator_preserves_input_order_under_concurrency():
    evaluator = IntentEvaluator(FakeRecognizer(), case_timeout_s=1, max_concurrency=2)
    cases = [
        IntentTestCase("slow", "query"),
        IntentTestCase("complaint", "complaint"),
    ]

    report = asyncio.run(evaluator.evaluate(cases))

    assert [case["message"] for case in report["cases"]] == ["slow", "complaint"]
