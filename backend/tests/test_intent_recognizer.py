from core.intent_recognizer import IntentCategory, IntentRecognizer, _TEMPLATES


def _recognizer():
    return IntentRecognizer(api_key="test-key", base_url="http://127.0.0.1:9")


def test_handoff_and_escalation_few_shots_do_not_overlap():
    escalation = set(_TEMPLATES[IntentCategory.ESCALATION])
    handoff = set(_TEMPLATES[IntentCategory.HUMAN_HANDOFF])

    assert escalation.isdisjoint(handoff)


def test_explicit_human_request_uses_handoff_intent():
    result = _recognizer()._pattern_recognize("机器人解决不了，请接人工坐席")

    assert result["intent"] is IntentCategory.HUMAN_HANDOFF


def test_supervisor_request_uses_escalation_intent():
    result = _recognizer()._pattern_recognize("请把这个问题升级给主管处理")

    assert result["intent"] is IntentCategory.ESCALATION


def test_dissatisfaction_without_handoff_stays_complaint():
    result = _recognizer()._pattern_recognize("我等了很久，体验太差了")

    assert result["intent"] is IntentCategory.COMPLAINT
