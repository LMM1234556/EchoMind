import asyncio

from core.intent_recognizer import (
    IntentCategory,
    IntentRecognizer,
    _INTENT_DEFINITIONS,
    _TEMPLATES,
)


def _recognizer():
    return IntentRecognizer(api_key="test-key", base_url="http://127.0.0.1:9")


def test_handoff_and_escalation_few_shots_do_not_overlap():
    escalation = set(_TEMPLATES[IntentCategory.ESCALATION])
    handoff = set(_TEMPLATES[IntentCategory.HUMAN_HANDOFF])

    assert escalation.isdisjoint(handoff)


def test_intent_definitions_cover_the_ontology_and_account_boundary():
    assert set(_INTENT_DEFINITIONS) == set(IntentCategory)
    assert "绑定手机号" in _INTENT_DEFINITIONS[IntentCategory.ACCOUNT]
    assert "异常登录" in _INTENT_DEFINITIONS[IntentCategory.ACCOUNT_SECURITY]


def test_explicit_human_request_uses_handoff_intent():
    result = _recognizer()._pattern_recognize("机器人解决不了，请接人工坐席")

    assert result["intent"] is IntentCategory.HUMAN_HANDOFF


def test_supervisor_request_uses_escalation_intent():
    result = _recognizer()._pattern_recognize("请把这个问题升级给主管处理")

    assert result["intent"] is IntentCategory.ESCALATION


def test_dissatisfaction_without_handoff_stays_complaint():
    result = _recognizer()._pattern_recognize("我等了很久，体验太差了")

    assert result["intent"] is IntentCategory.COMPLAINT


def test_generic_few_shots_match_their_declared_boundaries():
    assert _TEMPLATES[IntentCategory.QUERY] == [
        "客服服务时间是什么？", "平台支持哪些服务？", "在哪里查看使用说明？",
    ]
    assert _TEMPLATES[IntentCategory.BILLING] == [
        "账单金额如何计算？", "套餐费用怎么收取？", "查看本月账单明细",
    ]
    assert _TEMPLATES[IntentCategory.TECHNICAL] == [
        "接口请求超时", "数据同步失败", "配置保存后未生效",
    ]


def test_negated_crash_keywords_stay_general_technical():
    recognizer = _recognizer()

    for message in (
        "接口响应很慢，但没有报错也没有宕机",
        "页面加载超时，应用并未闪退",
        "服务偶尔延迟，不存在 500 或崩溃现象",
    ):
        result = recognizer._pattern_recognize(message)
        assert result["intent"] is IntentCategory.TECHNICAL


def test_affirmative_crash_keyword_stays_specific():
    result = _recognizer()._pattern_recognize("提交时页面报 500 错误")

    assert result["intent"] is IntentCategory.TECHNICAL_CRASH


def test_failed_llm_does_not_bypass_confidence_threshold_or_pollute_cache():
    recognizer = _recognizer()

    async def failed_llm(message, history):
        return {
            "intent": IntentCategory.OTHER,
            "confidence": 0.0,
            "reasoning": "LLM 失败 (test)",
            "failed": True,
        }

    async def weak_embedding(message):
        return {"intent": IntentCategory.HUMAN_HANDOFF, "confidence": 0.2}

    recognizer._llm_recognize = failed_llm
    recognizer._embedding_recognize = weak_embedding

    result = asyncio.run(recognizer.recognize("没有规则命中的模糊文本"))

    assert result.intent is IntentCategory.OTHER
    assert result.confidence == 0.2
    assert result.source_scores["llm_failed"] == 1.0
    assert recognizer.cache_stats["size"] == 0


def test_failed_llm_can_use_strong_pattern_fallback():
    recognizer = _recognizer()
    intent, confidence, scores = recognizer._vote(
        {"intent": IntentCategory.OTHER, "confidence": 0.0, "failed": True},
        {"intent": IntentCategory.QUERY, "confidence": 0.2},
        {"intent": IntentCategory.HUMAN_HANDOFF, "confidence": 0.5},
    )

    assert intent is IntentCategory.HUMAN_HANDOFF
    assert confidence == 0.5
    assert scores["llm_failed"] == 1.0
